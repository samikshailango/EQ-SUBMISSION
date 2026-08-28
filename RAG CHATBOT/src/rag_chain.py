import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    GEMINI_API_KEY, MODEL_NAME, LLM_TEMPERATURE, REFUSAL_MESSAGE,
    ENABLE_CLARIFICATION, CLARIFICATION_SCORE_THRESHOLD,
    CLARIFICATION_MIN_WORDS, ENABLE_FOLLOWUPS, NUM_FOLLOWUPS,
    RETRIEVAL_FETCH_K, TOP_K, CONTEXT_RELEVANCE_MIN_SCORE,
    ENABLE_GREETING, SUGGESTED_TOPICS,
    CONFIDENCE_HIGH_SCORE, CONFIDENCE_MEDIUM_SCORE,
)
from vectorstore import get_retriever, hybrid_search_with_scores
from pii_filter import query_requests_pii, redact_pii
from eval_db import log_interaction
from logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a mortgage document assistant. Answer ONLY using
the provided context chunks. Rules:

1. If the answer is not contained in the context, say you don't have
   information on that -- do not guess or use outside knowledge.
2. Never state a specific individual's name, home/property address, personal
   email, personal phone number, Social Security number, or a specific
   loan/account number, even if it appears in the context. Refer to people
   by role instead (e.g. "the loan officer", "the settlement agent") and
   to loans generically ("this loan") if the document names someone or
   identifies a loan number.
3. Start your response with the direct answer to the question -- no
   preamble like "Based on the provided context" or "According to the
   document". Then cite. Be concise: only include information that
   actually answers what was asked, not other facts from the context just
   because they were nearby.
4. Some of the context chunks below may not actually be relevant to the
   question (retrieval isn't perfect) -- silently ignore any chunk that
   doesn't help answer the question, and never mention chunk relevance to
   the user. Do not invent a citation tag that wasn't given to you in the
   context below.
5. Do NOT include citation tags, bracketed source references, or filenames
   in your answer text (e.g. do not write "[sample_loan_estimate.txt#0]" or
   similar). The UI shows sources separately in its own "Sources" panel, so
   your answer should read as plain, standalone prose with no source
   markers of any kind.
Context:
{context}
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _format_context(docs):
    parts = []
    for d in docs:
        tag = d.metadata.get("chunk_id", "unknown#0")
        parts.append(f"[{tag}]\n{d.page_content}")
    return "\n\n".join(parts)


def _get_llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY,
                                   temperature=LLM_TEMPERATURE)


_FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are helping a user explore mortgage/loan documents. Given the "
     "question they just asked, the answer they were given, and the "
     "source context, suggest {n} short, natural follow-up questions "
     "they might reasonably ask next. Rules: base every suggestion on "
     "topics that actually appear in the context (don't invent loan "
     "details that aren't there); keep each under 12 words; no numbering, "
     "no quotes, no explanation -- output ONLY the questions, one per line."),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nContext:\n{context}"),
])


def _generate_followups(question: str, answer: str, docs, n: int = NUM_FOLLOWUPS) -> list[str]:
    """LLM-suggested follow-up questions grounded in the retrieved chunks,
    so the chat stays a conversation instead of one-shot Q&A. Best-effort:
    on any failure we just return no suggestions rather than break the
    turn.

    These are generated from the same raw retrieved context as the
    answer, so they carry the same PII exposure risk (e.g. the model
    proposing "What is Michael Jones's role in this transaction?"
    grounded in a chunk that names the borrower) -- each suggestion goes
    through the same redact_pii() guardrail as the answer and source
    snippets before it's returned. A redacted suggestion that reads
    awkwardly is simply dropped rather than shown half-redacted."""
    if not ENABLE_FOLLOWUPS or not docs:
        return []
    try:
        chain = _FOLLOWUP_PROMPT | _get_llm() | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "answer": answer,
            "context": _format_context(docs),
            "n": n,
        })
        lines = [ln.strip(" -*\u2022\t") for ln in raw.splitlines() if ln.strip()]
        safe_lines = []
        for ln in lines[:n]:
            redacted, redacted_types = redact_pii(ln)
            if redacted_types:
                continue  # drop rather than show a suggestion with [REDACTED-...] in it
            safe_lines.append(redacted)
        return safe_lines
    except Exception:
        return []


def _confidence_label(best_score: float) -> str:
    """Map the fused hybrid retrieval score to a high/medium/low label.
    NOT shown in the UI (a raw score-derived badge isn't actionable for an
    end user) -- kept purely as an internal/logged signal for later
    analysis, e.g. correlating logged interactions' retrieval strength
    with their RAGAS scores or thumbs-down feedback."""
    if best_score >= CONFIDENCE_HIGH_SCORE:
        return "high"
    if best_score >= CONFIDENCE_MEDIUM_SCORE:
        return "medium"
    return "low"


_GREETING_RE = re.compile(
    r"^(hi|hello|hey|hiya|howdy|yo|sup|greetings"
    r"|good\s*(morning|afternoon|evening)"
    r"|what'?s\s*up)[\s!.,]*$",
    re.IGNORECASE,
)


def _is_greeting(question: str) -> bool:
    """Simple, deliberately narrow small-talk detector -- only fires on
    messages that are ENTIRELY a greeting (e.g. "hi", "good morning!"),
    not "hi, what's the interest rate" (that's a real question and should
    be answered as one)."""
    return bool(_GREETING_RE.match(question.strip()))


def _greeting_response() -> dict:
    """Friendly, conversational reply to a plain greeting -- welcomes the
    user and offers example topics (as tappable follow-up buttons in the
    UI, reusing the same "followups" mechanism as post-answer suggestions)
    instead of dumping raw document filenames at them."""
    answer = (
        "Hi there! \U0001F44B I'm here to help answer questions about your "
        "mortgage and loan documents -- things like interest rates, closing "
        "costs, escrow, or how a Loan Estimate compares to a Closing "
        "Disclosure. What would you like to know?"
    )
    return {
        "answer": answer,
        "sources": [],
        "blocked": False,
        "clarification": False,
        "greeting": True,
        "redacted_entities": [],
        "followups": list(SUGGESTED_TOPICS[:4]),
        "confidence": None,
        "interaction_id": None,
    }


def _clarification_response(question: str, reason: str) -> dict:
    """Build an interactive "help me narrow this down" response instead of
    forcing an answer out of a weak/ambiguous retrieval match. Offers
    example topics/questions (not raw document filenames -- a list of PDF
    names isn't a helpful answer to a confused user) as tappable
    follow-ups, same mechanism as _greeting_response."""
    prompt = (
        f"{reason} Could you tell me a bit more about what you're looking "
        "for? For example, you could ask about a specific loan term (like "
        "the interest rate, APR, or closing costs) or a general mortgage "
        "concept (like how escrow works)."
    )

    return {
        "answer": prompt,
        "sources": [],
        "blocked": False,
        "clarification": True,
        "redacted_entities": [],
        "followups": list(SUGGESTED_TOPICS[:4]),
        "confidence": None,
        "interaction_id": None,
    }



_NO_INFO_RE = re.compile(
    r"(don'?t|do\s*not|doesn'?t|does\s*not)\s+have\s+(?:any\s+|specific\s+|that\s+|the\s+|enough\s+)?information"
    r"|no\s+information\s+(?:on|about|regarding)"
    r"|not\s+(?:contained|found|mentioned|covered|available)\s+in\s+(?:this|these|the|my)?\s*(?:provided\s+)?(?:mortgage\s+)?(?:context|document|documents)"
    r"|(?:can'?t|cannot|couldn'?t|unable\s+to)\s+find\s+(?:that|this|any)\s+information"
    r"|(?:I\s+)?(?:do\s*not|don'?t)\s+know\s+that",
    re.IGNORECASE,
)


def _is_no_info_answer(answer: str) -> bool:
    """True if the model answered with some form of "I don't have
    information on that" per SYSTEM_PROMPT rule 1, rather than an actual
    answer grounded in the retrieved context."""
    return bool(_NO_INFO_RE.search(answer))


def answer_question(question: str) -> dict:
    """
    Returns a dict:
      {
        "answer": str,             # final, PII-redacted answer text
        "sources": [ {source, page, chunk_id, snippet}, ... ],
        "blocked": bool,           # True if refused due to PII-seeking query
        "clarification": bool,     # True if we asked a clarifying question
                                    #   instead of answering (interactivity)
        "redacted_entities": [...] # entity types redacted from the answer, if any
        "followups": [...]         # LLM-suggested next questions, if any
        "confidence": str|None,    # "high"/"medium"/"low", or None if N/A
        "interaction_id": str|None # eval_data row id, for thumbs up/down feedback
      }
    """
    turn_start = time.monotonic()
    logger.info("question received (%d chars)", len(question))


    if ENABLE_GREETING and _is_greeting(question):
        result = _greeting_response()
        result["interaction_id"] = log_interaction(question, result["answer"], [], [], clarification=True)
        logger.info("greeting handled in %.3fs", time.monotonic() - turn_start)
        return result

   
    if query_requests_pii(question):
        result = {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "blocked": True,
            "clarification": False,
            "redacted_entities": [],
            "followups": [],
            "confidence": None,
        }
        result["interaction_id"] = log_interaction(question, result["answer"], [], [], blocked=True)
        logger.warning("query blocked by PII guardrail in %.3fs", time.monotonic() - turn_start)
        return result

    if ENABLE_CLARIFICATION and len(question.split()) < CLARIFICATION_MIN_WORDS:
        result = _clarification_response(
            question, "That's a pretty short question -- I want to make sure I answer the right thing."
        )
        result["interaction_id"] = log_interaction(question, result["answer"], [], [], clarification=True)
        logger.info("clarification requested (too short) in %.3fs", time.monotonic() - turn_start)
        return result

    scored = hybrid_search_with_scores(question, k=RETRIEVAL_FETCH_K)
    scored.sort(key=lambda pair: pair[1], reverse=True)  

    if not scored:
        result = _clarification_response(question, "I couldn't find anything matching that in the mortgage documents.")
        result["interaction_id"] = log_interaction(question, result["answer"], [], [], clarification=True)
        logger.info("clarification requested (no retrieval hits) in %.3fs", time.monotonic() - turn_start)
        return result


    best_score = scored[0][1]
    if ENABLE_CLARIFICATION and best_score < CLARIFICATION_SCORE_THRESHOLD:
        result = _clarification_response(
            question, "I'm not confident that matches what's in the mortgage documents."
        )
        result["interaction_id"] = log_interaction(question, result["answer"], [], [], clarification=True)
        logger.info("clarification requested (weak match, score=%.3f) in %.3fs",
                    best_score, time.monotonic() - turn_start)
        return result

    relevant = [(d, s) for d, s in scored if s >= CONTEXT_RELEVANCE_MIN_SCORE]
    if len(relevant) < TOP_K:
        relevant = scored[:TOP_K]
    docs = [d for d, _s in relevant[:RETRIEVAL_FETCH_K]]

    logger.info(
        "retrieved %d candidate(s), kept %d after relevance filter (best_score=%.3f)",
        len(scored), len(docs), best_score,
    )

    context = _format_context(docs)

    # --- Generate ---
    chain = _prompt | _get_llm() | StrOutputParser()
    llm_start = time.monotonic()
    try:
        raw_answer = chain.invoke({"context": context, "question": question})
    except Exception:
        logger.exception("LLM generation failed")
        raise
    logger.info("LLM generation took %.3fs", time.monotonic() - llm_start)

    safe_answer, redacted_types = redact_pii(raw_answer)


    bracket_groups = re.findall(r"\[([^\[\]]+)\]", raw_answer)
    cited_tags = set()
    for group in bracket_groups:
        for tag in group.split(","):
            tag = tag.strip()
            if re.fullmatch(r"[^\[\]#]+#\d+", tag):
                cited_tags.add(tag)
    cited_docs = [d for d in docs if d.metadata.get("chunk_id") in cited_tags]


    no_info = _is_no_info_answer(safe_answer)

    if no_info:
        top_docs = []
    elif not cited_docs:

        if cited_tags:
            logger.warning(
                "answer cited tags %s but none matched retrieved chunk_ids %s -- "
                "falling back to top retrieved chunk",
                cited_tags, [d.metadata.get("chunk_id") for d in docs],
            )
        top_docs = docs[:1]
    else:
        top_docs = cited_docs[:1] 


    sources = []
    source_redacted_types = set()
    for d in top_docs:
        redacted_snippet, snippet_types = redact_pii(d.page_content[:220].strip())
        source_redacted_types.update(snippet_types)
        sources.append({
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "chunk_id": d.metadata.get("chunk_id"),
            "snippet": redacted_snippet + ("..." if len(d.page_content) > 220 else ""),
        })

    all_redacted_types = sorted(set(redacted_types) | source_redacted_types)

    sources_withheld = bool(source_redacted_types)
    user_facing_sources = [] if sources_withheld else sources
    if no_info:
        followups = list(SUGGESTED_TOPICS[:NUM_FOLLOWUPS])
    else:
        followups = _generate_followups(question, safe_answer, docs)

    result = {
        "answer": safe_answer,
        "sources": user_facing_sources,
        "sources_withheld": sources_withheld,
        "blocked": False,
        "clarification": False,
        "redacted_entities": all_redacted_types,
        "followups": followups,
        "confidence": _confidence_label(best_score),
    }

    result["interaction_id"] = log_interaction(
        question=question,
        answer=safe_answer,
        retrieved_contexts=[d.page_content for d in docs],
        sources=sources,
        blocked=False,
        redacted_entities=all_redacted_types,
        followups=followups,
    )

    if all_redacted_types:
        logger.info("redacted entity types from answer: %s", all_redacted_types)
    logger.info("turn completed in %.3fs (confidence=%s)",
                time.monotonic() - turn_start, result["confidence"])

    return result