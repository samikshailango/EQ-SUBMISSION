import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import GEMINI_API_KEY, MODEL_NAME, LLM_TEMPERATURE, DOCS_DIR
from document_loader import load_documents, chunk_documents
from pii_filter import redact_pii

_MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following excerpt from a mortgage/loan "
               "document in 2-4 sentences. Preserve concrete facts (dollar "
               "amounts, interest rates, dates, deadlines, fees). Do not "
               "include any individual's name, address, email, phone "
               "number, or loan/account number -- refer to roles "
               "(e.g. 'the borrower', 'the lender') and loans generically "
               "instead."),
    ("human", "{chunk}"),
])

_REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are given several partial summaries of sections of the "
               "same mortgage/loan document, in order. Combine them into a "
               "single, well-organized summary of the WHOLE document, using "
               "short headed sections if the document covers multiple "
               "topics. Do not include any individual's name, address, "
               "email, phone number, or loan/account number -- refer to "
               "roles and loans generically instead. Do not omit any "
               "concrete figure, rate, fee, or deadline mentioned in the "
               "partial summaries.\n\n"
               "Target length: {length_instruction} Prioritize the most "
               "important figures and terms if you have to trim."),
    ("human", "{partial_summaries}"),
])


SUMMARY_LENGTH_PRESETS = {
    "short": "About 75-100 words, 3-5 sentences, one paragraph.",
    "medium": "About 150-250 words, a short paragraph or two.",
    "long": "About 350-500 words, using headed sections if the document "
            "covers multiple topics (e.g. loan terms, costs, key dates).",
}
DEFAULT_SUMMARY_LENGTH = "medium"


def _get_llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY,
                                   temperature=LLM_TEMPERATURE)


def summarize_document(
    filename: str,
    docs_dir: str = DOCS_DIR,
    length: str = DEFAULT_SUMMARY_LENGTH,
) -> dict:
    """Summarize a single document by filename (must exist in docs_dir).

    `length` selects how long the final (REDUCE-step) summary should be --
    one of SUMMARY_LENGTH_PRESETS' keys ("short", "medium", "long"), or a
    custom free-text instruction (e.g. "about 40 words") for callers that
    want finer control than the three presets.
    """
    length_instruction = SUMMARY_LENGTH_PRESETS.get(length, length)
    path = os.path.join(docs_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found in {docs_dir}")

    all_docs = load_documents(docs_dir)
    doc_pages = [d for d in all_docs if d.metadata.get("source") == filename]
    if not doc_pages:
        raise ValueError(f"No content loaded for {filename}")

    chunks = chunk_documents(doc_pages)

    llm = _get_llm()
    map_chain = _MAP_PROMPT | llm | StrOutputParser()
    reduce_chain = _REDUCE_PROMPT | llm | StrOutputParser()

    # MAP step
    partial_summaries = []
    for c in chunks:
        s = map_chain.invoke({"chunk": c.page_content})
        partial_summaries.append(f"- {s}")

    # REDUCE step
    combined = "\n".join(partial_summaries)
    final_summary = reduce_chain.invoke({
        "partial_summaries": combined,
        "length_instruction": length_instruction,
    })

    safe_summary, redacted_types = redact_pii(final_summary)

    return {
        "document": filename,
        "summary": safe_summary,
        "num_chunks_processed": len(chunks),
        "redacted_entities": redacted_types,
        "length": length,
    }