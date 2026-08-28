import os
import streamlit as st

from config import DOCS_DIR, VECTORSTORE_DIR, ENABLE_KEY_TERMS, ENABLE_RED_FLAGS, SUGGESTED_TOPICS
from vectorstore import build_vectorstore
from rag_chain import answer_question
from summarizer import summarize_document, SUMMARY_LENGTH_PRESETS, DEFAULT_SUMMARY_LENGTH
from eval_db import update_feedback, feedback_summary
from key_terms import extract_key_terms, compare_documents, FIELD_LABELS
from red_flags import scan_document
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

st.set_page_config(page_title="Mortgage Document Assistant", page_icon="🏦", layout="wide")

st.title("🏦 Mortgage Document RAG Assistant")
st.caption(
    "Ask questions about loan estimates, closing disclosures, and the "
    "mortgage process. Every answer is grounded in the source documents "
    "with citations, and personal/confidential details about specific "
    "borrowers, sellers, or loans are never disclosed."
)


def _doc_files():
    return [f for f in os.listdir(DOCS_DIR)
            if f.lower().endswith((".pdf", ".docx", ".txt"))]


# --- Sidebar: index management ---
with st.sidebar:
    st.header("Knowledge Base")
    doc_files = _doc_files()
    if doc_files:
        st.write(f"**{len(doc_files)} document(s) loaded:**")
        for f in doc_files:
            st.write(f"- {f}")
    else:
        st.warning(f"No documents found in `{DOCS_DIR}`. Add .pdf/.docx/.txt files there.")

    st.subheader("📤 Add documents")
    uploaded = st.file_uploader(
        "Upload loan/mortgage documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Files are saved into the knowledge base folder and indexed "
             "when you click 'Save & rebuild index' below.",
    )
    if st.button("💾 Save & rebuild index", use_container_width=True,
                 disabled=not uploaded, type="primary"):
        saved = []
        for uf in uploaded:
            safe_name = os.path.basename(uf.name)
            dest = os.path.join(DOCS_DIR, safe_name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(safe_name)
        with st.spinner(f"Indexing {len(saved)} new file(s)..."):
            build_vectorstore()
        st.success(f"Added and indexed: {', '.join(saved)}")
        st.rerun()

    if st.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding vector index..."):
            build_vectorstore()
        st.success("Index rebuilt.")

    if doc_files:
        with st.expander("🗑️ Remove a document"):
            to_remove = st.selectbox("Choose a document to remove", doc_files, key="remove_select")
            if st.button("Remove & rebuild index", use_container_width=True):
                os.remove(os.path.join(DOCS_DIR, to_remove))
                remaining = _doc_files()
                with st.spinner("Rebuilding vector index..."):
                    if remaining:
                        build_vectorstore()
                    else:
                        st.warning("No documents left -- index not rebuilt.")
                st.success(f"Removed {to_remove}.")
                st.rerun()

    st.divider()
    st.subheader("📋 Summarize a document")
    if doc_files:
        chosen = st.selectbox("Choose a document", doc_files, key="summarize_select")
        length_choice = st.select_slider(
            "Summary length",
            options=list(SUMMARY_LENGTH_PRESETS.keys()),
            value=DEFAULT_SUMMARY_LENGTH,
            key="summary_length_select",
        )
        if st.button("Summarize", use_container_width=True):
            with st.spinner(f"Summarizing {chosen} ({length_choice})..."):
                result = summarize_document(chosen, length=length_choice)
            st.session_state["summary_result"] = result

    if "summary_result" in st.session_state:
        r = st.session_state["summary_result"]
        st.markdown(f"**Summary of {r['document']}** ({r.get('length', 'medium')}):")
        st.write(r["summary"])
        if r["redacted_entities"]:
            st.caption(f"🔒 Redacted: {', '.join(r['redacted_entities'])}")


    if ENABLE_KEY_TERMS and doc_files:
        st.divider()
        st.subheader("🔑 Key loan terms")
        kt_doc = st.selectbox("Choose a document", doc_files, key="key_terms_select")
        if st.button("Extract key terms", use_container_width=True):
            with st.spinner(f"Scanning {kt_doc}..."):
                st.session_state["key_terms_result"] = extract_key_terms(kt_doc)

        if "key_terms_result" in st.session_state:
            kt = st.session_state["key_terms_result"]
            st.caption(f"{kt['found_count']}/{kt['total_fields']} fields found in {kt['document']}")
            rows = [
                {"Field": label, "Value": kt.get(key) or "-- not found --"}
                for key, label in FIELD_LABELS.items()
            ]
            st.table(rows)

        st.markdown("**⚖️ Compare two documents**")
        if len(doc_files) >= 2:
            col_a, col_b = st.columns(2)
            doc_a = col_a.selectbox("Document A", doc_files, key="compare_a")
            doc_b = col_b.selectbox(
                "Document B", doc_files,
                index=min(1, len(doc_files) - 1), key="compare_b",
            )
            if st.button("Compare", use_container_width=True):
                if doc_a == doc_b:
                    st.warning("Choose two different documents to compare.")
                else:
                    with st.spinner("Comparing..."):
                        st.session_state["compare_result"] = compare_documents(doc_a, doc_b)

            if "compare_result" in st.session_state:
                cmp = st.session_state["compare_result"]
                rows = []
                for row in cmp["rows"]:
                    marker = "⚠️ " if row["differs"] else ""
                    rows.append({
                        "Field": marker + row["field"],
                        cmp["document_a"]: row[cmp["document_a"]],
                        cmp["document_b"]: row[cmp["document_b"]],
                    })
                st.table(rows)
                st.caption("⚠️ = the two documents disagree on this field.")
        else:
            st.caption("Add a second document to enable comparison.")

    # --- Unique feature: red-flag scanner ---
    if ENABLE_RED_FLAGS and doc_files:
        st.divider()
        st.subheader("⚠️ Loan red-flag scan")
        # st.caption("Deterministic keyword scan, no LLM call -- flags loan "
        #             "features worth a second look.")
        rf_doc = st.selectbox("Choose a document", doc_files, key="red_flags_select")
        if st.button("Scan for red flags", use_container_width=True):
            with st.spinner(f"Scanning {rf_doc}..."):
                st.session_state["red_flags_result"] = (rf_doc, scan_document(rf_doc))

        if "red_flags_result" in st.session_state:
            scanned_doc, flags = st.session_state["red_flags_result"]
            if not flags:
                st.success(f"No red flags detected in {scanned_doc}.")
            else:
                st.caption(f"{len(flags)} item(s) found in {scanned_doc}:")
                for fl in flags:
                    icon = "🔴" if fl["severity"] == "warning" else "🟡"
                    with st.expander(f"{icon} {fl['label']}"):
                        st.write(fl["why"])
                        st.caption(f"\u201c...{fl['snippet']}...\u201d")

    # --- Feedback tally ---
    fb = feedback_summary()
    if fb["up"] or fb["down"]:
        st.divider()
        st.caption(f"👍 {fb['up']}  ·  👎 {fb['down']} across all chat answers")

# --- Main chat area ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
# When the user taps a suggested follow-up button, we stash the question
# here and re-run so it's picked up as if typed into chat_input below.
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None


def _render_assistant_turn(result: dict, key_prefix: str = ""):
    st.write(result["answer"])
    if result.get("sources"):
        with st.expander("📎 Sources"):
            for s in result["sources"]:
                if s.get("chunk_id") is None and s.get("page") is None:
                    # Sources from "summarize all documents" -- whole-file
                    # references, not a specific chunk/page.
                    st.markdown(f"**{s['source']}**")
                else:
                    st.markdown(f"**{s['chunk_id']}** (page {s['page']})")
                    st.caption(s["snippet"])
    # if result.get("blocked"):
    #     st.info("🔒 This request was blocked by the PII guardrail.")
    # elif result.get("clarification"):
    #     st.info("🤔 I asked a clarifying question instead of guessing -- reply below to narrow it down.")
    # elif result.get("redacted_entities"):
    #     st.caption(f"🔒 Redacted from answer: {', '.join(result['redacted_entities'])}")
    if result.get("sources_withheld"):
        st.caption("🔒 Sources withheld: the matching document(s) contain personal information.")

    interaction_id = result.get("interaction_id")
    if interaction_id and not result.get("blocked") and not result.get("clarification"):
        fb_key = f"fb_{key_prefix}_{interaction_id}"
        already = st.session_state.get(fb_key)
        col1, col2, _rest = st.columns([1, 1, 8])
        if col1.button("👍" if already != 1 else "✅👍", key=f"{fb_key}_up"):
            update_feedback(interaction_id, positive=True)
            st.session_state[fb_key] = 1
            st.rerun()
        if col2.button("👎" if already != -1 else "✅👎", key=f"{fb_key}_down"):
            update_feedback(interaction_id, positive=False)
            st.session_state[fb_key] = -1
            st.rerun()


for i, turn in enumerate(st.session_state["chat_history"]):
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            _render_assistant_turn(turn, key_prefix=f"hist{i}")

            if turn.get("followups") and i == len(st.session_state["chat_history"]) - 1:
                st.caption("Continue the conversation:")
                cols = st.columns(len(turn["followups"]))
                for col, fq in zip(cols, turn["followups"]):
                    if col.button(fq, key=f"followup_{i}_{fq}"):
                        st.session_state["pending_question"] = fq
        else:
            st.write(turn["content"])


_asked_before = [
    t["content"] for t in st.session_state["chat_history"] if t["role"] == "user"
]
_suggestion_pool = list(dict.fromkeys(SUGGESTED_TOPICS + _asked_before))
draft = st.text_input(
    "🔎 Start typing to see suggested questions (optional) -- send below",
    key="query_draft",
    placeholder="e.g. escrow, APR, closing costs...",
)
if draft.strip():
    matches = [s for s in _suggestion_pool if draft.lower() in s.lower()][:5]
    if matches:
        st.caption("Suggestions:")
        cols = st.columns(len(matches))
        for col, m in zip(cols, matches):
            if col.button(m, key=f"suggest_{m}"):
                st.session_state["pending_question"] = m
                st.session_state["query_draft"] = ""
                st.rerun()

question = st.chat_input("Ask about the loan or mortgage documents...")
if st.session_state["pending_question"]:
    question = st.session_state["pending_question"]
    st.session_state["pending_question"] = None

if question:
    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_question(question)
            except Exception:
                logger.exception("answer_question failed for question: %r", question)
                st.error("Something went wrong answering that -- please try again.")
                st.stop()
        _render_assistant_turn(result, key_prefix="live")
        if result.get("followups"):
            st.caption("Continue the conversation:")
            cols = st.columns(len(result["followups"]))
            for col, fq in zip(cols, result["followups"]):
                if col.button(fq, key=f"followup_live_{fq}"):
                    st.session_state["pending_question"] = fq
                    st.rerun()

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": result["answer"],
        "answer": result["answer"],
        "sources": result["sources"],
        "blocked": result["blocked"],
        "clarification": result.get("clarification", False),
        "redacted_entities": result["redacted_entities"],
        "followups": result.get("followups", []),
        "confidence": result.get("confidence"),
        "interaction_id": result.get("interaction_id"),
    })
