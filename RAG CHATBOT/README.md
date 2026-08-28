# Mortgage Document RAG Assistant

A retrieval-augmented chatbot for mortgage/loan documents (Loan Estimates,
Closing Disclosures, and related CFPB guidance). Ask questions in plain
English and get answers grounded in — and cited to — the actual source
documents, plus document summarization, key-term extraction, side-by-side
document comparison, and a loan red-flag scanner. Built with hybrid
(BM25 + embedding) retrieval, PII redaction, and offline RAGAS + retrieval-
accuracy evaluation.

## 1. Features

- **Hybrid document search** — dense embeddings (sentence-transformers,
  local/offline) fused with BM25 keyword search, so both paraphrased
  questions and exact-term lookups (APR, escrow, specific clause names)
  retrieve well. See `src/vectorstore.py`.
- **Grounded chat answers** with citations back to `filename#chunk (page N)`.
- **Document summarization** with a **user-selectable length** (short /
  medium / long), via map-reduce so it scales past the LLM's context window.
- **PII redaction** — names, addresses, emails, phone numbers, SSNs, and
  loan/account numbers are stripped from both chat answers and summaries
  (Presidio + custom recognizers).
- **Key loan-term extraction** and **two-document comparison**, pulled with
  regex (not the LLM) so numbers are never paraphrased.
- **Red-flag scanner** — deterministic keyword scan for risk-relevant loan
  features (prepayment penalty, balloon payment, negative amortization, etc.).
- **Two front doors** to the same backend: a Streamlit UI (`src/app.py`)
  and a FastAPI service (`src/fastapi_app.py`).
- **Evaluation**: reference-free RAGAS scoring of live chat quality
  (`src/evaluate_ragas.py`) *and* a held-out retrieval-accuracy benchmark
  (`src/evaluate_retrieval.py`) — see [Evaluation](#5-evaluation) below.

## 2. Setup

**Requirements:** Python 3.10+, a free [Google AI Studio](https://aistudio.google.com/apikey) Gemini API key.

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_lg   # required by Presidio's PII detector
```

Create a `.env` file in the project root (do **not** commit this file):

```
GEMINI_API_KEY=your-key-here
MODEL_NAME=gemini-3.1-flash-lite      # optional override
```

> **Security note:** the copy of this project you may have received
> includes a `.env` with a live key checked in. Treat any key that has
> ever been shared/zipped as compromised — rotate it in AI Studio and
> replace it with a fresh one before using this outside a local sandbox.

## 3. Add documents & build the index

Sample CFPB Loan Estimate / Closing Disclosure documents are already in
`sample_docs/`. To (re)fetch them or add more:

```bash
python scripts/download_dataset.py        
python src/build_index.py                
```

You can also drop your own `.pdf` / `.docx` / `.txt` files into
`sample_docs/` and re-run `build_index.py`, or use the "Add documents"
uploader in the Streamlit sidebar (which rebuilds the index for you).

## 4. Run it

**Streamlit UI (recommended for a demo):**
```bash
streamlit run src/app.py
```
Opens at http://localhost:8501 — chat on the right, document management
and tools (summarize, key terms, compare, red-flag scan) in the sidebar.

**FastAPI service (for programmatic / curl access):**
```bash
uvicorn fastapi_app:app --reload --app-dir src --port 8000
```
Interactive docs at http://localhost:8000/docs. Example:
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the interest rate on this loan?"}'

curl "http://localhost:8000/documents/<filename>/summary?length=short"
```

## 5. Evaluation

Two separate, complementary evaluations are included:

**a) Retrieval accuracy** (`src/evaluate_retrieval.py`) — builds a
held-out test set from the corpus itself: samples chunks per source
document, has the LLM write a natural borrower-style question each chunk
would answer, runs that question through the same hybrid retriever the
chatbot uses, and checks whether the originating document/chunk comes
back in the top-K. Reports Document Hit Rate@K, Chunk Hit Rate@K, and MRR@K.

```bash
python src/evaluate_retrieval.py --queries-per-doc 2 --k 4
```

**b) Answer/generation quality** (`src/evaluate_ragas.py`) — reference-free
RAGAS scoring (faithfulness, answer relevancy, context utilization) over
real logged chatbot interactions in `eval_data/interactions.db`. Use the
chatbot a few times first, then run:

```bash
python src/evaluate_ragas.py --limit 50
```

Both scripts write a per-row CSV report to `eval_data/`.

## 6. Project layout

```
src/
  document_loader.py   # load + clean + chunk PDFs/DOCX/TXT
  vectorstore.py        # hybrid (FAISS dense + BM25 sparse) index & fusion
  rag_chain.py           # answer_question(): retrieval -> LLM -> citations
  summarizer.py          # map-reduce document summarization, adjustable length
  key_terms.py            # regex-based key field extraction + doc comparison
  red_flags.py             # deterministic risky-loan-feature scanner
  pii_filter.py             # Presidio-based PII redaction
  eval_db.py                 # SQLite log of chat interactions, for RAGAS
  evaluate_ragas.py            # generation-quality evaluation
  evaluate_retrieval.py         # retrieval-accuracy evaluation (this assignment's Task 4)
  build_index.py                 # one-off: (re)build the index from sample_docs/
  app.py                          # Streamlit UI
  fastapi_app.py                   # FastAPI service
scripts/
  download_dataset.py    # fetch the public CFPB sample corpus
  pii_benchmark.py         # precision/recall benchmark for the PII filter
sample_docs/              # CFPB Loan Estimate / Closing Disclosure samples
eval_data/                # SQLite interaction log + CSV eval reports
vectorstore_data/         # persisted FAISS index + BM25 pickle
REPORT.md                 # methodology, evaluation results, and challenges
```

See `REPORT.md` for the full write-up (data prep rationale, retrieval/
summarization methodology, evaluation results, and challenges).
