# Report: Document Search and Summarization Using LLMs
### Mortgage Document RAG Assistant

## 1. Corpus

The corpus is the Consumer Financial Protection Bureau's (CFPB) public
Loan Estimate / Closing Disclosure sample and guidance documents
(`sample_docs/`, fetched via `scripts/download_dataset.py`). These are
U.S. federal public-domain works, so there's no licensing friction, and
CFPB deliberately fills the "completed sample" forms in with illustrative
(fictional) borrower PII — which doubles as a realistic test bed for the
PII-redaction guardrail, not just the search/summarization pipeline.

## 2. Data preparation

`src/document_loader.py` handles ingestion in three stages:

1. **Load** — `PyPDFLoader` / `Docx2txtLoader` / `TextLoader` per file
   extension, so PDF, DOCX, and TXT are all supported uniformly.
2. **Clean** — raw extraction (especially from PDFs) is noisy in ways
   that hurt both retrieval and summarization if left in:
   - Extractors frequently insert extra whitespace/line breaks
     mid-sentence around column layouts and form fields — collapsed to
     single spaces/paragraph breaks.
   - Repeated page furniture ("Page 1 of 3", standalone page numbers)
     appears on every page and, left in, gets embedded into every chunk
     from that document, polluting the embedding with boilerplate that
     has nothing to do with a page's actual content. Lines matching an
     unambiguous page-furniture pattern are stripped — never lines that
     merely *contain* a number, so no dollar figure or date is ever at risk.
   - Blank/near-blank pages (cover sheets, scanned-image pages with no
     extractable text) are dropped entirely rather than indexed as
     empty, content-free chunks.
3. **Chunk** — `RecursiveCharacterTextSplitter` (800 chars, 120 overlap)
   with per-source, stable `chunk_id`s (e.g. `loan_estimate.pdf#3`) so
   citations survive re-indexing and don't shift when unrelated
   documents are added or removed.

**Why this design:** the cleaning step is deliberately conservative — it
only removes whitespace noise and lines that are unambiguously furniture.
A mortgage document's value is almost entirely in its concrete figures
(rates, fees, dates); a preprocessing step aggressive enough to risk
dropping or altering one of those would be worse than doing nothing.

## 3. Document search methodology

`src/vectorstore.py` implements **hybrid retrieval**:

- **Dense side**: local `sentence-transformers/all-MiniLM-L6-v2`
  embeddings in a FAISS index, L2-normalized so cosine similarity is
  computed correctly (`normalize_L2=True` — a common langchain/FAISS
  pitfall documented in the module).
- **Sparse side**: BM25 (`rank_bm25`) over the same chunks, which
  reliably catches exact-term queries (APR, escrow, specific clause
  labels) that embeddings alone sometimes place further apart than
  they should be.
- **Fusion**: both sides are scored on the *entire* corpus (not
  truncated to a small top-k window each before fusing — see the
  docstring in `hybrid_search_with_scores` for why early truncation
  drops correct answers), converted to independent [0,1] scales, and
  combined as a weighted sum (default 0.5 dense / 0.5 sparse, tunable
  in `config.py`).

Retrieval also drives two secondary behaviors: a **clarification check**
(if the best fused score is below a threshold, the bot asks a follow-up
question instead of forcing an answer from a weak match) and a
**confidence label** logged with every answered turn.

## 4. Summarization methodology

`src/summarizer.py` uses **map-reduce** summarization so it scales past
a single document's ability to fit in one LLM call:

- **Map**: each chunk is summarized independently (2-4 sentences,
  preserving concrete figures, no PII).
- **Reduce**: partial summaries are combined into one coherent summary,
  with an explicit **target length** the user selects — `short` (~75-100
  words), `medium` (~150-250 words), or `long` (~350-500 words, headed
  by topic). Word-count targets are used rather than vague labels because
  LLMs follow numeric budgets far more reliably than adjectives.
- The final summary is passed through the same PII redactor used for
  chat answers before being returned.

## 5. Evaluation

Two separate evaluations are provided, because "does search return the
right document" and "is the generated answer good" are different
questions requiring different methodology.

### 5a. Retrieval accuracy (`src/evaluate_retrieval.py`)

Directly implements the assignment's evaluation procedure: a held-out
test set is built by sampling chunks across the indexed corpus (multiple,
randomly-chosen chunks per source document, not just the first chunk, to
avoid the result being an artifact of distinctive opening text); for each
sampled chunk, the LLM is prompted to write a natural, paraphrased
borrower-style question that chunk would answer; that question is run
through the same hybrid retriever the chatbot uses; and it's checked
whether the originating document (**Document Hit Rate@K**) and the exact
originating chunk (**Chunk Hit Rate@K**, a stricter variant) appear in the
top-K results, plus **MRR@K** to reward ranking it higher rather than
just present. Run with:

```bash
python src/evaluate_retrieval.py --queries-per-doc 2 --k 4
```

*Numbers weren't captured in this write-up* — running it requires a live
Gemini API call per test query, which wasn't available in the environment
this report was drafted in. Run the command above locally and paste the
printed summary + `eval_data/retrieval_report.csv` here; with hybrid
retrieval over a corpus this size, a target of Hit Rate@4 ≥ 0.85 and
MRR@4 ≥ 0.6 is a reasonable bar to evaluate against.

### 5b. Answer/generation quality (`src/evaluate_ragas.py`)

Reference-free RAGAS scoring (faithfulness, answer relevancy, context
utilization) over real logged chat interactions (`eval_data/interactions.db`),
using the same Gemini model as judge. This is deliberately *not* a
hand-labeled gold-answer set — it scores the system against its own
actual usage, so it stays representative as the corpus or query mix
changes, at the cost of needing some real interaction volume before it's
statistically meaningful. Existing reports from prior runs are in
`eval_data/ragas_report*.csv`.

### 5c. PII-redaction benchmark (`scripts/pii_benchmark.py`)

Not required by the assignment, but included because the PII guardrail is
core to this domain: precision/recall of the redaction filter against
labeled sentences (including the `ai4privacy/pii-masking-300k` public
dataset), so redaction aggressiveness can be tuned against actual
false-positive/false-negative rates rather than by eye.

## 6. Challenges and solutions

- **FAISS cosine similarity pitfall** — passing `DistanceStrategy.COSINE`
  into langchain's FAISS wrapper does *not*, on its own, normalize
  vectors; without `normalize_L2=True` the underlying index is really
  computing L2 distance, and `_cosine_relevance_score_fn`'s naive
  `1 - distance` produces meaningless numbers. Fixed by normalizing on
  both index-build and query, and converting L2→cosine with the exact
  closed-form identity for unit vectors instead of trusting the
  library's relevance-score plumbing.
- **BM25/dense score scales are incompatible** — BM25 scores are
  unbounded; naively min-max normalizing them per-query makes even a
  bad match's best BM25 hit score 1.0, destroying any absolute
  "how relevant, really" signal needed for the clarification/confidence
  logic. Solved with a saturating curve (`score / (score + K)`) that
  keeps the score on an absolute, query-independent scale.
- **Truncate-then-fuse loses correct answers** — scoring only each
  side's own top-k window before fusing let a large, verbose document's
  chunks flood a small BM25 window on lexical overlap alone, crowding out
  a smaller document's chunk that would have won on the *fused* score.
  Fixed by scoring the full corpus on both sides before truncating.
- **Long documents don't fit one summarization call** — solved with
  map-reduce chunk-level summarization rather than truncating the
  document or relying on a huge-context model.
- **No API access in this evaluation environment** — the retrieval and
  RAGAS evaluation scripts both require live LLM calls; this report
  documents the methodology and how to run them, but actual numeric
  results should be captured by running both scripts locally with a
  valid `GEMINI_API_KEY` and pasted into this section.

## 7. Scalability & efficiency notes

- Embeddings are local/offline (no per-call API cost or latency for
  indexing), so re-indexing scales with CPU, not API rate limits.
- BM25 and FAISS are both flat/in-memory structures appropriate for the
  current corpus size (a few thousand chunks); for a much larger corpus,
  FAISS's `IndexFlatIP` would need to move to an approximate index
  (e.g. HNSW/IVF) and BM25 to a proper inverted-index library, but the
  hybrid fusion logic itself is index-implementation-agnostic.
- Retrieval scores the full corpus on both sides before truncating (see
  §6) — cheap at current scale, but the first thing to revisit if the
  corpus grows by orders of magnitude.
