import argparse
import csv
import os
import random
import sys

from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import GEMINI_API_KEY, MODEL_NAME, LLM_TEMPERATURE, VECTORSTORE_DIR, TOP_K
from vectorstore import load_vectorstore, hybrid_search_with_scores

_QUERY_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are simulating a mortgage borrower using a document Q&A "
     "chatbot. Below is one excerpt from a loan/mortgage document. Write "
     "ONE short, natural question a borrower would realistically ask "
     "that this excerpt would answer. Paraphrase -- do NOT copy exact "
     "phrases longer than a couple of words from the excerpt (a good "
     "test query uses the borrower's own words, not the document's). "
     "Return ONLY the question, no preamble."),
    ("human", "{chunk}"),
])


def _sample_test_chunks(queries_per_doc: int, seed: int = 42):
    """Group all indexed chunks by source document, then randomly sample
    `queries_per_doc` chunks from each source to form the held-out test
    set. Using multiple, well-separated chunks per document (rather than
    always the first chunk) avoids the accuracy number being an artifact
    of documents that happen to start with distinctive text."""
    vs = load_vectorstore(VECTORSTORE_DIR)
    by_source: dict[str, list] = {}
    for doc in vs.docstore._dict.values():
        by_source.setdefault(doc.metadata.get("source", "unknown"), []).append(doc)

    rng = random.Random(seed)
    sampled = []
    for source, chunks in by_source.items():
        candidates = [c for c in chunks if len(c.page_content) > 200] or chunks
        rng.shuffle(candidates)
        sampled.extend(candidates[:queries_per_doc])
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-per-doc", type=int, default=1,
                         help="How many test queries to generate per source document (default: 1).")
    parser.add_argument("--k", type=int, default=TOP_K,
                         help=f"Top-K cutoff for hit-rate/MRR (default: TOP_K={TOP_K}).")
    parser.add_argument("--out", type=str, default="eval_data/retrieval_report.csv",
                         help="Where to write the per-query CSV report.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed, for reproducibility.")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
        print("No index found. Run `python src/build_index.py` first.")
        sys.exit(1)

    test_chunks = _sample_test_chunks(args.queries_per_doc, seed=args.seed)
    if not test_chunks:
        print("No chunks available to build a test set from.")
        sys.exit(1)

    print(f"Built a test set of {len(test_chunks)} chunk(s) "
          f"across the indexed corpus. Generating queries...")

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY,
                                  temperature=LLM_TEMPERATURE)
    query_chain = _QUERY_GEN_PROMPT | llm | StrOutputParser()

    rows = []
    doc_hits, chunk_hits, reciprocal_ranks = [], [], []

    for i, chunk in enumerate(test_chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", f"{source}#?")
        query = query_chain.invoke({"chunk": chunk.page_content}).strip()

        results = hybrid_search_with_scores(query, k=args.k)
        result_sources = [d.metadata.get("source") for d, _ in results]
        result_chunk_ids = [d.metadata.get("chunk_id") for d, _ in results]

        doc_hit = source in result_sources
        chunk_hit = chunk_id in result_chunk_ids
        rank = (result_chunk_ids.index(chunk_id) + 1) if chunk_hit else None
        rr = (1.0 / rank) if rank else 0.0

        doc_hits.append(doc_hit)
        chunk_hits.append(chunk_hit)
        reciprocal_ranks.append(rr)

        print(f"[{i}/{len(test_chunks)}] {chunk_id}: "
              f"doc_hit={doc_hit} chunk_hit={chunk_hit} rank={rank}  "
              f"query={query!r}")

        rows.append({
            "source_document": source,
            "source_chunk_id": chunk_id,
            "generated_query": query,
            "doc_hit_at_k": doc_hit,
            "chunk_hit_at_k": chunk_hit,
            "rank_of_correct_chunk": rank,
            "reciprocal_rank": rr,
            "top_result_sources": " | ".join(result_sources),
        })

    n = len(rows)
    hit_rate_doc = sum(doc_hits) / n
    hit_rate_chunk = sum(chunk_hits) / n
    mrr = sum(reciprocal_ranks) / n

    print("\n=== Retrieval accuracy (k={}) ===".format(args.k))
    print(f"{'Document Hit Rate@k':>24}: {hit_rate_doc:.3f}  "
          f"(correct source document appeared in top-{args.k})")
    print(f"{'Chunk Hit Rate@k':>24}: {hit_rate_chunk:.3f}  "
          f"(the exact source chunk appeared in top-{args.k})")
    print(f"{'MRR@k':>24}: {mrr:.3f}  "
          f"(rewards ranking the correct chunk higher, not just present)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-query report written to {args.out}")

    logger.info(
        "Retrieval eval: n=%d doc_hit_rate=%.3f chunk_hit_rate=%.3f mrr=%.3f (k=%d)",
        n, hit_rate_doc, hit_rate_chunk, mrr, args.k,
    )


if __name__ == "__main__":
    main()
