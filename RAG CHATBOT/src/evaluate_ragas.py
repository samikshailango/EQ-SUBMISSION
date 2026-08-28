import argparse
import sys

from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

from eval_db import fetch_interactions, count_interactions


def main():
    parser = argparse.ArgumentParser(description="Evaluate logged RAG interactions with RAGAS.")
    parser.add_argument("--limit", type=int, default=50,
                         help="Max number of most-recent logged turns to evaluate (default: 50).")
    parser.add_argument("--out", type=str, default="eval_data/ragas_report.csv",
                         help="Where to write the per-row CSV report.")
    args = parser.parse_args()

    total = count_interactions()
    if total == 0:
        print("No interactions logged yet. Use the chatbot (streamlit run src/app.py) "
              "a few times first -- every answered question is logged automatically.")
        sys.exit(0)

    rows = fetch_interactions(limit=args.limit)
    if not rows:
        print("No answerable (non-blocked, non-clarification) interactions logged yet.")
        sys.exit(0)

    print(f"Evaluating {len(rows)} logged interaction(s) out of {total} total logged...")
    logger.info("Starting RAGAS evaluation of %d interaction(s)", len(rows))


    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI

    from config import GEMINI_API_KEY, JUDGE_MODEL_NAME
    from vectorstore import get_embeddings

    def _load_metric(old_name, class_name):
        try:
            import ragas.metrics as m
            return getattr(m, old_name)
        except AttributeError:
            import ragas.metrics as m
            return getattr(m, class_name)()

    faithfulness = _load_metric("faithfulness", "Faithfulness")
    answer_relevancy = _load_metric("answer_relevancy", "AnswerRelevancy")
    context_utilization = _load_metric("context_utilization", "ContextUtilization")

    logger.info("Using judge model: %s", JUDGE_MODEL_NAME)
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=JUDGE_MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0.0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    ragas_dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["retrieved_contexts"] for r in rows],
    })

    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_utilization],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    df.insert(0, "interaction_id", [r["id"] for r in rows])

    print("\n=== Aggregate scores ===")
    for metric in ("faithfulness", "answer_relevancy", "context_utilization"):
        if metric in df.columns:
            mean = df[metric].dropna().mean()
            n_scored = df[metric].notna().sum()
            print(f"{metric:>20}: {mean:.3f}  (scored {n_scored}/{len(df)} rows)")
            logger.info("%s: %.3f (scored %d/%d rows)", metric, mean, n_scored, len(df))

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nPer-row report written to {args.out}")


if __name__ == "__main__":
    main()