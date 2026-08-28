
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

_LABEL_MAP_SUBSTRINGS = [
    "NAME", "EMAIL", "PHONE", "STREET", "CITY", "ADDRESS", "SSN",
    "BUILDING", "ZIPCODE", "STATE", "COUNTY",
]


def _row_has_mapped_pii(privacy_mask) -> bool:
    return any(
        any(sub in str(item.get("label", "")).upper() for sub in _LABEL_MAP_SUBSTRINGS)
        for item in privacy_mask
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="Number of dataset rows to sample.")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency: pip install datasets", file=sys.stderr)
        sys.exit(1)

    from pii_filter import redact_pii

    print(f"Loading ai4privacy/pii-masking-300k ({args.split}, streaming first {args.n} rows)...")
    ds = load_dataset("ai4privacy/pii-masking-300k", split=args.split, streaming=True)

    tp = fp = fn = tn = 0
    n_seen = 0
    for row in ds:
        if n_seen >= args.n:
            break
        n_seen += 1

        text = row.get("unmasked_text") or row.get("source_text") or ""
        privacy_mask = row.get("privacy_mask", []) or []
        if not text:
            continue

        truly_has_pii = _row_has_mapped_pii(privacy_mask)
        _redacted_text, found_types = redact_pii(text)
        flagged = len(found_types) > 0

        if flagged and truly_has_pii:
            tp += 1
        elif flagged and not truly_has_pii:
            fp += 1
        elif not flagged and truly_has_pii:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

    print(f"\nSampled {n_seen} rows from ai4privacy/pii-masking-300k")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision = {precision:.3f}")
    print(f"  recall    = {recall:.3f}")
    print(f"  f1        = {f1:.3f}")
    print("\n(Sentence-level PII presence/absence, not exact span overlap -- "
          "good for tracking regressions, not a substitute for a full NER eval.)")


if __name__ == "__main__":
    main()
