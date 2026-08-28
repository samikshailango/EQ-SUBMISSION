
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")


DATASET = {
    "loan_estimate_fixed": (
        "cfpb_loan_estimate_fixed_rate_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_fixed-rate-loan-sample-H24B.pdf",
        "Completed sample Loan Estimate -- fixed rate purchase loan",
    ),
    "loan_estimate_refinance": (
        "cfpb_loan_estimate_refinance_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_refinance-sample-H24D.pdf",
        "Completed sample Loan Estimate -- refinance",
    ),
    "loan_estimate_balloon": (
        "cfpb_loan_estimate_balloon_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_baloon-payment-H24E.pdf",
        "Completed sample Loan Estimate -- balloon payment loan",
    ),
    "loan_estimate_arm": (
        "cfpb_loan_estimate_arm_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_interest-only-adjustable-rate-loan-sample-H24C.pdf",
        "Completed sample Loan Estimate -- interest-only adjustable rate loan",
    ),
    "closing_disclosure_fixed": (
        "cfpb_closing_disclosure_fixed_rate_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25B.pdf",
        "Completed sample Closing Disclosure -- fixed rate purchase loan",
    ),
    "closing_disclosure_refinance": (
        "cfpb_closing_disclosure_refinance_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25E.pdf",
        "Completed sample Closing Disclosure -- refinance",
    ),
    "closing_disclosure_second_lien": (
        "cfpb_closing_disclosure_second_lien_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25C.pdf",
        "Completed sample Closing Disclosure page 3 -- simultaneous second-lien transaction",
    ),
    "service_provider_list_sample": (
        "cfpb_service_provider_list_sample.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_mortgage-loans-transactions_cover_H27B.pdf",
        "Sample written list of settlement service providers",
    ),
    "loan_estimate_blank_model": (
        "cfpb_loan_estimate_blank_model_form.pdf",
        "https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_model-form-H24.pdf",
        "Blank Loan Estimate model form (no PII -- baseline/reference structure)",
    ),
    "trid_compliance_guide": (
        "cfpb_trid_guide_loan_estimate_closing_disclosure.pdf",
        "https://files.consumerfinance.gov/f/documents/cfpb_kbyo_guide-to-loan-estimate-and-closing-disclosure-forms_v2.0.pdf",
        "CFPB's full guide to the Loan Estimate and Closing Disclosure forms "
        "(long, narrative, PII-free -- good for summarization/citation testing)",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", nargs="*", default=None,
                         help="Subset of keys to download (default: all). See --list.")
    parser.add_argument("--list", action="store_true", help="List available keys and exit.")
    args = parser.parse_args()

    if args.list:
        for key, (fname, url, desc) in DATASET.items():
            print(f"{key:32s} {desc}")
        return

    try:
        import requests
    except ImportError:
        print("Missing dependency: pip install requests", file=sys.stderr)
        sys.exit(1)

    keys = args.which or list(DATASET.keys())
    os.makedirs(DOCS_DIR, exist_ok=True)

    for key in keys:
        if key not in DATASET:
            print(f"Unknown key: {key} (see --list)", file=sys.stderr)
            continue
        fname, url, desc = DATASET[key]
        dest = os.path.join(DOCS_DIR, fname)
        print(f"Downloading {desc} ...")
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception as e:
            print(f"  FAILED: {url} ({e})", file=sys.stderr)
            continue
        with open(dest, "wb") as f:
            f.write(resp.content)
        print(f"  Saved {dest} ({len(resp.content) / 1024:.0f} KB)")

    print(f"\nDone. {len(keys)} document(s) requested -> {DOCS_DIR}")
    print("Next: python src/build_index.py   # rebuild the FAISS index to include them")


if __name__ == "__main__":
    main()
