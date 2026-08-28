import os
import re

from config import DOCS_DIR
from document_loader import load_documents

_PATTERNS = {
    "loan_amount": (
        "Loan Amount",
        re.compile(r"Loan Amount[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)", re.IGNORECASE),
    ),
    "interest_rate": (
        "Interest Rate",
        re.compile(r"Interest Rate[:\s]*([\d.]+)\s*%", re.IGNORECASE),
    ),
    "apr": (
        "APR",
        re.compile(
            r"(?:Annual Percentage Rate(?:\s*\(APR\))?|\bAPR\b)[^%\d]{0,40}?([\d.]+)\s*%",
            re.IGNORECASE,
        ),
    ),
    "monthly_principal_interest": (
        "Monthly Principal & Interest",
        re.compile(
            r"Monthly Principal (?:&|and) Interest[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
    ),
    "total_monthly_payment": (
        "Estimated Total Monthly Payment",
        re.compile(
            r"(?:Estimated )?Total Monthly Payment[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
    ),
    "closing_costs": (
        "Estimated Closing Costs",
        re.compile(
            r"(?:Estimated |Total )?Closing Costs[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
    ),
    "cash_to_close": (
        "Cash to Close",
        re.compile(
            r"(?:Estimated )?Cash to Close[:\s]*\$?\s*([\d,]+(?:\.\d{2})?)",
            re.IGNORECASE,
        ),
    ),
    "loan_term": (
        "Loan Term",
        re.compile(r"Loan Term[:\s]*([\d]+\s*years?)", re.IGNORECASE),
    ),
    "loan_purpose": (
        "Loan Purpose",
        re.compile(r"Loan Purpose[:\s]*([A-Za-z][A-Za-z /-]{2,20})", re.IGNORECASE),
    ),
    "loan_type": (
        "Loan Type",
        re.compile(r"Loan Type[:\s]*([A-Za-z][A-Za-z /-]{2,20})", re.IGNORECASE),
    ),
}

_FLAG_PATTERNS = {
    "prepayment_penalty": re.compile(r"Prepayment Penalty[:\s]*(Yes|No|X)", re.IGNORECASE),
    "balloon_payment": re.compile(r"Balloon Payment[:\s]*(Yes|No|X)", re.IGNORECASE),
}


def _clean_value(v: str) -> str:
    return " ".join(v.split()).rstrip(".,")


def extract_key_terms(filename: str, docs_dir: str = DOCS_DIR) -> dict:
    """Pull headline loan terms out of one document by filename.

    Returns a dict of {field_key: value_or_None}, plus "document" and a
    "found_count" so the UI can show "6/11 fields found" instead of
    silently rendering a mostly-empty table.
    """
    path = os.path.join(docs_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found in {docs_dir}")

    all_docs = load_documents(docs_dir)
    text = "\n".join(
        d.page_content for d in all_docs if d.metadata.get("source") == filename
    )
    if not text.strip():
        raise ValueError(f"No content loaded for {filename}")

    result = {"document": filename}
    found = 0
    for key, (_label, pattern) in _PATTERNS.items():
        m = pattern.search(text)
        if m:
            result[key] = _clean_value(m.group(1))
            found += 1
        else:
            result[key] = None

    for key, pattern in _FLAG_PATTERNS.items():
        m = pattern.search(text)
        if m:
            raw = m.group(1).strip().upper()
            result[key] = "Yes" if raw in ("YES", "X") else "No"
            found += 1
        else:
            result[key] = None

    result["found_count"] = found
    result["total_fields"] = len(_PATTERNS) + len(_FLAG_PATTERNS)
    return result


FIELD_LABELS = {k: v[0] for k, v in _PATTERNS.items()}
FIELD_LABELS.update({
    "prepayment_penalty": "Prepayment Penalty",
    "balloon_payment": "Balloon Payment",
})


def compare_documents(filename_a: str, filename_b: str, docs_dir: str = DOCS_DIR) -> dict:
    """Side-by-side key terms for two documents, one row per field, so a
    borrower can compare two loan offers (or a Loan Estimate against the
    Closing Disclosure that followed it) without reading both in full."""
    terms_a = extract_key_terms(filename_a, docs_dir)
    terms_b = extract_key_terms(filename_b, docs_dir)

    rows = []
    for key, label in FIELD_LABELS.items():
        val_a, val_b = terms_a.get(key), terms_b.get(key)
        rows.append({
            "field": label,
            filename_a: val_a if val_a is not None else "-- not found --",
            filename_b: val_b if val_b is not None else "-- not found --",
            "differs": (val_a is not None and val_b is not None and val_a != val_b),
        })
    return {"document_a": filename_a, "document_b": filename_b, "rows": rows}
