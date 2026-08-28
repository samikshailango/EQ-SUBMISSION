import os
import re

from config import DOCS_DIR
from document_loader import load_documents
from pii_filter import redact_pii

_RULES = [
    (
        "prepayment_penalty_yes",
        "Prepayment penalty",
        "warning",
        re.compile(r"Prepayment Penalty[:\s]*(?:Yes|X)\b", re.IGNORECASE),
        "You could be charged a fee for paying off or refinancing this "
        "loan early.",
    ),
    (
        "balloon_payment_yes",
        "Balloon payment",
        "warning",
        re.compile(r"Balloon Payment[:\s]*(?:Yes|X)\b", re.IGNORECASE),
        "A large lump-sum payment is due at the end of the loan term, "
        "on top of the regular monthly payments.",
    ),
    (
        "rate_can_increase",
        "Interest rate can increase",
        "warning",
        re.compile(
            r"interest rate (?:can|may) increase|adjustable[- ]rate|"
            r"interest[- ]only",
            re.IGNORECASE,
        ),
        "This isn't a fixed rate for the life of the loan -- your rate "
        "(and payment) can change.",
    ),
    (
        "negative_amortization",
        "Negative amortization possible",
        "warning",
        re.compile(r"negative amortization", re.IGNORECASE),
        "Your loan balance could grow over time instead of shrinking, if "
        "payments don't cover the interest due.",
    ),
    (
        "demand_feature",
        "Demand feature",
        "warning",
        re.compile(r"demand feature", re.IGNORECASE),
        "The lender may be able to require full repayment of the loan "
        "before the scheduled end date.",
    ),
    (
        "pmi_required",
        "Mortgage insurance required",
        "note",
        re.compile(r"Mortgage Insurance[:\s]*\$\s*(?!0\b)[\d]", re.IGNORECASE),
        "An extra monthly cost on top of principal & interest -- ask when "
        "it can be removed.",
    ),
    (
        "high_late_fee",
        "Late fee above 5%",
        "note",
        re.compile(r"late fee of\s*([6-9]|\d{2,})\s*%", re.IGNORECASE),
        "The late-payment fee is higher than the 5%-of-payment figure "
        "seen on most standard Loan Estimates -- worth double-checking.",
    ),
    (
        "no_assumption",
        "Loan is not assumable",
        "note",
        re.compile(r"will not\s+allow.{0,40}assume this loan", re.IGNORECASE | re.DOTALL),
        "A future buyer of the property won't be able to take over this "
        "loan's existing rate and terms.",
    ),
]


def scan_document(filename: str, docs_dir: str = DOCS_DIR) -> list[dict]:
    """Scan one document by filename, return a list of matched flags with
    a short snippet of surrounding context for each."""
    path = os.path.join(docs_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found in {docs_dir}")

    all_docs = load_documents(docs_dir)
    text = "\n".join(
        d.page_content for d in all_docs if d.metadata.get("source") == filename
    )

    flags = []
    for flag_id, label, severity, pattern, why in _RULES:
        m = pattern.search(text)
        if not m:
            continue
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        snippet, _ = redact_pii(" ".join(text[start:end].split()))
        flags.append({
            "id": flag_id,
            "label": label,
            "severity": severity,
            "why": why,
            "snippet": snippet,
        })
    return flags


def scan_all(docs_dir: str = DOCS_DIR) -> dict:
    """Scan every supported document in docs_dir. Returns {filename: [flags]}."""
    results = {}
    for fname in sorted(os.listdir(docs_dir)):
        if os.path.splitext(fname)[1].lower() not in (".pdf", ".docx", ".txt"):
            continue
        try:
            results[fname] = scan_document(fname, docs_dir)
        except Exception:
            results[fname] = []
    return results
