
import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import CONFIDENTIAL_ENTITIES

_STREET_ADDRESS_PATTERN = Pattern(
    name="street_address_pattern",

    regex=r"\b\d{1,4}[A-Za-z]?\s+[A-Za-z][A-Za-z]*(?:\s[A-Za-z]+){0,4}\s"
          r"(?:Road|Street|St\.|Avenue|Ave\.|Lane|Ln\.|Drive|Dr\.|"
          r"Block|Sector|Floor|Tower|Apartment|Apt\.)\b",
    score=0.6,
)
_street_address_recognizer = PatternRecognizer(
    supported_entity="STREET_ADDRESS",
    patterns=[_STREET_ADDRESS_PATTERN],
)

_LOAN_ACCOUNT_NUMBER_PATTERN = Pattern(
    name="loan_account_number_pattern",
    regex=r"\b(?:Loan\s*(?:ID|Number|No\.?)|MIC\s*#|Account\s*(?:#|No\.?|Number))"
          r"[\s:#]*(\d[\d-]{5,25}\d)\b",
    score=0.7,
)
_loan_account_number_recognizer = PatternRecognizer(
    supported_entity="LOAN_ACCOUNT_NUMBER",
    patterns=[_LOAN_ACCOUNT_NUMBER_PATTERN],
)

_nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
}
_nlp_engine = NlpEngineProvider(nlp_configuration=_nlp_config).create_engine()

_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine, supported_languages=["en"])
_analyzer.registry.add_recognizer(_street_address_recognizer)
_analyzer.registry.add_recognizer(_loan_account_number_recognizer)
_anonymizer = AnonymizerEngine()

_ALL_CONFIDENTIAL_ENTITIES = CONFIDENTIAL_ENTITIES + ["STREET_ADDRESS", "LOAN_ACCOUNT_NUMBER"]

_PII_FIELD_KEYWORDS = [
    "email address", "email id", "email", "e-mail",
    "phone number", "contact number", "mobile number", "phone",
    "contact info", "contact details",
    "address", "home address", "residence", "residential address",
    "lives", "live at", "where does", "where is",
    "ssn", "social security",
    "personal details", "personal information", "personal info",
    "date of birth", "dob", "birthdate",
    "loan number", "loan id", "account number", "account #", "mic #",
    "loan account",
]

_PII_INTENT_PATTERNS = [
    r"\bwho('s| is| are)\b",  # "who is John Smith" / "who's John Smith" / "who are the borrowers"
    r"'s\s+number\b", r"\b(?:his|her|their)\s+number\b",
]


_PII_ROLE_KEYWORDS = [
    "loan officer", "settlement agent", "closing agent", "escrow officer",
    "escrow agent", "title agent", "borrower", "borrowers", "co-borrower",
    "co-borrowers", "seller", "sellers", "buyer", "buyers", "lender",
    "loan originator", "mortgage broker", "real estate agent", "realtor",
    "notary", "attorney",
]

_DOCUMENT_SCOPE_PATTERN = re.compile(
    r"\b(?:on (?:the|this)|for (?:the|this)|in (?:the|this)|"
    r"this (?:loan|transaction|document)|the transaction|"
    r"loan estimate|closing disclosure)\b",
    re.IGNORECASE,
)

_BULK_EXTRACTION_PATTERN = re.compile(
    r"\b(?:print|dump|extract|list|show|reveal)\b[^.?!]{0,25}\b(?:all|every)\b"
    r"[^.?!]{0,20}\b(?:pii|personal (?:info|information|data|details)|"
    r"confidential (?:info|information|data)|private (?:info|information|data))\b",
    re.IGNORECASE,
)


_THIRD_PERSON_PRONOUN_PATTERN = re.compile(
    r"\b(?:his|her|their|him|them|he|she)\b", re.IGNORECASE
)


def query_requests_pii(query: str) -> bool:
    """Layer 1: does the query itself look like a PII lookup?"""
    q = query.lower()

    if _BULK_EXTRACTION_PATTERN.search(q):
        return True

    has_pii_field_keyword = any(kw in q for kw in _PII_FIELD_KEYWORDS)
    has_intent_pattern = any(re.search(p, q) for p in _PII_INTENT_PATTERNS)
    has_role_keyword = any(kw in q for kw in _PII_ROLE_KEYWORDS)

    if has_role_keyword and has_pii_field_keyword:
        return True

    if (has_role_keyword and _DOCUMENT_SCOPE_PATTERN.search(q)
            and re.search(r"\bwho('s| is| are)\b", q)):
        return True


    if not (has_pii_field_keyword or has_intent_pattern):
        results = _analyzer.analyze(text=query, language="en",
                                     entities=_ALL_CONFIDENTIAL_ENTITIES)
        person_present = any(r.entity_type == "PERSON" for r in results)
        contact_present = any(
            r.entity_type in ("LOCATION", "STREET_ADDRESS", "EMAIL_ADDRESS",
                               "PHONE_NUMBER", "US_SSN", "LOAN_ACCOUNT_NUMBER")
            for r in results
        )
        return person_present and contact_present

    results = _analyzer.analyze(text=query, language="en", entities=["PERSON"])
    if len(results) > 0:
        return True

    if _THIRD_PERSON_PRONOUN_PATTERN.search(q):
        return True

    is_identity_lookup = bool(re.search(r"\bwho('s| is| are)\b", q))
    return is_identity_lookup and has_pii_field_keyword


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Layer 2: redact confidential entities from a block of text
    (e.g. the LLM's draft answer) before it's shown to the user.

    Returns (redacted_text, list_of_entity_types_found).
    """
    results = _analyzer.analyze(text=text, language="en",
                                 entities=_ALL_CONFIDENTIAL_ENTITIES)
    if not results:
        return text, []

    operators = {
        r.entity_type: OperatorConfig("replace",
                                       {"new_value": f"[REDACTED-{r.entity_type}]"})
        for r in results
    }
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results,
                                        operators=operators)
    found_types = sorted({r.entity_type for r in results})
    return anonymized.text, found_types
