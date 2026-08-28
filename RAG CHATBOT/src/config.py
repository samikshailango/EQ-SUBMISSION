import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash")
LLM_TEMPERATURE = 0.0             # deterministic answers for a compliance-adjacent bot


JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", MODEL_NAME)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HYBRID_DENSE_WEIGHT = 0.5
HYBRID_SPARSE_WEIGHT = 0.5

BM25_TOP_K = 7

BM25_SATURATION_K = 5.0

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval ---
TOP_K = 4

RETRIEVAL_FETCH_K = TOP_K + 3

# --- Storage ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")

VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore_data")


CONFIDENTIAL_ENTITIES = [
    "PERSON",          # borrower, co-borrower, seller, loan officer, agent names
    "LOCATION",        # covers cities/places; catches most "address" leakage too
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
]


REFUSAL_MESSAGE = (
    "I can't share personal details (like names, addresses, contact info, "
    "or loan/account numbers) tied to a specific borrower or loan. I can "
    "help with general questions about loan terms and the mortgage process "
    "instead -- for example, ask me what a document says rather than who "
    "it belongs to."
)



CLARIFICATION_SCORE_THRESHOLD = 0.28

CLARIFICATION_MIN_WORDS = 2
ENABLE_CLARIFICATION = True


CONTEXT_RELEVANCE_MIN_SCORE = CLARIFICATION_SCORE_THRESHOLD * 1.15

SUGGESTED_TOPICS = [
    "What's the interest rate and APR on this loan?",
    "What are the total closing costs?",
    "How does an escrow account work?",
    "Is there a prepayment penalty or balloon payment?",
    "What's the difference between a Loan Estimate and a Closing Disclosure?",
]
ENABLE_GREETING = True


ENABLE_FOLLOWUPS = True
NUM_FOLLOWUPS = 2


EVAL_DB_PATH = os.path.join(BASE_DIR, "eval_data", "interactions.db")

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "chatbot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB
LOG_BACKUP_COUNT = 3


CONFIDENCE_HIGH_SCORE = CLARIFICATION_SCORE_THRESHOLD * 2.2
CONFIDENCE_MEDIUM_SCORE = CLARIFICATION_SCORE_THRESHOLD * 1.5


ENABLE_KEY_TERMS = True

ENABLE_RED_FLAGS = True
