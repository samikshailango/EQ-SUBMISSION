import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

from config import DOCS_DIR, ENABLE_KEY_TERMS, ENABLE_RED_FLAGS
from vectorstore import build_vectorstore
from rag_chain import answer_question
from summarizer import summarize_document
from eval_db import update_feedback, feedback_summary
from key_terms import extract_key_terms, compare_documents
from red_flags import scan_document
import os

app = FastAPI(
    title="Mortgage Document RAG Chatbot API",
    description="Ask questions about mortgage/loan documents (Loan "
                "Estimates, Closing Disclosures, and the CFPB compliance "
                "guide) with citations and a PII guardrail.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    logger.info("%s %s -> %d (%.3fs)", request.method, request.url.path,
                response.status_code, duration)
    return response




class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")


class SourceOut(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut] = []
    sources_withheld: bool = False
    blocked: bool = False
    clarification: bool = False
    redacted_entities: list[str] = []
    followups: list[str] = []
    interaction_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    interaction_id: str
    positive: bool


class CompareRequest(BaseModel):
    document_a: str
    document_b: str




@app.get("/health")
def health():
    """Liveness/readiness check -- also confirms the knowledge base has
    at least one document indexed."""
    docs = [f for f in os.listdir(DOCS_DIR)
            if f.lower().endswith((".pdf", ".docx", ".txt"))] if os.path.isdir(DOCS_DIR) else []
    return {"status": "ok", "documents_indexed": len(docs)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Ask a question about the indexed mortgage/loan documents."""
    try:
        result = answer_question(req.question)
    except Exception:
        logger.exception("Error answering question: %r", req.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")
    return result


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Attach a thumbs up/down vote to a previously returned interaction_id."""
    ok = update_feedback(req.interaction_id, req.positive)
    if not ok:
        raise HTTPException(status_code=404, detail="interaction_id not found.")
    return {"status": "recorded"}


@app.get("/feedback/summary")
def feedback_tally():
    return feedback_summary()


@app.get("/documents")
def list_documents():
    """List document filenames currently in the knowledge base folder."""
    if not os.path.isdir(DOCS_DIR):
        return {"documents": []}
    return {"documents": sorted(
        f for f in os.listdir(DOCS_DIR)
        if f.lower().endswith((".pdf", ".docx", ".txt"))
    )}


@app.post("/documents/rebuild-index")
def rebuild_index():
    """Rebuild the FAISS index from whatever is currently in DOCS_DIR."""
    try:
        build_vectorstore()
    except Exception:
        logger.exception("Index rebuild failed")
        raise HTTPException(status_code=500, detail="Index rebuild failed.")
    return {"status": "rebuilt"}


@app.get("/documents/{filename}/summary")
def summarize(filename: str, length: str = "medium"):
    """`length`: "short" | "medium" | "long" (see summarizer.SUMMARY_LENGTH_PRESETS)."""
    try:
        return summarize_document(filename, length=length)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{filename} not found.")
    except Exception:
        logger.exception("Summarization failed for %s", filename)
        raise HTTPException(status_code=500, detail="Summarization failed.")


if ENABLE_KEY_TERMS:
    @app.get("/documents/{filename}/key-terms")
    def key_terms(filename: str):
        try:
            return extract_key_terms(filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"{filename} not found.")

    @app.post("/documents/compare")
    def compare(req: CompareRequest):
        try:
            return compare_documents(req.document_a, req.document_b)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))


if ENABLE_RED_FLAGS:
    @app.get("/documents/{filename}/red-flags")
    def red_flags(filename: str):
        try:
            return {"document": filename, "flags": scan_document(filename)}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"{filename} not found.")
