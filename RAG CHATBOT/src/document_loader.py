import os
import re

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


_PAGE_FURNITURE_RE = re.compile(
    r"^\s*(page\s+\d+\s+of\s+\d+|\d+\s*/\s*\d+|-?\s*\d+\s*-?)\s*$",
    re.IGNORECASE,
)

_MULTI_WHITESPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip page-furniture lines from one
    document's/page's raw extracted text. Deliberately conservative --
    only removes whitespace noise and lines that are unambiguously just
    a page number, never touches actual sentence content, so no dollar
    amount, rate, or defined term can ever be dropped by this step."""
    lines = text.split("\n")
    kept = [ln for ln in lines if not _PAGE_FURNITURE_RE.match(ln)]
    cleaned = "\n".join(kept)
    cleaned = _MULTI_WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _is_low_content(text: str, min_chars: int = 20) -> bool:
    """True for pages that are blank/near-blank after cleaning (cover
    sheets, dividers, scanned-image pages with no extractable text) --
    dropped before chunking so they don't waste index space or dilute
    retrieval with empty context."""
    return len(text) < min_chars


def load_documents(docs_dir: str = DOCS_DIR):
    """Load every supported file in docs_dir into LangChain Documents,
    with cleaning applied and low-content pages dropped."""
    documents = []
    for fname in sorted(os.listdir(docs_dir)):
        ext = os.path.splitext(fname)[1].lower()
        loader_cls = _LOADERS.get(ext)
        if not loader_cls:
            continue
        path = os.path.join(docs_dir, fname)
        loader = loader_cls(path)
        docs = loader.load()
        for i, d in enumerate(docs):
            d.page_content = _clean_text(d.page_content)
            d.metadata["source"] = fname
            d.metadata.setdefault("page", i)
        documents.extend(d for d in docs if not _is_low_content(d.page_content))
    return documents


def chunk_documents(documents):
    """Split loaded documents into overlapping chunks, preserving metadata
    (source + page) on every chunk so citations survive the split."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    per_source_counter: dict[str, int] = {}
    for c in chunks:
        source = c.metadata.get("source", "doc")
        idx = per_source_counter.get(source, 0)
        c.metadata["chunk_id"] = f"{source}#{idx}"
        per_source_counter[source] = idx + 1
    return chunks


def load_and_chunk(docs_dir: str = DOCS_DIR):
    return chunk_documents(load_documents(docs_dir))