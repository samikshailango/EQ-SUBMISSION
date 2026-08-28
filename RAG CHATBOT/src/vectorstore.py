import os
import pickle
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from config import (
    EMBEDDING_MODEL, VECTORSTORE_DIR, TOP_K, RETRIEVAL_FETCH_K,
    HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT, BM25_TOP_K,
    BM25_SATURATION_K,
)
from document_loader import load_and_chunk

_embeddings = None

BM25_INDEX_FILENAME = "bm25_store.pkl"


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings

def _bm25_tokenize(text: str) -> list[str]:
    """Deliberately simple tokenizer (lowercase alphanumeric runs) -- BM25
    just needs consistent term matching, not linguistic sophistication.
    Same function is used to index chunks and to tokenize queries, which
    matters more than any particular tokenization choice."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, BM25_INDEX_FILENAME)


def _build_bm25(chunks, persist_dir: str):
    """Build a BM25Okapi index over the same chunks used for the dense
    index and persist it as a single pickle file (index + the chunk
    Documents it was built over, so we can map scores back to text without
    depending on the FAISS docstore's internal object identity)."""
    from rank_bm25 import BM25Okapi  # local import: keeps BM25 optional-ish

    tokenized = [_bm25_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    os.makedirs(persist_dir, exist_ok=True)
    with open(_bm25_path(persist_dir), "wb") as f:
        pickle.dump({"bm25": bm25, "docs": chunks}, f)


def _load_bm25(persist_dir: str):
    """Load the persisted BM25 index, building it on the fly (from
    whatever's in the FAISS docstore) if it's missing -- mirrors
    load_vectorstore's "build if absent" fallback so an older
    vectorstore_data/ directory (built before hybrid retrieval existed)
    doesn't hard-fail, it just self-heals on first use."""
    path = _bm25_path(persist_dir)
    if not os.path.exists(path):
        vs = load_vectorstore(persist_dir)
        chunks = list(vs.docstore._dict.values())
        _build_bm25(chunks, persist_dir)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["docs"]


def _sparse_search_with_scores(query: str, persist_dir: str, k: int):
    """Raw BM25 hits, sorted best-first. Scores are unbounded
    (higher = more relevant), NOT yet normalized -- see
    `_bm25_score_to_unit` for the fusion-ready version."""
    bm25, docs = _load_bm25(persist_dir)
    scores = bm25.get_scores(_bm25_tokenize(query))
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def _bm25_score_to_unit(raw_score: float) -> float:
    if raw_score <= 0:
        return 0.0
    return raw_score / (raw_score + BM25_SATURATION_K)


# --------------------------------------------------------------------------
# Dense (FAISS/cosine) side
# --------------------------------------------------------------------------

def _l2sq_to_cosine(squared_l2_distance: float) -> float:

    sim = 1.0 - (squared_l2_distance / 2.0)
    return max(-1.0, min(1.0, sim))


def build_vectorstore(docs_dir=None, persist_dir: str = VECTORSTORE_DIR):
    """(Re)build the dense (FAISS, cosine) and sparse (BM25) indexes from
    scratch from the documents folder, then save both to disk."""
    chunks = load_and_chunk(docs_dir) if docs_dir else load_and_chunk()
    if not chunks:
        raise ValueError("No documents found to index. Add files to sample_docs/.")

    vs = FAISS.from_documents(
        chunks,
        embedding=get_embeddings(),
        distance_strategy=DistanceStrategy.COSINE,
        normalize_L2=True,  
    )
    os.makedirs(persist_dir, exist_ok=True)
    vs.save_local(persist_dir)

    _build_bm25(chunks, persist_dir)
    return vs


def load_vectorstore(persist_dir: str = VECTORSTORE_DIR):
    """Load an existing FAISS index from disk, or build one (+ its BM25
    sibling) if missing."""
    index_file = os.path.join(persist_dir, "index.faiss")
    if not os.path.exists(index_file):
        return build_vectorstore(persist_dir=persist_dir)

    return FAISS.load_local(
        persist_dir, get_embeddings(), allow_dangerous_deserialization=True
    )


def get_retriever(persist_dir: str = VECTORSTORE_DIR, k: int = TOP_K):
    vs = load_vectorstore(persist_dir)
    return vs.as_retriever(search_kwargs={"k": k})


def _dense_search_with_scores(query: str, persist_dir: str, k: int):
    """FAISS hits with squared-L2-on-unit-vectors converted to cosine
    similarity (higher = more similar), sorted best-first."""
    vs = load_vectorstore(persist_dir)
    raw = vs.similarity_search_with_score(query, k=k)
    scored = [(doc, _l2sq_to_cosine(dist)) for doc, dist in raw]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored



def _corpus_size(persist_dir: str) -> int:

    vs = load_vectorstore(persist_dir)
    return max(len(vs.docstore._dict), 1)


def hybrid_search_with_scores(
    query: str,
    persist_dir: str = VECTORSTORE_DIR,
    k: int = RETRIEVAL_FETCH_K,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    sparse_weight: float = HYBRID_SPARSE_WEIGHT,
):
    corpus_size = _corpus_size(persist_dir)
    dense = _dense_search_with_scores(query, persist_dir, k=corpus_size)
    sparse = _sparse_search_with_scores(query, persist_dir, k=corpus_size)

    docs_by_id = {}
    dense_by_id = {}
    for doc, cos_sim in dense:
        cid = doc.metadata.get("chunk_id") or id(doc)
        docs_by_id[cid] = doc
        dense_by_id[cid] = cos_sim

    sparse_by_id = {}
    for doc, bm25_raw in sparse:
        cid = doc.metadata.get("chunk_id") or id(doc)
        docs_by_id.setdefault(cid, doc)
        sparse_by_id[cid] = _bm25_score_to_unit(bm25_raw)

    fused = []
    for cid, doc in docs_by_id.items():
        d_score = dense_by_id.get(cid, 0.0)
        s_score = sparse_by_id.get(cid, 0.0)
        fused_score = dense_weight * d_score + sparse_weight * s_score
        fused.append((doc, fused_score))

    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused[:k]


def search_with_scores(query: str, persist_dir: str = VECTORSTORE_DIR, k: int = TOP_K):
    return hybrid_search_with_scores(query, persist_dir=persist_dir, k=k)


def list_indexed_sources(persist_dir: str = VECTORSTORE_DIR) -> list[str]:
    vs = load_vectorstore(persist_dir)
    sources = {
        d.metadata.get("source") for d in vs.docstore._dict.values()
        if d.metadata.get("source")
    }
    return sorted(sources)