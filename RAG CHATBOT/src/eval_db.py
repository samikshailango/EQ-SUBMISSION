import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

from config import EVAL_DB_PATH
from logging_config import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id                TEXT PRIMARY KEY,
    ts                REAL NOT NULL,
    question          TEXT NOT NULL,
    answer            TEXT NOT NULL,
    retrieved_contexts TEXT NOT NULL,   -- JSON list[str]
    sources           TEXT NOT NULL,    -- JSON list[dict] (chunk_id/source/page)
    blocked           INTEGER NOT NULL, -- 0/1, query-side PII guardrail fired
    clarification      INTEGER NOT NULL DEFAULT 0, -- 0/1, we asked instead of answered
    redacted_entities TEXT NOT NULL,    -- JSON list[str]
    followups         TEXT NOT NULL DEFAULT '[]' -- JSON list[str]
);
"""

# feedback: NULL = no vote yet, 1 = thumbs up, -1 = thumbs down. Added
# after the original schema, so it's applied as a migration (ALTER TABLE)
# rather than baked into _SCHEMA -- CREATE TABLE IF NOT EXISTS is a no-op
# against a database file that already exists from before this feature,
# so an old eval_data/interactions.db needs this extra step to gain the
# column instead of silently missing it.
_MIGRATIONS = [
    "ALTER TABLE interactions ADD COLUMN feedback INTEGER",
]


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(EVAL_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(EVAL_DB_PATH)
    try:
        conn.execute(_SCHEMA)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists -- migration already applied
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_interaction(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    sources: list[dict],
    blocked: bool = False,
    clarification: bool = False,
    redacted_entities: list[str] | None = None,
    followups: list[str] | None = None,
) -> str:
    """Persist one turn. Returns the row id."""
    row_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO interactions "
            "(id, ts, question, answer, retrieved_contexts, sources, blocked, "
            " clarification, redacted_entities, followups) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row_id,
                time.time(),
                question,
                answer,
                json.dumps(retrieved_contexts),
                json.dumps(sources),
                int(blocked),
                int(clarification),
                json.dumps(redacted_entities or []),
                json.dumps(followups or []),
            ),
        )
    logger.debug("logged interaction %s (blocked=%s, clarification=%s)",
                 row_id, blocked, clarification)
    return row_id


def update_feedback(row_id: str, positive: bool) -> bool:
    """Attach a thumbs up/down vote to an already-logged turn. Returns
    True if a row was actually updated (False if the id doesn't exist,
    e.g. a blocked/clarification turn whose id was never generated)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (1 if positive else -1, row_id),
        )
        return cur.rowcount > 0


def fetch_interactions(
    limit: int | None = None,
    include_blocked: bool = False,
    include_clarifications: bool = False,
) -> list[dict]:
    """Read logged interactions back out, most recent first.

    By default excludes blocked/clarification turns since those have no
    real "answer" to score against context -- RAGAS's core metrics assume
    a genuine attempt at answering.
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM interactions"
        clauses = []
        if not include_blocked:
            clauses.append("blocked = 0")
        if not include_clarifications:
            clauses.append("clarification = 0")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY ts DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()

    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "ts": r["ts"],
            "question": r["question"],
            "answer": r["answer"],
            "retrieved_contexts": json.loads(r["retrieved_contexts"]),
            "sources": json.loads(r["sources"]),
            "blocked": bool(r["blocked"]),
            "clarification": bool(r["clarification"]),
            "redacted_entities": json.loads(r["redacted_entities"]),
            "followups": json.loads(r["followups"]),
            "feedback": r["feedback"] if "feedback" in r.keys() else None,
        })
    return out


def count_interactions() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]


def feedback_summary() -> dict:
    """Quick thumbs up/down tally for the sidebar -- a live signal of real
    user-perceived answer quality, independent of (and much cheaper than)
    a RAGAS run."""
    with _connect() as conn:
        up = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE feedback = 1"
        ).fetchone()[0]
        down = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE feedback = -1"
        ).fetchone()[0]
    return {"up": up, "down": down}
