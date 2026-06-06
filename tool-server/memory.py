"""Chat memory service — SQLite per-chat semantic recall.

Uses the same schema as the router's memory DB. Embeddings via Fireworks
API. Designed to be called by both the thin router coordinator and the
orchestrator harness.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import pathlib
import re
import sqlite3
import struct
import time
from typing import Optional

import httpx

logger = logging.getLogger("memory")
logging.basicConfig(level=logging.INFO)

DB_PATH = os.getenv("CHAT_MEMORY_DB_PATH", "/app/backend/data/router_mem.db")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "accounts/fireworks/models/nomic-embed-text-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
EMBEDDING_URL = "https://api.fireworks.ai/inference/v1/embeddings"

MEMORY_TOP_K = int(os.getenv("CHAT_MEMORY_TOP_K", "6"))
MEMORY_MIN_TURNS = int(os.getenv("CHAT_MEMORY_MIN_TURNS", "8"))
MEMORY_MAX_PER_CHAT = int(os.getenv("CHAT_MEMORY_MAX_TURNS_PER_CHAT", "100"))

# ── Regex patterns ──────────────────────────────────────────────────────
THINKING_RE = re.compile(r"<thinking[\s\S]*?<\s*/thinking>", re.IGNORECASE)
ROUTER_STATE_RE = re.compile(
    r"<!--\s*ROUTER_STATE:.*?-->\s*|\[ROUTER_STATE:.*?\]\s*", re.IGNORECASE
)
UNVERIFIED_TRAILER_RE = re.compile(
    r"\n*\s*---+\s*\n\s*UNVERIFIED.*?(?:\n|$)", re.IGNORECASE | re.DOTALL
)
ACK_ONLY_RE = re.compile(
    r"^\s*(ok|okay|sure|got it|done|will do|noted|understood)\s*$", re.IGNORECASE
)
ROUTE_TAG_RE = re.compile(
    r"^\s*(?:[🔍🧲📝💬🎨🤖🔧]\s*)?[A-Z_]+\s*(?:_SEARCH)?\s*(?:🔍)?\s*\n", re.IGNORECASE
)

_memory_conn: Optional[sqlite3.Connection] = None
_memory_disabled = False
_embedding_dim: Optional[int] = None
_embedding_dim_warned = False
_init_lock = asyncio.Lock()


# ── Helpers ──────────────────────────────────────────────────────────────

def _f32_pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _f32_unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _clean_content(text: str) -> str:
    """Strip artifacts before storing for semantic recall."""
    t = THINKING_RE.sub("", text)
    t = ROUTER_STATE_RE.sub("", t)
    t = UNVERIFIED_TRAILER_RE.sub("", t)
    t = ROUTE_TAG_RE.sub("", t)
    return t.strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def _fts5_safe_query(query: str) -> str:
    """Escape FTS5 special chars and prefix-match the last token."""
    safe = query.replace('"', "").replace("'", "").replace("*", "")
    safe = re.sub(r"[\^\\\[\]\(\)]", "", safe)
    return " OR ".join(
        f'"{w}"*' if i == len(safe.split()) - 1 else f'"{w}"'
        for i, w in enumerate(safe.split()[:10])
        if len(w) >= 2
    )


# ── DB init ──────────────────────────────────────────────────────────────

async def get_conn() -> Optional[sqlite3.Connection]:
    global _memory_conn, _memory_disabled
    if _memory_disabled:
        return None
    if _memory_conn is not None:
        return _memory_conn

    async with _init_lock:
        if _memory_conn is not None or _memory_disabled:
            return _memory_conn
        try:
            db_dir = pathlib.Path(DB_PATH).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB,
                    created_at REAL NOT NULL,
                    is_summary INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_turns_hash "
                "ON chat_turns(chat_id, content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_turns_chat "
                "ON chat_turns(chat_id, created_at)"
            )

            # FTS5 for hybrid retrieval
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chat_turns_fts "
                "USING fts5(content, content='chat_turns', content_rowid='id')"
            )
            for trigger, timing, op in [
                ("chat_turns_fts_ai", "AFTER INSERT", "new"),
                ("chat_turns_fts_au", "AFTER UPDATE", "new"),
                ("chat_turns_fts_ad", "AFTER DELETE", "old"),
            ]:
                try:
                    conn.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {trigger} "
                        f"{timing} ON chat_turns BEGIN "
                        f"  INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) "
                        f"  VALUES ('delete', {op}.id, {op}.content); "
                        f"  INSERT INTO chat_turns_fts(rowid, content) VALUES ({op}.id, {op}.content); "
                        f"END"
                    )
                except sqlite3.OperationalError:
                    pass

            # Backfill FTS if behind
            try:
                src = conn.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0]
                fts = conn.execute("SELECT COUNT(*) FROM chat_turns_fts").fetchone()[0]
                if fts < src:
                    conn.execute(
                        "INSERT INTO chat_turns_fts(rowid, content) "
                        "SELECT id, content FROM chat_turns WHERE id NOT IN "
                        "(SELECT rowid FROM chat_turns_fts)"
                    )
                    logger.info(f"FTS backfill: {src - fts} rows")
            except Exception:
                pass

            conn.commit()
            _memory_conn = conn
            logger.info(f"Memory DB ready at {DB_PATH}")
            return conn
        except Exception as e:
            logger.error(f"Memory DB init failed: {e}")
            _memory_disabled = True
            return None


# ── Embedding ────────────────────────────────────────────────────────────

async def get_embedding(text: str) -> list[float]:
    if not FIREWORKS_API_KEY:
        logger.error("No FIREWORKS_API_KEY — embedding disabled")
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": EMBEDDING_MODEL, "input": text[:2000]},
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data["data"][0]["embedding"]
            return vec
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return []


# ── Public API ───────────────────────────────────────────────────────────

async def store(chat_id: str, role: str, content: str) -> bool:
    """Store a chat turn. Returns True if stored, False if duplicate or error."""
    conn = await get_conn()
    if not conn:
        return False

    cleaned = _clean_content(content)
    if not cleaned or ACK_ONLY_RE.match(cleaned):
        return False

    chash = _content_hash(cleaned)
    try:
        existing = conn.execute(
            "SELECT 1 FROM chat_turns WHERE chat_id=? AND content_hash=?",
            (chat_id, chash),
        ).fetchone()
        if existing:
            return False

        vec = await get_embedding(cleaned[:2000])
        if not vec:
            return False

        global _embedding_dim, _embedding_dim_warned
        if _embedding_dim is None:
            _embedding_dim = len(vec)
        elif len(vec) != _embedding_dim and not _embedding_dim_warned:
            _embedding_dim_warned = True
            logger.warning(
                f"Embedding dim drift: got {len(vec)}, expected {_embedding_dim}"
            )

        blob = _f32_pack(vec)
        conn.execute(
            "INSERT INTO chat_turns (chat_id, role, content, content_hash, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, cleaned, chash, blob, time.time()),
        )

        # Enforce per-chat cap
        excess = conn.execute(
            "SELECT COUNT(*) - ? FROM chat_turns WHERE chat_id=?",
            (MEMORY_MAX_PER_CHAT, chat_id),
        ).fetchone()[0]
        if excess > 0:
            conn.execute(
                "DELETE FROM chat_turns WHERE id IN ("
                "  SELECT id FROM chat_turns WHERE chat_id=? ORDER BY created_at ASC LIMIT ?"
                ")",
                (chat_id, excess),
            )

        conn.commit()
        return True
    except Exception as e:
        logger.warning(f"Memory store failed: {e}")
        return False


async def sweep_chat(chat_id: str, ttl_days: int = 90) -> int:
    """Delete turns older than ttl_days for a given chat. Returns rows removed."""
    conn = await get_conn()
    if not conn:
        return 0
    try:
        cutoff = time.time() - ttl_days * 86400
        cur = conn.execute(
            "DELETE FROM chat_turns WHERE chat_id=? AND created_at < ?",
            (chat_id, cutoff),
        )
        removed = cur.rowcount
        conn.commit()
        if removed:
            logger.info(f"Memory sweep removed {removed} old rows for chat {chat_id}")
        return removed
    except Exception as e:
        logger.warning(f"Memory sweep failed: {e}")
        return 0


async def recall(
    chat_id: str,
    query: str,
    exclude_hashes: Optional[list[str]] = None,
    top_k: int = None,
) -> list[tuple[str, str]]:
    """Recall relevant prior turns. Returns list of (role, content) tuples."""
    conn = await get_conn()
    if not conn:
        return []

    top_k = top_k or MEMORY_TOP_K
    exclude_hashes = exclude_hashes or []

    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM chat_turns WHERE chat_id=? AND is_summary=0",
            (chat_id,),
        ).fetchone()[0]
        if total < MEMORY_MIN_TURNS:
            return []

        rows = conn.execute(
            "SELECT role, content, content_hash, embedding "
            "FROM chat_turns WHERE chat_id=? AND embedding IS NOT NULL",
            (chat_id,),
        ).fetchall()
        if not rows:
            return []

        qvec = await get_embedding(query[:2000])
        if not qvec:
            return []

        exclude_set = set(exclude_hashes)
        scored = []
        for role, content, chash, blob in rows:
            if chash in exclude_set:
                continue
            try:
                vec = _f32_unpack(blob)
                cosine = _cosine_similarity(qvec, vec)
            except Exception:
                cosine = 0.0

            # BM25 via FTS5
            bm25_score = 0.0
            try:
                fts_q = _fts5_safe_query(query)
                if fts_q:
                    fmatch = conn.execute(
                        "SELECT rowid, bm25(chat_turns_fts, 0.75, 0.0) "
                        "FROM chat_turns_fts WHERE chat_turns_fts MATCH ?",
                        (fts_q,),
                    ).fetchall()
                    bm25_lookup = {r[0]: r[1] for r in fmatch}
                    row_id = conn.execute(
                        "SELECT id FROM chat_turns WHERE chat_id=? AND content_hash=?",
                        (chat_id, chash),
                    ).fetchone()
                    if row_id and row_id[0] in bm25_lookup:
                        bm25_score = bm25_lookup[row_id[0]]
            except Exception:
                pass

            # Weighted hybrid: 60% cosine, 40% BM25
            final = 0.6 * cosine + 0.4 * min(1.0, max(0.0, -bm25_score / 10.0))
            scored.append((final, role, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        recall_score_threshold = 0.0
        result = [
            (role, content[:800])
            for score, role, content in scored[:top_k]
            if score > recall_score_threshold
        ]
        return result
    except Exception as e:
        logger.warning(f"Memory recall failed: {e}")
        return []
