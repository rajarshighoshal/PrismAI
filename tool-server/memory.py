"""Chat memory service — SQLite per-chat semantic recall.

Uses the same schema as the router's memory DB. Embeddings via Fireworks
API. Designed to be called by both the thin router coordinator and the
orchestrator harness.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import math
import json
import os
import pathlib
import re
import sqlite3
import struct
import time
import weakref
from typing import Optional

import httpx

import llm  # provider-chain LLM (deepseek-direct -> fireworks)

logger = logging.getLogger("memory")
logging.basicConfig(level=logging.INFO)

DB_PATH = os.getenv("CHAT_MEMORY_DB_PATH", "/app/backend/data/router_mem.db")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "accounts/fireworks/models/qwen3-embedding-8b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))
EMBEDDING_URL = "https://api.fireworks.ai/inference/v1/embeddings"

MEMORY_TOP_K = int(os.getenv("CHAT_MEMORY_TOP_K", "6"))
# The orchestrator now decides WHEN to recall (only when a conversation overflows
# its context budget), so this is just a floor against recalling from an essentially
# empty chat. 1 = retrieve whatever is stored when asked.
MEMORY_MIN_TURNS = int(os.getenv("CHAT_MEMORY_MIN_TURNS", "1"))
MEMORY_MAX_PER_CHAT = int(os.getenv("CHAT_MEMORY_MAX_TURNS_PER_CHAT", "100"))
# Keep a bounded version history per chat so an edit-heavy chat can't grow the
# deliverables table without limit.
DELIVERABLES_MAX_PER_CHAT = int(os.getenv("DELIVERABLES_MAX_PER_CHAT", "30"))
ENABLE_CHAT_MEMORY_COMPRESSION = os.getenv(
    "ENABLE_CHAT_MEMORY_COMPRESSION", "true"
).lower() not in {"0", "false", "no"}
MEMORY_COMPRESS_WHEN_OVER = int(os.getenv("CHAT_MEMORY_COMPRESS_WHEN_OVER", "60"))
MEMORY_COMPRESS_CHUNK = int(os.getenv("CHAT_MEMORY_COMPRESS_CHUNK", "20"))
COMPRESSION_MODEL = os.getenv(
    "COMPRESSION_MODEL", "accounts/fireworks/models/deepseek-v4-flash"
)
CHAT_COMPLETIONS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

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
# All SQLite work runs on ONE dedicated worker thread: the calls never block the event
# loop, yet are naturally serialized (single worker) and always touch the connection
# from the same thread — so no locks (no deadlock) and no cross-thread sqlite errors.
# The connection is opened check_same_thread=False so this off-loop thread may use it.
_db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="memdb")


async def _db(fn):
    """Run a synchronous DB block off the event loop on the dedicated DB thread."""
    return await asyncio.get_running_loop().run_in_executor(_db_executor, fn)
# Weak values so a chat's lock is dropped once no longer in use, instead of the
# dict growing one entry per chat_id forever.
_compression_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()


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
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")

            # Migration: add is_summary column if missing (older DBs)
            try:
                conn.execute("ALTER TABLE chat_turns ADD COLUMN is_summary INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # column already exists

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

            # Persistent deliverables — the verified document a turn produced (cover
            # letter, report, …). Append-only for version history, so a later turn can
            # surgically EDIT the real prior artifact instead of reconstructing it from
            # scratch (which produced a different document and blew the token budget).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deliverables (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    filename TEXT,
                    fmt TEXT,
                    content TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_deliverables_chat "
                "ON deliverables(chat_id, version)"
            )

            # Per-model-call usage (tokens; $ computed at query time from prices).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    model TEXT NOT NULL,
                    label TEXT,
                    in_tok INTEGER DEFAULT 0,
                    out_tok INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts)")
            for col in ("source_id TEXT", "user_id TEXT"):
                try:
                    conn.execute(f"ALTER TABLE usage ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            # Dedup for swept rows (OWUI chat messages re-seen across sweeps).
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_src "
                         "ON usage(source_id) WHERE source_id IS NOT NULL")
            conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

            # FTS5 for hybrid retrieval
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chat_turns_fts "
                "USING fts5(content, content='chat_turns', content_rowid='id')"
            )
            # FTS5 external-content triggers — each op needs its OWN body. A shared
            # 'delete'+'insert' body corrupts the index on INSERT: the 'delete'
            # targets a rowid not yet in the FTS index → "database disk image is
            # malformed" on the first write, which silently killed ALL memory.
            for trigger, timing, body in [
                ("chat_turns_fts_ai", "AFTER INSERT",
                 "INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);"),
                ("chat_turns_fts_ad", "AFTER DELETE",
                 "INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) "
                 "VALUES ('delete', old.id, old.content);"),
                ("chat_turns_fts_au", "AFTER UPDATE",
                 "INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) "
                 "VALUES ('delete', old.id, old.content); "
                 "INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);"),
            ]:
                try:
                    conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                    conn.execute(
                        f"CREATE TRIGGER {trigger} {timing} "
                        f"ON chat_turns BEGIN {body} END"
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


async def summarize_turns(turns: list[tuple[str, str]]) -> str:
    if not turns:
        return ""
    block = "\n".join(f"[{role}]: {content[:700]}" for role, content in turns)
    prompt = (
        "Summarize these conversation turns into one concise, information-dense "
        "memory note for later semantic recall. Preserve important facts, decisions, "
        "user preferences, constraints, unresolved TODOs, code paths, commands, "
        "URLs, paper titles, eval outcomes, and model/routing decisions. Omit "
        "greetings, pleasantries, repeated phrasing, and transient details unless "
        "they affect future work. Do not invent facts. Target 200-400 words in one "
        "paragraph.\n\n"
        f"CONVERSATION TURNS:\n{block}\n\n"
        "MEMORY SUMMARY:"
    )
    try:
        # DeepSeek-direct primary, Fireworks fallback (via llm chain). Compression is a cheap
        # high-volume classifier loop -> pin fast ("none"), else DeepSeek-direct defaults to "high".
        text = await llm.chat(COMPRESSION_MODEL, [{"role": "user", "content": prompt}],
                              max_tokens=500, temperature=0.0, reasoning_effort="none")
        return THINKING_RE.sub("", text).strip()
    except Exception as e:
        logger.warning(f"Memory compression LLM failed: {e}")
        return ""


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
        existing = await _db(lambda: conn.execute(
            "SELECT 1 FROM chat_turns WHERE chat_id=? AND content_hash=?",
            (chat_id, chash),
        ).fetchone())
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

        def _write():
            conn.execute(
                "INSERT INTO chat_turns (chat_id, role, content, content_hash, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, role, cleaned, chash, blob, time.time()),
            )
            # Enforce per-chat cap without deleting summary rows first.
            excess = conn.execute(
                "SELECT COUNT(*) - ? FROM chat_turns WHERE chat_id=?",
                (MEMORY_MAX_PER_CHAT, chat_id),
            ).fetchone()[0]
            if excess > 0:
                conn.execute(
                    "DELETE FROM chat_turns WHERE id IN ("
                    "  SELECT id FROM chat_turns WHERE chat_id=? AND is_summary=0 "
                    "  ORDER BY created_at ASC LIMIT ?"
                    ")",
                    (chat_id, excess),
                )
            conn.commit()

        await _db(_write)
        return True
    except Exception as e:
        logger.warning(f"Memory store failed: {e}")
        return False


async def store_deliverable(chat_id: str, content: str, filename: str = "", fmt: str = "") -> int:
    """Persist a delivered document for this chat. Append-only: each call is a new
    version, so edits keep history (Canvas-style). Returns the new version (0 on failure)."""
    conn = await get_conn()
    if not conn or not (chat_id and (content or "").strip()):
        return 0
    try:
        def _write():
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM deliverables WHERE chat_id=?",
                (chat_id,),
            ).fetchone()
            version = (row[0] or 0) + 1
            conn.execute(
                "INSERT INTO deliverables (chat_id, filename, fmt, content, version, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, filename, fmt, content, version, time.time()),
            )
            # Bound the per-chat history: drop versions older than the most recent N.
            conn.execute(
                "DELETE FROM deliverables WHERE chat_id=? AND version <= ?",
                (chat_id, version - DELIVERABLES_MAX_PER_CHAT),
            )
            conn.commit()
            return version
        return await _db(_write)
    except Exception as e:
        logger.warning(f"Deliverable store failed: {e}")
        return 0


async def get_deliverable(chat_id: str) -> Optional[dict]:
    """Return the LATEST deliverable for a chat (the document a follow-up edits), or None."""
    conn = await get_conn()
    if not conn or not chat_id:
        return None
    try:
        row = await _db(lambda: conn.execute(
            "SELECT filename, fmt, content, version FROM deliverables "
            "WHERE chat_id=? ORDER BY version DESC LIMIT 1",
            (chat_id,),
        ).fetchone())
        if not row:
            return None
        return {"filename": row[0], "fmt": row[1], "content": row[2], "version": row[3]}
    except Exception as e:
        logger.warning(f"Deliverable get failed: {e}")
        return None


async def log_usage(model: str, label: str, in_tok: int, out_tok: int, user_id: str = "") -> bool:
    """One row per model call (fire-and-forget from the orchestrator's tracer).
    Tokens only — $ is computed at query time from a price table, so price
    corrections re-price history. user_id is the OWUI user this call served."""
    conn = await get_conn()
    if not conn or not model:
        return False
    try:
        def _write():
            conn.execute(
                "INSERT INTO usage (ts, model, label, in_tok, out_tok, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), model, label or "?", int(in_tok or 0), int(out_tok or 0),
                 user_id or None),
            )
            conn.commit()
        await _db(_write)
        return True
    except Exception as e:
        logger.warning(f"usage log failed: {e}")
        return False


OWUI_DB_PATH = os.getenv("OWUI_DB_PATH", "/app/backend/data/webui.db")


async def sweep_owui_usage() -> int:
    """Pull token usage for DIRECT OWUI model chats (deepseek/glm/kimi/… used outside
    PrismAI) from webui.db into the ledger. OWUI persists usage on every assistant
    message; PrismAI's own turns are skipped (already ledgered at call time). Dedup via
    source_id, so re-sweeping is free. Returns rows added."""
    conn = await get_conn()
    if not conn:
        return 0
    try:
        def _sweep():
            wm_row = conn.execute("SELECT v FROM kv WHERE k='usage_sweep_watermark'").fetchone()
            wm = float(wm_row[0]) if wm_row else 0.0
            src = sqlite3.connect(f"file:{OWUI_DB_PATH}?mode=ro", uri=True)
            added, new_wm = 0, wm
            try:
                for cid, owner, blob, upd in src.execute(
                        "SELECT id, user_id, chat, updated_at FROM chat WHERE updated_at > ?",
                        (wm - 3600,)):  # 1h overlap; dedup absorbs the rescan
                    new_wm = max(new_wm, float(upd or 0))
                    try:
                        msgs = (json.loads(blob).get("history") or {}).get("messages") or {}
                    except Exception:
                        continue
                    for mid, m in msgs.items():
                        u = m.get("usage") or {}
                        model = str(m.get("model") or "")
                        if (m.get("role") != "assistant" or not u
                                or model.startswith("PrismAI") or not model):
                            continue
                        cur = conn.execute(
                            "INSERT OR IGNORE INTO usage (ts, model, label, in_tok, out_tok, source_id, user_id) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (float(m.get("timestamp") or upd or time.time()),
                             model.split("/")[-1], "owui-chat",
                             int(u.get("prompt_tokens") or u.get("input_tokens") or 0),
                             int(u.get("completion_tokens") or u.get("output_tokens") or 0),
                             f"{cid}:{mid}", owner or None))
                        added += cur.rowcount
            finally:
                src.close()
            conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES ('usage_sweep_watermark', ?)",
                         (str(new_wm),))
            conn.commit()
            return added
        added = await _db(_sweep)
        if added:
            logger.info(f"usage sweep: +{added} rows from OWUI chats")
        return added
    except Exception as e:
        logger.warning(f"usage sweep failed: {e}")
        return 0


def _owui_user_names() -> dict:
    """Map OWUI user_id -> display name/email, read straight from webui.db (best effort)."""
    try:
        src = sqlite3.connect(f"file:{OWUI_DB_PATH}?mode=ro", uri=True)
        try:
            return {uid: (name or email or uid)
                    for uid, name, email in src.execute("SELECT id, name, email FROM user")}
        finally:
            src.close()
    except Exception:
        return {}


async def usage_summary(cost_fn=None, since=None, until=None) -> dict:
    """Aggregate the usage ledger by user, model, job, month, and day over an optional
    [since, until) unix-time window (both None = all time). cost_fn(model, in, out)->$ is
    applied PER ROW, so by-user/by-job dollars stay correct even when a bucket mixes
    models. The API layer maps a period (a month / last-3 / all) to the time window."""
    conn = await get_conn()
    if not conn:
        return {}
    cost_fn = cost_fn or (lambda m, i, o: 0.0)
    try:
        def _read():
            q = "SELECT ts, model, label, in_tok, out_tok, user_id FROM usage"
            cond, args = [], []
            if since is not None:
                cond.append("ts >= ?"); args.append(since)
            if until is not None:
                cond.append("ts < ?"); args.append(until)
            if cond:
                q += " WHERE " + " AND ".join(cond)
            return conn.execute(q, args).fetchall()
        rows = await _db(_read)
        names = _owui_user_names()
        import datetime as _dt
        by_model, by_label, by_day, by_user, by_month = {}, {}, {}, {}, {}
        total = 0.0
        for ts, model, label, i, o, uid in rows:
            usd = cost_fn(model, i or 0, o or 0)
            total += usd
            when = _dt.datetime.utcfromtimestamp(ts)
            user = names.get(uid, uid) if uid else "(unattributed)"
            for bucket, key in ((by_model, model), (by_label, label),
                                (by_day, when.strftime("%Y-%m-%d")),
                                (by_month, when.strftime("%Y-%m")), (by_user, user)):
                b = bucket.setdefault(key, {"in": 0, "out": 0, "calls": 0, "usd": 0.0})
                b["in"] += i; b["out"] += o; b["calls"] += 1; b["usd"] += usd
        for bucket in (by_model, by_label, by_day, by_user, by_month):
            for b in bucket.values():
                b["usd"] = round(b["usd"], 4)
        return {"calls": len(rows), "total_usd": round(total, 2),
                "by_model": by_model, "by_label": by_label, "by_day": by_day,
                "by_month": by_month, "by_user": by_user}
    except Exception as e:
        logger.warning(f"usage summary failed: {e}")
        return {}


async def last_active(chat_id: str) -> Optional[float]:
    """Most recent stored-turn time for a chat (unix seconds), or None — lets the agent
    tell the model how long since the user's previous message (resume-after-gap awareness)."""
    conn = await get_conn()
    if not conn or not chat_id:
        return None
    try:
        row = await _db(lambda: conn.execute(
            "SELECT MAX(created_at) FROM chat_turns WHERE chat_id=?", (chat_id,)
        ).fetchone())
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _compression_lock(chat_id: str) -> asyncio.Lock:
    lock = _compression_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _compression_locks[chat_id] = lock
    return lock


async def maybe_compress_chat(chat_id: str) -> int:
    """Summarize oldest raw rows into one durable summary row when a chat grows.

    Returns the number of raw rows compacted. Fails soft and never deletes raw
    rows unless a summary row has been inserted or already exists.
    """
    if not ENABLE_CHAT_MEMORY_COMPRESSION or not chat_id:
        return 0
    lock = _compression_lock(chat_id)
    if lock.locked():
        return 0
    async with lock:
        return await _compress_chat_unlocked(chat_id)


async def _compress_chat_unlocked(chat_id: str) -> int:
    conn = await get_conn()
    if not conn:
        return 0
    try:
        def _read():
            total = conn.execute(
                "SELECT COUNT(*) FROM chat_turns WHERE chat_id=? AND is_summary=0",
                (chat_id,),
            ).fetchone()[0]
            if total <= MEMORY_COMPRESS_WHEN_OVER:
                return None
            return conn.execute(
                "SELECT id, role, content FROM chat_turns "
                "WHERE chat_id=? AND is_summary=0 "
                "ORDER BY created_at ASC LIMIT ?",
                (chat_id, MEMORY_COMPRESS_CHUNK),
            ).fetchall()
        rows = await _db(_read)
        if not rows or len(rows) < 5:
            return 0

        summary = await summarize_turns([(role, content) for _, role, content in rows])
        if len(summary) < 50:
            return 0

        chash = _content_hash(summary)
        existing = await _db(lambda: conn.execute(
            "SELECT 1 FROM chat_turns WHERE chat_id=? AND content_hash=?",
            (chat_id, chash),
        ).fetchone())
        ids = [row_id for row_id, _, _ in rows]
        qs = ",".join("?" * len(ids))

        blob = None
        if not existing:
            # Only delete the raw rows once a SEARCHABLE summary is in place. If the
            # embedding call fails, keep the raw rows and retry next time rather than
            # leaving a NULL-embedding summary that can never be recalled.
            vec = await get_embedding(summary[:2000])
            if not vec:
                logger.warning("Memory compression: embedding failed for chat=%s, keeping raw rows", str(chat_id)[:20])
                return 0
            blob = _f32_pack(vec)

        def _finalize():
            if not existing:
                conn.execute(
                    "INSERT INTO chat_turns "
                    "(chat_id, role, content, content_hash, embedding, created_at, is_summary) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chat_id, "summary", summary, chash, blob, time.time(), 1),
                )
            conn.execute(f"DELETE FROM chat_turns WHERE id IN ({qs})", tuple(ids))
            conn.commit()
        await _db(_finalize)
        logger.info(
            "Memory compression: chat=%s summarized %d raw rows",
            str(chat_id)[:20],
            len(ids),
        )
        return len(ids)
    except Exception as e:
        logger.warning(f"Memory compression failed: {e}")
        return 0


async def sweep_chat(chat_id: str, ttl_days: int = 90) -> int:
    """Delete turns older than ttl_days for a given chat. Returns rows removed."""
    conn = await get_conn()
    if not conn:
        return 0
    try:
        cutoff = time.time() - ttl_days * 86400

        def _delete():
            cur = conn.execute(
                "DELETE FROM chat_turns WHERE chat_id=? AND created_at < ?",
                (chat_id, cutoff),
            )
            removed = cur.rowcount
            conn.commit()
            return removed
        removed = await _db(_delete)
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
        def _read():
            total = conn.execute(
                "SELECT COUNT(*) FROM chat_turns WHERE chat_id=? AND embedding IS NOT NULL",
                (chat_id,),
            ).fetchone()[0]
            if total < MEMORY_MIN_TURNS:
                return None
            return conn.execute(
                "SELECT id, role, content, content_hash, embedding "
                "FROM chat_turns WHERE chat_id=? AND embedding IS NOT NULL",
                (chat_id,),
            ).fetchall()
        rows = await _db(_read)
        if not rows:
            return []

        qvec = await get_embedding(query[:2000])
        if not qvec:
            return []

        # BM25 over the whole chat in ONE query (FTS rowid == chat_turns.id),
        # instead of re-querying the index for every row.
        def _bm25():
            try:
                fts_q = _fts5_safe_query(query)
                if not fts_q:
                    return {}
                fmatch = conn.execute(
                    "SELECT rowid, bm25(chat_turns_fts, 0.75, 0.0) "
                    "FROM chat_turns_fts WHERE chat_turns_fts MATCH ?",
                    (fts_q,),
                ).fetchall()
                return {r[0]: r[1] for r in fmatch}
            except Exception:
                return {}
        bm25_lookup = await _db(_bm25)

        exclude_set = set(exclude_hashes)
        scored = []
        for row_id, role, content, chash, blob in rows:
            if chash in exclude_set:
                continue
            try:
                cosine = _cosine_similarity(qvec, _f32_unpack(blob))
            except Exception:
                cosine = 0.0
            bm25_score = bm25_lookup.get(row_id, 0.0)
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
