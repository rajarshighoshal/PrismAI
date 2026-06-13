"""Per-chat semantic memory: SQLite + FTS5 hybrid retrieval with compression."""

import asyncio
import hashlib
import logging
import math
import os
import random
import re
import sqlite3
import struct
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_ACK_ONLY_RE = re.compile(
    r"^\s*(ok(ay)?|yes|yep|yeah|no|nope|thanks?|thank\s*you|cool|"
    r"got\s*it|sure|alright|fine|right|hmm+|uh+|mm+)[\s!.,?]*$",
    re.IGNORECASE,
)

_VERIFICATION_TRAILER_RE = re.compile(
    r"\n\n---\n⚠️\s*\*\*Verification note:.*?(?=\n*$|\Z)",
    re.DOTALL,
)

_THINKING_BLOCK_RE = re.compile(
    r"<think>.*?</think>|<tool_call>.*?</tool_call>", re.DOTALL
)

# Prompt-injection patterns to redact from user text injected into LLM prompts.
_INJECTION_RE = re.compile(
    r"(?i)\b("
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|messages?)"
    r"|disregard\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|messages?)"
    r"|forget\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|messages?)"
    r"|new\s+instructions?\s*:"
    r"|system\s+prompt\s*:"
    r")\b.*"
)


def _sanitize_user_text(text: str) -> str:
    """Remove prompt-injection patterns and code fences from user text."""
    return _INJECTION_RE.sub("[redacted]", text.replace("```", "")).strip()


_ROUTER_STATE_STRIP_RE = re.compile(
    r"\[ROUTER_STATE:\s*[A-Z]+(?:_SEARCH)?\]"
    r"|<!--\s*ROUTER_STATE:\s*[A-Z]+(?:_SEARCH)?\s*-->",
    re.IGNORECASE,
)

_ROUTE_HEADER_RE = re.compile(
    r"\A`[^\n`]*\b(?:FACTUAL|REASONING|CODING|RESEARCH|CASUAL|WRITING|FETCH)\b[^\n`]*`\s*\n(?:>\s*[^\n]*\n)?\s*",
    re.IGNORECASE,
)

_FOLLOWUP_STARTERS = re.compile(
    r"^\s*(and|also|now|but|so|or|plus|what\s+about|how\s+about|what\s+if|yeah|yes|ok|okay|more|again)\b",
    re.IGNORECASE,
)
_PRONOUN_REF = re.compile(
    r"\b(it|its|it's|that|this|these|those|they|them|their|he|she|him|her|hers|his)\b",
    re.IGNORECASE,
)


def _f32_pack(vec: list) -> bytes:
    """Serialize a float list to raw float32 bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _f32_unpack(blob: bytes) -> list:
    """Deserialize float32 bytes back to a Python list."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _memory_content_hash(text: str) -> str:
    """SHA-256 of normalized text. Used for chat-scoped dedup."""
    return hashlib.sha256(text.strip().encode("utf-8", errors="replace")).hexdigest()


def _first_name(raw: Optional[str]) -> str:
    """Pull a sensible first name out of an OWUI __user__.name.

    Handles common shapes:
      "Jane Doe"               → "Jane"
      "Alex Chen"              → "Alex"
      "jane@example.com"       → "jane"      (local-part of email)
      "admin"                  → "admin"     (single token, as-is)
      None / "" / punctuation  → ""
    """
    if not raw:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if "@" in s:
        s = s.split("@", 1)[0]
    m = re.match(r"[A-Za-z][A-Za-z'\-]{0,39}", s)
    return m.group(0) if m else ""


def _fts5_safe_query(text: str) -> str:
    """Turn free-form text into a safe FTS5 MATCH pattern.

    Default FTS5 syntax treats many chars specially (", *, :, (, )). We
    extract alphanumeric tokens, quote each, and join with OR so any
    token hit counts. Bounded at 16 tokens to keep queries cheap.
    Returns '' if no usable tokens — caller should skip BM25.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text or "")
    words = words[:16]
    if not words:
        return ""
    return " OR ".join(f'"{w}"' for w in words)


def _clean_for_memory(text: str) -> str:
    """Strip mechanical artifacts before storing a turn as memory.

    Removes (in order): thinking blocks, ROUTER_STATE tags, the route-tag
    header, and the verifier's UNVERIFIED trailer. Leaves the actual
    user-visible answer intact.
    """
    if not text:
        return ""
    text = _THINKING_BLOCK_RE.sub("", text)
    text = _ROUTER_STATE_STRIP_RE.sub("", text)
    text = _ROUTE_HEADER_RE.sub("", text, count=1)
    text = _VERIFICATION_TRAILER_RE.sub("", text)
    return text.strip()


def _cosine_similarity(v1: list, v2: list) -> float:
    """Cosine similarity between two float vectors (pure Python, no numpy)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return 0.0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)


def _is_followup_query(query: str) -> bool:
    """True when a query looks like a follow-up (short, starts with conjunction/pronoun)."""
    words = query.split()
    if not words or len(words) > 15:
        return False
    if _FOLLOWUP_STARTERS.match(query):
        return True
    if len(words) <= 6 and _PRONOUN_REF.search(query):
        return True
    return False


# ---------------------------------------------------------------------------
# ChatMemory — per-chat semantic recall with compression and sweeping
# ---------------------------------------------------------------------------

class ChatMemory:
    """Persistent per-chat semantic memory.

    Manages SQLite storage, embedding-based recall with hybrid BM25 scoring,
    lossy compression (summarize old turns), referential sweep (delete orphaned
    rows when OWUI chats are deleted), and TTL sweep.

    Dependencies (injected):
      - get_embedding: async fn(text) → list[float] or []
      - call_llm: async fn(prompt, model, max_tokens, fallback_chain, log_role, log_chat_id) → str or None
      - valves: object with config attrs (ENABLE_CHAT_MEMORY, CHAT_MEMORY_DB_PATH, etc.)
    """

    def __init__(
        self,
        get_embedding: Callable,
        call_llm: Callable,
        valves,
    ):
        self.get_embedding = get_embedding
        self.call_llm = call_llm
        self.valves = valves
        self._conn: Optional[sqlite3.Connection] = None
        self._disabled: bool = False
        self._init_lock = asyncio.Lock()
        # Per-chat compression locks — prevents simultaneous compression jobs
        # on the same chat_id, which would cost double LLM tokens and produce
        # duplicate summary rows.
        self._compression_locks: dict[str, asyncio.Lock] = {}
        # Expected embedding dimension, captured on the first stored row.
        self._embedding_dim: Optional[int] = None
        self._embedding_dim_warned: bool = False

    # ── DB connection ──────────────────────────────────────────────────

    async def _get_conn(self) -> Optional[sqlite3.Connection]:
        """Lazy-open the chat memory DB. None if disabled or init failed."""
        if not self.valves.ENABLE_CHAT_MEMORY or self._disabled:
            return None
        if self._conn is not None:
            return self._conn
        async with self._init_lock:
            if self._conn is not None:
                return self._conn
            try:
                path = self.valves.CHAT_MEMORY_DB_PATH
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                conn = sqlite3.connect(path, check_same_thread=False, timeout=5)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding BLOB,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_hash "
                    "ON chat_turns(chat_id, content_hash)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat ON chat_turns(chat_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_created ON chat_turns(created_at)"
                )
                # Usage / analytics log.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS request_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL NOT NULL,
                        chat_id TEXT,
                        model TEXT NOT NULL,
                        call_role TEXT,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER,
                        latency_ms INTEGER,
                        success INTEGER DEFAULT 1,
                        fallback INTEGER DEFAULT 0,
                        error TEXT
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_log_ts ON request_log(ts)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_log_chat_ts "
                    "ON request_log(chat_id, ts)"
                )
                # FTS5 for hybrid retrieval.
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chat_turns_fts USING fts5("
                    "content, content='chat_turns', content_rowid='id'"
                    ")"
                )
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

                # One-time FTS backfill for existing DBs.
                try:
                    src = conn.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0]
                    fts = conn.execute("SELECT COUNT(*) FROM chat_turns_fts").fetchone()[0]
                    if fts < src:
                        conn.execute(
                            "INSERT INTO chat_turns_fts(chat_turns_fts) VALUES('rebuild')"
                        )
                        logger.info("chat_turns_fts rebuilt from %d rows (was %d)", src, fts)
                except Exception as e:
                    logger.info("chat_turns_fts backfill skipped: %s", e)

                conn.commit()
                self._conn = conn
                logger.info("Chat memory DB ready at %s", path)
                return conn
            except Exception as e:
                logger.warning(
                    "Chat memory DB unavailable (%s) — memory disabled for this process.", e
                )
                self._disabled = True
                return None

    # ── Store ──────────────────────────────────────────────────────────

    async def store_turn(self, chat_id: str, role: str, raw_content: str) -> None:
        """Store one turn. Idempotent via (chat_id, content_hash).

        Strips mechanical detail first (think blocks, route tag, verification
        trailer) so stored text is just the actual human-visible content.
        Pure-acknowledgment turns ("ok", "thanks") are skipped.
        """
        if not chat_id:
            return
        content = _clean_for_memory(raw_content)
        if not content or _ACK_ONLY_RE.match(content):
            return
        conn = await self._get_conn()
        if conn is None:
            return
        try:
            ch = _memory_content_hash(content)
            already = conn.execute(
                "SELECT 1 FROM chat_turns WHERE chat_id=? AND content_hash=? LIMIT 1",
                (chat_id, ch),
            ).fetchone()
            if already:
                return
            vec = await self.get_embedding(content[:2000])
            emb_blob = None
            if vec:
                if self._embedding_dim is None:
                    self._embedding_dim = len(vec)
                elif len(vec) != self._embedding_dim and not self._embedding_dim_warned:
                    logger.warning(
                        "Embedding dimension changed: expected %d, got %d. "
                        "EMBEDDING_MODEL may have been updated.",
                        self._embedding_dim, len(vec),
                    )
                    self._embedding_dim_warned = True
                emb_blob = _f32_pack(vec)

            conn.execute(
                "INSERT OR IGNORE INTO chat_turns "
                "(chat_id, role, content, content_hash, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, role, content, ch, emb_blob, time.time()),
            )
            max_n = self.valves.CHAT_MEMORY_MAX_TURNS_PER_CHAT
            conn.execute(
                "DELETE FROM chat_turns WHERE id IN ("
                "  SELECT id FROM chat_turns WHERE chat_id=? "
                "  ORDER BY created_at DESC LIMIT -1 OFFSET ?"
                ")",
                (chat_id, max_n),
            )
            conn.commit()
        except Exception as e:
            logger.warning(
                "Chat memory store failed (chat=%s): %s", str(chat_id)[:20], e
            )

    # ── Recall ─────────────────────────────────────────────────────────

    async def recall(
        self,
        chat_id: str,
        query: str,
        exclude_hashes: set,
    ) -> list[tuple[str, str]]:
        """Top-K most relevant prior turns from THIS chat only.

        Uses hybrid scoring (cosine embedding + BM25 keyword) when
        ENABLE_HYBRID_RETRIEVAL is on. Falls back to cosine-only on FTS error.

        Returns list of (role, content). Empty when memory disabled, chat
        has < CHAT_MEMORY_MIN_TURNS rows, or query embedding fails.
        """
        if not chat_id or not query.strip():
            return []
        conn = await self._get_conn()
        if conn is None:
            return []
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM chat_turns WHERE chat_id=?", (chat_id,)
            ).fetchone()[0]
            if total < self.valves.CHAT_MEMORY_MIN_TURNS:
                return []

            rows = list(
                conn.execute(
                    "SELECT role, content, content_hash, embedding "
                    "FROM chat_turns WHERE chat_id=? AND embedding IS NOT NULL",
                    (chat_id,),
                )
            )
            if not rows:
                return []
            qvec = await self.get_embedding(query[:2000])
            if not qvec:
                return []

            content_by_hash: dict[str, tuple[str, str]] = {}
            cos_by_hash: dict[str, float] = {}
            qdim = len(qvec)
            for role, content, ch, emb_blob in rows:
                if ch in exclude_hashes or not emb_blob:
                    continue
                try:
                    vec = _f32_unpack(emb_blob)
                except Exception:
                    continue
                if len(vec) != qdim:
                    continue
                cos = _cosine_similarity(qvec, vec)
                content_by_hash[ch] = (role, content)
                cos_by_hash[ch] = (cos + 1.0) / 2.0

            # BM25 pool (optional, fail-open)
            bm25_by_hash: dict[str, float] = {}
            used_hybrid = False
            if getattr(self.valves, "ENABLE_HYBRID_RETRIEVAL", True):
                try:
                    fts_q = _fts5_safe_query(query)
                    if fts_q:
                        bm_rows = list(
                            conn.execute(
                                "SELECT ct.content_hash, bm25(chat_turns_fts) AS rank "
                                "FROM chat_turns_fts "
                                "JOIN chat_turns ct ON ct.id = chat_turns_fts.rowid "
                                "WHERE ct.chat_id = ? AND chat_turns_fts MATCH ? "
                                "ORDER BY rank LIMIT ?",
                                (chat_id, fts_q, self.valves.CHAT_MEMORY_TOP_K * 4),
                            )
                        )
                        if bm_rows:
                            ranks = [r[1] for r in bm_rows]
                            mn, mx = min(ranks), max(ranks)
                            span = mx - mn if mx != mn else 1.0
                            for ch, rank in bm_rows:
                                if ch in exclude_hashes:
                                    continue
                                bm25_by_hash[ch] = 1.0 - (rank - mn) / span
                            missing = set(bm25_by_hash) - set(content_by_hash)
                            if missing:
                                qs = ",".join("?" * len(missing))
                                for role, content, ch in conn.execute(
                                    f"SELECT role, content, content_hash "
                                    f"FROM chat_turns "
                                    f"WHERE chat_id=? AND content_hash IN ({qs})",
                                    (chat_id, *missing),
                                ):
                                    content_by_hash[ch] = (role, content)
                            used_hybrid = True
                except Exception as e:
                    logger.info("FTS5 hybrid skipped (cosine-only): %s", e)
                    bm25_by_hash = {}

            # Merge scores: 60% cosine, 40% BM25
            scored: list[tuple[float, str, str]] = []
            for ch, (role, content) in content_by_hash.items():
                cos = cos_by_hash.get(ch, 0.0)
                if used_hybrid:
                    bm = bm25_by_hash.get(ch, 0.0)
                    final = 0.6 * cos + 0.4 * bm
                else:
                    final = cos
                scored.append((final, role, content))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[: self.valves.CHAT_MEMORY_TOP_K]
            return [(role, content) for _, role, content in top]
        except Exception as e:
            logger.warning(
                "Chat memory recall failed (chat=%s): %s", str(chat_id)[:20], e
            )
            return []

    # ── Query rewrite for follow-ups ───────────────────────────────────

    async def rewrite_followup_query(
        self, query: str, messages: list, *, log_chat_id: str = ""
    ) -> Optional[str]:
        """Rewrite a short pronoun-y follow-up into a standalone retrieval query.

        Only fires when ENABLE_QUERY_REWRITE is on and the query looks like
        a follow-up. Returns the rewritten query, or None if not needed / failed.
        """
        if not getattr(self.valves, "ENABLE_QUERY_REWRITE", True):
            return None
        words = query.split()
        is_followup_like = _is_followup_query(query) or (
            len(words) <= 8 and _PRONOUN_REF.search(query)
        )
        if not is_followup_like:
            return None
        prior = [
            m for m in messages[:-1] if m.get("role") in ("user", "assistant")
        ]
        if not prior:
            return None

        ctx_lines: list[str] = []
        for m in prior[-3:]:
            role = m.get("role", "user")
            c = _sanitize_user_text(_text_of(m.get("content", ""))[:200])
            if c:
                ctx_lines.append(f"[{role}]: {c}")
        if not ctx_lines:
            return None

        prompt = (
            "Rewrite the user's latest message as a standalone search query by "
            "resolving pronouns and implicit references using the prior context. "
            "Be concise (max 20 words). Output ONLY the rewritten query — "
            "no preamble, no quotes, no explanation.\n\n"
            "Prior context:\n"
            + "\n".join(ctx_lines)
            + f"\n\nUser's latest message: {_sanitize_user_text(query)}\n\n"
            "Rewritten query:"
        )
        rewritten = await self.call_llm(
            prompt=prompt,
            model=self.valves.CLASSIFIER_MODEL,
            max_tokens=60,
            fallback_chain=[],  # passed through to call_llm
            log_role="rewrite",
            log_chat_id=log_chat_id,
        )
        if not rewritten:
            return None
        cleaned = _THINKING_BLOCK_RE.sub("", rewritten).strip().strip("'\"")
        if not cleaned or len(cleaned) > 300:
            return None
        return cleaned

    # ── Compression ────────────────────────────────────────────────────

    def _get_compression_lock(self, chat_id: str) -> asyncio.Lock:
        """Per-chat lock so concurrent outlets on the same chat don't both compress."""
        lock = self._compression_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._compression_locks[chat_id] = lock
        return lock

    async def maybe_compress(self, chat_id: str) -> None:
        """When a chat grows large, summarize its oldest turns into one row.

        Runs as a background task (asyncio.create_task) so the LLM call doesn't
        block the user's reply. Lossy compaction — original turns are DELETED
        after the summary is successfully stored.
        """
        if not getattr(self.valves, "ENABLE_CHAT_MEMORY_COMPRESSION", True) or not chat_id:
            return
        conn = await self._get_conn()
        if conn is None:
            return
        lock = self._get_compression_lock(chat_id)
        if lock.locked():
            return
        async with lock:
            await self._run_compression(conn, chat_id)

    async def _run_compression(self, conn: sqlite3.Connection, chat_id: str) -> None:
        """Body of compression; caller holds the per-chat lock."""
        try:
            trigger = self.valves.CHAT_MEMORY_COMPRESS_WHEN_OVER
            chunk = self.valves.CHAT_MEMORY_COMPRESS_CHUNK
            total = conn.execute(
                "SELECT COUNT(*) FROM chat_turns "
                "WHERE chat_id=? AND role != 'summary'",
                (chat_id,),
            ).fetchone()[0]
            if total <= trigger:
                return
            candidates = list(
                conn.execute(
                    "SELECT id, role, content FROM chat_turns "
                    "WHERE chat_id=? AND role != 'summary' "
                    "ORDER BY created_at ASC LIMIT ?",
                    (chat_id, chunk),
                )
            )
            if len(candidates) < 5:
                return
            block = "\n".join(
                f"[{role}]: {content[:500]}" for _, role, content in candidates
            )
            prompt = (
                "Summarize the following conversation turns into a concise, "
                "information-dense paragraph suitable for later semantic recall. "
                "Preserve: facts, decisions, user preferences, code/commands discussed, "
                "URLs, paper titles. Omit: greetings, pleasantries, exact phrasing. "
                "Target 200–400 words, single paragraph.\n\n"
                "CONVERSATION TURNS:\n"
                f"{block}\n\n"
                "SUMMARY:"
            )
            comp_model = getattr(self.valves, "COMPRESSION_MODEL", "") or self.valves.MAIN_MODEL
            summary = await self.call_llm(
                prompt=prompt,
                model=comp_model,
                max_tokens=500,
                fallback_chain=[],
                log_role="summary",
                log_chat_id=chat_id,
            )
            if not summary:
                return
            summary_clean = _THINKING_BLOCK_RE.sub("", summary).strip()
            if len(summary_clean) < 50:
                return
            ch = _memory_content_hash(summary_clean)
            already = conn.execute(
                "SELECT 1 FROM chat_turns WHERE chat_id=? AND content_hash=? LIMIT 1",
                (chat_id, ch),
            ).fetchone()
            vec = await self.get_embedding(summary_clean[:2000])
            emb_blob = _f32_pack(vec) if vec else None
            ids_to_delete = [c[0] for c in candidates]
            qs = ",".join("?" * len(ids_to_delete))
            if not already:
                conn.execute(
                    "INSERT INTO chat_turns "
                    "(chat_id, role, content, content_hash, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (chat_id, "summary", summary_clean, ch, emb_blob, time.time()),
                )
            conn.execute(
                f"DELETE FROM chat_turns WHERE id IN ({qs})",
                tuple(ids_to_delete),
            )
            conn.commit()
            logger.info(
                "Chat memory compression: chat=%s summarized %d turns",
                str(chat_id)[:20], len(ids_to_delete),
            )
        except Exception as e:
            logger.warning(
                "Chat memory compression failed (chat=%s): %s",
                str(chat_id)[:20], e,
            )

    # ── Sweeps ─────────────────────────────────────────────────────────

    async def sweep_referential(self) -> None:
        """Delete memory rows for chat_ids that no longer exist in webui.db.

        Runs on EVERY outlet — deterministic, NOT probabilistic. Cheap:
        two small SELECTs + set difference. When no orphans exist, the
        DELETE is skipped entirely.
        """
        conn = await self._get_conn()
        if conn is None:
            return
        webui_db = "/app/backend/data/webui.db"
        if not os.path.exists(webui_db):
            return
        try:
            alive = sqlite3.connect(
                f"file:{webui_db}?mode=ro", uri=True, timeout=2
            )
            try:
                alive_ids = {r[0] for r in alive.execute("SELECT id FROM chat")}
            finally:
                alive.close()
            if not alive_ids:
                return
            memory_ids = {
                r[0]
                for r in conn.execute("SELECT DISTINCT chat_id FROM chat_turns")
            }
            orphans = memory_ids - alive_ids
            if not orphans:
                return
            qs = ",".join("?" * len(orphans))
            removed = conn.execute(
                f"DELETE FROM chat_turns WHERE chat_id IN ({qs})",
                tuple(orphans),
            ).rowcount
            conn.commit()
            logger.info(
                "Chat memory referential sweep: %d rows across %d orphan chat(s)",
                removed, len(orphans),
            )
        except Exception as e:
            logger.warning("Chat memory referential sweep failed: %s", e)

    async def sweep_ttl(self) -> None:
        """TTL-based deletion. Probabilistic (~1% of calls) — age-based, not privacy."""
        if random.random() > 0.01:
            return
        conn = await self._get_conn()
        if conn is None:
            return
        try:
            cutoff = time.time() - self.valves.CHAT_MEMORY_TTL_DAYS * 86400
            removed = conn.execute(
                "DELETE FROM chat_turns WHERE created_at < ?", (cutoff,)
            ).rowcount
            conn.commit()
            if removed:
                logger.info("Chat memory TTL sweep: %d old rows removed", removed)
        except Exception as e:
            logger.warning("Chat memory TTL sweep failed: %s", e)

    # ── Request logging ────────────────────────────────────────────────

    async def log_request(
        self,
        chat_id: Optional[str],
        model: str,
        call_role: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency_ms: Optional[int] = None,
        success: bool = True,
        fallback: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Append one row to request_log for analytics. Fails silently."""
        conn = await self._get_conn()
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO request_log "
                "(ts, chat_id, model, call_role, prompt_tokens, completion_tokens, "
                " total_tokens, latency_ms, success, fallback, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    chat_id,
                    model,
                    call_role,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    latency_ms,
                    1 if success else 0,
                    1 if fallback else 0,
                    (error or "")[:300] if error else None,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.debug("request_log insert failed (non-fatal): %s", e)


# ── Standalone helper: extract text from multimodal content ───────────

def _text_of(content) -> str:
    """Extract plain text from message content that may be a string or a
    list of content-parts (vision+chat format)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return str(content)
