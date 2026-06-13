"""
title: PrismAI
description: Accuracy-first semantic router with model-driven classification, search dispatch,
  sticky routing, and vision proxy. Delegates memory, verification, and vision to independent
  modules (router_memory.py, router_verify.py, router_vision.py).
author: open-webui-community
version: 10.0
"""

import asyncio
import base64
import hashlib
import logging
import math
import os
import random
import re
import time
import uuid
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

from router_memory import ChatMemory, _clean_for_memory, _memory_content_hash, _text_of
from router_vision import VisionProxy
from router_verify import CitationVerifier

EventEmitter = Optional[Callable[[dict], Awaitable[Any]]]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_MARKER = "LIVE WEB SEARCH RESULTS"

FORCE_SEARCH_PATTERNS = [
    re.compile(r"\bsearch\b", re.IGNORECASE),
    re.compile(r"\bweb\s+search\b", re.IGNORECASE),
    re.compile(r"\blook\s+up\b", re.IGNORECASE),
    re.compile(r"\bgoogle\b", re.IGNORECASE),
    re.compile(r"\bfind\s+online\b", re.IGNORECASE),
    re.compile(r"\blatest\b", re.IGNORECASE),
    re.compile(r"\bcurrent(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bnowadays\b", re.IGNORECASE),
    re.compile(r"\brecently\b", re.IGNORECASE),
    re.compile(r"\bas\s+of\b", re.IGNORECASE),
    re.compile(r"\bup\s+to\s+date\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(year|month|week)\b", re.IGNORECASE),
    re.compile(r"\bbreaking\b", re.IGNORECASE),
    re.compile(r"\bnews\s+about\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+on\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s\s+the\s+latest\b", re.IGNORECASE),
]

URL_RE = re.compile(r"https?://[^\s<>\])}]+", re.IGNORECASE)
URL_FETCH_INTENT_PATTERNS = [
    re.compile(r"\b(fetch|open|read|summari[sz]e|quote|extract|inspect)\b", re.IGNORECASE),
    re.compile(r"\b(first|lead|intro(?:duction)?|specific)\s+section\b", re.IGNORECASE),
    re.compile(r"\b(the\s+)?(?:url|link|page|article|webpage|site)\b", re.IGNORECASE),
]

DOCUMENT_REQUEST_PATTERNS = [
    re.compile(
        r"\b(cover\s+letter|resume|cv|personal\s+statement|statement\s+of\s+purpose|"
        r"sop|motivation\s+letter|bio|profile|linkedin|email|letter|proposal|"
        r"essay|draft|rewrite|edit|polish|document|docx|pdf)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(write|draft|polish|tailor|personalize|personalise)\b", re.IGNORECASE),
]

DOCUMENT_WRITING_ARTIFACT_RE = re.compile(
    r"\b(cover\s+letter|resume|cv|personal\s+statement|statement\s+of\s+purpose|"
    r"sop|motivation\s+letter|bio|linkedin|email|letter|proposal|essay|placeholder)\b",
    re.IGNORECASE,
)
DOCUMENT_EXPORT_RE = re.compile(
    r"\b(export|download|downloadable|save|create|make|generate)\b.{0,80}"
    r"\b(docx|pdf|word\s+document|markdown|csv)\b"
    r"|\b(docx|pdf|word\s+document|markdown|csv)\b.{0,80}"
    r"\b(export|download|save)\b",
    re.IGNORECASE,
)
CODING_CONTEXT_RE = re.compile(
    r"\b(code|script|program|function|class|api|parse|parser|library|python|"
    r"javascript|typescript|java|bash|sql|regex|package|module)\b",
    re.IGNORECASE,
)

CATEGORY_NAMES = frozenset({"FACTUAL", "REASONING", "CODING", "RESEARCH", "CASUAL"})

# Models confirmed to natively accept image_url content parts.
VISION_CAPABLE_MODELS = frozenset({
    "accounts/fireworks/models/kimi-k2p5",
    "accounts/fireworks/models/kimi-k2p6",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
})

ORCHESTRATOR_MODEL_IDS = frozenset({"PrismAI"})

# Fallback model chains
CLASSIFIER_FALLBACK_CHAIN = [
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/kimi-k2p5",
    "accounts/fireworks/models/glm-5p1",
]

VERIFIER_FALLBACK_CHAIN = [
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/kimi-k2p5",
    "accounts/fireworks/models/glm-5p1",
]

CAPTION_FALLBACK_CHAIN = [
    "accounts/fireworks/models/kimi-k2p5",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "accounts/fireworks/models/kimi-k2p6",
]

# Provider routing
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
GROQ_PREFIX = "groq/"

THINKING_BLOCK_RE = re.compile(r"<think>.*?</think>|<tool_call>.*?</tool_call>", re.DOTALL)

# Prompt injection sanitization
_INJECTION_RE = re.compile(
    r"(?i)\b("
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|messages?)"
    r"|disregard\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|messages?)"
    r"|forget\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|messages?)"
    r"|new\s+instructions?\s*:"
    r"|system\s+prompt\s*:"
    r").*"
)

# Document style guidance
DOCUMENT_STYLE_PROMPT = (
    "DOCUMENT WRITING MODE:\n"
    "- Preserve the user's personal voice. Mirror their level of directness, energy, and vocabulary when there is enough prior context.\n"
    "- For cover letters, personal statements, bios, emails, and similar writing, use first person unless the user asks otherwise.\n"
    "- Avoid generic polished filler: no 'I am writing to express my interest', no empty enthusiasm, no buzzword stacks.\n"
    "- Use concrete details from the prompt and recalled chat context: role, company, project, skills, constraints, motivation, and stakes.\n"
    "- If web search results are provided, use them as background context. Do not put citations or a sources section into cover letters, personal statements, bios, or emails unless the user explicitly asks for cited writing.\n"
    "- Do not invent personal history, credentials, employers, publications, grades, locations, or achievements. If a required detail is missing, write [NEEDS DETAIL: ...].\n"
    "- For export requests, write the full final content and pass that same content to the export tool. Do not replace the answer with a meta-summary of what you wrote unless the user explicitly asks for only a file.\n"
    "- Make the result feel authored by this user, not by a template: specific, human, and purposeful.\n"
)

URL_FETCH_PROMPT = (
    "URL FETCH MODE:\n"
    "- The user supplied a specific URL. Use the fetch_url tool for that URL before summarizing, quoting, or describing the page unless the page text is already in the conversation.\n"
    "- Do not use Tavily search merely because a URL was provided. Use web search only when the user explicitly asks for broader search beyond the supplied page.\n"
    "- Respect the requested scope exactly, such as 'first section', 'introduction', or a requested sentence count.\n"
    "- Do not add a citations or sources section by default for single-page summaries; mention the source page only if it helps clarity.\n"
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

ROUTE_HEADER_RE = re.compile(
    r"\A`[^\n`]*\b(?:FACTUAL|REASONING|CODING|RESEARCH|CASUAL|WRITING|FETCH)\b[^\n`]*`\s*\n(?:>\s*[^\n]*\n)?\s*",
    re.IGNORECASE,
)

ROUTER_STATE_RE = re.compile(
    r"\[ROUTER_STATE:\s*([A-Z]+)(_SEARCH)?\]"
    r"|<!--\s*ROUTER_STATE:\s*([A-Z]+)(_SEARCH)?\s*-->",
    re.IGNORECASE,
)
ROUTER_STATE_STRIP_RE = re.compile(
    r"\[ROUTER_STATE:\s*[A-Z]+(?:_SEARCH)?\]"
    r"|<!--\s*ROUTER_STATE:\s*[A-Z]+(?:_SEARCH)?\s*-->",
    re.IGNORECASE,
)


class _NonRetryableError(Exception):
    pass


def _truncate_at_sentence(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last_period = chunk.rfind(".")
    last_newline = chunk.rfind("\n")
    cut = max(last_period, last_newline)
    return chunk[: cut + 1] if cut > max_chars // 2 else chunk


def _strip_thinking_blocks(text: str) -> str:
    return THINKING_BLOCK_RE.sub("", text)


def _sanitize_query(query: str) -> str:
    sanitized = query.replace("```", "")
    sanitized = _INJECTION_RE.sub("[redacted]", sanitized)
    return sanitized.strip()


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


async def _retry_request(coro_fn, max_retries: int = 2, base_delay: float = 0.5):
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except _NonRetryableError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            logger.warning("Retry %d/%d after error: %s — waiting %.1fs",
                           attempt + 1, max_retries, e, delay)
            await asyncio.sleep(delay)


async def _check_response(resp: aiohttp.ClientResponse) -> None:
    if resp.status == 200:
        return
    text = await resp.text()
    err = aiohttp.ClientResponseError(
        request_info=resp.request_info,
        history=resp.history,
        status=resp.status,
        message=f"HTTP {resp.status}: {text[:200]}",
    )
    if _is_retryable_status(resp.status):
        raise err
    raise _NonRetryableError(str(err)) from err


def _cosine_similarity(v1: list, v2: list) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return 0.0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)


def _is_followup_query(query: str) -> bool:
    words = query.split()
    if not words or len(words) > 15:
        return False
    if re.match(r"^\s*(and|also|now|but|so|or|plus|what\s+about|how\s+about|what\s+if|yeah|yes|ok|okay|more|again)\b",
                query, re.IGNORECASE):
        return True
    if len(words) <= 6 and re.search(
        r"\b(it|its|it's|that|this|these|those|they|them|their|he|she|him|her|hers|his)\b",
        query, re.IGNORECASE
    ):
        return True
    return False


def _is_document_request(query: str) -> bool:
    if not query or len(query) < 8:
        return False
    return any(p.search(query) for p in DOCUMENT_REQUEST_PATTERNS)


def _is_document_output_request(query: str) -> bool:
    if not _is_document_request(query):
        return False
    if DOCUMENT_WRITING_ARTIFACT_RE.search(query) or DOCUMENT_EXPORT_RE.search(query):
        return True
    return not CODING_CONTEXT_RE.search(query)


def _is_url_fetch_request(query: str) -> bool:
    if not query or not URL_RE.search(query):
        return False
    return any(p.search(query) for p in URL_FETCH_INTENT_PATTERNS)


# ---------------------------------------------------------------------------
# Filter class — thin coordinator delegating to extracted modules
# ---------------------------------------------------------------------------

    # ── Inlet helpers ──────────────────────────────────────────────────

    @staticmethod
    def _has_images_for_routing(messages: list) -> bool:
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            if any(p.get("type") == "image_url" for p in content if isinstance(p, dict)):
                return True
        return False

    @staticmethod
    def _extract_image_parts(messages: list) -> list[dict]:
        parts = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for p in content:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    parts.append(p)
        return parts

    @staticmethod
    def _inject_system_block(messages: list, block: str) -> None:
        existing = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if existing is not None:
            sys_content = _text_of(messages[existing]["content"])
            messages[existing]["content"] = sys_content + block
        else:
            messages.insert(0, {"role": "system", "content": block.strip()})


class Filter:
    class Valves(BaseModel):
        FIREWORKS_API_KEY: str = Field(default="", description="Your Fireworks.ai API Key.")
        GROQ_API_KEY: str = Field(default="", description="Your Groq API key.")
        TAVILY_API_KEY: str = Field(default="", description="Your Tavily API Key for web search.")
        EMBEDDING_MODEL: str = Field(default="nomic-ai/nomic-embed-text-v1.5", description="Embedding model for intent vectors.")
        CLASSIFIER_MODEL: str = Field(default="groq/llama-3.1-8b-instant", description="Routing model.")
        MAIN_MODEL: str = Field(default="accounts/fireworks/models/deepseek-v4-pro", description="Main chat model.")
        VERIFIER_MODEL: str = Field(default="groq/llama-3.3-70b-versatile", description="Citation auditor model.")
        ROUTING_THRESHOLD: float = Field(default=0.6, description="Cosine similarity threshold for embedding routing.")
        SHOW_ROUTE_TAG: bool = Field(default=True, description="Show the route category tag.")
        ENABLE_OUTLET_VERIFICATION: bool = Field(default=True, description="Run citation check on FACTUAL/RESEARCH responses.")
        VERIFIER_MODE: str = Field(default="hybrid", description="'regex', 'llm', or 'hybrid' verification mode.")
        VERIFIER_REGENERATE: bool = Field(default=False, description="Regenerate response on verification failure.")
        VERIFIER_MAX_TOKENS: int = Field(default=2500, description="Max tokens for verifier regeneration.")
        VERIFIER_MAX_RETRIES: int = Field(default=2, description="Max verifier regeneration retries.")
        OUTLET_VERIFY_TIMEOUT: int = Field(default=300, description="Verification loop timeout in seconds.")
        SEARCH_RESULTS_FACTUAL: int = Field(default=6, description="Tavily results for FACTUAL routes.")
        SEARCH_RESULTS_RESEARCH: int = Field(default=10, description="Tavily results for RESEARCH routes.")
        SEARCH_CACHE_TTL: int = Field(default=3600, description="Search cache TTL in seconds.")
        SEARCH_CACHE_MAX: int = Field(default=100, description="Max entries in search cache.")
        INLET_TIMEOUT: int = Field(default=90, description="Overall inlet timeout in seconds.")
        EMIT_STATUS_EVENTS: bool = Field(default=True, description="Emit status events to OWUI surface.")
        ENABLE_STICKY_ROUTING: bool = Field(default=True, description="Carry last category forward for follow-ups.")
        STICKY_MAX_CONVOS: int = Field(default=200, description="Max chats in sticky-route LRU.")
        STICKY_TTL_SECONDS: int = Field(default=86400, description="Sticky route TTL in seconds (24h default).")
        ENABLE_IMAGE_ROUTING: bool = Field(default=True, description="Caption images for routing classification.")
        IMAGE_CAPTION_MODEL: str = Field(default="accounts/fireworks/models/kimi-k2p5", description="Vision model for routing captions.")
        IMAGE_CAPTION_MAX_TOKENS: int = Field(default=80, description="Max tokens for routing caption.")
        ENABLE_VISION_PROXY: bool = Field(default=True, description="Caption images for non-vision models.")
        IMAGE_PROXY_MAX_TOKENS: int = Field(default=300, description="Max tokens for detailed vision proxy caption.")
        ENABLE_CHAT_MEMORY: bool = Field(default=True, description="Persistent per-chat semantic memory.")
        CHAT_MEMORY_DB_PATH: str = Field(default="/app/backend/data/router_mem.db", description="SQLite file for chat memory.")
        CHAT_MEMORY_MIN_TURNS: int = Field(default=15, description="Min turns before memory recall activates.")
        CHAT_MEMORY_TOP_K: int = Field(default=8, description="Number of prior turns to inject on recall.")
        CHAT_MEMORY_TTL_DAYS: int = Field(default=90, description="Prune memory rows older than this.")
        CHAT_MEMORY_MAX_TURNS_PER_CHAT: int = Field(default=500, description="Hard cap on stored turns per chat.")
        ADDRESS_USER_BY_NAME: bool = Field(default=True, description="Inject user name into system prompt.")
        ENABLE_HYBRID_RETRIEVAL: bool = Field(default=True, description="Combine BM25 + cosine for memory recall.")
        ENABLE_QUERY_REWRITE: bool = Field(default=True, description="Rewrite follow-up queries for recall.")
        ENABLE_DOCUMENT_STYLE_GUIDANCE: bool = Field(default=True, description="Inject voice-preserving style guide for writing.")
        DOCUMENT_STYLE_GUIDE: str = Field(default="", description="User-specific writing preferences.")
        ENABLE_CHAT_MEMORY_COMPRESSION: bool = Field(default=True, description="Summarize old turns when chat grows.")
        CHAT_MEMORY_COMPRESS_WHEN_OVER: int = Field(default=60, description="Trigger compression at N stored turns.")
        CHAT_MEMORY_COMPRESS_CHUNK: int = Field(default=20, description="Number of oldest turns to summarize.")
        COMPRESSION_MODEL: str = Field(default="accounts/fireworks/models/deepseek-v4-flash", description="Model for memory compression.")

    def __init__(self):
        self.valves = self.Valves()
        # Routing state
        self.anchor_embeddings: dict[str, list] = {}
        self._last_embedding_model: Optional[str] = None
        self.search_cache: OrderedDict[str, dict] = OrderedDict()
        self._last_tavily_key: Optional[str] = None
        self.sticky_routes: OrderedDict[str, dict] = OrderedDict()
        self._session_affinity_id = uuid.uuid4().hex
        # Shared HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: Optional[asyncio.Lock] = None
        self._embedding_lock: Optional[asyncio.Lock] = None
        # Embedding circuit breaker
        self._embedding_consec_fail: int = 0
        self._embedding_trip_until: float = 0.0
        # Category descriptions for embedding anchors
        self.categories = {
            "FACTUAL": "Objective inquiries requiring real-world verification. Focus on data, laws, prices, and current events.",
            "REASONING": "Mathematical, logical, and algorithmic reasoning. Focus on step-by-step proofs and theoretical math.",
            "CODING": "Programming, software engineering, and code execution. Focus on code generation and syntax.",
            "RESEARCH": "Academic literature, paper analysis, and scientific consensus. Focus on citations and academia.",
            "CASUAL": "Subjective, creative, or conversational interactions. Focus on opinions and greetings without factual constraints.",
        }
        # Extracted modules — lazily initialized
        self._memory: Optional[ChatMemory] = None
        self._vision: Optional[VisionProxy] = None
        self._verifier: Optional[CitationVerifier] = None

    # ── Extracted module accessors (lazy init) ──────────────────────────

    @property
    def memory(self) -> ChatMemory:
        if self._memory is None:
            self._memory = ChatMemory(
                get_embedding=self._get_embedding,
                call_llm=self._call_llm,
                valves=self.valves,
            )
        return self._memory

    @property
    def vision(self) -> VisionProxy:
        if self._vision is None:
            self._vision = VisionProxy(
                dispatch_model=self._dispatch_model,
                get_session=self._get_session,
                call_vision=self._call_vision_model,
                emit_status=self._emit_status,
                valves=self.valves,
            )
        return self._vision

    @property
    def verifier(self) -> CitationVerifier:
        if self._verifier is None:
            self._verifier = CitationVerifier(
                call_llm=self._call_llm,
                emit_status=self._emit_status,
                emit_replace=self._emit_replace,
                valves=self.valves,
            )
        return self._verifier

    # ── Lifecycle ────────────────────────────────────────────────────────

    def __del__(self):
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    on_shutdown = close

    # ── Session management ───────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                return self._session
            self._session = aiohttp.ClientSession()
            return self._session

    def _get_embedding_lock(self) -> asyncio.Lock:
        if self._embedding_lock is None:
            self._embedding_lock = asyncio.Lock()
        return self._embedding_lock

    # ── Model dispatch ───────────────────────────────────────────────────

    def _dispatch_model(self, model: str) -> tuple[str, str, str, dict]:
        """Resolve (base_url, api_key, stripped_model_id, extra_headers) for a model."""
        if model.startswith(GROQ_PREFIX):
            return (GROQ_BASE_URL, self.valves.GROQ_API_KEY,
                    model[len(GROQ_PREFIX):], {})
        return (FIREWORKS_BASE_URL, self.valves.FIREWORKS_API_KEY,
                model, {"x-session-affinity": self._session_affinity_id})

    # ── Embedding ────────────────────────────────────────────────────────

    async def _get_embedding(self, text: str) -> list:
        """Get embedding vector for text via Fireworks API. Returns [] on failure."""
        if time.time() < self._embedding_trip_until:
            return []
        try:
            async def _do_embed():
                session = await self._get_session()
                async with session.post(
                    "https://api.fireworks.ai/inference/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.valves.FIREWORKS_API_KEY}"},
                    json={"model": self.valves.EMBEDDING_MODEL, "input": text},
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    await _check_response(resp)
                    data = await resp.json()
                    return data["data"][0]["embedding"]
            result = await _retry_request(_do_embed)
            self._embedding_consec_fail = 0
            return result
        except _NonRetryableError as e:
            self._embedding_consec_fail += 1
            self._maybe_trip_embedding_breaker()
            logger.warning("Embedding non-retryable error for '%s…': %s", text[:50], e)
            return []
        except Exception as e:
            self._embedding_consec_fail += 1
            self._maybe_trip_embedding_breaker()
            logger.warning("Embedding failed after retries for '%s…': %s", text[:50], e)
            return []

    def _maybe_trip_embedding_breaker(self) -> None:
        if self._embedding_consec_fail >= 2:
            self._embedding_trip_until = time.time() + 60.0
            logger.warning("Embedding circuit breaker TRIPPED — 60s cool-down after %d failures.",
                           self._embedding_consec_fail)
            self._embedding_consec_fail = 0

    # ── Anchor embeddings ────────────────────────────────────────────────

    async def _ensure_anchor_embeddings(self) -> None:
        if self.anchor_embeddings:
            return
        async with self._get_embedding_lock():
            if self.anchor_embeddings:
                return
            tasks = [self._get_embedding(desc) for desc in self.categories.values()]
            results = await asyncio.gather(*tasks)
            for (cat, _), vec in zip(self.categories.items(), results):
                if vec:
                    self.anchor_embeddings[cat] = vec

    def _invalidate_stale_caches(self) -> None:
        if self.valves.EMBEDDING_MODEL != self._last_embedding_model:
            self.anchor_embeddings.clear()
            self._last_embedding_model = self.valves.EMBEDDING_MODEL
            logger.info("Embedding model changed — cleared anchor cache.")
        if self.valves.TAVILY_API_KEY != self._last_tavily_key:
            self.search_cache.clear()
            self._last_tavily_key = self.valves.TAVILY_API_KEY
            logger.info("Tavily API key changed — cleared search cache.")

    # ── LLM call infrastructure ──────────────────────────────────────────

    async def _call_llm(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 50,
        fallback_chain: Optional[list[str]] = None,
        log_role: str = "unknown",
        log_chat_id: Optional[str] = None,
    ) -> Optional[str]:
        """Call an LLM with automatic fallback across models."""
        models_to_try = [model]
        if fallback_chain:
            for fb in fallback_chain:
                if fb != model and fb not in models_to_try:
                    models_to_try.append(fb)
        if not self.valves.GROQ_API_KEY:
            models_to_try = [m for m in models_to_try if not m.startswith(GROQ_PREFIX)]

        last_err = None
        for i, m in enumerate(models_to_try):
            is_fallback = i > 0
            start = time.time()
            captured_usage: dict = {}

            try:
                async def _do_call(model_name=m):
                    base_url, api_key, model_id, extra_headers = self._dispatch_model(model_name)
                    session = await self._get_session()
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        **extra_headers,
                    }
                    async with session.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                        },
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        await _check_response(resp)
                        data = await resp.json()
                        captured_usage.update(data.get("usage") or {})
                        choice = data["choices"][0]
                        if choice.get("finish_reason") == "length":
                            logger.warning("LLM response truncated (model=%s)", model_name)
                        return choice["message"]["content"].strip()

                result = await _retry_request(_do_call)
                latency_ms = int((time.time() - start) * 1000)
                await self.memory.log_request(
                    chat_id=log_chat_id, model=m, call_role=log_role,
                    prompt_tokens=captured_usage.get("prompt_tokens"),
                    completion_tokens=captured_usage.get("completion_tokens"),
                    total_tokens=captured_usage.get("total_tokens"),
                    latency_ms=latency_ms, success=True, fallback=is_fallback,
                )
                if is_fallback:
                    logger.info("LLM fallback succeeded: %s (primary %s failed)",
                                m.split("/")[-1], model.split("/")[-1])
                return result
            except _NonRetryableError as e:
                last_err = e
                await self.memory.log_request(
                    chat_id=log_chat_id, model=m, call_role=log_role,
                    latency_ms=int((time.time() - start) * 1000),
                    success=False, fallback=is_fallback, error=str(e),
                )
                continue
            except Exception as e:
                last_err = e
                await self.memory.log_request(
                    chat_id=log_chat_id, model=m, call_role=log_role,
                    latency_ms=int((time.time() - start) * 1000),
                    success=False, fallback=is_fallback, error=str(e),
                )
                continue

        logger.error("All LLM models failed for prompt '%s…' (tried %s)",
                     prompt[:50], [m.split("/")[-1] for m in models_to_try])
        return None

    # ── Vision model calling (used by VisionProxy) ───────────────────────

    async def _call_vision_model(
        self,
        vision_content: list[dict],
        max_tokens: int,
        fallback_chain: list[str],
        log_role: str = "caption",
        log_chat_id: Optional[str] = None,
        event_emitter: EventEmitter = None,
    ) -> Optional[str]:
        """Call a vision model with automatic fallback. Returns text or None."""
        models_to_try = list(fallback_chain)
        if not self.valves.GROQ_API_KEY:
            models_to_try = [m for m in models_to_try if not m.startswith(GROQ_PREFIX)]

        last_err = None
        for i, model_name in enumerate(models_to_try):
            is_fallback = i > 0
            start = time.time()
            captured_usage: dict = {}
            try:
                async def _do_vision_call(mn=model_name):
                    base_url, api_key, model_id, extra_headers = self._dispatch_model(mn)
                    session = await self._get_session()
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        **extra_headers,
                    }
                    async with session.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": vision_content}],
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                        },
                        timeout=aiohttp.ClientTimeout(total=20),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            err_msg = f"HTTP {resp.status}: {text[:200]}"
                            if _is_retryable_status(resp.status):
                                raise aiohttp.ClientResponseError(
                                    request_info=resp.request_info, history=resp.history,
                                    status=resp.status, message=err_msg)
                            raise _NonRetryableError(err_msg)
                        data = await resp.json()
                        captured_usage.update(data.get("usage") or {})
                        return data["choices"][0]["message"]["content"].strip()

                result = await _retry_request(_do_vision_call)
                latency_ms = int((time.time() - start) * 1000)
                await self.memory.log_request(
                    chat_id=log_chat_id, model=model_name, call_role=log_role,
                    prompt_tokens=captured_usage.get("prompt_tokens"),
                    completion_tokens=captured_usage.get("completion_tokens"),
                    total_tokens=captured_usage.get("total_tokens"),
                    latency_ms=latency_ms, success=True, fallback=is_fallback,
                )
                if is_fallback and event_emitter:
                    await self._emit_status(
                        event_emitter,
                        f"🖼️ Caption fallback: using {model_name.split('/')[-1]} (primary was down).",
                    )
                return result
            except _NonRetryableError as e:
                last_err = str(e)[:150]
                await self.memory.log_request(
                    chat_id=log_chat_id, model=model_name, call_role=log_role,
                    latency_ms=int((time.time() - start) * 1000),
                    success=False, fallback=is_fallback, error=str(e),
                )
                continue
            except Exception as e:
                last_err = str(e)[:150]
                await self.memory.log_request(
                    chat_id=log_chat_id, model=model_name, call_role=log_role,
                    latency_ms=int((time.time() - start) * 1000),
                    success=False, fallback=is_fallback, error=str(e),
                )
                continue
        return None

    # ── Search ───────────────────────────────────────────────────────────

    async def _compress_search_query(self, query: str, log_chat_id: Optional[str] = None) -> str:
        """LLM-compress an over-long query into a Tavily-safe ≤400-char form."""
        if len(query) <= 400:
            return query
        snippet = query[:4000]
        prompt = (
            "Compress the text below into a concise web search query. "
            "Keep all key entities, dates, technical terms, and intent. "
            "Drop conversational filler and pronouns. Output ONLY the search "
            "query — no explanation, no quotes, no prefix. Strict 400-character "
            "maximum.\n\n"
            f"Text:\n{snippet}"
        )
        compressed = await self._call_llm(
            prompt=prompt, model=self.valves.CLASSIFIER_MODEL, max_tokens=140,
            fallback_chain=CLASSIFIER_FALLBACK_CHAIN, log_role="search_compress",
            log_chat_id=log_chat_id,
        )
        if compressed and len(compressed) <= 400:
            return compressed
        target = compressed if compressed else query
        return target[:400].rsplit(" ", 1)[0]

    async def _search_tavily(
        self, query: str, max_results: int = 4, log_chat_id: Optional[str] = None
    ) -> str:
        """Execute a Tavily search and return formatted results."""
        if not self.valves.TAVILY_API_KEY:
            return "[No Tavily API Key Provided]"
        if len(query) > 400:
            query = await self._compress_search_query(query, log_chat_id=log_chat_id)
        cache_key = hashlib.sha256(
            f"{query.strip().lower()}_{max_results}".encode()
        ).hexdigest()
        if cache_key in self.search_cache:
            entry = self.search_cache[cache_key]
            if time.time() - entry["timestamp"] < self.valves.SEARCH_CACHE_TTL:
                self.search_cache.move_to_end(cache_key)
                return entry["results"]
            del self.search_cache[cache_key]

        async def _do_search():
            session = await self._get_session()
            async with session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.valves.TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": max_results,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                await _check_response(resp)
                data = await resp.json()
                context = f"Tavily AI Summary: {data.get('answer', '')}\n\n"
                for res in data.get("results", []):
                    context += f"Source ({res.get('url')}):\n{res.get('content')}\n\n"
                return _truncate_at_sentence(context)

        try:
            result = await _retry_request(_do_search)
            self.search_cache[cache_key] = {"results": result, "timestamp": time.time()}
            self.search_cache.move_to_end(cache_key)
            while len(self.search_cache) > self.valves.SEARCH_CACHE_MAX:
                self.search_cache.popitem(last=False)
            return result
        except _NonRetryableError as e:
            logger.warning("Tavily non-retryable error: %s", e)
            return f"[Web Search Failed: {e}]"
        except Exception as e:
            logger.warning("Tavily search failed after retries: %s", e)
            return f"[Web Search Failed: {e}]"

    # ── Classification ───────────────────────────────────────────────────

    def _parse_llm_category(self, llm_response: Optional[str]) -> Optional[str]:
        if not llm_response:
            return None
        upper = llm_response.upper()
        for cat in CATEGORY_NAMES:
            if re.search(rf"\b{cat}\b", upper):
                return cat
        return None

    def _build_classifier_prompt(self, messages: list, image_caption: str = "") -> str:
        """Build the classifier prompt from recent messages + optional image caption."""
        recent = messages[-3:]
        lines = []
        for m in recent:
            content = _text_of(m.get("content")).strip()
            if not content:
                continue
            role = m.get("role", "user")
            if role == "assistant":
                content = ROUTE_HEADER_RE.sub("", content, count=1).strip()
            lines.append(f"[{role}]: {_sanitize_query(content[:300])}")
        convo = "\n".join(lines) if lines else "[user]: (empty)"

        image_hint = ""
        if image_caption:
            image_hint = (f"\nNOTE: The user also sent an image. "
                          f'Auto-generated description: "{image_caption}"\n')

        return (
            "Classify the LAST user message into ONE category based on its topic and the prior context. "
            "Resolve pronouns like 'it'/'that' using the earlier turns.\n"
            "Categories: FACTUAL, REASONING, CODING, RESEARCH, CASUAL.\n\n"
            f"Recent messages:\n{convo}\n"
            f"{image_hint}\n"
            "Return ONLY the category name."
        )

    # ── Sticky routing ───────────────────────────────────────────────────

    def _get_sticky(self, chat_id: Optional[str]) -> Optional[dict]:
        if not chat_id or not self.valves.ENABLE_STICKY_ROUTING:
            return None
        entry = self.sticky_routes.get(chat_id)
        if not entry:
            return None
        if time.time() - entry.get("timestamp", 0) > self.valves.STICKY_TTL_SECONDS:
            del self.sticky_routes[chat_id]
            return None
        self.sticky_routes.move_to_end(chat_id)
        return entry

    def _set_sticky(self, chat_id: Optional[str], category: str, searched: bool) -> None:
        if not chat_id or not self.valves.ENABLE_STICKY_ROUTING:
            return
        self.sticky_routes[chat_id] = {
            "category": category, "searched": searched, "timestamp": time.time(),
        }
        self.sticky_routes.move_to_end(chat_id)
        while len(self.sticky_routes) > self.valves.STICKY_MAX_CONVOS:
            self.sticky_routes.popitem(last=False)

    # ── Route content builder ────────────────────────────────────────────

    def _build_route_content(
        self, category: str, searched: bool, body_text: str,
        override_label: Optional[str] = None, override_emoji: Optional[str] = None,
        trailer: str = "",
    ) -> str:
        if not self.valves.SHOW_ROUTE_TAG:
            return body_text + trailer
        tag_emoji = {"FACTUAL": "🔍", "REASONING": "🧮", "CODING": "💻", "RESEARCH": "📚", "CASUAL": "💬"}
        emoji = override_emoji or tag_emoji.get(category, "💬")
        label = override_label or category
        header = f"`{emoji} {label}`\n"
        if searched:
            header += "> 🌐 **Tavily Search Executed**\n\n"
        else:
            header += "\n"
        return header + body_text + trailer

    # ── Event emission ───────────────────────────────────────────────────

    async def _emit_status(
        self, event_emitter: EventEmitter, description: str, done: bool = True
    ) -> None:
        if event_emitter is None or not self.valves.EMIT_STATUS_EVENTS:
            return
        try:
            await event_emitter({
                "type": "status", "data": {"description": description, "done": done},
            })
        except Exception as e:
            logger.warning("Event emit failed: %s", e)

    async def _emit_replace(self, event_emitter: EventEmitter, content: str) -> None:
        if event_emitter is None:
            return
        try:
            await event_emitter({"type": "replace", "data": {"content": content}})
        except Exception as e:
            logger.warning("Replace emit failed: %s", e)

    # ── Chat ID extraction ───────────────────────────────────────────────

    @staticmethod
    def _extract_chat_id(body: dict) -> Optional[str]:
        cid = body.get("chat_id") or body.get("id")
        if cid:
            return str(cid)
        meta = body.get("metadata") or {}
        cid = meta.get("chat_id") or meta.get("id")
        return str(cid) if cid else None

    @staticmethod
    def _extract_user_name(user_obj: Optional[dict]) -> str:
        from router_memory import _first_name
        if not user_obj:
            return ""
        return _first_name(user_obj.get("name") or user_obj.get("username"))

    # ── Orchestrator call ────────────────────────────────────────────────

    async def _call_orchestrator(self, messages, user_model, request_headers):
        """Forward the conversation to the PrismAI orchestrator harness."""
        url = "http://owui-orchestrator:8002/v1/chat/completions"
        payload = {"model": user_model, "messages": messages, "stream": False}
        headers = {}
        for k, v in (request_headers or {}).items():
            low = k.lower()
            if low.startswith("x-openwebui") or low.startswith("authorization"):
                headers[k] = v
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(url, json=payload, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=180)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choices = data.get("choices") or []
                        if choices:
                            return choices[0].get("message", {}).get("content", "")
                    return f"[Orchestrator unavailable (status {resp.status})]"
        except Exception as e:
            logger.warning(f"Orchestrator call failed: {e}")
            return f"[Orchestrator unavailable: {type(e).__name__}]"

    # ══════════════════════════════════════════════════════════════════════
    # INLET — thin coordination: memory recall → vision proxy → forwarding
    # ══════════════════════════════════════════════════════════════════════

    async def inlet(
        self, body: dict, __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        if not body.get("messages") or not self.valves.FIREWORKS_API_KEY:
            return body
        try:
            return await asyncio.wait_for(
                self._do_inlet(body, __user__, __event_emitter__),
                timeout=self.valves.INLET_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Inlet exceeded %ds — forwarding query unrouted.",
                           self.valves.INLET_TIMEOUT)
            await self._emit_status(
                __event_emitter__,
                f"⚠️ Router timed out after {self.valves.INLET_TIMEOUT}s — forwarding query unrouted.",
            )
            return body

    async def _do_inlet(
        self, body: dict, __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        self._invalidate_stale_caches()
        messages = body["messages"]
        query_raw = _text_of(messages[-1]["content"]).strip()
        model_id = body.get("model", "")
        is_raw = model_id not in ORCHESTRATOR_MODEL_IDS

        # ── Classification + search for raw models ──────────────────
        searched = False
        category = "ORCHESTRATOR"
        if is_raw:
            # Image caption for routing classification
            image_caption = ""
            if self.valves.ENABLE_IMAGE_ROUTING and _has_images_for_routing(messages):
                image_parts = _extract_image_parts(messages)
                if image_parts:
                    caption = await self.vision.caption_for_routing(
                        image_parts, query_raw, event_emitter=__event_emitter__
                    )
                    if caption:
                        image_caption = caption

            # LLM classifier
            prompt = self._build_classifier_prompt(messages, image_caption)
            raw_cat = await self._call_llm(
                prompt=prompt, model=self.valves.CLASSIFIER_MODEL, max_tokens=20,
                fallback_chain=CLASSIFIER_FALLBACK_CHAIN, log_role="classifier",
            )
            category = self._parse_llm_category(raw_cat) or "CASUAL"

            # Search for factual/research queries
            if category in ("FACTUAL", "RESEARCH"):
                await self._emit_status(__event_emitter__, f"🔍 Searching the web ({category})…")
                max_results = (self.valves.SEARCH_RESULTS_RESEARCH if category == "RESEARCH"
                               else self.valves.SEARCH_RESULTS_FACTUAL)
                search_results = await self._search_tavily(query_raw, max_results=max_results)
                if search_results and "Web Search Failed" not in search_results:
                    searched = True
                    block = (f"\n\n{SEARCH_MARKER}\n{search_results}\nEND OF {SEARCH_MARKER}\n")
                    _inject_system_block(messages, block)

        # Tag in metadata
        body.setdefault("metadata", {})
        body["metadata"]["_router_state"] = {
            "category": category, "searched": searched,
            "orch_model": model_id if not is_raw else "",
        }

        # ── Memory recall (all models) ──────────────────────────────
        chat_id = self._extract_chat_id(body)
        if self.valves.ENABLE_CHAT_MEMORY and chat_id:
            exclude_hashes = set()
            for m in messages:
                cleaned = _clean_for_memory(_text_of(m.get("content", "")))
                if cleaned:
                    exclude_hashes.add(_memory_content_hash(cleaned))
            recall_query = query_raw
            rewritten = await self.memory.rewrite_followup_query(
                query_raw, messages, log_chat_id=chat_id
            )
            if rewritten:
                recall_query = rewritten
                await self._emit_status(
                    __event_emitter__,
                    f"🧠 Memory query rewritten for recall: {rewritten[:80]}",
                )
            recalled = await self.memory.recall(chat_id, recall_query, exclude_hashes)
            if recalled:
                memory_block = (
                    "=========================================\n"
                    "RELEVANT PRIOR TURNS FROM THIS CHAT:\n"
                    "(semantic recall — use as additional context; each bracket is a prior turn)\n"
                    "=========================================\n"
                )
                for role, content in recalled:
                    memory_block += f"[{role}] {content[:800].strip()}\n\n"
                memory_block += "=========================================\n\n"
                existing_system = next(
                    (i for i, m in enumerate(messages) if m.get("role") == "system"), None
                )
                if existing_system is not None:
                    sys_content = _text_of(messages[existing_system]["content"])
                    messages[existing_system]["content"] = sys_content + f"\n\n{memory_block}"
                else:
                    messages.insert(0, {"role": "system", "content": memory_block})
                await self._emit_status(
                    __event_emitter__,
                    f"🧠 Recalled {len(recalled)} prior turn(s) from this chat.",
                )

        # ── Vision proxy ─────────────────────────────────────────────
        selected_model = body.get("model", "")
        is_vision_model = selected_model in VISION_CAPABLE_MODELS
        if self.valves.ENABLE_VISION_PROXY and not is_vision_model:
            await self.vision.run_vision_proxy(
                messages, query_raw, chat_id or "", event_emitter=__event_emitter__
            )

        return body

    # ══════════════════════════════════════════════════════════════════════
    # OUTLET — thin coordination: orchestrator call → memory store → sweeps
    # ══════════════════════════════════════════════════════════════════════

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        if not body.get("messages") or not self.valves.FIREWORKS_API_KEY:
            return body

        messages = body["messages"]
        model_id = body.get("model", "")
        request_headers = {}
        if __user__:
            request_headers["x-openwebui-user-id"] = __user__.get("id", "")
            request_headers["x-openwebui-user-email"] = __user__.get("email", "")
        request_headers["x-openwebui-chat-id"] = self._extract_chat_id(body) or ""

        orch_response = await self._call_orchestrator(messages, model_id, request_headers)
        body["messages"][-1]["content"] = orch_response
        await self._emit_replace(__event_emitter__, orch_response)

        # ── Memory store + sweeps ────────────────────────────────────
        chat_id = self._extract_chat_id(body)
        if self.valves.ENABLE_CHAT_MEMORY and chat_id:
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                last_user_text = _text_of(user_msgs[-1].get("content", ""))
                await self.memory.store_turn(chat_id, "user", last_user_text)
            await self.memory.store_turn(chat_id, "assistant", _text_of(orch_response))
            await self.memory.sweep_referential()
            await self.memory.sweep_ttl()
            if self.valves.ENABLE_CHAT_MEMORY_COMPRESSION:
                asyncio.create_task(self.memory.maybe_compress(chat_id))

        return body
