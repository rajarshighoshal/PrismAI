"""Ports: structural interfaces between PrismAI's core and the outside world.

Pure typing module — no runtime imports from providers, tools, or OpenWebUI. The
existing modules already satisfy these shapes structurally (Python Protocols are
duck-typed, so nothing here forces a change to them):

  LLMProvider          <- fireworks.py, openai_client.py, anthropic_client.py, gemini.py
                          (a local/self-hosted model becomes just another implementation)
  ToolProvider         <- toolserver.py (post to the tool-server HTTP API)
  MemoryStore          <- memory_client.py (chat turns, deliverables, plans, last-active)
  StyleProfileProvider <- style.py (per-user style profile, today read from webui.db)
  ArtifactSink         <- tool-server export endpoints, consumed via ToolProvider
  EventSink            <- the (kind, text) event channel run() streams to its caller

These exist so the core can be driven headless (tests, CLI, a future non-OWUI
frontend) by injecting fakes/alternatives at ONE seam instead of monkeypatching.
Adoption is incremental: call sites keep using the concrete modules until the
adapter step wires injection through.
"""
from typing import Any, AsyncGenerator, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """One chat-completion backend. The shared shape across all four current clients."""

    async def complete(
        self, messages, model: str, *,
        max_tokens: int, temperature: Optional[float] = None,
        session=None, label: str = "",
    ) -> str: ...

    async def chat(
        self, messages, model: str, *,
        max_tokens: int, temperature: Optional[float] = None,
        session=None, label: str = "",
    ) -> dict: ...

    def stream(
        self, messages, model: str, *,
        max_tokens: int, temperature: Optional[float] = None,
        session=None, label: str = "",
    ) -> AsyncGenerator[tuple[str, str], None]: ...


@runtime_checkable
class ToolProvider(Protocol):
    """HTTP tool execution (web fetch/export/grounding-audit) via the tool-server."""

    async def post(self, path: str, payload: dict, *, session=None, headers=None) -> Any: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Per-chat + per-user persistence. Implementations must be safe to call
    fire-and-forget (the orchestrator tracks them as background tasks)."""

    async def memory_store(self, chat_id: str, role: str, content: str, session=None) -> bool: ...

    async def memory_recall(self, chat_id: str, query: str, session=None) -> list: ...

    async def deliverable_store(self, chat_id: str, content: str, filename: str = "", fmt: str = "") -> bool: ...

    async def deliverable_get(self, chat_id: str) -> Optional[dict]: ...

    async def plan_store(self, chat_id: str, plan: dict) -> bool: ...

    async def plan_get(self, chat_id: str) -> Optional[dict]: ...

    async def plan_clear(self, chat_id: str) -> bool: ...


@runtime_checkable
class StyleProfileProvider(Protocol):
    """Per-user style/persona profile text (empty string when none/disabled)."""

    async def get_style_profile(self, user_id: str) -> str: ...


@runtime_checkable
class ArtifactSink(Protocol):
    """Produces a downloadable file from verified markdown. Returns (filename, url)
    or None on failure. Today: the tool-server export_* endpoints + OWUI files API."""

    async def export(self, markdown: str, filename: str, fmt: str, *, session=None, headers=None) -> Optional[tuple[str, str]]: ...


@runtime_checkable
class EventSink(Protocol):
    """Where a turn streams its events. kind is one of: content, reasoning, status,
    artifact. (The typed TurnEvent stream is a later step; this matches today's
    (kind, text) tuples.)"""

    async def emit(self, kind: str, text: str) -> None: ...
