"""Delivery + background-task helpers: track fire-and-forget writes, persist a turn to
chat memory, and re-export an already-verified deliverable. Split out of the agent
god-module (in-progress prism_core). Memory/tool access is via module attributes so the
test monkeypatch seam stays intact.
"""
import asyncio
import contextlib
import logging

from . import memory_client, toolserver
from prism_core.messages import _text_of
from .verifier import _WORD_RE
from .tools import _tool_path, _export_download

log = logging.getLogger(__name__)


# Strong references to fire-and-forget background writes. asyncio keeps only a
# WEAK reference to a running task, so a bare create_task() can be garbage
# collected mid-flight once the request returns — silently dropping the write.
_BG_TASKS: set = set()


def _track_task(task):
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


async def _cancel_mode_task(task) -> None:
    """Cancel the interaction-mode classifier when an early path handles the turn.
    Awaits the cancellation so no dangling task/exception is left behind."""
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _clip_memory_part(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _consolidated_user_memory(messages) -> str:
    """Raw last user message (clipped) for overflow recall. Storing raw means the embedding reflects what the user actually said, and a recalled turn matches the verbatim tail."""
    last_user = next(
        (_text_of(m.get("content")).strip() for m in reversed(messages)
         if m.get("role") == "user" and _text_of(m.get("content")).strip()),
        "",
    )
    return _clip_memory_part(last_user, 3000)


def _persist_turn(chat_id: str, messages: list[dict], assistant_text: str, session) -> None:
    """Persist one turn to chat memory: the consolidated user message + the assistant
    answer, fire-and-forget. Single home for what was duplicated across the plain-chat,
    edit, and agent-loop success paths (no-op without a chat_id)."""
    if not chat_id:
        return
    um = _consolidated_user_memory(messages)
    if um:
        _track_task(asyncio.create_task(memory_client._memory_store(chat_id, "user", um, session)))
    if assistant_text:
        _track_task(asyncio.create_task(memory_client._memory_store(chat_id, "assistant", assistant_text, session)))


async def _repackage_deliverable(content: str, filename: str, fmt: str, *, chat_id="", headers=None, session=None) -> str:
    """Re-export already-verified content under a new name/format — no writer or verifier needed (bytes don't change). Returns download-link markdown."""
    tool = (f"export_{fmt}" if fmt in ("docx", "pdf")
            else "export_markdown" if fmt in ("md", "markdown") else "export_docx")
    try:
        result = await toolserver.post(
            _tool_path(tool),
            {"markdown": content, "filename": filename or "document", "title": ""},
            session=session, headers=headers,
        )
        dl = _export_download(tool, result)
    except Exception as e:
        log.warning(f"[edit] re-export failed: {e}")
        return ""
    if not dl:
        return ""
    fn, url = dl
    if chat_id:
        _track_task(asyncio.create_task(memory_client._deliverable_store(chat_id, content, filename or fn, fmt)))
    return f"\n\n📎 [Download {fn}]({url})"


def _pending_prose_deliverable(pending) -> str:
    """Markdown of the largest pending prose export — the model writes the actual document in the export argument, not the chat message."""
    docs = [
        str(e.get("markdown") or "")
        for e in pending
        if e.get("tool") in ("export_docx", "export_pdf", "export_markdown")
    ]
    return max(docs, key=len) if docs else ""


def _same_doc(a: str, b: str) -> bool:
    """True when two texts are the same document — requires both similar length and high word overlap to avoid mistaking a summary for the document."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return False
    lo, hi = sorted((len(a), len(b)))
    if lo / hi < 0.6:  # very different lengths -> one is a summary/note, not the doc
        return False
    wa, wb = set(_WORD_RE.findall(a.lower())), set(_WORD_RE.findall(b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.6
