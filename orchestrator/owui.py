"""Parsing what OpenWebUI actually sends.

OWUI wraps/injects around the user's real words — file uploads arrive as <source>
blocks, and the RAG template is applied even with bypass on (issues #19281/#17720).
Everything here recovers the user's actual message and source material from that
wrapping. Pure functions: regex + string work only.
"""
import re

from prism_core.messages import _text_of, _same_message_source

_USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.S | re.I)

def _unwrap_owui(text: str) -> str:
    if not text:
        return ""
    m = _USER_QUERY_RE.search(text)
    if m:
        return m.group(1).strip()
    if "<context>" in text and "</context>" in text:
        tail = text.rsplit("</context>", 1)[1].strip()
        if tail:
            return tail
    return text

def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _unwrap_owui(_text_of(m.get("content")).strip())
    return ""

_SOURCE_BLOCK_RE = re.compile(r"<source\b[^>]*>(.*?)</source>", re.S | re.I)

def _owui_source_blocks(text: str) -> list[str]:
    """OpenWebUI injects an attached file's text (paperclip upload) into the chat
    as <source id=.. name=..>..</source> blocks — by default appended to the final
    user message, or to the system message when RAG_SYSTEM_CONTEXT is set. The whole
    injected document lives here verbatim, so this is the authoritative grounding
    source. Parsing it is what makes file attachments ground correctly WITHOUT the
    user touching the RAG / full-context toggle — the file's own content, not a
    fragile ≥120-char paragraph guess that drops short résumé lines."""
    return [m.strip() for m in _SOURCE_BLOCK_RE.findall(text or "") if m.strip()]

def _user_source(messages) -> str:
    # Grounding "source" has two origins, in priority order:
    #   1. Files the user ATTACHED — OWUI delivers these as <source> blocks (any
    #      role). The full document is authoritative; take it whole.
    #   2. Source-like material the user PASTED inline (quotes, code blocks, labeled
    #      sources/notes/resume, long paragraphs) in their own turns.
    # NOT ordinary conversational text — grounding casual follow-ups ("what's my
    # name?") was slow and leaked "the provided source" into answers. The full
    # conversation is still available to the model and auditors via `messages`.
    parts = []
    for m in messages:
        text = _text_of(m.get("content"))
        blocks = _owui_source_blocks(text)
        parts.extend(blocks)
        if m.get("role") == "user":
            # Strip the <source> blocks first so the paragraph heuristic neither
            # double-counts them nor pulls in their XML wrappers as noise.
            remainder = _SOURCE_BLOCK_RE.sub("", text) if blocks else text
            src = _same_message_source(remainder)
            if src:
                parts.append(src)
    seen, out = set(), []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return "\n\n".join(out).strip()

def _all_user_text(messages) -> str:
    """Every user turn joined — facts AND instructions. The honesty auditor needs
    the instructions too, so it can tell 'emphasize my 8 years' (an instruction)
    apart from a stated fact."""
    return "\n\n".join(
        _unwrap_owui(_text_of(m.get("content")).strip())
        for m in messages
        if m.get("role") == "user" and _text_of(m.get("content")).strip()
    )
