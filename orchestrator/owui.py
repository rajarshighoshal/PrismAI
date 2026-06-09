"""Parsing what OpenWebUI actually sends.

OWUI wraps/injects around the user's real words — file uploads arrive as <source>
blocks, and the RAG template is applied even with bypass on (issues #19281/#17720).
Everything here recovers the user's actual message and source material from that
wrapping. Pure functions: regex + string work only.
"""
import re


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


# OWUI's RAG path wraps its template around the user's real message EVEN with "Bypass
# Embedding and Retrieval" enabled (open-webui issues #19281, #17720). Verified against
# this instance's OWUI code (utils/middleware.py + utils/task.py rag_template):
#   - newer default templates embed the query in <user_query>…</user_query> tags;
#   - a template WITHOUT a {{QUERY}} placeholder (this instance's saved default) is
#     PREPENDED to the original message (add_or_update_user_message append=False), so the
#     user's real text is everything AFTER the last </context>.
# These are machine-generated structural delimiters from OWUI's renderer — parse them so
# the edit classifier, gates, and verifier see what the USER said, not boilerplate.
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


def _has_images(messages) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    return True
    return False


def _split_content_parts(content):
    if not isinstance(content, list):
        return [], []
    text_parts = [
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
    ]
    image_parts = [
        p
        for p in content
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    return text_parts, image_parts


def _same_message_source(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = []
    for match in re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S):
        parts.append(match.strip())
    for match in re.findall(r"(?m)((?:^>.*(?:\n|$))+)", text):
        parts.append(re.sub(r"(?m)^>\s?", "", match).strip())
    for match in re.findall(
        r"(?is)(?:^|\n)\s*(?:sources?|notes?|context|references?|resume|job posting)\s*[:\-]\s*(.+?)(?:\n\s*\n|$)",
        text,
    ):
        parts.append(match.strip())
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for paragraph in paragraphs[1:]:
        if len(paragraph) >= 120:
            parts.append(paragraph)
    seen = set()
    out = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return "\n\n".join(out).strip()


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
