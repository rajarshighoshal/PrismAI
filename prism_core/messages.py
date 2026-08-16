"""Channel-neutral chat-message helpers.

Pure functions over OpenAI-style message dicts (content as str or list-of-parts) —
no OpenWebUI, no provider, stdlib only. Anything OWUI-specific (RAG-template
unwrapping, <source> block parsing) lives in the adapter (orchestrator/owui.py).
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
    """Extract source-like material a user PASTED inline (code blocks, quotes, labeled
    sources/notes/resume sections, long paragraphs). Channel-agnostic heuristic; the
    adapter combines it with its own attachment parsing."""
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
