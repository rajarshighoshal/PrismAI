"""OpenWebUI background metadata task helpers.

OWUI sends chat-title and tag generation as ordinary chat-completion requests.
They should be detected before the PrismAI agent loop so they can run on a cheap
simple model with no tools, no verifier, no memory writes, and no progress UI.
"""
import re


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text") or "")
                elif isinstance(p.get("content"), str):
                    parts.append(p.get("content") or "")
        return "\n".join(parts)
    return ""


_TITLE_TASK_RE = re.compile(
    r"generate\s+(?:a\s+)?(?:concise[\s,]+)?(?:\d+\s*-\s*\d+\s+word\s+)?title\b",
    re.I | re.S,
)
_TAG_TASK_RE = re.compile(
    r"generate\s+(?:\d+\s*-\s*\d+\s+)?(?:broad\s+)?tags?\b|categor(?:y|iz)e.*\btags?\b",
    re.I | re.S,
)


def owui_metadata_task(messages: list[dict]) -> str:
    """Return 'title'/'tags' for standard OWUI title/tag prompts, else ''."""
    if len(messages or []) != 1:
        return ""
    text = text_of((messages[0] or {}).get("content")).strip()
    low = text.lower()
    if "### task" not in low or "chat history" not in low:
        return ""
    if _TITLE_TASK_RE.search(text):
        return "title"
    if _TAG_TASK_RE.search(text):
        return "tags"
    return ""


def metadata_fallback(kind: str) -> str:
    return '{"title":"New Chat"}' if kind == "title" else '{"tags":["General"]}'
