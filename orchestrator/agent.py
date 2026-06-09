"""Model-driven agent loop for the orchestrator.

The harness exposes tools and enforces verification. It does not classify the
turn into prewritten task flows; the model chooses tools, the harness executes
them, and final output is held until the grounding gate allows it.

The full turn lifecycle (unwrap → startup I/O → vision → edit engine → grounding →
agent loop → polish/voice → verify → deliver → persist), the model routing table,
and the standing invariants live in docs/ARCHITECTURE.md — read that first.

Map of this file (in lifecycle order):
  OWUI parsing      _unwrap_owui, _last_user_text, _owui_source_blocks, _user_source
  startup I/O       style profile (style.py), _deliverable_get, _last_active, _gap_note
  vision            _describe_images_for_agent
  edit engine       _classify_edit, _repackage_deliverable, _edit_inject
  memory client     _memory_recall, _memory_store (tool-server HTTP)
  agent loop        run() + tool execution, budgets, SYSTEM_TOOL_GUARD
  polish & voice    _prose_provider, _classify_voice_register, _voice_pass
  verification      _verified_or_blocked, _fact_audit, _refine_facts,
                    _claim_verbatim_in_source (the verbatim backstop)
  delivery          _export_final, _same_doc, _summarize_correction
"""

import aiohttp
import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta

from . import config, fireworks, gemini, openai_client, anthropic_client, prompt_security, search, style, toolserver
from .prompts import (
    TOOL_SCHEMAS, SYSTEM_AGENT, SYSTEM_VISION, SYSTEM_GATE, SYSTEM_REQUEST_GATE, SYSTEM_EDIT_INTENT,
    SYSTEM_FACT_AUDIT, SYSTEM_TOOL_GUARD, SYSTEM_CHANGE_SUMMARY, SYSTEM_VOICE_REGISTER,
    _PROSE_POLISH_SYS, _VOICE_REGISTER, _VOICE_PASS_SYS,
)

log = logging.getLogger(__name__)



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


async def _describe_images_for_agent(messages, *, session=None):
    # Transcribe every image-bearing message CONCURRENTLY — a chat with several image
    # turns (e.g. figures across a thread) used to caption them one after another, paying
    # the 9-18s vision latency per image in series. gather preserves order.
    async def _describe(m):
        content = m.get("content")
        text_parts, image_parts = _split_content_parts(content)
        if not image_parts:
            return dict(m)
        user_text = "\n".join(t.strip() for t in text_parts if t.strip())
        prompt = (
            "The user attached image(s) to this message. Preserve the visual evidence "
            "for a downstream text-only agent.\n\n"
            f"USER TEXT:\n{user_text or '(none)'}\n\n"
            "Return a faithful transcription/description. Quote visible text exactly. "
            "Do not answer the user's task."
        )
        vision_content = [{"type": "text", "text": prompt}] + image_parts
        try:
            description = await fireworks.complete(
                [{"role": "system", "content": SYSTEM_VISION},
                 {"role": "user", "content": vision_content}],
                config.VISION_MODEL,
                max_tokens=config.VISION_MAX_TOKENS,
                temperature=0.0,
                session=session,
                label="vision",
            )
        except Exception as e:
            log.warning(f"[vision] image description failed: {e}")
            description = "Image was attached, but the vision transcription failed."
        combined = user_text
        if description.strip():
            # Frame the vision output as the ASSISTANT's own sight of the image, not as
            # text the user pasted — otherwise the agent disclaims "I'm text-only, based
            # on the transcription you provided" (factually right, but it never saw a
            # user transcription; the system's vision step produced it).
            combined = (
                (combined + "\n\n") if combined else ""
            ) + "[What you see in the image the user attached:]\n" + description.strip()
        new_m = dict(m)
        new_m["content"] = combined or "Image was attached, but no text was available."
        return new_m

    return list(await asyncio.gather(*(_describe(m) for m in messages)))


def _with_system(messages, system_text):
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            base = _text_of(m.get("content"))
            m["content"] = (base + "\n\n" + system_text).strip() if base else system_text
            return out
    return [{"role": "system", "content": system_text}] + out


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


async def _memory_recall(chat_id: str, query: str, session=None) -> list[tuple[str, str]]:
    """Call tool-server memory recall. Uses its own session for independence."""
    if not chat_id:
        return []
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/memory/recall",
                json={"chat_id": chat_id, "query": query, "top_k": 6},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [(t["role"], t["content"]) for t in data.get("turns", [])]
    except Exception:
        pass
    return []


async def _memory_store(chat_id: str, role: str, content: str, session=None) -> bool:
    """Call tool-server memory store. Uses its own session to survive caller's session closure."""
    if not chat_id:
        return False
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/memory/store",
                json={"chat_id": chat_id, "role": role, "content": content},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("stored", False)
    except Exception:
        pass
    return False


async def _deliverable_store(chat_id: str, content: str, filename: str = "", fmt: str = "") -> bool:
    """Persist a delivered document so a LATER turn can edit the real artifact instead
    of rebuilding it from scratch. Fire-and-forget; failure just means no edit memory."""
    if not (chat_id and (content or "").strip()):
        return False
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/deliverable/store",
                json={"chat_id": chat_id, "content": content, "filename": filename, "fmt": fmt},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("stored", False)
    except Exception:
        pass
    return False


async def _deliverable_get(chat_id: str):
    """Fetch the latest delivered document for this chat (what a follow-up edits), or None."""
    if not chat_id:
        return None
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/deliverable/get",
                json={"chat_id": chat_id},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("deliverable")
    except Exception:
        pass
    return None


async def _last_active(chat_id: str):
    """Unix time of the chat's previous stored turn (for resume-after-a-gap awareness), or None."""
    if not chat_id:
        return None
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/memory/last_active",
                json={"chat_id": chat_id},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("last_active")
    except Exception:
        pass
    return None


def _gap_note(last_active) -> str:
    """One line telling the model the user is RESUMING after a gap — but only when the
    previous message was a prior calendar day or more (in local time). Same-day follow-ups
    get nothing, so a continuous session stays clean. Fixes 'replies days later, acts like
    no time passed' without timestamping every turn."""
    if not last_active:
        return ""
    try:
        tz = timezone(timedelta(minutes=config.LOCAL_TZ_OFFSET_MINUTES))
        prev = datetime.fromtimestamp(float(last_active), tz)
        days = (datetime.now(tz).date() - prev.date()).days
    except Exception:
        return ""
    if days < 1:
        return ""
    when = "yesterday" if days == 1 else f"{days} days ago"
    return (f"NOTE: the user is resuming this conversation after a gap — their previous "
            f"message was {when} ({prev:%A, %d %B %Y}), not today. Earlier turns are older; "
            f"don't treat the conversation as one continuous sitting.")


async def _classify_edit(last_user: str, prior: dict, *, session=None) -> dict:
    """Classify a follow-up against the chat's last delivered document: rename|reformat|
    edit|new (reasoning-on; a semantic judgement, not keyword matching). run() uses this to
    pick the rename/reformat fast path and whether to hand the writer the prior document.
    Returns {action, filename, format}."""
    last_user = (last_user or "").strip()
    if not (last_user and prior and prior.get("content")):
        return {"action": "new"}
    payload = {
        "latest_user": last_user[:1500],
        "current_filename": prior.get("filename") or "document",
        "current_format": prior.get("fmt") or "docx",
    }
    action, filename, fmt = "new", "", ""
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_EDIT_INTENT},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=True)}],
            config.GROUNDING_GATE_MODEL, max_tokens=160, temperature=0.0,
            reasoning_effort="low", session=session, label="gate:edit",
        )
        data = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
        a = str(data.get("action", "new")).lower()
        if a in ("rename", "reformat", "edit"):
            action = a
            filename = (data.get("filename") or "").strip()
            fmt = (data.get("format") or "").strip().lower()
    except Exception:
        pass
    log.info(f"[edit-intent] action={action} msg={last_user[:80]!r}")
    return {"action": action, "filename": filename, "format": fmt}


async def _repackage_deliverable(content: str, filename: str, fmt: str, *, chat_id="", headers=None, session=None) -> str:
    """Re-export already-verified content under a new name/format with NO writer and NO
    verifier — the bytes don't change, so there is nothing to write or re-check (the
    'just rename it' path). Returns a download-link markdown string, or "" on failure."""
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
        _track_task(asyncio.create_task(_deliverable_store(chat_id, content, filename or fn, fmt)))
    return f"\n\n📎 [Download {fn}]({url})"


def _edit_inject(prior: dict) -> str:
    """System directive for a content edit: revise the REAL prior document surgically,
    changing only what the user asks and leaving everything else identical."""
    return (
        "REVISION TASK — the user is revising a document you already delivered in this "
        "chat. Here is that document, verbatim:\n\n"
        "--- CURRENT DOCUMENT ---\n" + (prior.get("content") or "").strip()
        + "\n--- END CURRENT DOCUMENT ---\n\n"
        "Make ONLY the change the user now asks for. Keep every other word, sentence, "
        "heading, and the overall structure byte-for-byte identical — do not rewrite, "
        "re-order, or 'improve' anything else. Then export the revised document with the "
        "same export tool and filename as before, unless the user asks to rename it."
    )


# Strong references to fire-and-forget background writes. asyncio keeps only a
# WEAK reference to a running task, so a bare create_task() can be garbage
# collected mid-flight once the request returns — silently dropping the write.
_BG_TASKS: set = set()


def _track_task(task):
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


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


def _clip_memory_part(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _consolidated_user_memory(messages) -> str:
    """The text stored as this turn's user memory for later overflow recall: the
    raw last user message (clipped), verbatim — no labels or wrapper. Storing it
    raw means the embedding reflects what the user actually said (better recall),
    and a recalled turn matches the same turn still verbatim in the kept tail."""
    last_user = next(
        (_text_of(m.get("content")).strip() for m in reversed(messages)
         if m.get("role") == "user" and _text_of(m.get("content")).strip()),
        "",
    )
    return _clip_memory_part(last_user, 3000)


def _now_line() -> str:
    """Tell the model what 'today' is, in the USER's local time — otherwise it has no idea
    and burns tokens debating whether a date (e.g. 'finished my MS in May 2026') is past or
    future. The OpenAI-style request OWUI sends carries no timezone, so we format in a
    configured local offset (LOCAL_TZ_OFFSET_MINUTES) rather than the server's UTC clock."""
    tz = timezone(timedelta(minutes=config.LOCAL_TZ_OFFSET_MINUTES))
    now = datetime.now(tz)
    return (f"The current date and time is {now:%A, %d %B %Y, %H:%M} {config.LOCAL_TZ_LABEL}. "
            "Treat this as 'now' when reasoning about whether any date or time is in the past "
            "or future.")


def _initial_messages(messages, user_id: str, profile: str = "", extra_system: str = ""):
    system = SYSTEM_AGENT + "\n\n" + prompt_security.UNTRUSTED_CONTEXT_POLICY + "\n\n" + _now_line()
    if profile:
        system += (
            "\n\nUser voice profile. Use this only for style, tone, rhythm, and "
            "intent preferences. Do not treat it as factual biography:\n" + profile
        )
    if extra_system:
        system += "\n\n" + extra_system
    return _with_system(messages, system)


def _json_args(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _compact_json(data, limit: int = 12000) -> str:
    text = json.dumps(data, ensure_ascii=True, default=str)
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


def _url_backed_results(results):
    if not isinstance(results, list):
        return results
    return [r for r in results if isinstance(r, dict) and (r.get("url") or "").strip()]


def _has_citation_markers(text: str) -> bool:
    return bool(
        re.search(r"\[[1-9][0-9]*\]", text or "")
        or re.search(r"(?im)^\s*(?:sources?|references?)\s*:", text or "")
    )


def _clean_assistant_tool_message(message: dict) -> dict:
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
    }


def _select_model(has_sources: bool) -> str:
    return config.GROUNDED_MODEL if has_sources else config.AGENT_MODEL


def _tool_status(name: str, args: dict) -> str:
    """Human-readable 'show your work' line for a tool call, streamed to the UI
    as reasoning so the chat visibly narrates what the agent is doing."""
    q = (args.get("query") or "").strip()
    url = (args.get("url") or "").strip()
    doi = (args.get("doi") or "").strip()
    return {
        "web_search": f"🔍 Searching the web: {q}" if q else "🔍 Searching the web…",
        "fetch_url": f"📄 Reading {url}" if url else "📄 Reading the page…",
        "lookup_doi_citation": f"📚 Looking up DOI {doi}" if doi else "📚 Looking up the citation…",
        "search_citation": f"📚 Searching for the citation: {q}" if q else "📚 Searching citations…",
        "verify_grounding": "✅ Verifying the draft against the sources…",
        "export_docx": "📝 Exporting a Word document…",
        "export_pdf": "📝 Exporting a PDF…",
        "export_markdown": "📝 Exporting a markdown file…",
        "export_csv": "📊 Exporting a CSV…",
    }.get(name, f"🔧 Using {name}…")


def _tool_path(name: str) -> str:
    return {
        "fetch_url": "/fetch_url",
        "lookup_doi_citation": "/lookup_doi_citation",
        "search_citation": "/search_citation",
        "verify_grounding": "/verify_grounding",
        "export_docx": "/export/docx",
        "export_pdf": "/export/pdf",
        "export_markdown": "/export/markdown",
        "export_csv": "/export/csv",
    }[name]


async def _execute_tool(name: str, args: dict, *, session=None, headers=None):
    if name == "web_search":
        query = str(args.get("query") or "").strip()
        max_results = args.get("max_results")
        try:
            max_results = int(max_results) if max_results is not None else None
        except Exception:
            max_results = None
        return await search.search(query[: config.QUERY_MAX_CHARS], max_results=max_results, session=session)

    if name in {
        "fetch_url",
        "lookup_doi_citation",
        "search_citation",
        "verify_grounding",
        "export_docx",
        "export_pdf",
        "export_markdown",
        "export_csv",
    }:
        return await toolserver.post(
            _tool_path(name),
            args,
            session=session,
            headers=headers,
        )
    return {"error": True, "detail": f"unknown tool: {name}"}


async def _tool_allowed(name: str, args: dict, messages, source: str, *, session=None):
    if name != "web_search":
        return True, ""
    payload = {
        "latest_user": _last_user_text(messages),
        "source_available": bool(source.strip()),
        "proposed_tool": name,
        "proposed_args": args,
    }
    try:
        raw = await fireworks.complete(
            [
                {"role": "system", "content": SYSTEM_TOOL_GUARD},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            config.GROUNDING_GATE_MODEL,
            max_tokens=120,
            temperature=0.0,
            session=session,
            label="gate:tool",
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        return bool(data.get("allow")), str(data.get("reason") or "")
    except Exception:
        return True, ""


def _budgeted_tools(tool_call_count: int, web_search_count: int):
    if tool_call_count >= config.MAX_TOOL_CALLS_PER_TURN:
        return None
    if web_search_count >= config.MAX_WEB_SEARCHES_PER_TURN:
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] != "web_search"]
    return TOOL_SCHEMAS


def _export_download(name: str, result):
    """(filename, relative_url) for a successful export, else None."""
    if not name.startswith("export_") or not isinstance(result, list):
        return None
    for item in result:
        if isinstance(item, dict) and item.get("download_url"):
            return item.get("filename") or "file", item["download_url"]
    return None


def _pending_prose_deliverable(pending) -> str:
    """The markdown of the largest pending prose export (docx/pdf/md). The model
    typically writes the actual document in the export ARGUMENT and only a summary as
    its chat message, so this — not the chat content — is the real deliverable to
    verify, polish, and file."""
    docs = [
        str(e.get("markdown") or "")
        for e in pending
        if e.get("tool") in ("export_docx", "export_pdf", "export_markdown")
    ]
    return max(docs, key=len) if docs else ""


async def _export_final(pending, final_text, prose, messages, source, *, chat_id="", headers=None, session=None):
    """Build the deferred export files. When the verified chat deliverable IS this
    document (same text the honesty gate approved), the file carries that verified
    version — so a surgical correction lands in the file, not the model's raw export
    argument. When the chat is only a short confirmation and the document lives in the
    export argument, that argument is used (and polished) instead.

    Returns (links_str, filed_deliverable) — filed_deliverable is True when the file
    was built from the verified chat body, so the chat must NOT repeat it."""
    deliverable = (final_text or "").strip()
    out, filed_deliverable = [], False
    for exp in pending:
        raw = exp["markdown"]
        if deliverable and len(deliverable) >= config.POLISH_MIN_CHARS and _same_doc(deliverable, raw):
            md = deliverable          # verified/corrected version of THIS document
            filed_deliverable = True
        else:
            md = raw                  # document lives in the argument; polish the draft
            if prose is not None and len(md) >= config.POLISH_MIN_CHARS:
                try:
                    client, pmodel = prose
                    polished = await client.complete(
                        _prose_polish_messages(messages, md, source), pmodel,
                        max_tokens=config.DRAFT_MAX_TOKENS,
                        temperature=config.WRITER_TEMPERATURE, session=session,
                        label="polish:export",
                    )
                    if polished and polished.strip():
                        md = polished.strip()
                except Exception as e:
                    log.warning(f"[export] polish of deliverable failed, exporting draft: {e}")
        result = await toolserver.post(
            _tool_path(exp["tool"]),
            {"markdown": md, "filename": exp["filename"], "title": exp["title"]},
            session=session, headers=headers,
        )
        dl = _export_download(exp["tool"], result)
        if dl and dl not in out:
            out.append(dl)
        # Persist exactly what went INTO the file so a later turn edits the real artifact
        # (not a reconstruction). Fire-and-forget; never blocks the response.
        if chat_id and md.strip():
            fmt = "docx" if "docx" in exp["tool"] else "pdf" if "pdf" in exp["tool"] else "md"
            _track_task(asyncio.create_task(_deliverable_store(chat_id, md, exp["filename"], fmt)))
    links = ("\n\n" + "\n".join(f"📎 [Download {fn}]({url})" for fn, url in out)) if out else ""
    return links, filed_deliverable


def _source_from_tool(name: str, result) -> str:
    if name == "web_search" and isinstance(result, list):
        return search.format_context(_url_backed_results(result))
    if name == "fetch_url" and isinstance(result, dict) and not result.get("error"):
        return f"Fetched URL: {result.get('url', '')}\n\n{result.get('text', '')}".strip()
    if name in {"lookup_doi_citation", "search_citation"} and isinstance(result, dict) and not result.get("error"):
        return _compact_json(result, limit=8000)
    return ""


def _visible_tool_result(name: str, result):
    if name == "web_search":
        return {
            "results": _url_backed_results(result),
            "citation_rule": (
                "Only cite facts supported by these URL-backed results. Search "
                "summaries without URLs are not sources."
            ),
        }
    return toolserver.summarize_result(name, result)


def _combined_source(user_source: str, tool_sources: list[str]) -> str:
    parts = []
    if user_source:
        parts.append("USER-PROVIDED SOURCE MATERIAL:\n" + user_source)
    if tool_sources:
        parts.append("TOOL-GATHERED SOURCE MATERIAL:\n" + "\n\n".join(tool_sources))
    return "\n\n---\n\n".join(parts).strip()


async def _needs_verification(messages, candidate: str, source: str, *, session=None) -> bool:
    if not config.ENABLE_GROUNDING_GATE:
        return bool(source)
    if not candidate.strip():
        return False
    payload = {
        "latest_user": _last_user_text(messages),
        "source_available": bool(source.strip()),
        "draft": candidate[:6000],
    }
    try:
        raw = await fireworks.complete(
            [
                {"role": "system", "content": SYSTEM_GATE},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            config.GROUNDING_GATE_MODEL,
            max_tokens=120,
            temperature=0.0,
            session=session,
            label="gate:verify",
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        return bool(data.get("needs_verification"))
    except Exception:
        return bool(source)


async def _request_needs_work(messages, *, session=None) -> bool:
    """Plain-chat gate: does this turn need the agentic loop (tools/source/verify)?
    Uncertain -> True (use the safe buffered loop, never stream a risky turn)."""
    q = _last_user_text(messages).strip()[:2000]
    if not q:
        return True
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_REQUEST_GATE},
             {"role": "user", "content": q}],
            config.GROUNDING_GATE_MODEL, max_tokens=60, temperature=0.0, session=session,
            label="gate:work",
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        return bool(json.loads(match.group(0) if match else raw).get("needs_work", True))
    except Exception:
        return True


async def _refine_complete(prompt: str, user: str, *, prose=None, session=None) -> str:
    """Run a refine pass on the SAME prose model that wrote the draft when one was
    used (so paid Opus/GPT prose isn't silently reverted to the open model on a
    fix); otherwise use the default open REFINE_MODEL."""
    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    try:
        if prose is not None:
            client, model = prose
            return await client.complete(
                msgs, model, max_tokens=config.DRAFT_MAX_TOKENS,
                temperature=config.WRITER_TEMPERATURE, session=session,
                label="refine",
            )
        return await fireworks.complete(
            msgs, config.REFINE_MODEL,
            max_tokens=config.DRAFT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE, session=session,
            label="refine",
        )
    except Exception:
        return ""


# Corrections are SURGICAL by default — the draft was already shown to the user, so
# the verifier patches only the flagged spans and leaves the rest untouched. Only
# when a surgical patch can't clear the audit (the draft is too wrong to fix in
# place) do we escalate to a full rewrite-with-feedback by the upstream writer.
_SURGICAL_DIRECTIVE = (
    "Make the SMALLEST edit that fixes the flagged problems: change ONLY the specific "
    "words or sentences that are unsupported, and keep every other sentence, the "
    "wording, structure, tone, and length EXACTLY as written. Do not re-style, "
    "re-order, expand, or rewrite anything you are not explicitly fixing."
)
_REWRITE_DIRECTIVE = (
    "The draft has too many unsupported claims to patch in place. Write it again from "
    "the user's request and the SOURCE, fully addressing the feedback. Use ONLY real, "
    "supported facts — invent nothing — and keep the requested format and length."
)


def _edit_directive(rewrite: bool) -> str:
    return _REWRITE_DIRECTIVE if rewrite else _SURGICAL_DIRECTIVE


def _prose_provider(voice):
    """Map the agent-chosen polish voice to (client, model), honoring availability
    with graceful fallback. None if no provider is usable (stay on the open draft)."""
    if voice == "gpt-5.5" and openai_client.available():
        return openai_client, config.OPENAI_PROSE_MODEL_PREMIUM
    if voice == "opus" and anthropic_client.available():
        return anthropic_client, config.ANTHROPIC_PROSE_MODEL
    if voice == "sonnet" and anthropic_client.available():
        return anthropic_client, config.ANTHROPIC_STANDARD_MODEL
    # requested provider unavailable — fall back to any usable prose model
    if anthropic_client.available():
        return anthropic_client, config.ANTHROPIC_PROSE_MODEL
    if openai_client.available():
        return openai_client, config.OPENAI_PROSE_MODEL_PREMIUM
    if gemini.available():
        return gemini, config.GEMINI_PROSE_MODEL
    return None




def _prose_polish_messages(messages, candidate, source):
    """Build the polish request: the user's ask + (optional) source as untrusted
    reference + the open model's draft to rewrite. No tool-role messages /
    tool_calls (OpenAI-compat endpoints, esp. Gemini, choke on those)."""
    user_req = _all_user_text(messages)
    parts = [f"USER REQUEST:\n{user_req}"]
    if source.strip():
        parts.append(prompt_security.wrap_untrusted("gathered source material", source[:12000]))
    parts.append(f"DRAFT TO POLISH:\n{candidate}")
    return [
        {"role": "system", "content": _PROSE_POLISH_SYS},
        {"role": "user", "content": "\n\n".join(parts)},
    ]




async def _classify_voice_register(request, candidate, *, session=None) -> str:
    """Pick the voice-pass register (warm/formal/none) for an exported deliverable,
    naturally from the document — this replaces the model-chosen polish-tool argument
    so important documents still get the right warmth touch without a confusing tool."""
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_VOICE_REGISTER},
             {"role": "user", "content": f"REQUEST:\n{request[:1500]}\n\nDELIVERABLE (excerpt):\n{candidate[:1500]}"}],
            config.GROUNDING_GATE_MODEL, max_tokens=30, temperature=0.0,
            session=session, label="gate:voice")
        m = re.search(r"\{.*\}", raw, flags=re.S)
        reg = str(json.loads(m.group(0) if m else raw).get("register", "none")).lower()
        return reg if reg in ("warm", "formal") else "none"
    except Exception:
        return "none"


async def _voice_pass(candidate, register, *, session=None):
    """Optional sonnet voice-only pass at a register (warm/formal). Never alters facts."""
    if not anthropic_client.available():
        return candidate
    sys = _VOICE_PASS_SYS.replace("{register}", _VOICE_REGISTER.get(register, _VOICE_REGISTER["formal"]))
    try:
        out = await anthropic_client.complete(
            [{"role": "system", "content": sys}, {"role": "user", "content": f"DRAFT:\n{candidate}"}],
            config.ANTHROPIC_STANDARD_MODEL,
            max_tokens=config.AGENT_MAX_TOKENS, temperature=config.WRITER_TEMPERATURE, session=session,
            label="voice")
        return out.strip() if (out and out.strip()) else candidate
    except Exception as e:
        log.warning(f"[voice_pass] {register} failed, keeping draft: {e}")
        return candidate


def _is_clarification(text: str) -> bool:
    """Detect a clarifying-question turn (the agent asked the user for info rather
    than producing a deliverable) — these must NOT be prose-polished or treated as
    a final deliverable."""
    t = (text or "").strip().lower()
    if "?" not in t:
        return False
    # Heuristic: short-ish, question-led, asks for info before writing.
    cues = ("i need to know", "before i write", "could you clarify", "a few questions",
            "to write this", "which of", "can you tell me", "what is the", "let me know")
    return len(t) < 1200 and (t.count("?") >= 2 or any(c in t for c in cues))


def _all_user_text(messages) -> str:
    """Every user turn joined — facts AND instructions. The honesty auditor needs
    the instructions too, so it can tell 'emphasize my 8 years' (an instruction)
    apart from a stated fact."""
    return "\n\n".join(
        _unwrap_owui(_text_of(m.get("content")).strip())
        for m in messages
        if m.get("role") == "user" and _text_of(m.get("content")).strip()
    )


_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm_token_str(text) -> str:
    """Lowercase, reduce to [a-z0-9] tokens joined by single spaces and wrapped in
    spaces, so a substring test matches only on whole-token boundaries."""
    return " " + " ".join(_WORD_RE.findall(str(text).lower())) + " "


def _claim_verbatim_in_source(phrase, source_norm: str) -> bool:
    """True ONLY when a flagged phrase appears near-verbatim and CONTIGUOUS in the
    source — the narrow case where the auditor flagged something literally present, so
    it's a clear false positive to keep. Unlike a word-overlap score this never rescues
    a semantic inflation ('led' for a sourced 'collaborated', '$1.5M/year' for '/month',
    'p<0.001' for 'p<0.05'): those are not contiguous spans of the source, so the
    auditor's flag stands and the claim is stripped. A tight backstop BEHIND the
    full-source auditor, which makes the real semantic judgement."""
    toks = _WORD_RE.findall(str(phrase).lower())
    if len(toks) < 2:  # too short for a reliable verbatim match — defer to the auditor
        return False
    return (" " + " ".join(toks) + " ") in source_norm


def _same_doc(a: str, b: str) -> bool:
    """True when two texts are clearly the same document — used to tell 'the chat body
    IS the exported deliverable' from 'the chat is a short summary/confirmation while
    the document lives in the export argument'. Requires BOTH similar length and high
    word overlap: a summary OF a document shares its vocabulary but is far shorter, so
    the length gate stops it from being mistaken for the document itself."""
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


def _fit_audit_source(source: str, draft: str, budget: int) -> str:
    """Fit `source` into `budget` chars for the auditor WITHOUT head-truncating.

    Head-truncation was the over-strip bug: a long job posting in front pushed the
    résumé past the cut, so grounded credentials looked unsupported. Real documents
    (résumé + posting ~11k) fit the budget whole and return unchanged. Only a
    genuinely oversized source is trimmed — and then by *relevance to the draft*
    (keep the paragraphs whose vocabulary the draft actually uses, in original
    order), so the spans a claim is grounded in survive instead of the tail being
    silently dropped. For sources far beyond one model's window the correct path is
    chunk-and-audit (union of supported); this helper covers the realistic middle.
    """
    source = source.strip()
    if len(source) <= budget:
        return source
    draft_words = set(_WORD_RE.findall(draft.lower()))
    paras = [p for p in re.split(r"\n\s*\n", source) if p.strip()]
    scored = [
        (i, len(draft_words & set(_WORD_RE.findall(p.lower()))), p)
        for i, p in enumerate(paras)
    ]
    kept, used = [], 0
    for i, _score, p in sorted(scored, key=lambda t: t[1], reverse=True):
        if used + len(p) > budget:
            continue
        kept.append((i, p))
        used += len(p) + 2
    kept.sort()
    return "\n\n".join(p for _i, p in kept) or source[:budget]


async def _fact_audit(full_request: str, source: str, candidate: str, *, session=None, raw_source=None):
    """The single fact-integrity verifier for ANY written deliverable. Sees the USER
    REQUEST, the full SOURCE MATERIAL, and the DRAFT side by side and flags only
    unsupported FACTUAL claims (against the user's stated facts + SOURCE + common
    knowledge); motivation, opinion, and framing pass. `raw_source` (if given) mirrors
    `source` and is used only for the diagnostic.
    Returns {unsupported:[...], verdict:FABRICATION|CLEAN} or None on failure (fail-soft)."""
    if not candidate.strip():
        return None
    fitted = _fit_audit_source(source, candidate, config.AUDIT_SOURCE_BUDGET)
    # De-dup only: the source lives in SOURCE MATERIAL, so drop the identical <source>
    # blocks from the request (no information lost). NO truncation — the auditor sees
    # the whole request, the whole source, and the whole draft.
    request = _SOURCE_BLOCK_RE.sub("", full_request).strip()
    user = (
        f"USER REQUEST (instructions; the FACTS are in SOURCE MATERIAL):\n{request}\n\n"
        f"SOURCE MATERIAL:\n{fitted if fitted else '(none)'}\n\n"
        f"DRAFT:\n{candidate}"
    )
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_FACT_AUDIT},
             {"role": "user", "content": user}],
            config.HONESTY_MODEL,
            max_tokens=config.AUDIT_MAX_TOKENS,
            temperature=0.0,
            reasoning_effort=config.AUDIT_REASONING_EFFORT,
            session=session,
            label="audit",
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        if isinstance(data, dict) and config.LOG_SOURCE_DIAG:
            # How many flagged phrases appear verbatim in the source? A non-zero count
            # means the auditor mis-flagged something literally present — a false
            # positive the verbatim backstop will keep.
            source_norm = _norm_token_str(raw_source or fitted)
            flagged = data.get("unsupported") or []
            false_pos = sum(1 for f in flagged if _claim_verbatim_in_source(f, source_norm))
            log.info(f"[audit-diag] reasoning={config.AUDIT_REASONING_EFFORT} audit_src_chars={len(fitted)} "
                     f"verdict={data.get('verdict')} flagged={len(flagged)} verbatim_false_pos={false_pos}")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def _refine_facts(full_request: str, source: str, candidate: str, unsupported,
                        *, rewrite=False, prose=None, session=None) -> str:
    """Remove the unsupported FACTS the verifier flagged, keeping everything else —
    motivation, framing, tone, structure — exactly. Surgical by default; full rewrite
    only when the draft is too wrong to patch."""
    listed = "\n".join(f"- {u}" for u in unsupported) if unsupported else "(unsupported factual claims)"
    prompt = (
        _edit_directive(rewrite) + " "
        "Remove or neutrally rephrase each listed claim so the draft asserts no fact "
        "the user or SOURCE did not support. Do NOT invent replacements. If only PART of "
        "a flagged claim is unsupported (an invented elaboration attached to a fact the "
        "user did state), remove ONLY the unsupported part — never delete a fact the user "
        "themselves stated or the source contains. KEEP all motivation, interest, framing, "
        "tone, and structure exactly — only the unsupported FACTS change. If removing a "
        "claim leaves the draft thinner, that is fine; do not pad with invented detail. "
        "Output only the revised deliverable."
    )
    user = (
        f"USER REQUEST:\n{full_request}\n\n"
        f"SOURCE MATERIAL:\n{source if source.strip() else '(none)'}\n\n"
        f"UNSUPPORTED CLAIMS TO FIX:\n{listed}\n\n"
        f"DRAFT:\n{candidate}"
    )
    return await _refine_complete(prompt, user, prose=prose, session=session)


def _facts_block_msg(unsupported) -> str:
    listed = "\n".join(f"- {u}" for u in unsupported) if unsupported else ""
    return (
        "I can't present those claims as true from the facts you gave me. These "
        "details are unsupported:\n\n" + listed
        + "\n\nI can still write a truthful version using the facts you provided, "
        "or you can give me the real details and I'll include them accurately."
    )


async def _summarize_correction(before: str, after: str, *, session=None) -> str:
    """One cheap call describing WHAT the verifier changed, so the chat can show the
    correction in a sentence or two instead of re-dumping the whole deliverable."""
    if not (before.strip() and after.strip()):
        return ""
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_CHANGE_SUMMARY},
             {"role": "user", "content": f"BEFORE:\n{before[:8000]}\n\nAFTER:\n{after[:8000]}"}],
            config.HONESTY_MODEL, max_tokens=200, temperature=0.0, session=session,
            label="summarize",
        )
        return (raw or "").strip()
    except Exception:
        return ""


async def _verified_or_blocked(messages, candidate: str, source: str, *, recall_context: str = "", prose=None, force: bool = False, session=None):
    """ONE fact-integrity check for any kind of writing — email, resume, letter,
    report, research, chat. No document-type branching: flag only unsupported FACTS,
    leave motivation / opinion / framing untouched, surgically correct, escalate to a
    rewrite if a patch can't clear it, and block only if facts still can't be made
    truthful."""
    if not config.ENABLE_VERIFICATION:
        return "ok", candidate

    # Recalled facts are the user's OWN earlier statements, surfaced only when a long
    # chat overflowed the context budget (see run()). They are established context, so
    # the verifier must see them — otherwise a correctly-recalled fact gets flagged as
    # a fabrication and stripped. For normal chats recall_context is "" (no-op).
    _rc = (recall_context or "").strip()
    _recall_extra = ("\n\nEARLIER IN THIS CONVERSATION (the user already stated):\n" + _rc) if _rc else ""
    # The USER is the authority on their OWN facts. Anything they state in the conversation
    # ("I've now finished my MS", "I work on geometric probes") grounds their own claims —
    # alongside the uploaded source and recalled facts. Without this the verifier strips
    # facts the user ADDED in chat just because they aren't in the original files (the
    # exact bug that 'corrected out' the user's real geometric-probes research). The
    # honesty guarantee still holds: the MODEL can't invent anything the user didn't give.
    _user_said = _all_user_text(messages)
    grounding_source = "\n\n".join(p for p in (source, _rc, _user_said) if p and p.strip())

    # Verify only a FACTUAL DELIVERABLE — the cheap classifier decides, and an exported
    # file always counts (a document the user will rely on). The mere PRESENCE of a
    # source does not force a verification: that blanket rule made the honesty pass fire
    # on every casual turn that happened to have pasted text or an image, and strip
    # reasonable asides ("software like Keybr or TypingClub") from an opinion. Opinion,
    # assessment, and Q&A about an attachment are not deliverables and skip straight through.
    needs = force or await _needs_verification(messages, candidate, grounding_source, session=session)
    if not needs:
        return "ok", candidate

    # The auditor and refiner must know today's date too — the writer is TOLD the current
    # date (system prompt), so a dated letterhead is established context, not a fabrication
    # to strip (the bug that "corrected" a real date into a [Date] placeholder).
    full_request = _now_line() + "\n\n" + _all_user_text(messages) + _recall_extra

    if config.ENABLE_HONESTY_AUDIT:
        # Hand the auditor the FULL source (relevance-fitted inside _fact_audit), the
        # request, and the draft — side by side — and let it make the SEMANTIC call. No
        # distilled fact list in between: a lossy summary dropped real credentials (so the
        # auditor flagged them) and can't preserve verbatim figures anyway. With the whole
        # source in view it can tell 'led' is unsupported when the source says
        # 'collaborated'; a tight verbatim backstop then keeps only the flags whose exact
        # phrase is literally in the source (the auditor mis-flagging a quote).
        source_norm = _norm_token_str(grounding_source)

        async def _flags(text):
            """Genuinely-unsupported claims in `text`: the auditor's flags MINUS any whose
            exact phrase appears contiguously in the source (a literal false positive). The
            auditor judges meaning; the backstop only rescues verbatim quotes — it never
            keeps a same-words inflation. [] means clean."""
            a = await _fact_audit(full_request, grounding_source, text, session=session, raw_source=grounding_source)
            if not (a and str(a.get("verdict", "")).upper().startswith("FAB")):
                return []
            return [u for u in (a.get("unsupported") or []) if not _claim_verbatim_in_source(u, source_norm)]

        unsupported = await _flags(candidate)
        if unsupported:
            refined = await _refine_facts(full_request, grounding_source, candidate, unsupported, prose=prose, session=session)
            remaining = (await _flags(refined)) if refined else unsupported
            if remaining:  # surgical patch left genuine fabrications — one full-rewrite attempt
                refined = await _refine_facts(full_request, grounding_source, candidate, unsupported, rewrite=True, prose=prose, session=session)
                remaining = (await _flags(refined)) if refined else unsupported
            if refined and not remaining:
                candidate = refined
            else:
                return "unsupported_claims", _facts_block_msg(unsupported)

    # Citations must rest on real retrieved sources.
    if not grounding_source.strip() and _has_citation_markers(candidate):
        return (
            "citation_without_source",
            "The previous draft included citations or source labels, but no sources were actually supplied or retrieved.",
        )
    return "ok", candidate


def _norm_turn(content) -> str:
    """Collapse whitespace so a recalled turn (now stored verbatim) can be deduped
    against the same turn still verbatim in the kept tail."""
    return " ".join((content or "").split())


def _split_recent_history(messages, budget_chars: int):
    """Split a long history into (recent_tail, older_head). The tail is the most
    recent messages that fit in ~70% of the budget (leaving room for the recall
    block + the answer); the head is everything older, to be represented by
    recall. Always keeps at least the final message."""
    keep = max(1, int(budget_chars * 0.7))
    total, cut = 0, 0
    for i in range(len(messages) - 1, -1, -1):
        total += len(_text_of(messages[i].get("content")))
        if total > keep and i < len(messages) - 1:
            cut = i + 1
            break
    return messages[cut:], messages[:cut]


async def run(messages, *, user_id="", session=None, request_headers=None, user_model=""):
    """Drive one chat turn. Async generator of (kind, text)."""
    if not messages:
        yield ("content", "")
        return

    # Instant feedback: a heavy turn (uploaded doc, big paste) spends a few seconds
    # on the first generation before any other breadcrumb — show something now so
    # the user isn't staring at a blank.
    if config.SHOW_WORK:
        yield ("reasoning", "🧠 Reading your request…\n")

    user_final_model = (user_model or "").strip()
    is_user_model = bool(user_final_model)

    had_images = _has_images(messages)
    req_headers = request_headers or {}
    chat_id = req_headers.get("x-openwebui-chat-id", "")
    # Startup I/O in parallel: the style profile and the chat's last delivered document
    # (for multi-turn edits) both load off-thread, overlapped with image description.
    style_task = asyncio.ensure_future(style.get_style_profile(user_id))
    deliverable_task = asyncio.ensure_future(_deliverable_get(chat_id)) if chat_id else None
    active_task = asyncio.ensure_future(_last_active(chat_id)) if chat_id else None
    if had_images:
        if config.SHOW_WORK:
            yield ("reasoning", "🖼️ Reading image context…\n")
        messages = await _describe_images_for_agent(messages, session=session)

    style_profile = await style_task
    prior_deliverable = (await deliverable_task) if deliverable_task else None
    gap_note = _gap_note(await active_task) if active_task else ""

    # ── Multi-turn edit of the chat's last delivered document ──────────────────────
    # Do the LEAST the request needs. rename/reformat just re-package the verified bytes
    # (no writer, no verifier — nothing changed to re-check). A content edit injects the
    # REAL prior document so the model changes only what's asked. Anything else is normal.
    edit_directive, edit_baseline = "", ""
    if prior_deliverable and prior_deliverable.get("content"):
        intent = await _classify_edit(_last_user_text(messages), prior_deliverable, session=session)
        if intent["action"] in ("rename", "reformat"):
            fmt = intent.get("format") or prior_deliverable.get("fmt") or "docx"
            filename = intent.get("filename") or prior_deliverable.get("filename") or "document"
            link = await _repackage_deliverable(
                prior_deliverable["content"], filename, fmt,
                chat_id=chat_id, headers=req_headers, session=session)
            verb = "Renamed" if intent["action"] == "rename" else f"Re-exported as {fmt.upper()}"
            yield ("content", (f"📄 {verb} — download below.{link}") if link
                   else "I couldn't re-export that file — want me to try again?")
            return
        if intent["action"] == "edit":
            edit_directive = _edit_inject(prior_deliverable)
            edit_baseline = (prior_deliverable.get("content") or "").strip()

    # Extra system context handed to the writer: the resume-after-gap note (if any) and,
    # for a content edit, the prior document to revise.
    agent_extra = "\n\n".join(x for x in (gap_note, edit_directive) if x)

    # Chat-memory recall = OVERFLOW handler only. OWUI sends the full native
    # conversation history every turn, so for normal-length chats the model (and
    # the verifier) already have everything — running recall would just re-inject
    # what is already present AND risk the can't-lie layer flagging a recalled
    # fact it cannot see as a fabrication. Only when the history exceeds the
    # context budget do we keep the recent tail verbatim and recall the relevant
    # older facts to stand in for the trimmed head. recall_context is then handed
    # to the verifier so those facts count as established (not fabricated).
    recall_context = ""
    messages_for_verify = messages
    history_chars = sum(len(_text_of(m.get("content"))) for m in messages)
    if chat_id and history_chars > config.MEMORY_CONTEXT_BUDGET_CHARS:
        recent, _older = _split_recent_history(messages, config.MEMORY_CONTEXT_BUDGET_CHARS)
        recent_norm = {_norm_turn(_text_of(m.get("content"))) for m in recent}
        # Query recall on the CURRENT question (last user turn). Joining several
        # recent turns lets a big pasted message crowd out the actual intent once
        # clipped, so recall would search on filler instead of what's being asked.
        recall_query = next(
            (_text_of(m.get("content")).strip() for m in reversed(messages) if m.get("role") == "user"),
            "",
        )[:2000]
        # Split recalled turns by ROLE. Only the user's OWN earlier statements may
        # be treated as established/grounding context. Feeding the assistant's own
        # prior answers back into the verifier as "source" would let the model
        # ground a fresh claim on its own earlier (possibly unverified) output —
        # the verifier rubber-stamping itself. Assistant turns are kept for
        # continuity only, clearly labeled, and never grounded against.
        user_lines, asst_lines, seen = [], [], set()
        if recall_query.strip():  # no current question -> nothing meaningful to recall
            for role, content in await _memory_recall(chat_id, recall_query, session):
                c = _norm_turn(content)[:500]
                if not c or c in recent_norm or c in seen:
                    continue  # already visible in the kept tail, or a duplicate
                seen.add(c)
                (user_lines if role == "user" else asst_lines).append(c)
        if user_lines or asst_lines:
            scratch = _initial_messages(recent, user_id, style_profile, extra_system=agent_extra)
            messages_for_verify = recent
            recall_context = "\n".join(user_lines)  # USER-stated facts only -> verifier
            blocks = []
            if user_lines:
                blocks.append(
                    "Earlier in THIS conversation the user stated (established facts "
                    "they told you):\n" + recall_context
                )
            if asst_lines:
                blocks.append(
                    "Earlier assistant replies, for continuity only — NOT verified "
                    "facts; do not rely on them as sources:\n" + "\n".join(asst_lines)
                )
            scratch.append({
                "role": "system",
                "content": "This is a long conversation; earlier turns were trimmed "
                "to fit.\n\n" + "\n\n".join(blocks),
            })
        else:
            # Recall produced nothing (cold chat, service down, transient embed
            # failure). Don't trim blind: ~140k chars is still well within the
            # model window, so keep the full conversation rather than silently
            # dropping the older head with no replacement.
            scratch = _initial_messages(messages, user_id, style_profile, extra_system=agent_extra)
    else:
        scratch = _initial_messages(messages, user_id, style_profile, extra_system=agent_extra)

    # Grounding source = the user's pasted/quoted material across the WHOLE
    # conversation. verify_grounding takes the source independently of the model's
    # context budget, so a document pasted in a since-trimmed turn can still ground
    # a faithful quote; recall_context separately carries older user-stated facts.
    user_source = _user_source(messages)
    if edit_baseline:
        # The prior verified document grounds the edit: its carried-over facts count as
        # established, and a non-empty source routes the turn to the grounded writer and
        # the honesty gate (never the plain-chat fast path).
        user_source = (user_source + "\n\n" + edit_baseline).strip() if user_source.strip() else edit_baseline

    if config.LOG_SOURCE_DIAG:
        chars_by_role, source_blocks = {}, 0
        for m in messages:
            r = m.get("role", "?")
            t = _text_of(m.get("content"))
            chars_by_role[r] = chars_by_role.get(r, 0) + len(t)
            source_blocks += len(_owui_source_blocks(t))
        log.info(
            f"[source-diag] user_source_chars={len(user_source)} "
            f"owui_source_blocks={source_blocks} chars_by_role={chars_by_role}"
        )

    # Plain-chat fast path: stream the answer live when the turn needs no tools,
    # source, or verification. No verifier runs — there is nothing to ground or
    # audit. Anything uncertain falls through to the buffered loop below.
    if (config.STREAM_SIMPLE_CHAT and not is_user_model and not had_images
            and not user_source
            and not await _request_needs_work(messages, session=session)):
        streamed = []
        async for kind, tok in fireworks.stream(
            scratch, config.AGENT_MODEL,
            max_tokens=config.AGENT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE, session=session,
            label="chat",
        ):
            if kind == "content":
                streamed.append(tok)
            yield (kind, tok)
        answer = "".join(streamed).strip()
        if answer and chat_id:
            user_memory = _consolidated_user_memory(messages)
            if user_memory:
                _track_task(asyncio.create_task(_memory_store(chat_id, "user", user_memory, session)))
            _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", answer, session)))
        return

    # Heavy turn: show an instant status line so the user sees activity while the first
    # heavy generation runs. A deterministic line — not a model call: spending a flash
    # generation to emit filler added latency and cost and cluttered the thinking panel
    # with non-model text. (feat/progress-ux will make these stages user-visible.)
    if config.SHOW_WORK and config.STREAM_PREAMBLE:
        yield ("reasoning", "🧭 Planning the response…\n")

    tool_sources = []
    export_links = []        # immediate exports (csv) — link collected as they run
    pending_exports = []     # deferred prose exports (docx/pdf/md) — built from the final answer
    repair_steps = 0
    tool_call_count = 0
    web_search_count = 0
    budget_note_added = False
    edit_nudged = False
    polish_voice = None
    polish_voice_pass = None

    for _ in range(config.AGENT_MAX_STEPS):
        source = _combined_source(user_source, tool_sources)
        model = _select_model(bool(source)) if not is_user_model else config.AGENT_MODEL
        tools = _budgeted_tools(tool_call_count, web_search_count)

        if tools is None and not budget_note_added:
            budget_note_added = True
            scratch.append(
                {
                    "role": "system",
                    "content": (
                        "Internal harness note: tool budget reached. Produce the "
                        "best final answer from gathered evidence now. If evidence "
                        "is insufficient, say what cannot be verified."
                    ),
                }
            )

        # The open model always drives generation (it owns tool-calling + reasoning).
        # Prose polish happens AFTER a final answer is produced (see below), so it
        # applies whether or not tools were still on the table — pure writing tasks
        # finish on turn 1 with tools still offered and must still get polished.
        step_temp = config.TOOL_TEMPERATURE if tools is not None else config.WRITER_TEMPERATURE
        # Stream the open model's output live when this generation will BE the final
        # answer — not a draft about to be polished, and not a user-model regen
        # (those produce their own final text below). Streamed optimistically; a tool
        # step carries little prose and the loop just continues.
        stream_live = config.STREAM_ANSWER and not polish_voice and not is_user_model
        parts, message, tool_calls = [], {}, []
        async for kind, data in fireworks.stream_chat(
            scratch, model, max_tokens=config.AGENT_MAX_TOKENS, temperature=step_temp,
            session=session, tools=tools, tool_choice="auto" if tools is not None else None,
            label="agent",
        ):
            if kind == "reasoning":
                if config.SHOW_WORK:
                    yield ("reasoning", data)
            elif kind == "content":
                parts.append(data)
                if stream_live and not pending_exports:
                    yield ("content", data)
            elif kind == "final":
                message = {"role": "assistant", "content": data["content"],
                           "tool_calls": data["tool_calls"]}
                tool_calls = data["tool_calls"]

        if tool_calls:
            scratch.append(_clean_assistant_tool_message(message))
            executable = []  # search/fetch/etc. — collected here, then run concurrently
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                args = _json_args(fn.get("arguments") or "{}")
                if name in ("export_docx", "export_pdf", "export_markdown"):
                    # Defer prose exports to the FINAL answer (built after polish +
                    # verify below) so the downloaded file matches the polished chat
                    # answer rather than this rough mid-draft. Skip only an EXACT
                    # duplicate (same format + filename + content — a model
                    # double-calling export); genuinely distinct files in one turn
                    # (a resume AND a cover letter, several docx) are all kept.
                    exp = {
                        "tool": name,
                        "markdown": str(args.get("markdown") or ""),
                        "filename": args.get("filename") or "document",
                        "title": args.get("title") or "",
                    }
                    if exp not in pending_exports:
                        pending_exports.append(exp)
                    fmt = name.replace("export_", "")
                    scratch.append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or name,
                        "name": name,
                        "content": f"Acknowledged: the {fmt} file will be exported from your final, verified answer and delivered to the user. Do not call export again.",
                    })
                    continue
                tool_call_count += 1
                if name == "web_search":
                    web_search_count += 1
                if config.SHOW_WORK:
                    yield ("reasoning", _tool_status(name, args) + "\n")
                executable.append((call, name, args))

            # Run the batch CONCURRENTLY: when the model fires several searches/fetches
            # in one step, every result returns in a SINGLE round instead of N
            # sequential trips. Guard + execute run together, per call.
            async def _run_tool(call, name, args):
                allowed, reason = await _tool_allowed(name, args, messages, source, session=session)
                if allowed:
                    raw = await _execute_tool(name, args, session=session, headers=request_headers)
                else:
                    raw = {
                        "rejected": True,
                        "tool": name,
                        "reason": reason or "tool call was not necessary for this request",
                        "instruction": "Answer the user's actual question directly without this tool.",
                    }
                return call, name, raw

            for call, name, raw_result in await asyncio.gather(
                *[_run_tool(c, n, a) for c, n, a in executable]
            ):
                source_text = _source_from_tool(name, raw_result)
                if source_text:
                    tool_sources.append(source_text)
                dl = _export_download(name, raw_result)
                if dl and dl not in export_links:
                    export_links.append(dl)
                visible = _compact_json(_visible_tool_result(name, raw_result))
                # External/source-bearing tool output (web, fetched pages, citations)
                # is untrusted: wrap it so embedded instructions are treated as data.
                if source_text:
                    visible = prompt_security.wrap_untrusted(name, visible)
                scratch.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "name": name,
                    "content": visible,
                })
            continue

        candidate = (message.get("content") or "").strip()
        if not candidate:
            scratch.append(
                {
                    "role": "system",
                    "content": "Internal harness note: produce a final answer or call a tool.",
                }
            )
            continue

        # REVISION invariant, harness-enforced: an edit of a delivered file MUST re-export
        # the revised document. The live smoke run proved hope is not an invariant — the
        # model acknowledged the edit in 174 chars and never called the export tool, so no
        # new file existed. One bounded nudge; if it still refuses, the turn proceeds.
        if edit_baseline and not pending_exports and not export_links and not edit_nudged:
            edit_nudged = True
            scratch.append({"role": "assistant", "content": candidate})
            scratch.append({
                "role": "system",
                "content": (
                    "Internal harness note: this is a REVISION of a document you delivered "
                    "as a file. You have not re-exported it. Call the same export tool now "
                    "with the COMPLETE revised document (the prior document with only the "
                    "requested change applied) and the same filename."
                ),
            })
            continue

        # A file export carries the deliverable, so its body is kept out of the chat
        # (shown as a download + a what-changed note instead) — meaning nothing was
        # streamed live for it.
        streamed_live = stream_live and not pending_exports
        is_clar = _is_clarification(candidate)

        # User-chosen model: regenerate final answer with their model
        # Runs regardless of tool use — the user picks the model, they get it.
        if is_user_model and not is_clar:
            if config.SHOW_WORK:
                yield ("reasoning", f"✨ Writing with {user_final_model.split('/')[-1]}…\n")
            regen = await _regenerate_with_user_model(
                scratch, user_final_model, source, session
            )
            if regen:
                candidate = regen
            # if regen is empty (API failure or non-Fireworks model), keep agent draft
        elif is_user_model and is_clar:
            pass  # keep clarification as-is

        # If the model produced a prose FILE, the export ARGUMENT is the real
        # deliverable — the model often writes the document in the export call and only
        # a summary as its chat message. Make THAT the thing we polish, verify, and
        # file, so the file carries the verified document (not the summary) and the
        # chat is just a note. This also routes the document through the honesty gate.
        if not is_user_model and not is_clar:
            doc = _pending_prose_deliverable(pending_exports)
            if doc and len(doc) >= config.POLISH_MIN_CHARS:
                candidate = doc
                # Auto-polish any NEW exported document — but never a surgical EDIT of an
                # already-delivered one: v1 was already polished and voiced, and re-running
                # a full polish + voice pass on the whole doc both wastes ~40s and rewrites
                # text the user did not ask to change.
                if not edit_baseline:
                    polish_voice = polish_voice or config.AUTO_POLISH_MODEL
                    # ...and give important documents a voice pass at the register that
                    # fits them (cover letter -> formal, email -> warm, code -> none),
                    # decided from the document rather than always-on or hand-picked.
                    if polish_voice_pass is None:
                        polish_voice_pass = await _classify_voice_register(
                            _all_user_text(messages), candidate, session=session)

        # Polish runs on a substantial deliverable (auto-set above for exports), not a
        # clarifying question and not a user-chosen model.
        substantial = len(candidate) >= config.POLISH_MIN_CHARS
        prose = None
        if polish_voice and not is_clar and substantial and not is_user_model:
            prose = _prose_provider(polish_voice)
        if prose is not None:
            prose_client, prose_model = prose
            if config.SHOW_WORK:
                yield ("reasoning", "✨ Polishing…\n")
            # messages_for_verify (= the kept tail in overflow) bounds the premium
            # polish prompt. Stream the polished deliverable live — but if a file will
            # carry it, stream into the thinking panel (progress) instead of the chat,
            # so the document lands in the file, not duplicated in the conversation.
            pmsgs = _prose_polish_messages(messages_for_verify, candidate, source)
            to_chat = not pending_exports
            try:
                if config.STREAM_ANSWER:
                    pparts = []
                    async for k, t in prose_client.stream(
                        pmsgs, prose_model, max_tokens=config.AGENT_MAX_TOKENS,
                        temperature=config.WRITER_TEMPERATURE, session=session,
                        label="polish",
                    ):
                        if k == "content":
                            pparts.append(t)
                            yield ("content" if to_chat else "reasoning", t)
                    if "".join(pparts).strip():
                        candidate = "".join(pparts).strip()
                        streamed_live = to_chat
                else:
                    polish = await prose_client.complete(
                        pmsgs, prose_model, max_tokens=config.AGENT_MAX_TOKENS,
                        temperature=config.WRITER_TEMPERATURE, session=session,
                        label="polish",
                    )
                    if polish and polish.strip():
                        candidate = polish.strip()
            except Exception as e:
                log.warning(f"[prose_polish] {prose_model} failed, keeping open-model draft: {e}")
        # Stage 2 — optional voice-only register pass (sonnet); facts untouched.
        # A SECOND premium call, so reserve it for genuinely long-form prose.
        if (not streamed_live and polish_voice_pass and polish_voice_pass != "none"
                and not is_clar and len(candidate) >= config.POLISH_VOICE_MIN_CHARS):
            if config.SHOW_WORK:
                yield ("reasoning", f"✨ Voice pass ({polish_voice_pass})…\n")
            candidate = await _voice_pass(candidate, polish_voice_pass, session=session)

        # Don't emit status as reasoning once content has streamed live — OWUI renders a
        # post-content reasoning token as a stray <details> block visible IN the answer.
        if config.SHOW_WORK and not streamed_live:
            yield ("reasoning", "✍️ Verifying the answer…\n")
        status, text = await _verified_or_blocked(
            messages_for_verify,
            candidate,
            source,
            recall_context=recall_context,
            prose=prose,
            force=bool(pending_exports),  # an exported file is always a deliverable -> verify
            session=session,
        )
        links = ("\n\n" + "\n".join(f"📎 [Download {fn}]({url})" for fn, url in export_links)) if export_links else ""

        if status == "ok":
            # Build the deferred prose exports from the FINAL, verified text.
            file_links, filed_deliverable = await _export_final(
                pending_exports, text, prose, messages_for_verify, source,
                chat_id=chat_id, headers=request_headers, session=session,
            )
            links += file_links
            changed = text.strip() != candidate.strip()
            summary = await _summarize_correction(candidate, text, session=session) if (changed and filed_deliverable) else ""

            if filed_deliverable:
                # The file carries the deliverable. The chat never repeats its body —
                # just a what-changed note (if the verifier touched it) + the download.
                if changed:
                    yield ("content", "\n\n---\n\n*Corrected before saving the file:*\n\n"
                           + (summary or "- Tightened a few details to match your source.") + links)
                else:
                    yield ("content", ("\n\n" if streamed_live else "") + "📄 Your file is ready — download below." + links)
                final_text = text
            elif streamed_live:
                # Chat IS the deliverable and was already shown live. Add only what's new:
                # a what-changed note + corrected text if it was touched, else the links.
                if changed:
                    final_text = text + links
                    yield ("content", "\n\n---\n\n*On review I corrected a few unsupported details:*\n\n"
                           + (summary or "- Tightened to match your source.")
                           + "\n\n*Corrected version:*\n\n" + final_text)
                else:
                    final_text = candidate + links
                    if links:
                        yield ("content", links)
            else:
                final_text = text + links
                yield ("content", final_text)
            # Store in memory asynchronously (don't block the response). Hold a
            # strong reference (_track_task): the event loop keeps only a weak ref
            # to a bare create_task, so an orphan store could be GC'd mid-flight.
            if chat_id:
                user_memory = _consolidated_user_memory(messages)
                if user_memory:
                    _track_task(asyncio.create_task(_memory_store(chat_id, "user", user_memory, session)))
                _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", final_text, session)))
            return

        # Blocked. If we already streamed the answer, admit it openly and correct;
        # otherwise repair (re-generate) before showing anything.
        if streamed_live:
            yield ("content", "\n\n---\n\n⚠️ " + text)
            return

        if repair_steps < config.GROUNDING_REPAIR_STEPS:
            repair_steps += 1
            scratch.append(
                {
                    "role": "system",
                    "content": (
                        "Internal verification gate blocked the previous unshown "
                        f"draft: {text}\nUse tools to gather evidence or revise "
                        "so the final answer contains only verified claims. If "
                        "the issue was citation provenance, remove citations unless "
                        "they came from actual provided or retrieved sources. Do not "
                        "show the blocked draft."
                    ),
                }
            )
            continue

        yield ("content", text)
        return

    yield (
        "content",
        "I could not complete a verified answer within the configured tool budget. "
        "I am not going to present an unverified factual answer as final.",
    )


def _build_regeneration_context(scratch: list[dict], source: str) -> list[dict]:
    tool_outputs = []
    for m in scratch:
        if m.get("role") == "tool":
            name = m.get("name", "tool")
            content = (m.get("content") or "")[:2000]
            tool_outputs.append(f"[{name}]: {content}")
    tool_text = "\n\n".join(tool_outputs) if tool_outputs else ""

    if tool_text:
        system_msg = (
            "You have gathered information using tools. Answer the user's original "
            "question using ONLY the evidence below. Cite source URLs where available. "
            "If evidence is insufficient, say what cannot be verified. Be direct and "
            "natural — no preamble, no sign-off."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Tool results:\n\n{tool_text}\n\nOriginal question:\n{source[:4000]}"},
        ]
    else:
        # No tools used — pass the original question directly
        user_texts = []
        for m in scratch:
            if m.get("role") == "user":
                user_texts.append(_text_of(m.get("content")))
        last_user = user_texts[-1] if user_texts else source[:2000]
        return [
            {"role": "user", "content": last_user.strip()},
        ]


async def _regenerate_with_user_model(
    scratch: list[dict], user_model: str, source: str, session
) -> str:
    if not user_model.startswith("accounts/fireworks/"):
        log.warning(f"user_model={user_model} is not a Fireworks model — keeping agent draft")
        return ""
    messages = _build_regeneration_context(scratch, source)
    result = await fireworks.chat(
        messages,
        user_model,
        max_tokens=config.AGENT_MAX_TOKENS,
        temperature=config.WRITER_TEMPERATURE,
        session=session,
        tools=None,
        tool_choice=None,
    )
    return (result.get("message") or {}).get("content", "") or ""
