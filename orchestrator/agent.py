"""Model-driven agent loop for the orchestrator."""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from typing import Any, AsyncGenerator, Optional

from . import config, fireworks, gemini, openai_client, anthropic_client, prompt_security, search, style, toolserver
from .owui import (
    _text_of, _unwrap_owui, _last_user_text, _has_images, _split_content_parts,
    _same_message_source, _SOURCE_BLOCK_RE, _owui_source_blocks, _user_source, _all_user_text,
)
from .memory_client import (
    _memory_recall, _memory_store, _deliverable_store, _deliverable_get, _last_active,
    _plan_store, _plan_get, _plan_clear,
)
from .timectx import _now_line, _gap_note
from .verifier import _verified_or_blocked, _summarize_correction, _WORD_RE, _has_citation_markers, _fit_audit_source
from .prompts import (
    TOOL_SCHEMAS, SYSTEM_AGENT, SYSTEM_VISION, SYSTEM_GATE, SYSTEM_REQUEST_GATE, SYSTEM_EDIT_INTENT, SYSTEM_EDIT_PATCH,
    SYSTEM_FACT_AUDIT, SYSTEM_TOOL_GUARD, SYSTEM_CHANGE_SUMMARY, SYSTEM_VOICE_REGISTER,
    SYSTEM_LONGDOC_GATE, SYSTEM_OUTLINE, SYSTEM_PLAN_INTENT, SYSTEM_SECTION_WRITER,
    _PROSE_POLISH_SYS, _VOICE_REGISTER, _VOICE_PASS_SYS,
)

log = logging.getLogger(__name__)

# A tool call leaked into the CONTENT channel as text (DeepSeek DSML / <tool_calls>
# markup, incl. the fullwidth-pipe variant) — it never executed; detect and recover.
_TEXTUAL_TOOL_CALL_RE = re.compile(r"<\s*[｜|]?\s*(?:tool_calls?|invoke|DSML)\b", re.I)
_TEXTUAL_TOOL_BLOCK_RE = re.compile(
    r"<\s*[｜|]?\s*(?:tool_calls?|invoke|DSML)\b.*?(?:</\s*[｜|]?\s*(?:tool_calls?|invoke|DSML)\s*>|$)",
    re.I | re.S)


def _split_vision_output(text: str):
    """Split vision emission into (evidence_transcript, reading) for audit-grade grounding. Falls back to whole text when the two-part structure is absent."""
    t = (text or "").strip()
    if not t:
        return "", ""
    parts = re.split(r"(?im)^\s*#+\s*reading\b.*$", t, maxsplit=1)
    if len(parts) == 2:
        transcript = re.sub(r"(?im)^\s*#+\s*evidence transcript\b.*$", "", parts[0]).strip()
        reading = parts[1].strip()
        if transcript:
            return transcript, reading
    return t, t  # no clean split -> the whole emission serves as both source and context


# Per-image vision cache: the EVIDENCE TRANSCRIPT is a question-independent, literal read of
# the pixels, so once a given image is read we never pay the 15-50s M3 call again — OWUI
# re-sends the whole conversation every turn, so a multi-turn chat about one image used to
# re-read it on EVERY follow-up. Bounded; keyed by image-content hash.
_VISION_CACHE: dict = {}
_VISION_CACHE_MAX = int(getattr(config, "VISION_CACHE_MAX", 0) or 128)


def _image_hash(image_parts, user_id: str = "") -> str:
    # Scope the key by user so two different users who upload the byte-identical image don't
    # share one cached grounding source (the value is also request-shaped via the reading).
    h = hashlib.sha256()
    h.update((user_id or "").encode("utf-8", "ignore") + b"\x00")
    for p in image_parts:
        h.update(((p.get("image_url") or {}).get("url") or "").encode("utf-8", "ignore"))
    return h.hexdigest()


async def _describe_images_for_agent(messages, *, user_id="", session=None):
    """Replace each image with native-vision text (EVIDENCE TRANSCRIPT + READING). Returns (messages, transcript) for audit routing. Images read concurrently and cached."""
    async def _describe(m):
        content = m.get("content")
        text_parts, image_parts = _split_content_parts(content)
        if not image_parts:
            return dict(m), ""
        user_text = "\n".join(t.strip() for t in text_parts if t.strip())
        ihash = _image_hash(image_parts, user_id)
        cached = _VISION_CACHE.get(ihash)
        if cached:
            # Reuse the cached read (the question-independent transcript drives this; the text
            # reasoner answers THIS turn's question from it). Skips the expensive re-read.
            _VISION_CACHE[ihash] = _VISION_CACHE.pop(ihash)  # LRU promote: a hot image isn't evicted
            new_m = dict(m)
            new_m["content"] = (((user_text + "\n\n") if user_text else "")
                                + "[What you see in the attached image:]\n" + cached)
            return new_m, cached
        prompt = (
            "The user attached image(s) with this message. Read them natively and follow your "
            "two-part contract — EVIDENCE TRANSCRIPT, then READING.\n\n"
            f"USER REQUEST:\n{user_text or '(none)'}"
        )
        detail = (config.VISION_IMAGE_DETAIL or "").strip().lower()

        def _content(use_detail: bool):
            # High image detail makes the provider tile a large/dense image at full resolution
            # instead of downscaling it into a blur the model confabulates from (A/B-proven).
            parts = []
            for p in image_parts:
                if use_detail and detail and detail != "auto" and isinstance(p.get("image_url"), dict):
                    p = {**p, "image_url": {**p["image_url"], "detail": detail}}
                parts.append(p)
            return [{"type": "text", "text": prompt}] + parts

        out = ""
        readers = [m for m in (config.VISION_MODEL, config.VISION_FALLBACK_MODEL) if m]
        for i, model in enumerate(readers):
            # Primary (M3) gets high-detail tiling for faithfulness; the FALLBACK runs WITHOUT
            # it — a lighter, faster degrade that actually returns when the primary stalled on a
            # huge tiled image (both stalling = the image getting silently dropped, the bug).
            try:
                _read = fireworks.complete(
                    [{"role": "system", "content": SYSTEM_VISION},
                     {"role": "user", "content": _content(use_detail=(i == 0))}],
                    model, max_tokens=config.VISION_MAX_TOKENS, temperature=0.0,
                    session=session, label="vision",
                    reasoning_effort=config.VISION_REASONING_EFFORT)
                # Deadline the PRIMARY so a stall hands off to the light fallback promptly.
                if i == 0 and len(readers) > 1 and config.VISION_PRIMARY_TIMEOUT > 0:
                    _read = asyncio.wait_for(_read, timeout=config.VISION_PRIMARY_TIMEOUT)
                out = (await _read).strip()
                if out:
                    break
            except Exception as e:
                log.warning(f"[vision] {model.split('/')[-1]} read failed: {type(e).__name__}: {e}")
        new_m = dict(m)
        if out:
            # Cache + GROUND against the full output (transcript + cited READING), not the
            # literal transcript alone: a legitimate visual identification the reading makes
            # (a breed/landmark/object, cited to a region) is then IN the grounding source, so
            # the forced image audit doesn't flag it as an 'unsupported' world-fact. The
            # transcript is question-independent (durable), so the cache stays valid per image.
            if len(_VISION_CACHE) >= _VISION_CACHE_MAX:
                _VISION_CACHE.pop(next(iter(_VISION_CACHE)), None)  # evict oldest (LRU; hits promote)
            _VISION_CACHE[ihash] = out
            # Frame as the assistant's OWN sight of the image (not user-pasted text), so the
            # reasoner doesn't disclaim "based on the transcription you gave me".
            new_m["content"] = (((user_text + "\n\n") if user_text else "")
                                + "[What you see in the attached image:]\n" + out)
        else:
            # NEVER silently drop the image — without this the agent sees only the question and
            # claims "no image was attached". Keep an explicit signal so it owns the failure.
            new_m["content"] = (((user_text + "\n\n") if user_text else "")
                                + "[An image WAS attached, but the vision reader could not "
                                "process it this time. Tell the user you had trouble reading "
                                "the image and ask them to re-send it — do NOT say no image was "
                                "attached.]")
        return new_m, out   # "" on a failed read -> no grounding source, no forced audit

    pairs = list(await asyncio.gather(*(_describe(m) for m in messages)))
    msgs = [p[0] for p in pairs]
    transcript = "\n\n".join(p[1] for p in pairs if p[1].strip())
    return msgs, transcript


def _with_system(messages, system_text):
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            base = _text_of(m.get("content"))
            m["content"] = (base + "\n\n" + system_text).strip() if base else system_text
            return out
    return [{"role": "system", "content": system_text}] + out


async def _classify_edit_once(payload, *, session=None) -> dict:
    # The PRO model judges this gate: flash-at-low-reasoning misread even "can you
    # update the doc?" as 'new' (live smoke, repeatedly, regardless of prompt wording).
    # The gate only fires in chats that already delivered a document — pennies, and a
    # wrong verdict here silently drops the user's document.
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_EDIT_INTENT},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=True)}],
            config.GROUNDED_MODEL, max_tokens=600, temperature=0.0,
            session=session, label="gate:edit",
        )
        data = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
        a = str(data.get("action", "new")).lower()
        if a in ("rename", "reformat", "edit"):
            return {"action": a, "filename": (data.get("filename") or "").strip(),
                    "format": (data.get("format") or "").strip().lower()}
    except Exception:
        pass
    return {"action": "new", "filename": "", "format": ""}


async def _classify_edit(last_user: str, prior: dict, *, messages=None, session=None) -> dict:
    """Classify a follow-up against the last delivered document: rename|reformat|edit|new.

    The two misroute directions are NOT symmetric: new-misread-as-edit is self-correcting,
    while edit-misread-as-new silently drops the document — so a 'new' verdict must win TWICE."""
    last_user = (last_user or "").strip()
    if not (last_user and prior and prior.get("content")):
        return {"action": "new"}
    recent = ""
    if messages:
        turns = [f"[{m.get('role')}]: {_unwrap_owui(_text_of(m.get('content')))[:200]}"
                 for m in messages[:-1] if m.get("role") in ("user", "assistant")]
        recent = "\n".join(turns[-4:])
    payload = {
        "recent_conversation": recent,
        "latest_user": last_user[:1500],
        "current_filename": prior.get("filename") or "document",
        "current_format": prior.get("fmt") or "docx",
    }
    result = await _classify_edit_once(payload, session=session)
    if result["action"] == "new":
        second = await _classify_edit_once(payload, session=session)
        if second["action"] != "new":
            result = second
    log.info(f"[edit-intent] action={result['action']} msg={last_user[:80]!r}")
    return result


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
        _track_task(asyncio.create_task(_deliverable_store(chat_id, content, filename or fn, fmt)))
    return f"\n\n📎 [Download {fn}]({url})"


def _edit_inject(prior: dict) -> str:
    """System directive for a surgical content edit of the prior document."""
    return (
        "REVISION TASK — the user is revising a document you already delivered in this "
        "chat. Here is that document, verbatim:\n\n"
        "--- CURRENT DOCUMENT ---\n" + (prior.get("content") or "").strip()
        + "\n--- END CURRENT DOCUMENT ---\n\n"
        "Make the change the user asks for — their INTENT, not a literal find-and-replace "
        "of their words. A 'fix this line' touches one line; a 'review and make X "
        "consistent' means reading the whole document and rewording every passage the "
        "intent genuinely covers, with judgment. Leave everything the request does not "
        "cover exactly as it is — never rewrite, re-order, or 'improve' beyond the ask. "
        "If the instruction is ambiguous or you are unsure what they want, output ONLY a "
        "short clarifying question (no document) — the user explicitly prefers being "
        "asked over being guessed at, and their answer comes straight back to you."
    )


async def _try_patch_edit(baseline: str, instruction: str, *, session=None):
    """Apply targeted find→replace edits to the stored document. Returns patched text, or None when the change is too broad — caller falls back to a full re-emit."""
    if not baseline.strip():
        return None
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_EDIT_PATCH + "\n\n--- DOCUMENT ---\n" + baseline},
             {"role": "user", "content": instruction}],
            config.GROUNDED_MODEL, max_tokens=config.DRAFT_MAX_TOKENS, temperature=0.0,
            session=session, label="edit:patch")
        data = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
    except Exception:
        return None
    edits = data.get("edits")
    if data.get("broad") or not isinstance(edits, list) or not edits:
        return None
    text = baseline
    for e in edits:
        if not isinstance(e, dict):
            return None
        find = e.get("find") or ""
        repl = "" if e.get("replace") is None else str(e.get("replace"))
        if not find or text.count(find) != 1:   # must match exactly once, else fall back
            return None
        text = text.replace(find, repl, 1)
    return text if (text.strip() and text != baseline) else None


# Strong references to fire-and-forget background writes. asyncio keeps only a
# WEAK reference to a running task, so a bare create_task() can be garbage
# collected mid-flight once the request returns — silently dropping the write.
_BG_TASKS: set = set()


def _track_task(task):
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


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


def _clean_assistant_tool_message(message: dict) -> dict:
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
    }


def _select_model(has_sources: bool) -> str:
    return config.GROUNDED_MODEL if has_sources else config.AGENT_MODEL


def _tool_status(name: str, args: dict) -> str:
    """Human-readable action line for a tool call, streamed to the UI as reasoning."""
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
    """Markdown of the largest pending prose export — the model writes the actual document in the export argument, not the chat message."""
    docs = [
        str(e.get("markdown") or "")
        for e in pending
        if e.get("tool") in ("export_docx", "export_pdf", "export_markdown")
    ]
    return max(docs, key=len) if docs else ""


async def _export_final(pending, final_text, prose, messages, source, *, chat_id="", headers=None, session=None):
    """Build deferred export files from the verified draft or polished export argument. Returns (links_str, filed_deliverable)."""
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
    # The polisher gets today's date too — without it, gpt-5.5 letterheads a formal
    # document with a "[Date]" placeholder (it can't know the date, so it blanks it).
    return [
        {"role": "system", "content": _PROSE_POLISH_SYS + "\n\n" + _now_line()},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


async def _classify_voice_register(request, candidate, *, session=None) -> str:
    """Pick the voice-pass register (warm/formal/none) for an exported deliverable from the document itself."""
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


def _norm_turn(content) -> str:
    """Collapse whitespace so a recalled turn (now stored verbatim) can be deduped
    against the same turn still verbatim in the kept tail."""
    return " ".join((content or "").split())


def _split_recent_history(messages, budget_chars: int):
    """Split a long history into (recent_tail, older_head) for overflow recall. Always keeps at least the final message."""
    keep = max(1, int(budget_chars * 0.7))
    total, cut = 0, 0
    for i in range(len(messages) - 1, -1, -1):
        total += len(_text_of(messages[i].get("content")))
        if total > keep and i < len(messages) - 1:
            cut = i + 1
            break
    return messages[cut:], messages[:cut]


# ── Chunked section-writer ───────────────────────────────────────────────────────
# A long, multi-section document is OUTLINED -> approved -> written section-by-section ->
# assembled -> verified -> exported, instead of emitted in one capped shot. The outline is
# held as a pending PLAN (kv, per chat) between the propose-turn and the build-turn.
def _slug(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (title or "document")).strip("_").lower()
    return (s or "document")[:60]


_LONGDOC_CUES = (
    "paper", "thesis", "dissertation", "report", "essay", "review", "chapter", "white paper",
    "whitepaper", "case study", "proposal", "study guide", "manuscript", "literature review",
    "section", "comprehensive", "in-depth", "in depth", "detailed", "multi-part",
)


def _maybe_longdoc(messages) -> bool:
    """Cheap deterministic PREFILTER before the (flash) long-doc classifier: only spend the
    gate on plausibly-long requests. A short ask with no doc cues is never a long document,
    so skip the model call entirely (most heavy turns — short edits, summaries — land here)."""
    t = _last_user_text(messages).strip().lower()
    return len(t) > 140 or any(c in t for c in _LONGDOC_CUES)


async def _classify_longdoc(messages, *, session=None) -> dict:
    """Is this a request to WRITE a long, multi-section document (-> outline-first chunked
    writer)? One cheap flash gate; uncertain / parse failure -> not a long doc (normal flow)."""
    q = _last_user_text(messages).strip()[:2000]
    if not q:
        return {"longdoc": False, "doc_type": ""}
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_LONGDOC_GATE},
             {"role": "user", "content": q}],
            config.GROUNDING_GATE_MODEL, max_tokens=60, temperature=0.0, session=session,
            label="gate:longdoc",
        )
        m = re.search(r"\{.*\}", raw, flags=re.S)
        d = json.loads(m.group(0) if m else raw)
        return {"longdoc": bool(d.get("longdoc", False)), "doc_type": str(d.get("doc_type") or "").strip()}
    except Exception:
        return {"longdoc": False, "doc_type": ""}


def _outline_for_prompt(plan: dict) -> str:
    """The current sections, compactly, to hand back for an in-place revision."""
    return "\n".join(f"{i}. {s.get('heading','')}: {s.get('intent','')}"
                     for i, s in enumerate(plan.get("sections") or [], 1))


async def _generate_outline(request: str, source: str, *, current_outline: str = "",
                            change: str = "", session=None):
    """Plan a long document as {title, sections:[{heading,intent}]}. Returns plan or None."""
    user = f"USER REQUEST:\n{(request or '').strip()[:6000]}"
    if (current_outline or "").strip():
        user += "\n\nCURRENT OUTLINE (apply the requested change to THIS, keep the rest):\n" + current_outline
    if (change or "").strip():
        user += "\n\nREQUESTED CHANGE: " + change.strip()[:1000]
    if (source or "").strip():
        user += "\n\nSOURCE MATERIAL:\n" + source[:20000]
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_OUTLINE},
             {"role": "user", "content": user}],
            config.GROUNDED_MODEL, max_tokens=config.OUTLINE_MAX_TOKENS,
            temperature=0.0, session=session, label="outline",
        )
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0) if m else raw)
        title = str(data.get("title") or "").strip()
        sections = []
        for s in (data.get("sections") or [])[:config.CHUNKED_MAX_SECTIONS]:
            heading = str((s or {}).get("heading") or "").strip()
            intent = str((s or {}).get("intent") or "").strip()
            if heading:
                sections.append({"heading": heading, "intent": intent})
        if title and sections:
            return {"title": title, "sections": sections}
    except Exception as e:
        log.warning(f"[outline] generation failed: {e}")
    return None


def _render_outline(plan: dict, *, revised: bool = False) -> str:
    """The outline shown to the user for approval before any prose is written."""
    title = plan.get("title") or "Document"
    head = (f"Here's the proposed structure for **{title}**:" if not revised
            else f"Updated outline for **{title}**:")
    lines = [head, ""]
    for i, s in enumerate(plan.get("sections") or [], 1):
        heading = s.get("heading") or f"Section {i}"
        intent = s.get("intent") or ""
        lines.append(f"{i}. **{heading}**" + (f" — {intent}" if intent else ""))
    lines += ["", "Want me to **write it**? Or tell me what to change "
              "(add / remove / reorder a section, adjust the scope)."]
    return "\n".join(lines)


async def _classify_plan_intent(latest_user: str, plan: dict, *, session=None) -> dict:
    """Classify the user's reply to a shown outline: approve | revise | abandon. On a parse
    failure default to a no-op 'revise' (re-show the outline) — never silently build or drop."""
    latest = (latest_user or "").strip()
    if not latest:
        return {"action": "revise", "revision": ""}
    outline_txt = "\n".join(f"{i}. {s.get('heading','')}: {s.get('intent','')}"
                            for i, s in enumerate(plan.get("sections") or [], 1))
    payload = {"title": plan.get("title", ""), "outline": outline_txt, "user_reply": latest[:1500]}
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_PLAN_INTENT},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=True)}],
            config.GROUNDED_MODEL, max_tokens=300, temperature=0.0, session=session,
            label="gate:plan",
        )
        m = re.search(r"\{.*\}", raw, flags=re.S)
        d = json.loads(m.group(0) if m else raw)
        action = str(d.get("action") or "").lower()
        if action in ("approve", "revise", "abandon"):
            return {"action": action, "revision": str(d.get("revision") or "").strip()}
    except Exception:
        pass
    return {"action": "revise", "revision": ""}


async def _write_section(title: str, sections: list, idx: int, prior_recap: str,
                         source: str, *, session=None) -> str:
    """Write one section of the long document, aware of the whole outline. Returns Markdown, or '' on failure."""
    sec = sections[idx]
    outline_txt = "\n".join(
        f"{i+1}. {s.get('heading','')}" + ("  <- WRITE THIS ONE" if i == idx else "")
        for i, s in enumerate(sections))
    parts = [
        f"DOCUMENT TITLE: {title}",
        f"FULL OUTLINE:\n{outline_txt}",
        f"SECTION TO WRITE NOW:\n{sec.get('heading','')} — {sec.get('intent','')}",
    ]
    if (prior_recap or "").strip():
        parts.append("PRECEDING SECTIONS ALREADY COVERED (continue from these, don't repeat):\n" + prior_recap)
    if (source or "").strip():
        # Relevance-FIT the source to THIS section (its heading+intent) instead of a blind
        # head-truncation — so a section sees the source material that's actually about it,
        # not just whatever happened to be in the first 24k chars (matters for a long paper
        # with a big source where the relevant bits are deep in the document).
        sec_source = _fit_audit_source(source, f"{sec.get('heading','')} {sec.get('intent','')}", 24000)
        parts.append("SOURCE MATERIAL (assert only what this supports; never fabricate):\n" + sec_source)
    try:
        return (await fireworks.complete(
            [{"role": "system", "content": SYSTEM_SECTION_WRITER},
             {"role": "user", "content": "\n\n".join(parts)}],
            config.GROUNDED_MODEL, max_tokens=config.DRAFT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE, session=session, label="section:write",
        )).strip()
    except Exception as e:
        log.warning(f"[section] write failed for {sec.get('heading','')!r}: {e}")
        return ""


async def _present_outline(request: str, source: str, *, chat_id: str, filename: str = "",
                           fmt: str = "docx", session=None, revised: bool = False,
                           current_outline: str = "", change: str = "", revise_count: int = 0):
    """Generate or revise an outline, persist it as the pending plan, and yield it for approval."""
    plan = await _generate_outline(request, source, current_outline=current_outline,
                                   change=change, session=session)
    if not plan:
        yield ("content", "I couldn't draft a clear outline for that — tell me a bit more "
               "about the document you want and I'll plan it.")
        return
    plan["source"] = source or ""
    plan["request"] = request or ""        # the ORIGINAL request, stable across revisions
    plan["filename"] = filename or _slug(plan.get("title") or "document")
    plan["fmt"] = fmt or "docx"
    plan["created_at"] = time.time()       # TTL anchor so a never-approved plan expires
    plan["revise_count"] = revise_count
    if chat_id:
        await _plan_store(chat_id, plan)   # awaited: the NEXT turn reads this
    yield ("content", _render_outline(plan, revised=revised))


async def _build_from_plan(plan: dict, messages, user_id: str, chat_id: str, headers, session=None):
    """Build the approved long document section-by-section with live progress and per-section verification. Async generator of (kind, text)."""
    title = plan.get("title") or "Document"
    sections = plan.get("sections") or []
    source = plan.get("source") or ""
    filename = plan.get("filename") or _slug(title)
    fmt = plan.get("fmt") or "docx"

    yield ("content", f"📝 Writing **{title}** — {len(sections)} sections, one at a time.\n\n")
    assembled, recap, prev_tail, failures = [], "", "", []
    for i, sec in enumerate(sections):
        heading = sec.get("heading") or f"Section {i+1}"
        yield ("content", f"✍️ §{i+1} {heading}…\n")
        # Hand the writer the prior headings AND a tail of the previous section's real prose,
        # so it can actually pick up the thread instead of restating it.
        prompt_recap = recap + (f"\n\nThe previous section ended:\n…{prev_tail}" if prev_tail else "")
        section_md = await _write_section(title, sections, i, prompt_recap, source, session=session)
        if not section_md:                      # one retry on an empty/failed generation
            section_md = await _write_section(title, sections, i, prompt_recap, source, session=session)
        if not section_md:
            failures.append((i + 1, heading, "couldn't be generated"))
            yield ("content", "   ⚠️ couldn't write this section\n")
            continue
        # A from-scratch section (no source) that cites sources we never had is a fabrication —
        # the deterministic backstop, made source-aware here (the global guard is masked by the
        # always-present date/user-text in grounding_source).
        if not source.strip() and _has_citation_markers(section_md):
            failures.append((i + 1, heading, "cited sources that weren't provided"))
            yield ("content", "   ⚠️ cited unprovided sources — held back\n")
            continue
        # Per-section honesty pass. Force the audit when there's source to ground against;
        # with no source, let the gate decide (a from-scratch essay isn't a grounding task).
        status, checked = await _verified_or_blocked(
            messages, section_md, source, force=bool(source.strip()), session=session)
        if status != "ok":
            failures.append((i + 1, heading, "made claims I couldn't verify against your sources"))
            yield ("content", "   ⚠️ unverifiable claims — held back\n")
            continue
        section_md = checked
        assembled.append(section_md)
        prev_tail = section_md[-300:]
        recap += f"- {heading}: {sec.get('intent','')}\n"
        yield ("content", f"   ✓ {len(_WORD_RE.findall(section_md)):,} words\n")

    if failures or not assembled:
        detail = ("\n".join(f"- §{n} {h} — {why}" for n, h, why in failures)
                  or "- the document came back empty")
        yield ("content",
               "\n\n⚠️ I held this back rather than ship something unverified:\n\n" + detail
               + "\n\nYour outline is saved — add a source for those sections (or tell me to write "
               "them more conservatively) and say **write it** to rebuild.")
        return  # plan kept on purpose

    full_doc = (f"# {title}\n\n" + "\n\n".join(assembled)).strip()
    link = await _repackage_deliverable(full_doc, filename, fmt,
                                        chat_id=chat_id, headers=headers, session=session)
    if chat_id and not link:
        # Export failed AFTER the doc verified — don't lose the verified bytes: store them as
        # the deliverable so a 'export it as docx' (reformat path) recovers the file, no rebuild.
        await _deliverable_store(chat_id, full_doc, filename, fmt)
    if chat_id:
        await _plan_clear(chat_id)
        _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", full_doc[:4000], session)))
    words = len(_WORD_RE.findall(full_doc))
    if link:
        yield ("content", f"\n\n📄 **{title}** is ready — {len(assembled)} sections, {words:,} words. "
               f"Download below.{link}")
    else:
        yield ("content", f"\n\n📄 **{title}** is written and saved ({words:,} words), but the file "
               "export failed — say **export it as docx** and I'll produce the file.")


# ── AgentState import (replaces scattered boolean flags) ────────────────
from .agent_state import AgentState


# ═══════════════════════════════════════════════════════════════════════════
# Phase helpers — regular async functions that return data, plus a few
# async generators for streaming phases (plan dispatch, plain chat, longdoc).
# ═══════════════════════════════════════════════════════════════════════════

async def _read_images(messages, user_id, session):
    """Transcribe attached images. Returns (messages, transcript)."""
    if not _has_images(messages):
        return messages, ""
    return await _describe_images_for_agent(messages, user_id=user_id, session=session)


async def _gather_context(user_id, chat_id, session):
    """Parallel fetch: style profile, prior deliverable, last-active, pending plan.
    Returns (profile, prior_deliverable, gap_note, pending_plan)."""
    profile, prior, active, plan = "", None, None, None
    tasks = [asyncio.ensure_future(style.get_style_profile(user_id))]
    if chat_id:
        tasks.append(asyncio.ensure_future(_deliverable_get(chat_id)))
        tasks.append(asyncio.ensure_future(_last_active(chat_id)))
        if config.ENABLE_CHUNKED_WRITER:
            tasks.append(asyncio.ensure_future(_plan_get(chat_id)))
    results = await asyncio.gather(*tasks)
    profile = results[0]
    idx = 1
    if chat_id:
        prior = results[idx]; idx += 1
        active = results[idx]; idx += 1
        if config.ENABLE_CHUNKED_WRITER:
            plan = results[idx]; idx += 1
    return profile, prior, _gap_note(active) if active else "", plan


async def _dispatch_plan(messages, plan, chat_id, req_headers, session):
    """Handle pending outline. Yields output if plan handled; caller detects via flag."""
    if not plan:
        return
    created = float(plan.get("created_at") or 0)
    age = (time.time() - created) if created else 0
    revises = int(plan.get("revise_count") or 0)
    if age > config.CHUNKED_PLAN_TTL_SECONDS or revises > config.CHUNKED_MAX_REVISES:
        await _plan_clear(chat_id)
        return
    intent = await _classify_plan_intent(_last_user_text(messages), plan, session=session)
    if intent["action"] == "approve":
        async for kt in _build_from_plan(plan, messages, "user", chat_id, req_headers, session):
            yield kt
        return
    if intent["action"] == "revise":
        revision = intent.get("revision") or ""
        if revision:
            async for kt in _present_outline(
                plan.get("request") or "", plan.get("source") or "",
                chat_id=chat_id, filename=plan.get("filename") or "",
                fmt=plan.get("fmt") or "docx", session=session, revised=True,
                current_outline=_outline_for_prompt(plan), change=revision,
                revise_count=revises + 1):
                yield kt
        else:
            yield ("content", _render_outline(plan)
                   + "\n\n*(Say \"write it\" to build, name a change, or \"never mind\" to drop it.)*")
        return
    await _plan_clear(chat_id)


async def _dispatch_edit(messages, prior, chat_id, req_headers, session, show_work):
    """Handle multi-turn edit. Returns (handled, output_or_directive, baseline).
    If handled: output_or_directive is the response text, baseline is "".
    If not handled: output_or_directive is the edit directive, baseline is the prior doc."""
    if not (prior and prior.get("content")):
        return False, "", ""

    intent = await _classify_edit(_last_user_text(messages), prior, messages=messages, session=session)

    if intent["action"] in ("rename", "reformat"):
        fmt = intent.get("format") or prior.get("fmt") or "docx"
        filename = intent.get("filename") or prior.get("filename") or "document"
        link = await _repackage_deliverable(prior["content"], filename, fmt,
                                            chat_id=chat_id, headers=req_headers, session=session)
        verb = "Renamed" if intent["action"] == "rename" else f"Re-exported as {fmt.upper()}"
        return True, (f"📄 {verb} — download below.{link}" if link
                       else "I couldn't re-export that file — want me to try again?"), ""

    if intent["action"] != "edit":
        return False, "", ""

    baseline = (prior.get("content") or "").strip()

    # 1. Targeted in-place patches
    revised = await _try_patch_edit(baseline, _last_user_text(messages), session=session)
    patched = revised is not None

    # 2. Full rewrite if patches aren't clean
    if not patched:
        revised = ""
        try:
            revised = (await fireworks.complete(
                [{"role": "system", "content": _now_line() + "\n\n" + _edit_inject(prior)
                  + "\n\nOutput ONLY the complete revised document — no commentary."},
                 {"role": "user", "content": _last_user_text(messages)}],
                config.GROUNDED_MODEL, max_tokens=config.DRAFT_MAX_TOKENS,
                temperature=config.WRITER_TEMPERATURE, session=session, label="edit:write")).strip()
        except Exception as e:
            log.warning(f"[edit] directed revision failed, falling to normal flow: {e}")
        if revised and not _same_doc(revised, baseline) and "?" in revised and len(revised) < 1200:
            return True, revised, ""

    # 3. Verify + re-export
    if revised and (patched or _same_doc(revised, baseline)):
        src = ((_user_source(messages) + "\n\n" + baseline).strip()
               if _user_source(messages).strip() else baseline)
        status, text = await _verified_or_blocked(messages, revised, src, force=True, session=session)
        if status != "ok":
            return True, text, ""
        link = await _repackage_deliverable(text, prior.get("filename") or "document",
                                            prior.get("fmt") or "docx",
                                            chat_id=chat_id, headers=req_headers, session=session)
        summary = await _summarize_correction(baseline, text, session=session)
        output = ("📄 Updated — download below." + (("\n\n" + summary) if summary else "") + link) if link \
                 else "I couldn't rebuild the file — want me to try again?"
        if chat_id:
            um = _consolidated_user_memory(messages)
            if um:
                _track_task(asyncio.create_task(_memory_store(chat_id, "user", um, session)))
            _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", text, session)))
        return True, output, ""
    # Fell through — inject prior doc as context for normal agent loop
    return False, _edit_inject(prior), baseline


def _build_source(messages, image_transcript, edit_baseline):
    """Assemble grounding source: user files/pastes + image transcript + prior doc."""
    src = _user_source(messages)
    if image_transcript:
        tag = "IMAGE EVIDENCE TRANSCRIPT:\n" + image_transcript
        src = (src + "\n\n" + tag).strip() if src.strip() else tag
    if edit_baseline:
        src = (src + "\n\n" + edit_baseline).strip() if src.strip() else edit_baseline
    return src


async def _build_system_prompt(messages, user_id, chat_id, profile, extra, session):
    """Build system prompt, handling context overflow with memory recall.
    Returns (scratch, recall_context, messages_for_verify)."""
    recall_ctx = ""
    msg_for_verify = messages
    history_chars = sum(len(_text_of(m.get("content"))) for m in messages)

    if not (chat_id and history_chars > config.MEMORY_CONTEXT_BUDGET_CHARS):
        return _initial_messages(messages, user_id, profile, extra_system=extra), "", messages

    recent, _ = _split_recent_history(messages, config.MEMORY_CONTEXT_BUDGET_CHARS)
    recent_norm = {_norm_turn(_text_of(m.get("content"))) for m in recent}
    recall_query = next(
        (_text_of(m.get("content")).strip() for m in reversed(messages) if m.get("role") == "user"), "",
    )[:2000]
    user_lines, asst_lines, seen = [], [], set()
    if recall_query.strip():
        for role, content in await _memory_recall(chat_id, recall_query, session):
            c = _norm_turn(content)[:500]
            if not c or c in recent_norm or c in seen:
                continue
            seen.add(c)
            (user_lines if role == "user" else asst_lines).append(c)
    if not (user_lines or asst_lines):
        return _initial_messages(messages, user_id, profile, extra_system=extra), "", messages

    scratch = _initial_messages(recent, user_id, profile, extra_system=extra)
    msg_for_verify = recent
    recall_ctx = "\n".join(user_lines)
    blocks = []
    if user_lines:
        blocks.append("Earlier in THIS conversation the user stated:\n" + recall_ctx)
    if asst_lines:
        blocks.append("Earlier assistant replies (for continuity only — NOT verified facts):\n"
                      + "\n".join(asst_lines))
    scratch.append({"role": "system", "content": "This is a long conversation; earlier turns were trimmed.\n\n"
                     + "\n\n".join(blocks)})
    return scratch, recall_ctx, msg_for_verify


async def _try_plain_chat(messages, scratch, user_source, chat_id, session, is_user_model, had_images):
    """Stream a simple answer if no tools/sources needed. Yields output if handled."""
    if not config.STREAM_SIMPLE_CHAT or is_user_model or had_images or user_source:
        return
    if await _request_needs_work(messages, session=session):
        return
    streamed = []
    async for kind, tok in fireworks.stream(
        scratch, config.AGENT_MODEL, max_tokens=config.AGENT_MAX_TOKENS,
        temperature=config.WRITER_TEMPERATURE, session=session, label="chat"):
        if kind == "content":
            streamed.append(tok)
        yield (kind, tok)
    answer = "".join(streamed).strip()
    if answer and chat_id:
        um = _consolidated_user_memory(messages)
        if um:
            _track_task(asyncio.create_task(_memory_store(chat_id, "user", um, session)))
        _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", answer, session)))


async def _try_longdoc(messages, user_source, chat_id, session, is_user_model, edit_baseline):
    """Propose outline for long-document requests. Yields output if handled."""
    if not (config.ENABLE_CHUNKED_WRITER and chat_id and not is_user_model and not edit_baseline):
        return
    if not _maybe_longdoc(messages):
        return
    ld = await _classify_longdoc(messages, session=session)
    if not ld.get("longdoc"):
        return
    async for kt in _present_outline(_all_user_text(messages), user_source,
                                      chat_id=chat_id, filename="", fmt="docx", session=session):
        yield kt


def _build_regeneration_context(scratch: list[dict], source: str) -> list[dict]:
    tool_outputs = []
    for m in scratch:
        if m.get("role") == "tool":
            tool_outputs.append(f"[{m.get('name', 'tool')}]: {(m.get('content') or '')[:2000]}")
    tool_text = "\n\n".join(tool_outputs)
    if tool_text:
        return [
            {"role": "system", "content": (
                "You have gathered information using tools. Answer the user's original "
                "question using ONLY the evidence below. Cite source URLs where available.")},
            {"role": "user", "content": f"Tool results:\n\n{tool_text}\n\nOriginal question:\n{source[:4000]}"},
        ]
    user_texts = [_text_of(m.get("content")) for m in scratch if m.get("role") == "user"]
    last_user = user_texts[-1] if user_texts else source[:2000]
    return [{"role": "user", "content": last_user.strip()}]


async def _regenerate_with_user_model(scratch, user_model, source, session) -> str:
    if not user_model.startswith("accounts/fireworks/"):
        return ""
    messages = _build_regeneration_context(scratch, source)
    result = await fireworks.chat(messages, user_model, max_tokens=config.AGENT_MAX_TOKENS,
                                  temperature=config.WRITER_TEMPERATURE, session=session)
    return (result.get("message") or {}).get("content", "") or ""


# ═══════════════════════════════════════════════════════════════════════════
# Agent loop — the heavy path, now driven by AgentState
# ═══════════════════════════════════════════════════════════════════════════

async def _agent_loop(
    messages: list[dict],
    scratch: list[dict],
    messages_for_verify: list[dict],
    user_source: str,
    image_transcript: str,
    recall_context: str,
    edit_baseline: str,
    is_user_model: bool,
    user_final_model: str,
    chat_id: str,
    request_headers: dict,
    session,
) -> AsyncGenerator[tuple[str, str], None]:
    """Model-driven tool loop: calls → execute → verify → deliver."""
    st = AgentState()
    export_links: list = []

    for _ in range(config.AGENT_MAX_STEPS):
        source = _combined_source(user_source, st.tool_sources)
        model = _select_model(bool(source)) if not is_user_model else config.AGENT_MODEL
        tools = _budgeted_tools(st.tool_call_count, st.web_search_count)

        if tools is None and not st.budget_note_added:
            st.budget_note_added = True
            scratch.append({"role": "system", "content": (
                "Internal harness note: tool budget reached. Produce the best final answer "
                "from gathered evidence. If evidence insufficient, say what cannot be verified.")})

        step_temp = config.TOOL_TEMPERATURE if tools is not None else config.WRITER_TEMPERATURE
        stream_live = config.STREAM_ANSWER and not st.polish_voice and not is_user_model
        parts, message, tool_calls = [], {}, []

        async for kind, data in fireworks.stream_chat(
            scratch, model, max_tokens=config.AGENT_MAX_TOKENS, temperature=step_temp,
            session=session, tools=tools, tool_choice="auto" if tools is not None else None,
            label="agent"):
            if kind == "reasoning":
                if config.SHOW_WORK:
                    yield ("reasoning", data)
            elif kind == "content":
                parts.append(data)
                if stream_live and not st.pending_exports:
                    yield ("content", data)
            elif kind == "final":
                message = {"role": "assistant", "content": data["content"],
                           "tool_calls": data["tool_calls"]}
                tool_calls = data["tool_calls"]

        # ── Tool execution ────────────────────────────────────────────
        if tool_calls:
            scratch.append(_clean_assistant_tool_message(message))
            executable = []
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                args = _json_args(fn.get("arguments") or "{}")
                if name in ("export_docx", "export_pdf", "export_markdown"):
                    exp = {"tool": name, "markdown": str(args.get("markdown") or ""),
                           "filename": args.get("filename") or "document",
                           "title": args.get("title") or ""}
                    if exp not in st.pending_exports:
                        st.pending_exports.append(exp)
                    scratch.append({"role": "tool", "tool_call_id": call.get("id") or name,
                                    "name": name, "content": (
                        f"Acknowledged: the {name.replace('export_', '')} file will be exported "
                        "from your final, verified answer. Do not call export again.")})
                    continue
                st.tool_call_count += 1
                if name == "web_search":
                    st.web_search_count += 1
                if config.SHOW_WORK:
                    yield ("reasoning", _tool_status(name, args) + "\n")
                executable.append((call, name, args))

            async def _run_one(call, name, args):
                allowed, reason = await _tool_allowed(name, args, messages, source, session=session)
                raw = await _execute_tool(name, args, session=session, headers=request_headers) if allowed else {
                    "rejected": True, "tool": name, "reason": reason or "not necessary",
                    "instruction": "Answer the user's actual question directly without this tool."}
                return call, name, raw

            for call, name, raw in await asyncio.gather(*[_run_one(c, n, a) for c, n, a in executable]):
                src_text = _source_from_tool(name, raw)
                if src_text:
                    st.tool_sources.append(src_text)
                dl = _export_download(name, raw)
                if dl and dl not in export_links:
                    export_links.append(dl)
                visible = _compact_json(_visible_tool_result(name, raw))
                if src_text:
                    visible = prompt_security.wrap_untrusted(name, visible)
                scratch.append({"role": "tool", "tool_call_id": call.get("id") or name,
                                "name": name, "content": visible})
            continue

        candidate = (message.get("content") or "").strip()

        # Handle textual tool-call leaks (DeepSeek DSML in content)
        if _TEXTUAL_TOOL_CALL_RE.search(candidate):
            log.warning("[dsml-leak] textual tool-call detected in content channel — nudging model")
            if not st.textual_tool_nudged:
                st.textual_tool_nudged = True
                scratch.append({"role": "assistant", "content": candidate})
                scratch.append({"role": "system", "content": (
                    "Internal harness note: you wrote a tool call as plain text — it did NOT "
                    "execute. Issue it through the function-calling interface, or answer directly.")})
                continue
            log.warning("[dsml-leak] second occurrence — stripping DSML from candidate")
            candidate = _TEXTUAL_TOOL_BLOCK_RE.sub("", candidate).strip()

        if not candidate:
            scratch.append({"role": "system", "content": "Produce a final answer or call a tool."})
            continue

        # Edit re-export insurance
        if edit_baseline and not st.pending_exports and not export_links and not st.edit_nudged:
            st.edit_nudged = True
            scratch.append({"role": "assistant", "content": candidate})
            scratch.append({"role": "system", "content": (
                "This is a REVISION of a document you delivered as a file. Call the same export "
                "tool now with the COMPLETE revised document and the same filename.")})
            continue

        streamed_live = stream_live and not st.pending_exports
        is_clar = _is_clarification(candidate)

        # User-chosen model regeneration
        if is_user_model and not is_clar:
            if config.SHOW_WORK:
                yield ("reasoning", f"✨ Writing with {user_final_model.split('/')[-1]}…\n")
            regen = await _regenerate_with_user_model(scratch, user_final_model, source, session)
            if regen:
                candidate = regen

        # Auto-polish new exports (never surgical edits)
        if not is_user_model and not is_clar:
            doc = _pending_prose_deliverable(st.pending_exports)
            if doc and len(doc) >= config.POLISH_MIN_CHARS:
                candidate = doc
                if not edit_baseline:
                    st.polish_voice = st.polish_voice or config.AUTO_POLISH_MODEL
                    if st.polish_voice_pass is None:
                        st.polish_voice_pass = await _classify_voice_register(
                            _all_user_text(messages), candidate, session=session)

        substantial = len(candidate) >= config.POLISH_MIN_CHARS
        prose = _prose_provider(st.polish_voice) if (st.polish_voice and not is_clar and substantial and not is_user_model) else None

        # Premium prose polish
        if prose is not None:
            pclient, pmodel = prose
            if config.SHOW_WORK:
                yield ("content" if st.pending_exports else "reasoning",
                       f"✨ Polishing the document ({pmodel.split('/')[-1]})…\n\n")
            pmsgs = _prose_polish_messages(messages_for_verify, candidate, source)
            to_chat = not st.pending_exports
            try:
                if config.STREAM_ANSWER and to_chat:
                    pparts = []
                    async for k, t in pclient.stream(pmsgs, pmodel, max_tokens=config.AGENT_MAX_TOKENS,
                                                      temperature=config.WRITER_TEMPERATURE,
                                                      session=session, label="polish"):
                        if k == "content":
                            pparts.append(t)
                            yield ("content", t)
                    if "".join(pparts).strip():
                        candidate = "".join(pparts).strip()
                        streamed_live = True
                else:
                    polished = await pclient.complete(pmsgs, pmodel, max_tokens=config.AGENT_MAX_TOKENS,
                                                       temperature=config.WRITER_TEMPERATURE,
                                                       session=session, label="polish")
                    if polished and polished.strip():
                        candidate = polished.strip()
            except Exception as e:
                log.warning(f"[prose_polish] {pmodel} failed, keeping open-model draft: {e}")

        # Voice pass (long-form only)
        if (not streamed_live and st.polish_voice_pass and st.polish_voice_pass != "none"
                and not is_clar and len(candidate) >= config.POLISH_VOICE_MIN_CHARS):
            if config.SHOW_WORK:
                yield ("reasoning", f"✨ Voice pass ({st.polish_voice_pass})…\n")
            candidate = await _voice_pass(candidate, st.polish_voice_pass, session=session)

        if config.SHOW_WORK and not streamed_live:
            yield ("reasoning", "✍️ Verifying the answer…\n")
        status, text = await _verified_or_blocked(
            messages_for_verify, candidate, source,
            recall_context=recall_context, prose=prose,
            force=bool(st.pending_exports or (image_transcript and config.VISION_FORCE_AUDIT)),
            session=session)

        links_str = ("\n\n" + "\n".join(f"📎 [Download {fn}]({url})" for fn, url in export_links)) if export_links else ""

        if status == "ok":
            file_links, filed = await _export_final(
                st.pending_exports, text, prose, messages_for_verify, source,
                chat_id=chat_id, headers=request_headers, session=session)
            links_str += file_links
            changed = text.strip() != candidate.strip()
            summary = ""
            if changed and filed:
                summary = await _summarize_correction(candidate, text, session=session) or ""

            if filed:
                yield ("content", ("\n\n---\n\n*Corrected before saving:*\n\n"
                       + (summary or "- Tightened to match your source.") + links_str) if changed
                       else (("\n\n" if streamed_live else "") + "📄 Your file is ready — download below." + links_str))
                final_text = text
            elif streamed_live:
                if changed:
                    final_text = text + links_str
                    yield ("content", "\n\n---\n\n*Corrected:*\n\n" + (summary or "- Tightened.")
                           + "\n\n*Corrected version:*\n\n" + final_text)
                else:
                    final_text = candidate + links_str
                    if links_str:
                        yield ("content", links_str)
            else:
                final_text = text + links_str
                yield ("content", final_text)

            if chat_id:
                um = _consolidated_user_memory(messages)
                if um:
                    _track_task(asyncio.create_task(_memory_store(chat_id, "user", um, session)))
                _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", final_text, session)))
            return

        # Blocked
        if streamed_live:
            yield ("content", "\n\n---\n\n⚠️ " + text)
            return
        if st.repair_steps < config.GROUNDING_REPAIR_STEPS:
            st.repair_steps += 1
            scratch.append({"role": "system", "content": (
                f"Internal verification gate blocked: {text}\n"
                "Use tools to gather evidence or revise. Do not show the blocked draft.")})
            continue
        yield ("content", text)
        return

    yield ("content", "I could not complete a verified answer within the configured tool budget.")


# ═══════════════════════════════════════════════════════════════════════════
# run() — thin phase orchestrator
# ═══════════════════════════════════════════════════════════════════════════

async def run(
    messages: list[dict], *,
    user_id: str = "",
    session: Optional[Any] = None,
    request_headers: Optional[dict] = None,
    user_model: str = "",
) -> AsyncGenerator[tuple[str, str], None]:
    """Drive one chat turn through the full phase pipeline."""
    if not messages:
        yield ("content", "")
        return

    user_final_model = (user_model or "").strip()
    is_user_model = bool(user_final_model)
    had_images = _has_images(messages)
    req_headers = request_headers or {}
    chat_id = req_headers.get("x-openwebui-chat-id", "")

    # Phase 1: Vision
    if had_images:
        if config.SHOW_WORK:
            yield ("content", "🖼️ Reading the image…\n\n")
        messages, image_transcript = await _read_images(messages, user_id, session)
    else:
        image_transcript = ""

    # Phase 2: Startup I/O (parallel)
    profile, prior_doc, gap_note, pending_plan = await _gather_context(user_id, chat_id, session)

    # Phase 3: Plan dispatch
    handled = False
    async for kt in _dispatch_plan(messages, pending_plan, chat_id, req_headers, session):
        handled = True
        yield kt
    if handled:
        return

    # Phase 4: Edit dispatch
    edit_handled, edit_output, edit_baseline = await _dispatch_edit(
        messages, prior_doc, chat_id, req_headers, session, config.SHOW_WORK)
    if edit_handled:
        yield ("content", edit_output)
        return
    edit_directive = edit_output  # when not handled, output is the directive

    # Phase 5: System prompt + context budget
    agent_extra = "\n\n".join(x for x in (gap_note, edit_directive) if x)
    scratch, recall_context, messages_for_verify = await _build_system_prompt(
        messages, user_id, chat_id, profile, agent_extra, session)

    # Phase 6: Grounding source
    user_source = _build_source(messages, image_transcript, edit_baseline)
    if config.LOG_SOURCE_DIAG:
        chars_by_role, blocks = {}, 0
        for m in messages:
            r = m.get("role", "?")
            chars_by_role[r] = chars_by_role.get(r, 0) + len(_text_of(m.get("content")))
            blocks += len(_owui_source_blocks(_text_of(m.get("content"))))
        log.info(f"[source-diag] user_source_chars={len(user_source)} "
                 f"owui_source_blocks={blocks} chars_by_role={chars_by_role}")

    # Phase 7: Plain chat fast path
    handled = False
    async for kt in _try_plain_chat(messages, scratch, user_source, chat_id,
                                      session, is_user_model, had_images):
        handled = True
        yield kt
    if handled:
        return

    # Phase 8: Chunked long-document writer
    handled = False
    async for kt in _try_longdoc(messages, user_source, chat_id, session,
                                   is_user_model, edit_baseline):
        handled = True
        yield kt
    if handled:
        return

    # Phase 9: Heavy turn preamble
    if config.SHOW_WORK and config.STREAM_PREAMBLE:
        yield ("content", "🧭 Working on it…\n\n")

    # Phase 10: Agentic tool loop
    async for kt in _agent_loop(messages, scratch, messages_for_verify, user_source,
                                  image_transcript, recall_context, edit_baseline, is_user_model,
                                  user_final_model, chat_id, req_headers, session):
        yield kt


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
