"""Model-driven agent loop for the orchestrator.

The harness exposes tools and enforces verification. It does not classify the
turn into prewritten task flows; the model chooses tools, the harness executes
them, and final output is held until the grounding gate allows it.

The full turn lifecycle (unwrap → startup I/O → vision → edit engine → grounding →
agent loop → polish/voice → verify → deliver → persist), the model routing table,
and the standing invariants live in docs/ARCHITECTURE.md — read that first.

Module map (in lifecycle order):
  owui.py           parsing what OWUI sends: _unwrap_owui, _user_source, source blocks
  timectx.py        _now_line (what 'now' is), _gap_note (resume-after-gap)
  memory_client.py  tool-server HTTP: chat memory, deliverable store, last-active
  style.py          per-user voice profile (read-only, off-thread)
  verifier.py       the can't-lie gate: _verified_or_blocked, _fact_audit,
                    _refine_facts, the verbatim backstop
  THIS FILE         vision (_describe_images_for_agent); the edit engine
                    (_classify_edit → directed pipeline / _repackage_deliverable);
                    the agent loop (run(), tool execution, budgets, SYSTEM_TOOL_GUARD);
                    polish & voice (_prose_provider, _voice_pass); delivery
                    (_export_final, _same_doc).
"""

import asyncio
import json
import logging
import re
import time

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
from .verifier import _verified_or_blocked, _summarize_correction, _WORD_RE, _has_citation_markers
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
    """Split the vision model's emission into (evidence_transcript, reading). The transcript is
    the audit-grade SOURCE (literal, verbatim); the reading is the interpretation for the
    reasoner. Falls back to using the whole text for both when the model didn't emit the
    two-part structure (e.g. the kimi fallback produced a flat caption)."""
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


async def _describe_images_for_agent(messages, *, session=None):
    """Replace each image with NATIVE-vision text: the model SEES the pixels and emits a
    structured EVIDENCE TRANSCRIPT (audit-grade source) + a cited READING (context for the
    text reasoner). Returns (messages, image_transcript): the transcript is surfaced
    SEPARATELY so run() can route it into the grounding source, where the text auditor can
    verify image-derived claims against it. Images are read CONCURRENTLY (order preserved)."""
    async def _describe(m):
        content = m.get("content")
        text_parts, image_parts = _split_content_parts(content)
        if not image_parts:
            return dict(m), ""
        user_text = "\n".join(t.strip() for t in text_parts if t.strip())
        prompt = (
            "The user attached image(s) with this message. Read them natively and follow your "
            "two-part contract — EVIDENCE TRANSCRIPT, then READING.\n\n"
            f"USER REQUEST:\n{user_text or '(none)'}"
        )
        # Pin high image detail so the provider tiles a large/dense image at full resolution
        # instead of downscaling it into a blur the model confabulates from (A/B-proven).
        detail = (config.VISION_IMAGE_DETAIL or "").strip().lower()
        imgs = []
        for p in image_parts:
            if detail and detail != "auto" and isinstance(p.get("image_url"), dict):
                p = {**p, "image_url": {**p["image_url"], "detail": detail}}
            imgs.append(p)
        vision_content = [{"type": "text", "text": prompt}] + imgs
        out = ""
        # M3 (native vision) first; degrade to the fallback reader if its call fails/empties.
        for model in (config.VISION_MODEL, config.VISION_FALLBACK_MODEL):
            if not model:
                continue
            try:
                out = (await fireworks.complete(
                    [{"role": "system", "content": SYSTEM_VISION},
                     {"role": "user", "content": vision_content}],
                    model, max_tokens=config.VISION_MAX_TOKENS, temperature=0.0,
                    session=session, label="vision")).strip()
                if out:
                    break
            except Exception as e:
                log.warning(f"[vision] {model.split('/')[-1]} read failed: {e}")
        transcript, _reading = _split_vision_output(out)
        new_m = dict(m)
        if out:
            # Frame as the assistant's OWN sight of the image (not user-pasted text), so the
            # reasoner doesn't disclaim "based on the transcription you gave me".
            new_m["content"] = (((user_text + "\n\n") if user_text else "")
                                + "[What you see in the attached image:]\n" + out)
        else:
            new_m["content"] = user_text or "Image was attached, but the vision read failed."
        return new_m, transcript

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
    """Classify a follow-up against the chat's last delivered document: rename|reformat|
    edit|new (reasoning-on; a semantic judgement, not keyword matching).

    The judge gets the RECENT CONVERSATION, not just the lone message — follow-ups are
    anaphoric ("also add…", "connect it to this position") and unclassifiable without
    their antecedent (live smoke proved a context-starved judge reads them as 'new').
    And the two misroute directions are NOT symmetric: new-misread-as-edit is
    self-correcting (the revision fails _same_doc and falls back to the normal flow),
    while edit-misread-as-new silently drops the user's document — so a 'new' verdict
    must win TWICE, biasing the rare flake toward the recoverable direction."""
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
    """In-place edit, the Canvas/Artifact way: ask the model for targeted find→replace
    changes and apply them to the stored document, leaving everything else byte-for-byte
    identical. Returns the patched text, or None when the change is too broad for clean
    patches or a 'find' doesn't match exactly once — the caller then falls back to a full
    re-emit (exactly how Claude/ChatGPT choose targeted-edit vs rewrite). The model emits
    only the change (~1/40th the tokens of regenerating the whole document), and untouched
    text cannot drift because the model never re-types it."""
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
    """Plan a long document: {title, sections:[{heading,intent}]}. Grounded model at MAX
    reasoning (label not a gate). For a revision, current_outline + change are supplied so the
    model adjusts THAT outline in place (positional edits land right). Returns plan or None."""
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
    """Write ONE section (focused, bounded, MAX reasoning), aware of the whole outline and
    what earlier sections covered. Returns the section's Markdown, or '' on failure."""
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
        parts.append("SOURCE MATERIAL (assert only what this supports; never fabricate):\n" + source[:24000])
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
    """Generate an outline (or revise the current one in place), persist it as the pending
    plan with a timestamp + revise counter, and yield it for approval. Async gen of (kind, text)."""
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
    """Build the approved long document SECTION-BY-SECTION with live progress. Each section is
    verified on its OWN — a small input, so the honesty audit reasons within budget (no
    dilution across a giant doc) and the refiner re-emits only that section (never the
    64k-truncation a whole-doc rewrite would hit). If any section can't be written or grounded,
    NOTHING ships — the plan is kept so the user can add a source / adjust and rebuild.
    Async generator of (kind, text)."""
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


async def run(messages, *, user_id="", session=None, request_headers=None, user_model=""):
    """Drive one chat turn. Async generator of (kind, text)."""
    if not messages:
        yield ("content", "")
        return

    # Instant feedback: a heavy turn (uploaded doc, big paste) spends a few seconds
    # on the first generation before any other breadcrumb — show something now so
    # the user isn't staring at a blank.
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
    plan_task = (asyncio.ensure_future(_plan_get(chat_id))
                 if chat_id and config.ENABLE_CHUNKED_WRITER else None)
    image_transcript = ""
    if had_images:
        if config.SHOW_WORK:
            yield ("content", "🖼️ Reading the image…\n\n")
        messages, image_transcript = await _describe_images_for_agent(messages, session=session)

    style_profile = await style_task
    prior_deliverable = (await deliverable_task) if deliverable_task else None
    gap_note = _gap_note(await active_task) if active_task else ""

    # ── Pending long-doc outline awaiting approval (chunked writer, turn 2+) ─────────
    # A plan is pending only mid-flow (we proposed an outline last turn). The user's reply
    # is ABOUT that outline — approve (build it), revise (adjust + re-show), or abandon —
    # so this is decided BEFORE the edit/chat paths, which would misread "go" as a new turn.
    pending_plan = (await plan_task) if plan_task else None
    if pending_plan:
        # Escape hatch: a never-approved plan must not trap the chat forever (the intent
        # classifier biases to 'revise' when unsure). Expire on age or on too many revises.
        # A missing created_at means "unknown age" -> don't expire on age (only on revises).
        created = float(pending_plan.get("created_at") or 0)
        age = (time.time() - created) if created else 0
        revises = int(pending_plan.get("revise_count") or 0)
        if age > config.CHUNKED_PLAN_TTL_SECONDS or revises > config.CHUNKED_MAX_REVISES:
            await _plan_clear(chat_id)
            pending_plan = None
    if pending_plan:
        intent = await _classify_plan_intent(_last_user_text(messages), pending_plan, session=session)
        if intent["action"] == "approve":
            async for kt in _build_from_plan(pending_plan, messages, user_id, chat_id, req_headers, session):
                yield kt
            return
        if intent["action"] == "revise":
            revision = intent.get("revision") or ""
            if revision:
                # Revise the CURRENT outline in place (so 'remove section 3' lands), keeping the
                # ORIGINAL request stable across revisions; bump the revise counter.
                async for kt in _present_outline(
                        pending_plan.get("request") or "", pending_plan.get("source") or "",
                        chat_id=chat_id, filename=pending_plan.get("filename") or "",
                        fmt=pending_plan.get("fmt") or "docx", session=session, revised=True,
                        current_outline=_outline_for_prompt(pending_plan), change=revision,
                        revise_count=revises + 1):
                    yield kt
            else:
                # Couldn't read a change (or empty reply) — re-show, but make the state escapable.
                yield ("content", _render_outline(pending_plan)
                       + "\n\n*(Say “write it” to build, name a change, or “never mind” to drop it.)*")
            return
        # abandon -> drop the plan and fall through to the normal flow with this message
        await _plan_clear(chat_id)

    # ── Multi-turn edit of the chat's last delivered document ──────────────────────
    # Do the LEAST the request needs, DETERMINISTICALLY. rename/reformat re-package the
    # verified bytes (no writer, no verifier). A content edit runs a DIRECTED pipeline:
    # one writer call (revised document as its plain response — no tool choice involved),
    # verify, then the HARNESS exports with the stored filename/format. The smoke harness
    # proved any path where the model must CHOOSE to call export is a coin flip.
    edit_directive, edit_baseline = "", ""
    if prior_deliverable and prior_deliverable.get("content"):
        intent = await _classify_edit(_last_user_text(messages), prior_deliverable,
                                      messages=messages, session=session)
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
            if config.SHOW_WORK:
                yield ("content", "✏️ Revising the document…\n\n")
            baseline = (prior_deliverable.get("content") or "").strip()
            # 1. In-place targeted edit first (the Canvas/Artifact way): the model returns
            #    only the find→replace changes, applied to the stored document so untouched
            #    text is preserved byte-for-byte at ~1/40th the tokens of a rewrite.
            revised = await _try_patch_edit(baseline, _last_user_text(messages), session=session)
            patched = revised is not None
            # 2. Broad change (or no clean patch) -> regenerate the whole document.
            if not patched:
                revised = ""
                try:
                    revised = (await fireworks.complete(
                        [{"role": "system", "content":
                            _now_line() + "\n\n" + _edit_inject(prior_deliverable)
                            + "\n\nOutput ONLY the complete revised document — no commentary, "
                            "no preamble, no tool calls."},
                         {"role": "user", "content": _last_user_text(messages)}],
                        config.GROUNDED_MODEL, max_tokens=config.DRAFT_MAX_TOKENS,
                        temperature=config.WRITER_TEMPERATURE, session=session,
                        label="edit:write")).strip()
                except Exception as e:
                    log.warning(f"[edit] directed revision failed, falling to normal flow: {e}")
                # The writer may ask instead of guess (re-emit path only): a short,
                # question-marked, non-document response IS a clarifying question — ship it
                # straight to the user; their reply re-enters this path, document still stored.
                if (revised and not _same_doc(revised, baseline)
                        and "?" in revised and len(revised) < 1200):
                    yield ("content", revised)
                    return
            # 3. Ship the revision (patched in-place OR fully re-emitted), if it's a real doc.
            if revised and (patched or _same_doc(revised, baseline)):
                if config.SHOW_WORK:
                    yield ("content", "🛡️ Verifying facts…\n\n")
                src = ((_user_source(messages) + "\n\n" + baseline).strip()
                       if _user_source(messages).strip() else baseline)
                status, text = await _verified_or_blocked(
                    messages, revised, src, force=True, session=session)
                if status != "ok":
                    yield ("content", text)
                    return
                link = await _repackage_deliverable(
                    text, prior_deliverable.get("filename") or "document",
                    prior_deliverable.get("fmt") or "docx",
                    chat_id=chat_id, headers=req_headers, session=session)
                summary = await _summarize_correction(baseline, text, session=session)
                yield ("content", ("📄 Updated — download below."
                       + (("\n\n" + summary) if summary else "") + link) if link
                       else "I couldn't rebuild the file — want me to try again?")
                if chat_id:
                    user_memory = _consolidated_user_memory(messages)
                    if user_memory:
                        _track_task(asyncio.create_task(_memory_store(chat_id, "user", user_memory, session)))
                    _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", text, session)))
                return
            edit_directive = _edit_inject(prior_deliverable)
            edit_baseline = baseline

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
    if image_transcript:
        # The native-vision EVIDENCE TRANSCRIPT is the auditable proxy for the pixels: routing
        # it into the grounding source lets the text auditor verify image-derived claims (a
        # fabricated chart value / invented table cell is flagged because it isn't in the
        # transcript) — turning the unauditable-image problem into the auditable-text one.
        tag = ("IMAGE EVIDENCE TRANSCRIPT (what the vision reader saw in the attached image; "
               "authoritative source for any claim about the image):\n" + image_transcript)
        user_source = (user_source + "\n\n" + tag).strip() if user_source.strip() else tag
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

    # ── NEW long document -> outline-first chunked writer ──────────────────────────
    # We're past the plain-chat fast path, so this is a heavy turn. If the user is asking
    # for a long, multi-section document, don't emit it in one capped shot — propose an
    # outline now, then write it section-by-section once they approve (see Hook above).
    # Skips edits (handled) and user-model regens (they picked a model). A cheap deterministic
    # prefilter (_maybe_longdoc) gates the flash classifier so it never fires on short heavy
    # turns (quick edits, summaries) — only on plausibly-long requests.
    if (config.ENABLE_CHUNKED_WRITER and chat_id and not is_user_model and not edit_baseline
            and _maybe_longdoc(messages)
            and (await _classify_longdoc(messages, session=session)).get("longdoc")):
        async for kt in _present_outline(
                _all_user_text(messages), user_source, chat_id=chat_id,
                filename="", fmt="docx", session=session):
            yield kt
        return

    # Heavy turn: show an instant status line so the user sees activity while the first
    # heavy generation runs. A deterministic line — not a model call: spending a flash
    # generation to emit filler added latency and cost and cluttered the thinking panel
    # with non-model text. (feat/progress-ux will make these stages user-visible.)
    if config.SHOW_WORK and config.STREAM_PREAMBLE:
        yield ("content", "🧭 Working on it…\n\n")

    tool_sources = []
    export_links = []        # immediate exports (csv) — link collected as they run
    pending_exports = []     # deferred prose exports (docx/pdf/md) — built from the final answer
    repair_steps = 0
    tool_call_count = 0
    web_search_count = 0
    budget_note_added = False
    edit_nudged = False
    textual_tool_nudged = False
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
                        "instruction": (
                            "Answer the user's actual question directly without this tool. "
                            "If you genuinely cannot answer it from what you know, say so "
                            "plainly and state what you'd need — NEVER say you will look it "
                            "up or search later; there is no later, this answer is final."
                        ),
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

        # DeepSeek sometimes writes its tool call as PLAIN TEXT ("<tool_calls>…") instead
        # of the function-calling channel — nothing executes and the user would see raw
        # markup as the answer (live report). Same defense the tool-server has for
        # exports: nudge once so the model re-issues it properly; strip as last resort.
        if _TEXTUAL_TOOL_CALL_RE.search(candidate):
            if not textual_tool_nudged:
                textual_tool_nudged = True
                scratch.append({"role": "assistant", "content": candidate})
                scratch.append({
                    "role": "system",
                    "content": (
                        "Internal harness note: you wrote a tool call as plain text — it "
                        "did NOT execute and the user would see raw markup. Issue it "
                        "through the function-calling interface, or answer directly "
                        "without the tool."
                    ),
                })
                continue
            candidate = _TEXTUAL_TOOL_BLOCK_RE.sub("", candidate).strip()

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
                yield ("content" if pending_exports else "reasoning", f"✨ Polishing the document ({prose_model.split('/')[-1]})…\n\n")
            # messages_for_verify (= the kept tail in overflow) bounds the premium
            # polish prompt. Stream the polished deliverable live ONLY when the chat
            # carries it. When a FILE carries it, the body goes nowhere near the thinking
            # panel — thinking is the model's reasoning + a clean stage log, never a
            # scratch dump of the whole letter scrolling by.
            pmsgs = _prose_polish_messages(messages_for_verify, candidate, source)
            to_chat = not pending_exports
            try:
                if config.STREAM_ANSWER and to_chat:
                    pparts = []
                    async for k, t in prose_client.stream(
                        pmsgs, prose_model, max_tokens=config.AGENT_MAX_TOKENS,
                        temperature=config.WRITER_TEMPERATURE, session=session,
                        label="polish",
                    ):
                        if k == "content":
                            pparts.append(t)
                            yield ("content", t)
                    if "".join(pparts).strip():
                        candidate = "".join(pparts).strip()
                        streamed_live = True
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
