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

from . import config, fireworks, gemini, openai_client, anthropic_client, prompt_security, search, style, toolserver
from .owui import (
    _text_of, _unwrap_owui, _last_user_text, _has_images, _split_content_parts,
    _same_message_source, _SOURCE_BLOCK_RE, _owui_source_blocks, _user_source, _all_user_text,
)
from .memory_client import (
    _memory_recall, _memory_store, _deliverable_store, _deliverable_get, _last_active,
)
from .timectx import _now_line, _gap_note
from .verifier import _verified_or_blocked, _summarize_correction, _WORD_RE
from .prompts import (
    TOOL_SCHEMAS, SYSTEM_AGENT, SYSTEM_VISION, SYSTEM_GATE, SYSTEM_REQUEST_GATE, SYSTEM_EDIT_INTENT,
    SYSTEM_FACT_AUDIT, SYSTEM_TOOL_GUARD, SYSTEM_CHANGE_SUMMARY, SYSTEM_VOICE_REGISTER,
    _PROSE_POLISH_SYS, _VOICE_REGISTER, _VOICE_PASS_SYS,
)

log = logging.getLogger(__name__)


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


async def _classify_edit_once(payload, *, session=None) -> dict:
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
                yield ("reasoning", "✏️ Revising the document…\n")
            baseline = (prior_deliverable.get("content") or "").strip()
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
            # Sanity: the revision must still BE the document (not an ack/refusal). If it
            # isn't, fall through to the normal loop with the injected-doc directive.
            if revised and _same_doc(revised, baseline):
                if config.SHOW_WORK:
                    yield ("reasoning", "✍️ Verifying the revision…\n")
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
                yield ("reasoning", f"✨ Polishing the document ({prose_model.split('/')[-1]})…\n")
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
