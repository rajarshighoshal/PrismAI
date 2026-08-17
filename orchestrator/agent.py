"""Model-driven agent loop for the orchestrator."""

import asyncio
import contextlib
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Optional

from . import config, escalation, fireworks, prompt_security, style, toolserver, interaction_mode
from .owui import (
    _unwrap_owui, _last_user_text,
    _owui_source_blocks, _user_source, _all_user_text,
)
from prism_core.messages import _text_of, _has_images
from . import memory_client
from .agent_state import AgentState
from .timectx import _now_line, _gap_note
from .verifier import _verified_or_blocked, _summarize_correction
from .prompts import (
    TOOL_SCHEMAS, SYSTEM_AGENT, SYSTEM_REQUEST_GATE,
)
# Vision phase lives in its own module now; re-exported names keep agent.run() and the
# existing tests (agent._VISION_CACHE, agent._split_vision_output) working unchanged.
from .vision import _read_images, _split_vision_output, _VISION_CACHE  # noqa: F401
# Prose polish + voice pass live in their own module; imported for use in the agent loop.
from .prose import _prose_provider, _prose_polish_messages  # noqa: F401
# Tool layer (schemas, guard gate, execution, result shaping) lives in its own module.
from .tools import (  # noqa: F401
    _budgeted_tools, _tool_status, _tool_path, _execute_tool, _tool_allowed,
    _source_from_tool, _visible_tool_result, _combined_source,
    _json_args, _compact_json, _clean_assistant_tool_message, _export_download,
)
# Delivery/persist + background-task helpers live in their own module; _BG_TASKS is
# re-exported (same set object) because the tests drain it via agent._BG_TASKS.
from .delivery import (  # noqa: F401
    _BG_TASKS, _track_task, _cancel_mode_task, _persist_turn,
    _same_doc, _pending_prose_deliverable,
)
# Multi-turn editing and the chunked long-doc writer live in their own modules; the
# longdoc names are re-exported because the tests drive them via agent.*.
from .editing import _dispatch_edit  # noqa: F401
from .longdoc import (  # noqa: F401
    _dispatch_plan, _try_longdoc,
    _maybe_longdoc, _classify_longdoc, _generate_outline, _render_outline,
)

log = logging.getLogger(__name__)

# A tool call leaked into the CONTENT channel as text (DeepSeek DSML / <tool_calls>
# markup, incl. the fullwidth-pipe variant) — it never executed; detect and recover.
_TEXTUAL_TOOL_CALL_RE = re.compile(r"<\s*[｜|]?\s*(?:tool_calls?|invoke|DSML)\b", re.I)
_TEXTUAL_TOOL_BLOCK_RE = re.compile(
    r"<\s*[｜|]?\s*(?:tool_calls?|invoke|DSML)\b.*?(?:</\s*[｜|]?\s*(?:tool_calls?|invoke|DSML)\s*>|$)",
    re.I | re.S)

def _with_system(messages, system_text):
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            base = _text_of(m.get("content"))
            m["content"] = (base + "\n\n" + system_text).strip() if base else system_text
            return out
    return [{"role": "system", "content": system_text}] + out

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

def _select_model(has_sources: bool) -> str:
    return config.GROUNDED_MODEL if has_sources else config.AGENT_MODEL

_PROGRESS_FALLBACK = {
    "start": "🧭 I’m setting up the work now: reading your request, checking the available sources, and deciding the safest path.\n\n",
    "drafted": "✓ Draft written. Next I’m checking whether it follows your request and stays grounded.\n",
    "polish": "✨ The factual draft is ready; I’m polishing the wording now without changing the facts.\n\n",
    "voice": "✨ The content is polished; I’m doing a light voice pass now so it reads naturally.\n",
    "verify": "🔍 Now I’m verifying the factual claims against your sources before anything gets finalized.\n",
    "export": "📦 Verification passed; I’m building the downloadable file now.\n",
}

async def _progress_note(stage: str, messages, *, detail: str = "", session=None) -> str:
    """One short user-visible progress sentence for serious/buffered work.

    The model is allowed to vary wording like Claude/ChatGPT, but not to invent
    task-specific facts or expose internals. If it fails, use the deterministic
    fallback so the user still sees movement.
    """
    fallback = _PROGRESS_FALLBACK.get(stage, "Working…\n")
    if not (config.SHOW_WORK and getattr(config, "ENABLE_MODEL_PROGRESS", True)):
        return fallback
    try:
        raw = await fireworks.complete(
            [
                {"role": "system", "content": (
                    "Write ONE brief, natural progress update for the user. "
                    "Explain what is happening now and, if useful, what comes next. "
                    "Do not mention model names, APIs, prompts, hidden chain-of-thought, "
                    "or internal implementation. Do not assert facts about the user's "
                    "document; this is only a process update. Max 22 words.")},
                {"role": "user", "content": json.dumps({
                    "stage": stage,
                    "detail": detail[:300],
                    "latest_user_request": _last_user_text(messages)[:600],
                }, ensure_ascii=True)},
            ],
            config.PROGRESS_MODEL,
            max_tokens=60,
            temperature=0.4,
            session=session,
            label="gate:progress",
        )
        note = " ".join((raw or "").strip().split())
        if not note:
            return fallback
        note = re.sub(r"(?i)\b(api|prompt|chain.of.thought|hidden|system message)\b", "", note).strip()
        return note.rstrip(".!?") + "…\n"
    except Exception:
        return fallback

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
            _track_task(asyncio.create_task(memory_client._deliverable_store(chat_id, md, exp["filename"], fmt)))
    links = ("\n\n" + "\n".join(f"📎 [Download {fn}]({url})" for fn, url in out)) if out else ""
    return links, filed_deliverable

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

async def _adherence_check(messages, candidate: str, *, export_pending: bool = False, session=None) -> dict:
    """Check task-contract adherence, separate from truth/grounding."""
    if not (config.ENABLE_ADHERENCE_GATE and candidate.strip()):
        return {"followed": True, "severity": "none", "misses": []}
    draft = candidate
    if len(draft) > config.ADHERENCE_MAX_DRAFT_CHARS:
        draft = draft[:config.ADHERENCE_MAX_DRAFT_CHARS] + "\n...[truncated]"
    try:
        raw = await fireworks.complete(
            [
                {"role": "system", "content": (
                    "You are an instruction-adherence checker. Compare the user's request "
                    "to the draft. Check only whether the draft followed the requested task, "
                    "format, sections, length, file/export intent, and explicit must/avoid "
                    "constraints. Do NOT judge factual truth; another verifier handles that. "
                    "If export_pending=true, do NOT flag the absence of a download link, file "
                    "attachment, or docx/pdf packaging in the draft itself; the surrounding "
                    "system handles the actual file export after this check. "
                    "Return strict JSON only: {\"followed\": boolean, \"severity\": "
                    "\"none\"|\"minor\"|\"major\", \"misses\": [\"brief issue\", ...]}. "
                    "Use severity=major only when the output clearly fails the user's core ask.")},
                {"role": "user", "content": json.dumps({
                    "request": _all_user_text(messages)[:6000],
                    "draft": draft,
                    "export_pending": bool(export_pending),
                }, ensure_ascii=True)},
            ],
            config.GROUNDING_GATE_MODEL,
            max_tokens=300,
            temperature=0.0,
            session=session,
            label="gate:adherence",
        )
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0) if m else raw)
        misses = data.get("misses") if isinstance(data.get("misses"), list) else []
        sev = str(data.get("severity") or "none").lower()
        return {"followed": bool(data.get("followed", True)),
                "severity": sev if sev in ("none", "minor", "major") else "none",
                "misses": [str(x)[:180] for x in misses[:5]]}
    except Exception:
        return {"followed": True, "severity": "none", "misses": []}

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
async def _gather_context(user_id, chat_id, session):
    """Parallel fetch: style profile, prior deliverable, last-active, pending plan.
    Returns (profile, prior_deliverable, gap_note, pending_plan)."""
    profile, prior, active, plan = "", None, None, None
    tasks = [asyncio.ensure_future(style.get_style_profile(user_id))]
    if chat_id:
        tasks.append(asyncio.ensure_future(memory_client._deliverable_get(chat_id)))
        tasks.append(asyncio.ensure_future(memory_client._last_active(chat_id)))
        if config.ENABLE_CHUNKED_WRITER:
            tasks.append(asyncio.ensure_future(memory_client._plan_get(chat_id)))
    results = await asyncio.gather(*tasks)
    profile = results[0]
    idx = 1
    if chat_id:
        prior = results[idx]; idx += 1
        active = results[idx]; idx += 1
        if config.ENABLE_CHUNKED_WRITER:
            plan = results[idx]; idx += 1
    return profile, prior, _gap_note(active) if active else "", plan

def _build_source(messages, image_transcript, edit_baseline):
    """Assemble grounding source: user files/pastes + image transcript + prior doc."""
    src = _user_source(messages)
    if image_transcript:
        tag = "IMAGE EVIDENCE TRANSCRIPT:\n" + image_transcript
        src = (src + "\n\n" + tag).strip() if src.strip() else tag
    if edit_baseline:
        src = (src + "\n\n" + edit_baseline).strip() if src.strip() else edit_baseline
    return src

def _source_coverage_note(messages, source: str) -> str:
    if not (config.SHOW_SOURCE_COVERAGE and (source or "").strip()):
        return ""
    blocks = sum(len(_owui_source_blocks(_text_of(m.get("content")))) for m in messages)
    if blocks:
        return f"\n\n_Checked against {blocks} attached source block{'s' if blocks != 1 else ''}._"
    return "\n\n_Checked against the source material you provided in chat._"

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
        for role, content in await memory_client._memory_recall(chat_id, recall_query, session):
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
    if answer:
        _persist_turn(chat_id, messages, answer, session)

#═══════════════════════════════════════════════════════════════════════════
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
                if config.SHOW_WORK:
                    yield ("content" if st.pending_exports else "reasoning",
                           await _progress_note("drafted", messages, session=session))
                if not edit_baseline:
                    st.polish_voice = st.polish_voice or config.AUTO_POLISH_MODEL

        substantial = len(candidate) >= config.POLISH_MIN_CHARS
        visible_progress = bool(st.pending_exports and substantial)
        prose = _prose_provider(st.polish_voice) if (st.polish_voice and not is_clar and substantial and not is_user_model) else None

        # Premium prose polish
        if prose is not None:
            pclient, pmodel = prose
            if config.SHOW_WORK:
                yield ("content" if visible_progress else "reasoning",
                       await _progress_note("polish", messages, detail=pmodel.split("/")[-1], session=session))
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

        if visible_progress:
            if config.SHOW_WORK:
                yield ("reasoning", "📋 Checking that the draft follows your requested format and constraints…\n")
            adherence = await _adherence_check(messages_for_verify, candidate, export_pending=bool(st.pending_exports), session=session)
            if (adherence.get("severity") == "major" or not adherence.get("followed", True)):
                if st.adherence_steps < config.ADHERENCE_REPAIR_STEPS:
                    st.adherence_steps += 1
                    misses = "\n".join(f"- {m}" for m in adherence.get("misses") or ["missed the user's requested format/constraints"])
                    scratch.append({"role": "assistant", "content": candidate})
                    scratch.append({"role": "system", "content": (
                        "Internal instruction-adherence gate blocked the draft. Fix these issues "
                        "and produce the complete corrected answer/export argument; do not mention "
                        "the gate to the user:\n" + misses)})
                    continue

        if config.SHOW_WORK and not streamed_live:
            yield ("content" if visible_progress else "reasoning",
                   await _progress_note("verify", messages, session=session))
        status, text = await _verified_or_blocked(
            messages_for_verify, candidate, source,
            recall_context=recall_context, prose=prose,
            force=bool(st.pending_exports or (image_transcript and config.VISION_FORCE_AUDIT)),
            session=session)

        links_str = ("\n\n" + "\n".join(f"📎 [Download {fn}]({url})" for fn, url in export_links)) if export_links else ""

        if status == "ok":
            if config.SHOW_WORK and visible_progress:
                yield ("content", await _progress_note("export", messages, session=session))
            file_links, filed = await _export_final(
                st.pending_exports, text, prose, messages_for_verify, source,
                chat_id=chat_id, headers=request_headers, session=session)
            links_str += file_links
            changed = text.strip() != candidate.strip()
            summary = ""
            if changed and filed:
                summary = await _summarize_correction(candidate, text, session=session) or ""

            if filed:
                coverage = _source_coverage_note(messages_for_verify, source) if visible_progress else ""
                yield ("content", ("\n\n---\n\n*Corrected before saving:*\n\n"
                       + (summary or "- Tightened to match your source.") + coverage + links_str) if changed
                       else (("\n\n" if streamed_live else "") + "📄 Your file is ready — download below." + coverage + links_str))
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

            _persist_turn(chat_id, messages, final_text, session)
            return

        if streamed_live:
            yield ("content", "\n\n---\n\n⚠️ " + text)
            return
        if st.repair_steps < config.GROUNDING_REPAIR_STEPS:
            st.repair_steps += 1
            scratch.append({"role": "system", "content": (
                f"Internal verification gate blocked: {text}\n"
                "Use tools to gather evidence or revise. Do not show the blocked draft.")})
            continue
        # Repair budget exhausted — a genuine capability shortfall. Optionally escalate ONCE
        # to a stronger model and re-verify (inert unless ESCALATION_MODEL is set; hardness-
        # gated). Never fires on a first block or a transient error. Non-Fireworks targets
        # no-op here until provider routing is added in Phase 2.
        if not st.escalated and await escalation.should_escalate(
                status, st.repair_steps, messages=messages, session=session):
            st.escalated = True
            if config.SHOW_WORK:
                yield ("reasoning",
                       f"🔼 Escalating to {config.ESCALATION_MODEL.split('/')[-1]} and re-verifying…\n")
            regen = await _regenerate_with_user_model(scratch, config.ESCALATION_MODEL, source, session)
            if regen and regen.strip():
                esc_status, esc_text = await _verified_or_blocked(
                    messages_for_verify, regen, source,
                    recall_context=recall_context, prose=None, force=True, session=session)
                if esc_status == "ok":
                    yield ("content", esc_text)
                    _persist_turn(chat_id, messages, esc_text, session)
                    return
                text = esc_text  # escalated draft still blocked — surface its (often clearer) message
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
    # Interaction mode is only UX polish. Start it alongside context I/O so it
    # does not add a serial round trip to ordinary chat turns — this overlap is
    # what lets the tiny on-path budget (below) catch the result on normal turns.
    # Tradeoff (intentional): on plan/edit early-return turns the call is
    # started then cancelled, so a few cheap flash calls are wasted. That is
    # observable via label=gate:interaction (traces + usage ledger) and fully
    # disabled by ENABLE_INTERACTION_MODE=false. Starting later instead would
    # make the note miss its budget and rarely fire on the common chat path.
    mode_task = asyncio.create_task(interaction_mode.classify(messages, session=session))
    profile, prior_doc, gap_note, pending_plan = await _gather_context(user_id, chat_id, session)

    # Phase 3: Plan dispatch
    handled = False
    async for kt in _dispatch_plan(messages, pending_plan, chat_id, req_headers, session):
        handled = True
        yield kt
    if handled:
        await _cancel_mode_task(mode_task)
        return

    # Phase 4: Edit dispatch
    edit_handled, edit_output, edit_baseline = await _dispatch_edit(
        messages, prior_doc, chat_id, req_headers, session, config.SHOW_WORK)
    if edit_handled:
        await _cancel_mode_task(mode_task)
        yield ("content", edit_output)
        return
    edit_directive = edit_output  # when not handled, output is the directive

    # Phase 5: System prompt + context budget
    # Cheap style/persona adapter: helps the assistant behave like the right kind
    # of helper for this turn (student tutor, practical tech support, creative
    # brainstormer, etc.). It is deliberately style-only and does not affect
    # source/tool/verification policy. It is pure UX polish, so it never delays
    # the turn: consume it only if it is already done, otherwise allow a tiny
    # residual budget and then drop it (fail open).
    mode_note = ""
    if mode_task.done():
        with contextlib.suppress(Exception):
            mode_note = mode_task.result() or ""
    else:
        try:
            mode_note = await asyncio.wait_for(
                asyncio.shield(mode_task), timeout=config.INTERACTION_MODE_ONPATH_BUDGET)
            mode_note = mode_note or ""
        except Exception:
            mode_note = ""
            await _cancel_mode_task(mode_task)
    agent_extra = "\n\n".join(x for x in (gap_note, edit_directive, mode_note) if x)
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
        yield ("content", await _progress_note("start", messages, session=session))

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
