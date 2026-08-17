"""Chunked long-document writer: outline proposal/approval, then per-section
generation with its own audit + recap. Split out of the agent god-module.
"""
import asyncio
import json
import re
import time
import logging

from . import config, fireworks, memory_client
from .owui import _last_user_text, _all_user_text
from .prompts import SYSTEM_LONGDOC_GATE, SYSTEM_OUTLINE, SYSTEM_PLAN_INTENT, SYSTEM_SECTION_WRITER
from .verifier import _verified_or_blocked, _has_citation_markers, _WORD_RE
from prism_core.verifier import fit_audit_source as _fit_audit_source
from .delivery import _repackage_deliverable, _track_task

log = logging.getLogger(__name__)

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
        await memory_client._plan_store(chat_id, plan)   # awaited: the NEXT turn reads this
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
        await memory_client._deliverable_store(chat_id, full_doc, filename, fmt)
    if chat_id:
        await memory_client._plan_clear(chat_id)
        _track_task(asyncio.create_task(memory_client._memory_store(chat_id, "assistant", full_doc[:4000], session)))
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

async def _dispatch_plan(messages, plan, chat_id, req_headers, session):
    """Handle pending outline. Yields output if plan handled; caller detects via flag."""
    if not plan:
        return
    created = float(plan.get("created_at") or 0)
    age = (time.time() - created) if created else 0
    revises = int(plan.get("revise_count") or 0)
    if age > config.CHUNKED_PLAN_TTL_SECONDS or revises > config.CHUNKED_MAX_REVISES:
        await memory_client._plan_clear(chat_id)
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
    await memory_client._plan_clear(chat_id)

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


#═══════════════════════════════════════════════════════════════════════════
# Agent loop — the heavy path, now driven by AgentState
# ═══════════════════════════════════════════════════════════════════════════

def _slug(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (title or "document")).strip("_").lower()
    return (s or "document")[:60]


_LONGDOC_CUES = (
    "paper", "thesis", "dissertation", "report", "essay", "review", "chapter", "white paper",
    "whitepaper", "case study", "proposal", "study guide", "manuscript", "literature review",
    "section", "comprehensive", "in-depth", "in depth", "detailed", "multi-part",
)
