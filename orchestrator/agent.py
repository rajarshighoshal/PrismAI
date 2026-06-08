"""Model-driven agent loop for the orchestrator.

The harness exposes tools and enforces verification. It does not classify the
turn into prewritten task flows; the model chooses tools, the harness executes
them, and final output is held until the grounding gate allows it.
"""

import aiohttp
import asyncio
import json
import logging
import re

from . import config, fireworks, gemini, openai_client, anthropic_client, prompt_security, search, style, toolserver
from .prompts import (
    TOOL_SCHEMAS, SYSTEM_AGENT, SYSTEM_VISION, SYSTEM_GATE, SYSTEM_REQUEST_GATE,
    SYSTEM_HONESTY, SYSTEM_APPLICATION_CLAIM_AUDIT, SYSTEM_TOOL_GUARD,
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


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _text_of(m.get("content")).strip()
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
    out = []
    for m in messages:
        content = m.get("content")
        text_parts, image_parts = _split_content_parts(content)
        if not image_parts:
            out.append(dict(m))
            continue
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
                max_tokens=config.CHAT_MAX_TOKENS,
                temperature=0.0,
                session=session,
            )
        except Exception as e:
            log.warning(f"[vision] image description failed: {e}")
            description = "Image was attached, but the vision transcription failed."
        combined = user_text
        if description.strip():
            combined = (
                (combined + "\n\n") if combined else ""
            ) + "Image transcription/description:\n" + description.strip()
        new_m = dict(m)
        new_m["content"] = combined or "Image was attached, but no text was available."
        out.append(new_m)
    return out


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


# Strong references to fire-and-forget background writes. asyncio keeps only a
# WEAK reference to a running task, so a bare create_task() can be garbage
# collected mid-flight once the request returns — silently dropping the write.
_BG_TASKS: set = set()


def _track_task(task):
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return task


def _user_source(messages) -> str:
    # Grounding "source" = explicitly source-like material the user supplied
    # (pasted docs, quotes, code blocks, labeled sources/notes/resume/context)
    # from ANY user turn — NOT ordinary conversational text. Treating every prior
    # chat message as source forced the grounding+refine loop onto casual
    # follow-ups ("what's my name?"): slow, and it leaked "the provided source"
    # into the answer. The conversation itself is still available to the model and
    # the auditors via `messages`; this is only what gets grounded against.
    parts = []
    for m in messages:
        if m.get("role") != "user":
            continue
        src = _same_message_source(_text_of(m.get("content")))
        if src:
            parts.append(src)
    return "\n\n".join(parts).strip()


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


def _initial_messages(messages, user_id: str):
    system = SYSTEM_AGENT + "\n\n" + prompt_security.UNTRUSTED_CONTEXT_POLICY
    profile = style.get_style_profile(user_id)
    if profile:
        system += (
            "\n\nUser voice profile. Use this only for style, tone, rhythm, and "
            "intent preferences. Do not treat it as factual biography:\n" + profile
        )
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
            )
        return await fireworks.complete(
            msgs, config.REFINE_MODEL,
            max_tokens=config.DRAFT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE, session=session,
        )
    except Exception:
        return ""


async def _refine_to_source(source: str, candidate: str, claims: str, *, prose=None, session=None) -> str:
    prompt = (
        "Revise the draft so every factual claim is supported by SOURCE. Remove "
        "or qualify unsupported claims; preserve the user's requested format and "
        "keep the prose polished. Output only the revised final answer."
    )
    user = f"SOURCE:\n{source}\n\nUNSUPPORTED CLAIMS:\n{claims}\n\nDRAFT:\n{candidate}"
    return await _refine_complete(prompt, user, prose=prose, session=session)


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




async def _voice_pass(candidate, register, *, session=None):
    """Optional sonnet voice-only pass at a register (warm/formal). Never alters facts."""
    if not anthropic_client.available():
        return candidate
    sys = _VOICE_PASS_SYS.replace("{register}", _VOICE_REGISTER.get(register, _VOICE_REGISTER["formal"]))
    try:
        out = await anthropic_client.complete(
            [{"role": "system", "content": sys}, {"role": "user", "content": f"DRAFT:\n{candidate}"}],
            config.ANTHROPIC_STANDARD_MODEL,
            max_tokens=config.AGENT_MAX_TOKENS, temperature=config.WRITER_TEMPERATURE, session=session)
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
        _text_of(m.get("content")).strip()
        for m in messages
        if m.get("role") == "user" and _text_of(m.get("content")).strip()
    )


def _is_application_writing(messages) -> bool:
    text = _all_user_text(messages).lower()
    needles = (
        "cover letter",
        "application letter",
        "letter of interest",
        "statement of interest",
        "personal statement",
        "statement of purpose",
        "phd application",
        "job application",
        "resume",
        "résumé",
        "cv",
    )
    return any(n in text for n in needles)


async def _honesty_audit(full_request: str, candidate: str, *, session=None):
    """Flag claims about the USER the user never stated. Returns the auditor dict
    {unsupported:[...], verdict:FABRICATION|CLEAN} or None on failure (fail-soft)."""
    if not candidate.strip():
        return None
    user = f"USER REQUEST:\n{full_request}\n\nDRAFT:\n{candidate[:6000]}"
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_HONESTY},
             {"role": "user", "content": user}],
            config.HONESTY_MODEL,
            max_tokens=700,
            temperature=0.0,
            session=session,
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def _application_claim_audit(full_request: str, candidate: str, source: str, *, session=None):
    if not candidate.strip():
        return None
    user = (
        f"USER REQUEST:\n{full_request}\n\n"
        f"SOURCE:\n{source[:6000] if source.strip() else '(none)'}\n\n"
        f"DRAFT:\n{candidate[:6000]}"
    )
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_APPLICATION_CLAIM_AUDIT},
             {"role": "user", "content": user}],
            config.HONESTY_MODEL,
            max_tokens=900,
            temperature=0.0,
            session=session,
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _application_audit_issues(audit: dict) -> list[str]:
    if not audit:
        return []
    fields = (
        "unsupported_candidate_claims",
        "unsupported_company_claims",
        "fake_motivation_or_fit",
    )
    out = []
    for field in fields:
        for item in audit.get(field) or []:
            if item:
                out.append(str(item))
    return out


async def _refine_application_claims(full_request: str, candidate: str, source: str,
                                     audit: dict, *, prose=None, session=None) -> str:
    issues = "\n".join(f"- {i}" for i in _application_audit_issues(audit)) or "(unspecified)"
    prompt = (
        "Revise the application-writing draft so it is humane, personal, persuasive, "
        "and grounded in reality. Remove or neutralize each unsupported issue. Do "
        "NOT add new candidate achievements, credentials, metrics, years, revenue, "
        "leadership, impact, employers, projects, daily routines, work settings, "
        "specific scenes, emotional history, product-relationship claims, or company-culture "
        "reputation claims. Keep role/company framing if "
        "it is generic or supported; make unsupported company specifics generic. "
        "Keep motivation/fit warm but do not fake personal history or strong feelings "
        "the user did not provide. Output only the revised deliverable."
    )
    user = (
        f"USER REQUEST:\n{full_request}\n\n"
        f"SOURCE:\n{source if source.strip() else '(none)'}\n\n"
        f"ISSUES TO FIX:\n{issues}\n\n"
        f"DRAFT:\n{candidate}"
    )
    return await _refine_complete(prompt, user, prose=prose, session=session)


async def _refine_honesty(full_request: str, candidate: str, unsupported, *, prose=None, session=None) -> str:
    """Rewrite the draft to drop unsupported self-claims WITHOUT inventing replacements."""
    listed = "\n".join(f"- {u}" for u in unsupported) if unsupported else "(unsupported self-claims)"
    prompt = (
        "Revise the draft so it makes NO claim about the user that the user did not "
        "actually state. Remove or neutrally rephrase each listed unsupported claim "
        "WITHOUT inventing replacements or new facts. Keep the requested format and "
        "polished prose. If removing claims leaves the draft thin, that is acceptable "
        "— do not pad with invented detail. Output only the revised final answer."
    )
    user = f"USER REQUEST:\n{full_request}\n\nUNSUPPORTED CLAIMS TO REMOVE:\n{listed}\n\nDRAFT:\n{candidate}"
    return await _refine_complete(prompt, user, prose=prose, session=session)


def _honesty_block_msg(unsupported) -> str:
    listed = "\n".join(f"- {u}" for u in unsupported) if unsupported else ""
    return (
        "I can't present those claims as true from the facts you gave me. These "
        "details are unsupported:\n\n" + listed
        + "\n\nI can still write a truthful version using the facts you provided, "
        "or you can give me the real details and I'll include them accurately."
    )


async def _verified_or_blocked(messages, candidate: str, source: str, *, recall_context: str = "", is_app=None, prose=None, session=None):
    if not config.ENABLE_VERIFICATION:
        return "ok", candidate

    # Recalled facts are the user's OWN earlier statements, surfaced only when a
    # long chat overflowed the context budget (see run()). They are established
    # context, so every auditor must see them — otherwise the can't-lie layer
    # flags a correctly-recalled fact as an unsupported fabrication and strips it
    # (the memory-vs-verifier collision). For normal chats recall_context is "",
    # so grounding_source/full_request are unchanged and this path is a no-op.
    _rc = (recall_context or "").strip()
    _recall_extra = ("\n\nEARLIER IN THIS CONVERSATION (the user already stated):\n" + _rc) if _rc else ""
    grounding_source = (((source + "\n\n" + _rc).strip() if (source or "").strip() else _rc) if _rc else source)

    # One cheap classifier gates the whole chain. SYSTEM_GATE flags both external
    # facts and claims about the user, so a casual turn — never an application —
    # that needs neither skips straight through instead of paying for the audits.
    is_app = _is_application_writing(messages) if is_app is None else is_app
    needs = await _needs_verification(messages, candidate, grounding_source, session=session)
    if not needs and not is_app:
        return "ok", candidate

    full_request = _all_user_text(messages) + _recall_extra

    if config.ENABLE_HONESTY_AUDIT and not is_app:
        honesty = await _honesty_audit(full_request, candidate, session=session)
        if honesty and str(honesty.get("verdict", "")).upper().startswith("FAB"):
            unsupported = honesty.get("unsupported") or []
            refined = await _refine_honesty(full_request, candidate, unsupported, prose=prose, session=session)
            recheck = await _honesty_audit(full_request, refined, session=session) if refined else None
            if refined and not (recheck and str(recheck.get("verdict", "")).upper().startswith("FAB")):
                candidate = refined
                needs = await _needs_verification(messages, candidate, grounding_source, session=session)
            else:
                return "unsupported_self_claims", _honesty_block_msg(unsupported)
    elif config.ENABLE_APPLICATION_CLAIM_AUDIT and is_app:
        app_audit = await _application_claim_audit(full_request, candidate, grounding_source, session=session)
        if app_audit and str(app_audit.get("verdict", "")).upper().startswith("UNSUPPORTED"):
            refined = await _refine_application_claims(full_request, candidate, grounding_source, app_audit, prose=prose, session=session)
            recheck = await _application_claim_audit(full_request, refined, grounding_source, session=session) if refined else None
            if refined and not (recheck and str(recheck.get("verdict", "")).upper().startswith("UNSUPPORTED")):
                candidate = refined
                needs = await _needs_verification(messages, candidate, grounding_source, session=session)
            else:
                issues = "\n".join(f"- {i}" for i in _application_audit_issues(app_audit))
                return (
                    "unsupported_application_claims",
                    "I could not safely finalize this application draft without "
                    "unsupported claims:\n\n" + issues,
                )

    if not grounding_source.strip() and _has_citation_markers(candidate):
        return (
            "citation_without_source",
            "The previous draft included citations or source labels, but no sources were actually supplied or retrieved.",
        )

    if not needs:
        return "ok", candidate
    if not grounding_source.strip():
        return (
            "needs_source",
            "The previous draft made factual claims without source material.",
        )

    audit = await toolserver.verify_grounding(grounding_source, candidate, session=session)
    if audit is None:
        return (
            "blocked",
            "I could not reach the verification service, so I am not going to present an unverified factual answer.",
        )
    if audit.get("grounded"):
        return "ok", candidate

    claims = (audit.get("unsupported_claims") or "").strip() or "Unsupported claims were found."
    refined = await _refine_to_source(grounding_source, candidate, claims, prose=prose, session=session)
    if refined:
        second = await toolserver.verify_grounding(grounding_source, refined, session=session)
        if second is not None and second.get("grounded"):
            return "ok", refined
    return (
        "unsupported",
        "I could not safely finalize this without unsupported claims. The verification gate flagged:\n\n"
        + claims,
    )


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

    user_final_model = (user_model or "").strip()
    is_user_model = bool(user_final_model)

    had_images = _has_images(messages)
    if had_images:
        if config.SHOW_WORK:
            yield ("reasoning", "🖼️ Reading image context…\n")
        messages = await _describe_images_for_agent(messages, session=session)

    req_headers = request_headers or {}
    chat_id = req_headers.get("x-openwebui-chat-id", "")

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
            scratch = _initial_messages(recent, user_id)
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
            scratch = _initial_messages(messages, user_id)
    else:
        scratch = _initial_messages(messages, user_id)

    # Grounding source = the user's pasted/quoted material across the WHOLE
    # conversation. verify_grounding takes the source independently of the model's
    # context budget, so a document pasted in a since-trimmed turn can still ground
    # a faithful quote; recall_context separately carries older user-stated facts.
    user_source = _user_source(messages)

    # Plain-chat fast path: stream the answer live when the turn needs no tools,
    # source, or verification. No verifier runs — there is nothing to ground or
    # audit. Anything uncertain falls through to the buffered loop below.
    if (config.STREAM_SIMPLE_CHAT and not is_user_model and not had_images
            and not user_source and not _is_application_writing(messages)
            and not await _request_needs_work(messages, session=session)):
        streamed = []
        async for kind, tok in fireworks.stream(
            scratch, config.AGENT_MODEL,
            max_tokens=config.AGENT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE, session=session,
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

    tool_sources = []
    repair_steps = 0
    tool_call_count = 0
    web_search_count = 0
    budget_note_added = False
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
        result = await fireworks.chat(
            scratch,
            model,
            max_tokens=config.AGENT_MAX_TOKENS,
            temperature=step_temp,
            session=session,
            tools=tools,
            tool_choice="auto" if tools is not None else None,
        )
        message = result.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            scratch.append(_clean_assistant_tool_message(message))
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                args = _json_args(fn.get("arguments") or "{}")
                if name == "polish":
                    polish_voice = args.get("model") or args.get("voice") or polish_voice
                    polish_voice_pass = args.get("voice_pass") or polish_voice_pass
                    note = f"Acknowledged: polish with {polish_voice}"
                    note += (f" + {polish_voice_pass} voice pass." if polish_voice_pass
                             and polish_voice_pass != "none" else ".")
                    scratch.append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or name,
                        "name": name,
                        "content": note,
                    })
                    continue
                tool_call_count += 1
                if name == "web_search":
                    web_search_count += 1
                if config.SHOW_WORK:
                    yield ("reasoning", _tool_status(name, args) + "\n")
                allowed, reason = await _tool_allowed(
                    name,
                    args,
                    messages,
                    source,
                    session=session,
                )
                if allowed:
                    raw_result = await _execute_tool(
                        name,
                        args,
                        session=session,
                        headers=request_headers,
                    )
                else:
                    raw_result = {
                        "rejected": True,
                        "tool": name,
                        "reason": reason or "tool call was not necessary for this request",
                        "instruction": "Answer the user's actual question directly without this tool.",
                    }
                source_text = _source_from_tool(name, raw_result)
                if source_text:
                    tool_sources.append(source_text)
                visible = _compact_json(_visible_tool_result(name, raw_result))
                # External/source-bearing tool output (web, fetched pages,
                # citations) is untrusted: wrap it so embedded instructions are
                # treated as data, not commands. Local tools (export/verify) are
                # not externally sourced and pass through unwrapped.
                if source_text:
                    visible = prompt_security.wrap_untrusted(name, visible)
                scratch.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or name,
                        "name": name,
                        "content": visible,
                    }
                )
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

        # Polish the final answer only when the agent asked for it (polish tool),
        # it isn't a clarifying question, AND it is substantial prose. A short
        # factual/numeric/conversational answer must not pay for a premium Opus
        # rewrite. Skip polish for user-chosen models.
        substantial = len(candidate) >= config.POLISH_MIN_CHARS
        prose = None
        if polish_voice and not is_clar and substantial and not is_user_model:
            prose = _prose_provider(polish_voice)
        if prose is not None:
            prose_client, prose_model = prose
            if config.SHOW_WORK:
                yield ("reasoning", "✨ Polishing…\n")
            try:
                polish = await prose_client.complete(
                    # messages_for_verify (= the kept tail in overflow) keeps the
                    # premium polish call bounded; the full older history would
                    # otherwise be concatenated uncapped into the prompt.
                    _prose_polish_messages(messages_for_verify, candidate, source),
                    prose_model,
                    max_tokens=config.AGENT_MAX_TOKENS,
                    temperature=config.WRITER_TEMPERATURE,
                    session=session,
                )
                if polish and polish.strip():
                    candidate = polish.strip()
            except Exception as e:
                log.warning(f"[prose_polish] {prose_model} failed, keeping open-model draft: {e}")
        # Stage 2 — optional voice-only register pass (sonnet); facts untouched.
        # A SECOND premium call, so reserve it for genuinely long-form prose.
        if (polish_voice_pass and polish_voice_pass != "none" and not is_clar
                and len(candidate) >= config.POLISH_VOICE_MIN_CHARS):
            if config.SHOW_WORK:
                yield ("reasoning", f"✨ Voice pass ({polish_voice_pass})…\n")
            candidate = await _voice_pass(candidate, polish_voice_pass, session=session)

        if config.SHOW_WORK:
            yield ("reasoning", "✍️ Verifying the answer…\n")
        status, text = await _verified_or_blocked(
            messages_for_verify,
            candidate,
            source,
            recall_context=recall_context,
            is_app=_is_application_writing(messages),
            prose=prose,
            session=session,
        )
        if status == "ok":
            # Store in memory asynchronously (don't block the response). Hold a
            # strong reference (_track_task): the event loop keeps only a weak ref
            # to a bare create_task, so an orphan store could be GC'd mid-flight
            # after the request returns and the write silently lost.
            if chat_id:
                user_memory = _consolidated_user_memory(messages)
                if user_memory:
                    _track_task(asyncio.create_task(_memory_store(chat_id, "user", user_memory, session)))
                _track_task(asyncio.create_task(_memory_store(chat_id, "assistant", text, session)))
            yield ("content", text)
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
