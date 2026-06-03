"""Model-driven agent loop for the orchestrator.

The harness exposes tools and enforces verification. It does not classify the
turn into prewritten task flows; the model chooses tools, the harness executes
them, and final output is held until the grounding gate allows it.
"""
import json
import re

from . import config, fireworks, gemini, prompt_security, search, style, toolserver

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current or external facts. Returns numbered results with snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Concise search query, <= 400 characters."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a specific public URL and extract readable text. Use for exact URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 50000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_doi_citation",
            "description": "Look up a DOI in CrossRef and return APA citation metadata, with optional title verification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string"},
                    "expected_title": {"type": "string"},
                },
                "required": ["doi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_citation",
            "description": "Search CrossRef by title/author/year when the DOI is unknown. Use instead of guessing DOIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "rows": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_grounding",
            "description": "Audit a draft against provided source text and return unsupported claims. Use before finalizing source-bound factual writing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "draft": {"type": "string"},
                },
                "required": ["source", "draft"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_docx",
            "description": "Export complete final markdown as a Word .docx file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string"},
                    "filename": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["markdown"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_pdf",
            "description": "Export complete final markdown as a PDF file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string"},
                    "filename": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["markdown"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_markdown",
            "description": "Export complete final markdown as a downloadable .md file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string"},
                    "filename": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["markdown"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_csv",
            "description": "Export tabular rows as a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "filename": {"type": "string"},
                },
                "required": ["rows"],
            },
        },
    },
]

SYSTEM_AGENT = (
    "You are the controller for one user-facing assistant. The user should never "
    "see tool mechanics; they should just get the right result. Decide whether "
    "to call tools, which tools to chain, and when enough evidence exists. Do "
    "not call tools for ordinary conversation, stable conceptual explanations, "
    "editing, brainstorming, or writing from provided material. Use web_search "
    "only for current facts, source-grounded research, or external facts that "
    "must be checked; do not search definitions of terms unless the user asks "
    "for definitions. Use fetch_url for exact URLs, citation tools for DOI/"
    "reference work, export tools only when the user asks for a file, and "
    "verify_grounding before finalizing source-bound factual writing. For "
    "research, gather a small amount of high-signal evidence, then synthesize; "
    "do not keep searching once you have enough to answer. Cite current or "
    "source-grounded claims only to actual retrieved URLs; do not cite search "
    "summaries or invent a source list. "
    "Do not invent facts, citations, dates, numbers, credentials, or sources. "
    "When the user says to use only provided facts, do not infer impact, scope, "
    "mechanism, metrics, or outcomes beyond those facts; keep every concrete "
    "claim traceable to the supplied text. "
    "If evidence is insufficient, say what cannot be verified. Final answers "
    "should be polished, direct, and natural.\n"
    "Return ONLY what the user asked for: no preamble (no 'Here's a...'), no "
    "sign-off commentary, no offers to tailor it further. Match the requested "
    "format and length exactly. Add nothing beyond the given or verified facts — "
    "no invented flourish or padding."
)

SYSTEM_VISION = (
    "Answer directly and concretely. If an image is present, use it as visual "
    "context. Do not invent details you cannot see."
)

SYSTEM_GATE = (
    "Decide if a draft needs grounding verification before it can be shown. "
    "Return JSON only: {\"needs_verification\": boolean, \"reason\": string}. "
    "needs_verification=true for source-bound writing, current/external factual "
    "claims, citations, claims about user credentials/history, numbers, dates, "
    "or anything that would be a problem if fabricated. false for greetings, "
    "purely creative writing, opinion, harmless brainstorming, or simple code "
    "with no external factual claims."
)

SYSTEM_HONESTY = (
    "You are an honesty auditor for an assistant's draft written for a user. You are "
    "given the full USER REQUEST (which mixes facts the user states about themselves "
    "with instructions about what to write) and the DRAFT.\n"
    "Flag every claim in the DRAFT that asserts something about the USER — years of "
    "experience, seniority, leadership, employers, education, credentials, metrics, "
    "revenue, achievements — that the user did NOT actually state as true about "
    "themselves. CRITICAL: an INSTRUCTION to include or 'emphasize' a claim is NOT "
    "evidence the claim is true. If the user says 'emphasize my 8 years of leadership' "
    "but never states they HAVE 8 years, a draft asserting it is UNSUPPORTED. Do not "
    "flag genuine stylistic wording, the user's real stated facts, or reasonable "
    "paraphrase of them.\n"
    "Output strict JSON only: {\"unsupported\": [\"exact phrase\", ...], \"verdict\": "
    "\"FABRICATION\" if any unsupported self-claims exist, else \"CLEAN\"}."
)

SYSTEM_TOOL_GUARD = (
    "Decide whether a proposed tool call is necessary for the user's actual "
    "request. Return JSON only: {\"allow\": boolean, \"reason\": string}. "
    "Allow web_search for current/latest/recent facts, source-grounded research, "
    "citations, market/prices/schedules/laws, or external facts that must be "
    "checked. Reject web_search for stable conceptual explanations, ordinary "
    "conversation, editing, brainstorming, or writing from provided source "
    "material. Prefer answering the user's question directly over keyword-adjacent "
    "research."
)


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


def _user_source(messages) -> str:
    user_texts = [
        _text_of(m.get("content")).strip()
        for m in messages
        if m.get("role") == "user"
    ]
    if not user_texts:
        return ""
    prior = "\n\n".join(t for t in user_texts[:-1] if t).strip()
    same = _same_message_source(user_texts[-1])
    return "\n\n".join(p for p in (prior, same) if p).strip()


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


SYSTEM_PROSE_CLASSIFIER = (
    "Classify whether this user request is asking for FORMAL DELIVERABLE prose "
    "(cover letter, resume, CV, research paper, thesis, formal letter, proposal, "
    "report, personal statement, executive summary, manuscript, professional email "
    "to external parties) vs CASUAL output (conversation, brainstorming, quick "
    "answers, internal notes, code, debugging, explanations, informal chat).\n"
    "Return JSON only: {\"tier\": \"formal\" | \"casual\", \"reason\": \"brief\"}."
)


async def _classify_prose_tier(messages, *, session=None) -> str:
    """Classify if request is formal deliverable or casual. Returns 'formal' or 'casual'."""
    user_text = _last_user_text(messages)
    if not user_text:
        return "casual"
    try:
        raw = await fireworks.complete(
            [
                {"role": "system", "content": SYSTEM_PROSE_CLASSIFIER},
                {"role": "user", "content": user_text[:2000]},
            ],
            config.PROSE_CLASSIFIER_MODEL,
            max_tokens=80,
            temperature=0.0,
            session=session,
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        return "formal" if data.get("tier") == "formal" else "casual"
    except Exception:
        return "casual"


def _has_export_request(tool_calls) -> bool:
    """Check if any tool call is an export (docx/pdf/markdown/csv)."""
    for call in tool_calls or []:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        if name in {"export_docx", "export_pdf", "export_markdown", "export_csv"}:
            return True
    return False


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


async def _refine_to_source(source: str, candidate: str, claims: str, *, session=None) -> str:
    prompt = (
        "Revise the draft so every factual claim is supported by SOURCE. Remove "
        "or qualify unsupported claims; preserve the user's requested format and "
        "keep the prose polished. Output only the revised final answer."
    )
    user = f"SOURCE:\n{source}\n\nUNSUPPORTED CLAIMS:\n{claims}\n\nDRAFT:\n{candidate}"
    try:
        return await fireworks.complete(
            [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
            config.REFINE_MODEL,
            max_tokens=config.DRAFT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE,
            session=session,
        )
    except Exception:
        return ""


def _all_user_text(messages) -> str:
    """Every user turn joined — facts AND instructions. The honesty auditor needs
    the instructions too, so it can tell 'emphasize my 8 years' (an instruction)
    apart from a stated fact."""
    return "\n\n".join(
        _text_of(m.get("content")).strip()
        for m in messages
        if m.get("role") == "user" and _text_of(m.get("content")).strip()
    )


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


async def _refine_honesty(full_request: str, candidate: str, unsupported, *, session=None) -> str:
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
    try:
        return await fireworks.complete(
            [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
            config.REFINE_MODEL,
            max_tokens=config.DRAFT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE,
            session=session,
        )
    except Exception:
        return ""


def _honesty_block_msg(unsupported) -> str:
    listed = "\n".join(f"- {u}" for u in unsupported) if unsupported else ""
    return (
        "I can't include claims about you that you haven't actually told me — that "
        "would be fabrication, and avoiding exactly that is the point of this "
        "assistant. These weren't supported by anything you stated:\n\n" + listed
        + "\n\nGive me the real details (or confirm them) and I'll write it accurately."
    )


async def _verified_or_blocked(messages, candidate: str, source: str, *, session=None):
    if not config.ENABLE_VERIFICATION:
        return "ok", candidate

    # Honesty audit FIRST — independent of the grounding gate, which wrongly waves
    # through "creative writing" that inflates the user's credentials. This is the
    # founding can't-lie guarantee: a request to assert experience the user never
    # stated must not produce a confident fabrication.
    if getattr(config, "ENABLE_HONESTY_AUDIT", True):
        full_request = _all_user_text(messages)
        honesty = await _honesty_audit(full_request, candidate, session=session)
        if honesty and str(honesty.get("verdict", "")).upper().startswith("FAB"):
            unsupported = honesty.get("unsupported") or []
            refined = await _refine_honesty(full_request, candidate, unsupported, session=session)
            if refined:
                recheck = await _honesty_audit(full_request, refined, session=session)
                if not recheck or not str(recheck.get("verdict", "")).upper().startswith("FAB"):
                    candidate = refined  # cleaned draft continues through grounding checks
                else:
                    return "unsupported_self_claims", _honesty_block_msg(unsupported)
            else:
                return "unsupported_self_claims", _honesty_block_msg(unsupported)

    if not source.strip() and _has_citation_markers(candidate):
        return (
            "citation_without_source",
            "The previous draft included citations or source labels, but no sources were actually supplied or retrieved.",
        )

    needs = await _needs_verification(messages, candidate, source, session=session)
    if not needs:
        return "ok", candidate
    if not source.strip():
        return (
            "needs_source",
            "The previous draft made factual claims without source material.",
        )

    audit = await toolserver.verify_grounding(source, candidate, session=session)
    if audit is None:
        return (
            "blocked",
            "I could not reach the verification service, so I am not going to present an unverified factual answer.",
        )
    if audit.get("grounded"):
        return "ok", candidate

    claims = (audit.get("unsupported_claims") or "").strip() or "Unsupported claims were found."
    refined = await _refine_to_source(source, candidate, claims, session=session)
    if refined:
        second = await toolserver.verify_grounding(source, refined, session=session)
        if second is not None and second.get("grounded"):
            return "ok", refined
    return (
        "unsupported",
        "I could not safely finalize this without unsupported claims. The verification gate flagged:\n\n"
        + claims,
    )


async def run(messages, *, user_id="", session=None, request_headers=None):
    """Drive one chat turn. Async generator of (kind, text)."""
    if not messages:
        yield ("content", "")
        return

    if _has_images(messages):
        async for kind, text in fireworks.stream(
            _with_system(messages, SYSTEM_VISION),
            config.VISION_MODEL,
            max_tokens=config.CHAT_MAX_TOKENS,
            session=session,
        ):
            yield (kind, text)
        return

    scratch = _initial_messages(messages, user_id)
    user_source = _user_source(messages)
    tool_sources = []
    repair_steps = 0
    tool_call_count = 0
    web_search_count = 0
    budget_note_added = False
    export_requested = False
    prose_tier_cached = None

    for _ in range(config.AGENT_MAX_STEPS):
        source = _combined_source(user_source, tool_sources)
        model = _select_model(bool(source))
        tools = _budgeted_tools(tool_call_count, web_search_count)
        use_gemini = False

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

        # When generating final prose (tools exhausted), check if Gemini should be used.
        # Export-triggered (zero cost) or LLM-classified (cheap gpt-oss call).
        if tools is None and gemini.available():
            if export_requested:
                use_gemini = True
            else:
                if prose_tier_cached is None:
                    prose_tier_cached = await _classify_prose_tier(messages, session=session)
                if prose_tier_cached == "formal":
                    use_gemini = True

        # Split temperature by the turn's job: when tools are still on the table
        # this is primarily a routing/decide turn (low temp = reliable tool choice
        # + tight instruction-following); once the tool budget is exhausted the
        # turn can only write the final artifact, so use the warmer writer temp
        # for natural prose.
        step_temp = config.TOOL_TEMPERATURE if tools is not None else config.WRITER_TEMPERATURE

        if use_gemini:
            if config.SHOW_WORK:
                yield ("reasoning", "✨ Polishing…\n")
            result = await gemini.chat(
                scratch,
                config.GEMINI_PROSE_MODEL,
                max_tokens=config.AGENT_MAX_TOKENS,
                temperature=step_temp,
                session=session,
            )
        else:
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
            if _has_export_request(tool_calls):
                export_requested = True
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                args = _json_args(fn.get("arguments") or "{}")
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

        if gemini.available() and not use_gemini:
            if prose_tier_cached is None:
                prose_tier_cached = await _classify_prose_tier(messages, session=session)
            if prose_tier_cached == "formal" or export_requested:
                if config.SHOW_WORK:
                    yield ("reasoning", "✨ Polishing…\n")
                gemini_messages = [
                    {"role": m["role"], "content": _text_of(m.get("content"))}
                    for m in scratch
                    if m.get("role") in ("system", "user", "assistant")
                    and not m.get("tool_calls")
                ]
                try:
                    gemini_result = await gemini.chat(
                        gemini_messages,
                        config.GEMINI_PROSE_MODEL,
                        max_tokens=config.AGENT_MAX_TOKENS,
                        temperature=config.WRITER_TEMPERATURE,
                        session=session,
                    )
                    gemini_candidate = (gemini_result.get("message", {}).get("content") or "").strip()
                    if gemini_candidate:
                        candidate = gemini_candidate
                        if config.SHOW_WORK:
                            yield ("reasoning", "✅ Gemini polished\n")
                except Exception as e:
                    if config.SHOW_WORK:
                        yield ("reasoning", f"⚠️ Gemini failed: {e}\n")

        if config.SHOW_WORK:
            yield ("reasoning", "✍️ Writing and verifying the answer…\n")
        status, text = await _verified_or_blocked(
            messages,
            candidate,
            source,
            session=session,
        )
        if status == "ok":
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
