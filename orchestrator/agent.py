"""Model-driven agent loop for the orchestrator.

The harness exposes tools and enforces verification. It does not classify the
turn into prewritten task flows; the model chooses tools, the harness executes
them, and final output is held until the grounding gate allows it.
"""
import json
import logging
import re

from . import config, fireworks, gemini, openai_client, anthropic_client, prompt_security, search, style, toolserver

log = logging.getLogger(__name__)

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
    "no invented flourish or padding.\n"
    "CLARIFICATION QUESTIONS: When writing formal deliverables (cover letters, "
    "resumes, proposals, applications), if key information is missing or you would "
    "need to make assumptions, DO NOT write the deliverable with placeholders. "
    "Instead, STOP and ask 2-4 brief clarifying questions AS YOUR ENTIRE RESPONSE. "
    "For example: 'Before I write this cover letter, I need to know: 1) Why this "
    "specific role/field? 2) Which of your experiences should I emphasize?' "
    "WAIT for the user's answers. THEN write the complete deliverable with no "
    "placeholders. Never put [NEEDS DETAIL] or similar markers in formal output. "
    "Use the EXACT technical language from the user's provided materials — do not "
    "invent connections, analogies, or claims not explicitly stated."
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
    "Classify what TYPE OF OUTPUT the user is requesting — ignore how casually "
    "they asked, focus on WHAT they want produced.\n"
    "FORMAL: cover letter, resume, CV, application letter, research paper, thesis, "
    "formal letter, proposal, report, personal statement, executive summary, "
    "manuscript, professional email, grant application, PhD application.\n"
    "CASUAL: conversation, brainstorming, quick answers, internal notes, code, "
    "debugging, explanations, informal chat, questions, summaries for self.\n"
    "If the user asks you to WRITE or DRAFT something that would be sent to "
    "employers, universities, clients, or external parties → FORMAL.\n"
    "Return JSON only: {\"tier\": \"formal\" | \"casual\", \"reason\": \"brief\"}."
)

SYSTEM_QUALITY_CLASSIFIER = (
    "Given a formal writing request, classify the QUALITY TIER needed:\n"
    "- STANDARD: routine cover letters, simple proposals, internal reports, "
    "standard professional emails. Good prose is enough.\n"
    "- QUALITY: important cover letters for competitive roles, research paper "
    "abstracts, grant proposals, client-facing reports. Needs polished prose.\n"
    "- PREMIUM: executive briefs to C-suite, thesis/dissertation chapters, "
    "high-stakes research papers, board presentations, anything where prose "
    "quality is critical to outcome.\n"
    "Return JSON only: {\"quality\": \"standard\" | \"quality\" | \"premium\", "
    "\"reason\": \"brief\"}."
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
        tier = "formal" if data.get("tier") == "formal" else "casual"
        log.info(f"[prose_tier] input='{user_text[:100]}...' result={tier} raw={data}")
        return tier
    except Exception as e:
        log.warning(f"[prose_tier] error: {e}")
        return "casual"


async def _classify_quality_tier(messages, *, session=None) -> str:
    """Classify quality tier for formal prose. Returns 'standard', 'quality', or 'premium'."""
    user_text = _last_user_text(messages)
    if not user_text:
        return "standard"
    try:
        raw = await fireworks.complete(
            [
                {"role": "system", "content": SYSTEM_QUALITY_CLASSIFIER},
                {"role": "user", "content": user_text[:2000]},
            ],
            config.PROSE_CLASSIFIER_MODEL,
            max_tokens=80,
            temperature=0.0,
            session=session,
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        tier = data.get("quality", _default_quality_tier())
        tier = tier if tier in ("standard", "quality", "premium") else _default_quality_tier()
        log.info(f"[quality_tier] result={tier} raw={data}")
        return tier
    except Exception as e:
        log.warning(f"[quality_tier] error: {e}; defaulting to {_default_quality_tier()}")
        return _default_quality_tier()


def _default_quality_tier() -> str:
    """When the quality classifier is unsure/errors, prefer a tier whose provider
    is actually available — so a formal deliverable still gets real prose instead
    of falling to a 429'd provider. Opus (quality) is the strong default; fall to
    standard, then premium, by availability."""
    if anthropic_client.available():
        return "quality"
    if openai_client.available():
        return "standard"
    return "premium"


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


def _prose_provider(prose_quality):
    """Map a quality tier to (client_module, model) honoring availability, with
    graceful fallback. Returns None if no prose provider is usable (caller then
    stays on the open-model path). Keeps generation AND refine on the SAME model
    so paid prose quality isn't silently reverted to the open model on a refine.
    """
    if prose_quality == "premium" and openai_client.available():
        return openai_client, config.OPENAI_PROSE_MODEL_PREMIUM
    if prose_quality == "quality" and anthropic_client.available():
        return anthropic_client, config.ANTHROPIC_PROSE_MODEL
    if prose_quality == "standard":
        # Standard tier = Sonnet (better prose than GPT-4o per benchmarks, and on
        # the working Anthropic key); GPT-4o is only a fallback if Anthropic is down.
        if anthropic_client.available():
            return anthropic_client, config.ANTHROPIC_STANDARD_MODEL
        if openai_client.available():
            return openai_client, config.OPENAI_PROSE_MODEL
    if prose_quality and gemini.available():
        return gemini, config.GEMINI_PROSE_MODEL
    return None


_PROSE_POLISH_SYS = (
    "You are a prose editor. Rewrite the DRAFT into clearer, more natural, more "
    "engaging prose for the user's request. STRICT RULES: change only wording and "
    "flow — do NOT add facts, claims, numbers, names, or experience not already in "
    "the draft or the SOURCE; do not invent. Keep the requested format and length. "
    "No preamble, no sign-off commentary — output ONLY the rewritten deliverable."
)


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
        "I can't include claims about you that you haven't actually told me — that "
        "would be fabrication, and avoiding exactly that is the point of this "
        "assistant. These weren't supported by anything you stated:\n\n" + listed
        + "\n\nGive me the real details (or confirm them) and I'll write it accurately."
    )


async def _verified_or_blocked(messages, candidate: str, source: str, *, prose=None, session=None):
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
            refined = await _refine_honesty(full_request, candidate, unsupported, prose=prose, session=session)
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
    refined = await _refine_to_source(source, candidate, claims, prose=prose, session=session)
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

        # PROSE POLISH — runs on the FINAL answer (no tool calls this turn),
        # regardless of whether tools were still available. Formal deliverables
        # (cover letters, statements, reports) get re-written by a stronger prose
        # model; casual/chat stays on the open model. This is gated on the final
        # answer (not "tools is None") so pure-writing turns that finish on turn 1
        # still get polished. The polish is source-aware and the resulting `prose`
        # provider is reused for any verification refine so paid prose isn't
        # reverted to the open model.
        prose = None
        if not _is_clarification(candidate):
            if prose_tier_cached is None:
                prose_tier_cached = await _classify_prose_tier(messages, session=session)
            if prose_tier_cached == "formal" or export_requested:
                prose_quality = await _classify_quality_tier(messages, session=session)
                prose = _prose_provider(prose_quality)
        if prose is not None:
            prose_client, prose_model = prose
            label = {
                config.OPENAI_PROSE_MODEL_PREMIUM: "Premium polish (GPT-5.5 Pro)",
                config.ANTHROPIC_PROSE_MODEL: "Quality polish (Opus)",
                config.ANTHROPIC_STANDARD_MODEL: "Polishing (Sonnet)",
                config.OPENAI_PROSE_MODEL: "Polishing (GPT-4o)",
                config.GEMINI_PROSE_MODEL: "Polishing (Gemini)",
            }.get(prose_model, "Polishing")
            if config.SHOW_WORK:
                yield ("reasoning", f"✨ {label}…\n")
            try:
                polish = await prose_client.complete(
                    _prose_polish_messages(messages, candidate, source),
                    prose_model,
                    max_tokens=config.AGENT_MAX_TOKENS,
                    temperature=config.WRITER_TEMPERATURE,
                    session=session,
                )
                if polish and polish.strip():
                    candidate = polish.strip()
            except Exception as e:
                log.warning(f"[prose_polish] {prose_model} failed, keeping open-model draft: {e}")

        if config.SHOW_WORK:
            yield ("reasoning", "✍️ Verifying the answer…\n")
        status, text = await _verified_or_blocked(
            messages,
            candidate,
            source,
            prose=prose,
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
