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
    {
        "type": "function",
        "function": {
            "name": "polish",
            "description": (
                "Polish a final written deliverable when writing quality matters (not for "
                "plain chat, quick answers, or code). Two stages, NEITHER may change or inflate facts.\n"
                "'model' — writer for the substance rewrite; pick by what THIS piece must DO, "
                "not by a static cost tier: "
                "gpt-5.5 is the default for calibrated academic/formal substance (research prose, "
                "PhD statements, academic cover letters, source-sensitive writing). Opus is the "
                "default for corporate/job-market persuasion (resume/CV bullets, recruiter-facing "
                "cover letters, pitches, bios) and when the user wants a bolder/high-visibility "
                "style. Sonnet is best as a voice-only warmth pass. The reality audit still applies "
                "after every polish.\n"
                "'voice_pass' — optional final voice-only register pass: 'warm' (personal writing), "
                "'formal' (academic/professional), or 'none'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "enum": ["gpt-5.5", "sonnet", "opus"]},
                    "voice_pass": {"type": "string", "enum": ["none", "warm", "formal"]},
                },
                "required": ["model"],
            },
        },
    },
]

SYSTEM_AGENT = (
    "You are the controller for one user-facing assistant. The user sees the "
    "answer, never the tool mechanics. Decide which tools to call, how to chain "
    "them, and when you have enough evidence to answer.\n"
    "\n"
    "TOOL USE\n"
    "- web_search: for current facts, external verification, or source-grounded "
    "research. Do not search definitions unless explicitly asked.\n"
    "- verify_grounding: before finalizing any source-bound factual writing.\n"
    "- polish: before writing a deliverable whose quality matters (cover letter, "
    "statement, application/research prose, important email). Pick the writer model "
    "by what this piece must do, using the model map in the tool description.\n"
    "- export tools: only when the user asks for a file.\n"
    "- Gather high-signal evidence, then synthesize. Do not over-search — stop "
    "once you can answer.\n"
    "\n"
    "TRUTH\n"
    "Every specific claim (number, date, credential, metric, scope, motivation, "
    "lived experience) MUST be traceable to one of: user-provided material, "
    "tool-retrieved sources, or genuine common knowledge. If a claim isn't "
    "traceable: say it's uncertain — don't invent it.\n"
    "Every citation [N] maps to a real retrieved URL. No invented sources.\n"
    "When the user says to use only provided facts: do not infer impact, scope, "
    "or mechanisms beyond those facts.\n"
    "\n"
    "WHEN ASKED TO OVERSTATE\n"
    "If the request asks you to assert things the user has NOT given — inflate "
    "experience, credentials, metrics, or leadership they never stated, or summarize "
    "or cite a source you cannot verify — do NOT fabricate, and do NOT merely refuse "
    "or ask one terse question. In ONE response: (1) briefly name what you can't "
    "assert and why (unsupported / unverifiable); (2) deliver the honest version "
    "using ONLY the real facts; (3) ask for the specific real details that would let "
    "you say more, naming them concretely; and where useful offer real, checkable "
    "alternatives. Be genuinely helpful within the truth — the honest answer should "
    "still be the most useful one in the room.\n"
    "\n"
    "OUTPUT\n"
    "- Answer directly. No preamble (\"Here's a...\"), no sign-off, no offers to "
    "tailor further.\n"
    "- Match the requested format and length exactly.\n"
    "- If you need a fact you were not given, ask the user for it — never invent it "
    "or leave a bracketed placeholder/blank.\n"
    "\n"
    "CLARIFICATION\n"
    "For formal deliverables: first decide if you can produce a credible, useful "
    "result from what the user already gave. If yes — write it immediately, ask "
    "nothing. Only ask when a single ESSENTIAL fact is missing without which the "
    "deliverable cannot be credible. Then ask exactly one short question naming "
    "only that fact, as your entire response. Never more than one question, never "
    "ask for nice-to-haves, never web-search for the missing fact."
)

SYSTEM_VISION = (
    "Transcribe and describe the image for a text-only agent. Quote all visible "
    "text exactly when possible, then summarize layout, context, and likely user "
    "intent. Do not answer the user's task; produce only faithful image context."
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

SYSTEM_APPLICATION_CLAIM_AUDIT = (
    "You are a calibrated application-writing claim auditor. The goal is NOT sterile "
    "writing: cover letters and applications should be humane, personal, persuasive, "
    "and grounded in reality. Classify claims in the DRAFT against the USER REQUEST "
    "and any SOURCE. Hard candidate claims must be explicitly supplied by the user "
    "(achievements, metrics, years, credentials, revenue, leadership, employers, "
    "projects, impact, daily routines, work settings, frequency, specific tool/product "
    "history, or concrete scenes). Company/role framing may be persuasive if it is "
    "generic or supported; specific company/product facts should be in the request/"
    "source or kept generic. Motivation/fit language is allowed when light and "
    "plausible, but it must not falsely attribute specific lived feelings, habits, "
    "personal attachment, insider knowledge, or first-person product relationships "
    "the user did not give. Flag over-personalized autobiographical texture, "
    "unsupplied habitual-use claims, unsupplied emotional-history claims, and "
    "specific company-culture/reputation claims unless supported by the request/"
    "source. Output strict JSON only: "
    "{\"unsupported_candidate_claims\":[],\"unsupported_company_claims\":[],"
    "\"fake_motivation_or_fit\":[],\"acceptable_framing\":[],"
    "\"verdict\":\"CLEAN\"|\"UNSUPPORTED\"}."
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
                timeout=aiohttp.ClientTimeout(total=config.HTTP_TIMEOUT),
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
                timeout=aiohttp.ClientTimeout(total=config.HTTP_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("stored", False)
    except Exception:
        pass
    return False


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


def _clip_memory_part(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _consolidated_user_memory(messages, user_source: str) -> str:
    """Compact per-chat memory: current ask plus the context that informed it."""
    user_texts = [
        _text_of(m.get("content")).strip()
        for m in messages
        if m.get("role") == "user" and _text_of(m.get("content")).strip()
    ]
    if not user_texts and not user_source.strip():
        return ""
    last_user = user_texts[-1] if user_texts else ""
    parts = []
    if last_user:
        parts.append("Current user request:\n" + _clip_memory_part(last_user, 3000))
    if user_source.strip() and user_source.strip() != last_user.strip():
        parts.append(
            "Relevant provided/prior context:\n" + _clip_memory_part(user_source, 6000)
        )
    return "\n\n---\n\n".join(parts).strip()


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


_PROSE_POLISH_SYS = (
    "You are a prose editor. Rewrite the DRAFT into clearer, more natural, more "
    "engaging prose for the user's request. STRICT RULES: change only wording and "
    "flow — do NOT add facts, claims, numbers, names, or experience not already in "
    "the draft or the SOURCE; do not invent. Do NOT inflate, strengthen, or reframe "
    "what the writer has done or claimed: keep the draft's level of confidence and "
    "every hedge, and never imply more was accomplished or solved than the draft "
    "states. Keep the requested format and length. No preamble, no sign-off "
    "commentary — output ONLY the rewritten deliverable."
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


_VOICE_REGISTER = {
    "warm": "natural and lightly warm, with a little personality; contractions are fine",
    "formal": "polished and formal, professional; no contractions",
}
_VOICE_PASS_SYS = (
    "You are a VOICE editor, not a content editor. Improve ONLY the voice of the DRAFT — its "
    "rhythm and naturalness, so it reads like a skilled human wrote it — at this register: "
    "{register}. STRICT: do NOT add, remove, strengthen, soften, or simplify any claim, fact, "
    "number, or hedge; do not add confidence or imply more was accomplished than stated; do not "
    "change meaning or materially change length. Output ONLY the revised text, no preamble."
)


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


async def _verified_or_blocked(messages, candidate: str, source: str, *, prose=None, session=None):
    if not config.ENABLE_VERIFICATION:
        return "ok", candidate

    # Honesty audit FIRST — independent of the grounding gate, which wrongly waves
    # through "creative writing" that inflates the user's credentials. This is the
    # founding can't-lie guarantee: a request to assert experience the user never
    # stated must not produce a confident fabrication.
    #
    # On APPLICATION writing the calibrated application-claim audit below owns this:
    # it catches the same unsupported candidate facts but ALLOWS grounded persuasive
    # framing. Running both is a double-audit (extra latency) AND lets the strict
    # honesty pass over-block the color a cover letter should have — so run exactly
    # ONE auditor: honesty for non-application deliverables, the app audit for apps.
    is_app = _is_application_writing(messages)
    if getattr(config, "ENABLE_HONESTY_AUDIT", True) and not is_app:
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

    if getattr(config, "ENABLE_APPLICATION_CLAIM_AUDIT", True) and is_app:
        full_request = _all_user_text(messages)
        app_audit = await _application_claim_audit(full_request, candidate, source, session=session)
        if app_audit and str(app_audit.get("verdict", "")).upper().startswith("UNSUPPORTED"):
            refined = await _refine_application_claims(
                full_request, candidate, source, app_audit, prose=prose, session=session
            )
            if refined:
                recheck = await _application_claim_audit(full_request, refined, source, session=session)
                if not recheck or not str(recheck.get("verdict", "")).upper().startswith("UNSUPPORTED"):
                    candidate = refined
                else:
                    issues = "\n".join(f"- {i}" for i in _application_audit_issues(app_audit))
                    return (
                        "unsupported_application_claims",
                        "I could not safely finalize this application draft without "
                        "unsupported claims:\n\n" + issues,
                    )
            else:
                issues = "\n".join(f"- {i}" for i in _application_audit_issues(app_audit))
                return (
                    "unsupported_application_claims",
                    "I could not safely finalize this application draft without "
                    "unsupported claims:\n\n" + issues,
                )

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


async def run(messages, *, user_id="", session=None, request_headers=None, user_model=""):
    """Drive one chat turn. Async generator of (kind, text)."""
    if not messages:
        yield ("content", "")
        return

    user_final_model = (user_model or "").strip()
    is_user_model = bool(user_final_model)

    if _has_images(messages):
        if config.SHOW_WORK:
            yield ("reasoning", "🖼️ Reading image context…\n")
        messages = await _describe_images_for_agent(messages, session=session)

    scratch = _initial_messages(messages, user_id)
    user_source = _user_source(messages)

    # Memory recall: inject relevant prior turns into system prompt context
    req_headers = request_headers or {}
    chat_id = req_headers.get("x-openwebui-chat-id", "")
    if chat_id:
        user_texts = [
            _text_of(m.get("content")).strip()
            for m in messages if m.get("role") == "user"
        ]
        recall_query = " ".join(user_texts)[:2000] or ""
        recalled = await _memory_recall(chat_id, recall_query, session)
        # Inject ONLY recalled USER turns (facts the user actually stated). Never
        # inject recalled ASSISTANT turns: a past "I don't have that" answer would
        # be fed back as context that instructs the model to deny again — and that
        # denial then gets stored, a self-poisoning loop. Strip the consolidated-
        # memory labels and frame the rest as established facts.
        facts, seen = [], set()
        for role, content in recalled:
            if role != "user":
                continue
            c = (content or "")
            for label in ("Current user request:", "Relevant provided/prior context:"):
                c = c.replace(label, " ")
            c = " ".join(c.split())
            if c and c not in seen:
                seen.add(c)
                facts.append(c[:500])
        if facts:
            memory_block = (
                "Earlier in THIS conversation the user already told you the following. "
                "Treat these as established facts the user has stated and use them to "
                "answer the current message:\n\n"
                + "\n".join(f"- {f}" for f in facts)
            )
            scratch.append({"role": "system", "content": memory_block})

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

        # Polish the final answer only when the agent asked for it (polish tool)
        # and it isn't a clarifying question. Skip polish for user-chosen models.
        prose = None
        if polish_voice and not is_clar:
            prose = _prose_provider(polish_voice)
        if prose is not None:
            prose_client, prose_model = prose
            if config.SHOW_WORK:
                yield ("reasoning", "✨ Polishing…\n")
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
        # Stage 2 — optional voice-only register pass (sonnet); facts untouched.
        if polish_voice_pass and polish_voice_pass != "none" and not is_clar:
            if config.SHOW_WORK:
                yield ("reasoning", f"✨ Voice pass ({polish_voice_pass})…\n")
            candidate = await _voice_pass(candidate, polish_voice_pass, session=session)

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
            # Store in memory asynchronously (don't block the response)
            if chat_id:
                user_memory = _consolidated_user_memory(messages, user_source)
                if user_memory:
                    asyncio.create_task(_memory_store(chat_id, "user", user_memory, session))
                asyncio.create_task(_memory_store(chat_id, "assistant", text, session))
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
