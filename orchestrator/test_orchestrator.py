"""Offline contract tests for the agentic orchestrator.

Run from the repo root:
  python -m orchestrator.test_orchestrator

No network and no prod. Fireworks, search, and tool-server calls are monkey
patched so these tests assert harness behavior rather than model quality.
"""
import asyncio
import json

from orchestrator import agent, config, dedup, fireworks, pipeline, search, toolserver

_calls = {
    "chat_models": [],
    "chat_messages": [],
    "complete_models": [],
    "search": [],
    "post": [],
    "verify": [],
    "fact_audit": [],
    "refine_prompts": [],
}
_chat_queue = []
_gate_queue = []
_tool_gate_queue = []
_honesty_queue = []
_verify_queue = []
_post_queue = []
_request_work_queue = []
_stream_out = []


def _tool_call(name, args, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _chat_content(text):
    return {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}


def _chat_tools(*calls):
    return {
        "message": {"role": "assistant", "content": "", "tool_calls": list(calls)},
        "finish_reason": "tool_calls",
    }


async def _fake_chat(messages, model, *, max_tokens, temperature=None, session=None, tools=None, tool_choice=None):
    _calls["chat_models"].append(model)
    _calls["chat_messages"].append(messages)
    if not _chat_queue:
        return _chat_content("")
    return _chat_queue.pop(0)


async def _fake_stream_chat(messages, model, *, max_tokens, temperature=None, session=None, tools=None, tool_choice=None):
    _calls["chat_models"].append(model)
    _calls["chat_messages"].append(messages)
    item = _chat_queue.pop(0) if _chat_queue else _chat_content("")
    msg = item.get("message", {})
    content = msg.get("content") or ""
    if content:
        yield ("content", content)
    yield ("final", {"content": content, "tool_calls": msg.get("tool_calls") or [],
                     "finish_reason": item.get("finish_reason")})


async def _fake_complete(messages, model, *, max_tokens, temperature=None, session=None):
    _calls["complete_models"].append(model)
    sys = messages[0]["content"] if messages else ""
    if model == config.VISION_MODEL:
        return "VISIBLE TEXT: Apply for this PhD by Friday.\nCONTEXT: screenshot of an application email."
    if "Decide if a draft needs grounding verification" in sys:
        value = _gate_queue.pop(0) if _gate_queue else False
        return json.dumps({"needs_verification": value, "reason": "test"})
    if "needs tools, external" in sys:
        value = _request_work_queue.pop(0) if _request_work_queue else True
        return json.dumps({"needs_work": value})
    if "Decide whether a proposed tool call is necessary" in sys:
        value = _tool_gate_queue.pop(0) if _tool_gate_queue else True
        return json.dumps({"allow": value, "reason": "test"})
    if "fact-integrity verifier" in sys:  # the one unified fact auditor
        _calls["fact_audit"].append(messages[1]["content"] if len(messages) > 1 else "")
        if _honesty_queue:
            return json.dumps(_honesty_queue.pop(0))
        return json.dumps({"unsupported": [], "verdict": "CLEAN"})
    if "Make the SMALLEST edit" in sys or "Write it again from" in sys:  # surgical or rewrite refine
        _calls["refine_prompts"].append(sys)
        return "Corrected final answer."
    return "completion"


async def _fake_stream(messages, model, *, max_tokens, temperature=None, session=None):
    _calls["chat_models"].append(model)
    for chunk in (_stream_out or ["streamed answer"]):
        yield ("content", chunk)


async def _fake_search(query, *, max_results=None, session=None):
    _calls["search"].append((query, max_results))
    return [
        {"title": "Source One", "url": "https://example.com/one", "snippet": "Verified fact one."},
        {"title": "Source Two", "url": "https://example.com/two", "snippet": "Verified fact two."},
    ]


async def _fake_post(path, payload, *, session=None, headers=None):
    _calls["post"].append((path, payload, headers or {}))
    if _post_queue:
        return _post_queue.pop(0)
    return {"ok": True}


async def _fake_verify(source, draft, *, session=None):
    _calls["verify"].append((source, draft))
    if _verify_queue:
        return _verify_queue.pop(0)
    return {"grounded": True, "unsupported_claims": ""}


async def _collect(messages, **kw):
    out = []
    async for kind, text in pipeline.run(messages, **kw):
        out.append((kind, text))
    return out


def _content(events):
    return "".join(t for k, t in events if k == "content")


def _reset():
    for value in _calls.values():
        value.clear()
    _chat_queue.clear()
    _gate_queue.clear()
    _tool_gate_queue.clear()
    _honesty_queue.clear()
    _verify_queue.clear()
    _post_queue.clear()
    _request_work_queue.clear()
    _stream_out.clear()


async def _run_tests():
    fails = []

    def check(name, cond):
        print(f"{'PASS' if cond else 'FAIL'}: {name}")
        if not cond:
            fails.append(name)

    # --- unit: OWUI file-attachment grounding via <source> blocks ---
    # OWUI delivers a paperclip upload as a <source> block (default: appended to the
    # user message). The old extractor only knew pasted prose and dropped short
    # resume lines (<120 chars), so grounded credentials looked unsupported and got
    # stripped. These lock in that the file's content now becomes grounding source.
    owui_msg = [{"role": "user", "content": (
        "Write a cover letter for this posting.\n\n"
        '<source id="1" name="Resume.docx">\n'
        "Jane Doe - Backend Engineer\n"
        "Acme Corp | Payments platform, 2M req/day\n"
        "Globex | Search relevance across 13 services\n"
        "Initech | cut infra spend ~$1.5M/year\n"
        "</source>"
    )}]
    owui_src = agent._user_source(owui_msg)
    check("source: OWUI <source> file block captured", "Acme Corp" in owui_src)
    check("source: short resume lines survive (no >=120 drop)", "Globex | Search relevance across 13 services" in owui_src)
    check("source: <source> wrapper tags stripped", "<source" not in owui_src and "</source>" not in owui_src)
    sys_inject = [
        {"role": "system", "content": '<source id="2" name="r.docx">Initech | fraud savings ~$100K/year</source>'},
        {"role": "user", "content": "Draft it."},
    ]
    check("source: <source> in system role also captured", "Initech | fraud savings" in agent._user_source(sys_inject))
    check("source: casual chat is not grounding source",
          agent._user_source([{"role": "user", "content": "what's my name?"}]) == "")

    fireworks.chat = _fake_chat
    fireworks.stream_chat = _fake_stream_chat
    fireworks.complete = _fake_complete
    fireworks.stream = _fake_stream
    search.search = _fake_search
    toolserver.post = _fake_post
    toolserver.verify_grounding = _fake_verify
    config.ENABLE_VERIFICATION = True
    config.ENABLE_GROUNDING_GATE = True
    config.ENABLE_OPENAI_PROSE = False
    config.ENABLE_ANTHROPIC_PROSE = False
    config.ENABLE_GEMINI_PROSE = False
    config.AGENT_MAX_STEPS = 6
    config.GROUNDING_REPAIR_STEPS = 2
    config.STREAM_SIMPLE_CHAT = False  # buffered loop for the existing suite; streaming has its own test
    config.STREAM_PREAMBLE = False     # off for the existing suite; preamble has its own test
    config.STREAM_ANSWER = False        # buffered/verify-before-show for the existing suite; optimistic has its own test

    tool_names = {t["function"]["name"] for t in agent.TOOL_SCHEMAS}
    check("tools: all required tools exposed", {
        "web_search",
        "fetch_url",
        "export_docx",
        "export_pdf",
        "export_markdown",
        "export_csv",
        "lookup_doi_citation",
        "search_citation",
        "verify_grounding",
    } <= tool_names)

    # Plain chat: model finalizes, gate says no verification needed.
    _reset()
    _chat_queue.append(_chat_content("Plain answer."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "hey"}])
    check("chat: returns direct answer", _content(ev) == "Plain answer.")
    check("chat: used agent model", _calls["chat_models"] == [config.AGENT_MODEL])

    # Plain-chat live streaming: request gate says no work -> answer streams live,
    # in chunks, with no buffered loop and no verifier.
    _reset()
    config.STREAM_SIMPLE_CHAT = True
    _request_work_queue.append(False)
    _stream_out[:] = ["Hi ", "there!"]
    ev = await _collect([{"role": "user", "content": "hey"}])
    check("stream: plain chat streams the answer live", _content(ev) == "Hi there!")
    check("stream: arrived as multiple content chunks", sum(1 for k, _ in ev if k == "content") >= 2)
    check("stream: no verifier ran on plain chat", _calls["fact_audit"] == [])
    config.STREAM_SIMPLE_CHAT = False

    # Fast preamble: a heavy turn streams a quick acknowledgment as reasoning first,
    # before the real answer (which still goes through the buffered loop).
    _reset()
    config.STREAM_PREAMBLE = True
    _stream_out[:] = ["Let me ", "draft that."]
    _chat_queue.append(_chat_content("Plain answer."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "write something for me"}])
    reasoning = "".join(t for k, t in ev if k == "reasoning")
    check("preamble: streamed as reasoning before the answer", "Let me draft that." in reasoning)
    check("preamble: answer still produced", _content(ev) == "Plain answer.")
    config.STREAM_PREAMBLE = False

    # Optimistic streaming: the open-model answer streams live; a clean turn shows
    # it as-is, a flagged turn shows it then openly self-corrects.
    _reset()
    config.STREAM_ANSWER = True
    _chat_queue.append(_chat_content("This is the streamed answer."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "tell me something"}])
    check("optimistic: answer streamed live (no buffering)", _content(ev) == "This is the streamed answer.")

    _reset()
    _chat_queue.append(_chat_content("Bitcoin is exactly $1 today."))
    _gate_queue.append(True)  # needs verification; unsupported stat -> blocked
    _honesty_queue.extend([{"unsupported": ["exactly $1"], "verdict": "FABRICATION"}] * 3)  # persists through patch + rewrite
    ev = await _collect([{"role": "user", "content": "what is bitcoin worth?"}])
    body = _content(ev)
    check("optimistic: streamed draft is shown", "exactly $1" in body)
    check("optimistic: then openly self-corrects", "⚠️" in body)
    config.STREAM_ANSWER = False

    # Model-driven web search: tool call first, then grounded final is verified.
    _reset()
    _chat_queue.extend([
        _chat_tools(_tool_call("web_search", {"query": "mars news", "max_results": 2})),
        _chat_content("Grounded answer [1]."),
    ])
    _gate_queue.append(True)
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})
    ev = await _collect([{"role": "user", "content": "what is the latest mars news?"}])
    body = _content(ev)
    check("agent: executed web_search tool", _calls["search"] == [("mars news", 2)])
    check("agent: final answer returned after verification", body == "Grounded answer [1].")
    check("agent: switched to grounded model after source", _calls["chat_models"] == [
        config.AGENT_MODEL,
        config.GROUNDED_MODEL,
    ])
    check("agent: verify saw tool source", "Source One" in _calls["fact_audit"][0])
    # Untrusted-context hardening (borrowed from Odysseus): web/tool source text
    # the model sees must be wrapped so embedded instructions are treated as data.
    _tool_msgs = [
        m for msgs in _calls["chat_messages"] for m in msgs if m.get("role") == "tool"
    ]
    check("security: tool output wrapped as untrusted",
          any("UNTRUSTED SOURCE DATA" in (m.get("content") or "") for m in _tool_msgs))

    # Search summaries without URLs are useful hints but not citable evidence.
    _reset()
    async def _fake_search_with_summary(query, *, max_results=None, session=None):
        _calls["search"].append((query, max_results))
        return [
            {"title": "Tavily AI summary", "url": "", "snippet": "uncited summary claim"},
            {"title": "Real Source", "url": "https://example.com/real", "snippet": "real source claim"},
        ]

    search.search = _fake_search_with_summary
    _chat_queue.extend([
        _chat_tools(_tool_call("web_search", {"query": "open model news"})),
        _chat_content("Real source claim [1]."),
    ])
    _gate_queue.append(True)
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})
    await _collect([{"role": "user", "content": "latest open model news with sources"}])
    last_context = json.dumps(_calls["chat_messages"][-1])
    check("search: uncited summary hidden from model-visible tool result", "uncited summary claim" not in last_context)
    check("search: uncited summary excluded from verification source", "uncited summary claim" not in _calls["fact_audit"][0])
    check("search: URL-backed source remains visible", "real source claim" in last_context)
    search.search = _fake_search

    # Honesty audit (the can't-lie layer): unsupported self-claims get refined out;
    # if they persist the draft is BLOCKED, never shown. The founding guarantee —
    # exercised explicitly here, not left to fail-soft.
    _reset()
    _chat_queue.append(_chat_content("I have 10 years of leadership and drove $5M in revenue."))
    _gate_queue.append(True)  # gate flags claims about the user -> audit runs
    _honesty_queue.append({"unsupported": ["10 years of leadership", "$5M in revenue"], "verdict": "FABRICATION"})
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})  # recheck after refine
    ev = await _collect([{"role": "user", "content": "Write a one-line professional bio emphasizing my leadership and revenue impact."}])
    check("honesty: fabrication refined out (original claim not shown)",
          "10 years" not in _content(ev) and _content(ev) == "Corrected final answer.")
    check("honesty: first correction is SURGICAL (minimal edit, not a rewrite)",
          _calls["refine_prompts"] and "SMALLEST edit" in _calls["refine_prompts"][0])

    _reset()
    _saved_repair = config.GROUNDING_REPAIR_STEPS
    config.GROUNDING_REPAIR_STEPS = 0  # surface the block now instead of re-prompting for a repair
    _chat_queue.append(_chat_content("I led a 50-person team for 12 years."))
    _gate_queue.append(True)  # gate flags claims about the user -> audit runs
    _honesty_queue.append({"unsupported": ["50-person team for 12 years"], "verdict": "FABRICATION"})
    # persists through BOTH the surgical patch recheck and the rewrite-with-feedback recheck
    _honesty_queue.append({"unsupported": ["50-person team for 12 years"], "verdict": "FABRICATION"})
    _honesty_queue.append({"unsupported": ["50-person team for 12 years"], "verdict": "FABRICATION"})
    ev = await _collect([{"role": "user", "content": "Write a one-line bio about my management track record."}])
    check("honesty: persistent fabrication is BLOCKED, not shown",
          "can't present those claims" in _content(ev).lower())
    config.GROUNDING_REPAIR_STEPS = _saved_repair

    # Polish chain: when the agent calls the polish tool, the chosen paid writer
    # actually runs on the final draft (and the result still passes verification).
    _reset()
    import orchestrator.openai_client as _oc
    _oc_avail, _oc_complete = _oc.available, _oc.complete
    _oc.available = lambda: True
    async def _fake_prose(messages, model, *, max_tokens, temperature=None, session=None):
        _calls.setdefault("prose", []).append(model)
        return "POLISHED FINAL LETTER."
    _oc.complete = _fake_prose
    config.ENABLE_OPENAI_PROSE = True
    _chat_queue.append(_chat_tools(_tool_call("polish", {"model": "gpt-5.5", "voice_pass": "none"})))
    # A real deliverable (>= POLISH_MIN_CHARS) — polish is gated to substantial
    # prose now, so a one-liner would (correctly) skip the premium writer.
    _chat_queue.append(_chat_content(
        "Dear Hiring Team, I am writing to apply for the data analyst role. Over the "
        "past two years I have worked as a data analyst, building dashboards and "
        "running experiments to inform decisions. I would welcome the chance to bring "
        "that experience to your team and contribute from day one. Thank you for your "
        "consideration; I look forward to hearing from you."
    ))
    ev = await _collect([{"role": "user", "content": "Write a short cover letter. My facts: 2 years as a data analyst."}])
    check("polish: paid writer ran on the final draft", "POLISHED FINAL LETTER." in _content(ev))
    check("polish: routed to the chosen gpt-5.5 writer", _calls.get("prose") == [config.OPENAI_PROSE_MODEL_PREMIUM])
    _oc.available, _oc.complete = _oc_avail, _oc_complete
    config.ENABLE_OPENAI_PROSE = False

    # If a model tries factual output with no source, the gate pushes it back
    # into the tool loop instead of showing the unverified draft.
    _reset()
    _chat_queue.extend([
        _chat_content("Bitcoin is exactly $1 today."),
        _chat_tools(_tool_call("web_search", {"query": "bitcoin price"})),
        _chat_content("The source reports the current price changes continuously."),
    ])
    _gate_queue.extend([True, True])
    # first draft's stat is unsupported and can't be patched (no source) -> blocked ->
    # repair -> search; the grounded answer after search verifies clean.
    _honesty_queue.extend([{"unsupported": ["exactly $1"], "verdict": "FABRICATION"}] * 3
                          + [{"unsupported": [], "verdict": "CLEAN"}])
    ev = await _collect([{"role": "user", "content": "what is bitcoin worth right now?"}])
    body = _content(ev)
    check("gate: blocked ungrounded factual draft", "exactly $1" not in body)
    check("gate: repair led to search", _calls["search"][0][0] == "bitcoin price")

    # If the model reaches for search on a stable conceptual explanation, the
    # harness rejects it and pushes the model back to the user's actual ask.
    _reset()
    _chat_queue.extend([
        _chat_tools(_tool_call("web_search", {"query": "prompt definition"})),
        _chat_content("Prompt engineering helps, but it cannot guarantee reliability because the model can still misunderstand, lack facts, or invent details."),
    ])
    _tool_gate_queue.append(False)
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "Explain why prompt engineering alone is not enough to make a model reliable, in plain language."}])
    body = _content(ev)
    check("tool guard: rejected unnecessary web search", _calls["search"] == [])
    check("tool guard: returns direct conceptual answer", "cannot guarantee reliability" in body)

    # Source-bound output is verified before display; unsupported draft is
    # refined and only the corrected answer is shown.
    _reset()
    _chat_queue.append(_chat_content("Rajarshi built a RAG tool and managed 25 people."))
    _gate_queue.append(True)
    _honesty_queue.extend([
        {"unsupported": ["managed 25 people"], "verdict": "FABRICATION"},
        {"unsupported": [], "verdict": "CLEAN"},  # recheck after refine
    ])
    ev = await _collect([
        {"role": "user", "content": "Notes: Rajarshi built a RAG email-triage tool."},
        {"role": "user", "content": "Write a resume bullet from those notes."},
    ])
    body = _content(ev)
    check("verify: does not show unsupported draft", "25 people" not in body)
    check("verify: shows corrected final", body == "Corrected final answer.")

    # Export tools are exposed to the model, but base64 output is compacted
    # before being fed back into the context.
    _reset()
    _post_queue.append([
        "data:text/markdown;base64,ZmFrZS1maWxl",
        {"status": "success", "filename": "draft.md", "mime_type": "text/markdown",
         "download_url": "/api/v1/files/abc123/content/draft.md"},
    ])
    _chat_queue.extend([
        _chat_tools(_tool_call("export_markdown", {"markdown": "# Draft", "filename": "draft"})),
        _chat_content("Exported draft.md."),
    ])
    _gate_queue.append(False)
    ev = await _collect(
        [{"role": "user", "content": "export this as markdown"}],
        request_headers={"x-open-webui-chat-id": "c1", "x-open-webui-message-id": "m1"},
    )
    tool_context = json.dumps(_calls["chat_messages"][-1])
    check("export: endpoint called", _calls["post"][0][0] == "/export/markdown")
    check("export: attach headers forwarded", _calls["post"][0][2]["x-open-webui-chat-id"] == "c1")
    check("export: base64 not returned to model", "ZmFrZS1maWxl" not in tool_context)
    check("export: final text returned", _content(ev).startswith("Exported draft.md."))
    check("export: download link surfaced", "/api/v1/files/abc123/content/draft.md" in _content(ev))
    check("export: file built from the deliverable, deferred to the final answer", _calls["post"][0][1]["markdown"] == "# Draft")

    # No double-dump: when a file carries the deliverable, the chat shows a link + a
    # short note — NOT a second copy of the document body (file = deliverable,
    # chat = pointer). And the file is built from the VERIFIED final text.
    _reset()
    report = ("# Quarterly Report\n\n"
              + "Revenue grew steadily across all regions this quarter. " * 12
              + "\n\nOperational costs held flat while headcount rose modestly.")
    _post_queue.append([
        {"status": "success", "filename": "report.docx",
         "download_url": "/api/v1/files/r1/content/report.docx"},
    ])
    _chat_queue.extend([
        _chat_tools(_tool_call("export_docx", {"markdown": report, "filename": "report"})),
        _chat_content(report),
    ])
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "Turn my notes into a quarterly report and export as docx."}])
    body = _content(ev)
    check("nodupe: deliverable body is NOT repeated in chat", "Revenue grew steadily" not in body)
    check("nodupe: chat shows the download link", "/api/v1/files/r1/content/report.docx" in body)
    check("nodupe: file is built from the verified deliverable", _calls["post"][0][1]["markdown"] == report)

    # Regression (the KTH cover-letter bug): the model writes the DOCUMENT in the
    # export argument and only a SUMMARY as its chat message. The file must carry the
    # DOCUMENT (verified), never the summary, and the chat must not contain the body.
    _reset()
    letter = ("Dear Admissions Committee,\n\n"
              + "I am applying for the doctoral position because my machine-learning research aligns with the project. " * 9
              + "\n\nSincerely,\nJane Doe")
    summ = "Your cover letter has been generated and exported. It opens with your interest, then three paragraphs on research fit, and closes."
    _post_queue.append([
        {"status": "success", "filename": "letter.docx",
         "download_url": "/api/v1/files/L1/content/letter.docx"},
    ])
    _chat_queue.extend([
        _chat_tools(_tool_call("export_docx", {"markdown": letter, "filename": "letter"})),
        _chat_content(summ),
    ])
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "Write a cover letter for the PhD and export as docx."}])
    body = _content(ev)
    filed = _calls["post"][0][1]["markdown"]
    check("export-arg: FILE holds the document, not the summary",
          "Dear Admissions Committee" in filed and "three paragraphs on research fit" not in filed)
    check("export-arg: chat does NOT contain the document body",
          "Dear Admissions Committee" not in body and "applying for the doctoral position" not in body)
    check("export-arg: chat shows the download link", "/api/v1/files/L1/content/letter.docx" in body)

    # A model that double-calls export with IDENTICAL args yields ONE file + ONE link.
    _reset()
    rep = "# Report\n\n" + "Quarterly numbers held steady across all regions. " * 12
    _post_queue.append([
        {"status": "success", "filename": "r.docx", "download_url": "/api/v1/files/d/content/r.docx"},
    ])
    _chat_queue.extend([
        _chat_tools(_tool_call("export_docx", {"markdown": rep, "filename": "r"}),
                    _tool_call("export_docx", {"markdown": rep, "filename": "r"}, "call_2")),
        _chat_content(rep),
    ])
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "Write a report and export as docx."}])
    body = _content(ev)
    check("export dedup: identical double export renders one file", len(_calls["post"]) == 1)
    check("export dedup: one download link", body.count("/api/v1/files/d/content/r.docx") == 1)

    # But DISTINCT files in one turn (a resume AND a cover letter) are all kept.
    _reset()
    resume = "# Resume\n\n" + "Engineer with a broad systems and ML background. " * 10
    cover = "# Cover Letter\n\n" + "I am writing to express strong interest in this role. " * 10
    _post_queue.extend([
        [{"status": "success", "filename": "resume.docx", "download_url": "/api/v1/files/a/content/resume.docx"}],
        [{"status": "success", "filename": "cover.docx", "download_url": "/api/v1/files/b/content/cover.docx"}],
    ])
    _chat_queue.extend([
        _chat_tools(_tool_call("export_docx", {"markdown": resume, "filename": "resume"}),
                    _tool_call("export_docx", {"markdown": cover, "filename": "cover"}, "call_2")),
        _chat_content("Both documents are ready."),
    ])
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "Make a resume and a cover letter, export both as docx."}])
    body = _content(ev)
    check("export: distinct files in one turn are all kept", len(_calls["post"]) == 2)
    check("export: both download links shown", "resume.docx" in body and "cover.docx" in body)

    # Vision is transcribed first, then the normal agent loop answers.
    _reset()
    _chat_queue.append(_chat_content("The image says the PhD application is due Friday."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
    ]}])
    check("vision: transcribes then answers through agent", "PhD application is due Friday" in _content(ev))
    check("vision: used vision model for transcription", _calls["complete_models"][0] == config.VISION_MODEL)
    check("vision: used agent loop after transcription",
          _calls["chat_models"] in ([config.AGENT_MODEL], [config.GROUNDED_MODEL]))
    check("vision: injected image context into agent", "VISIBLE TEXT" in json.dumps(_calls["chat_messages"][0]))

    # The verifier flags a fabricated FACT but leaves motivation/voice untouched —
    # for any kind of writing, not a hand-coded "application" category.
    _reset()
    _chat_queue.append(_chat_content(
        "I am drawn to your mission because it resonates with me. "
        "I led analytics dashboards that transformed executive decision-making."
    ))
    _honesty_queue.extend([
        {"unsupported": ["led analytics dashboards that transformed executive decision-making"], "verdict": "FABRICATION"},
        {"unsupported": [], "verdict": "CLEAN"},  # recheck after refine
    ])
    _gate_queue.append(True)
    ev = await _collect([{"role": "user", "content": (
        "Write a short cover letter for a data analyst role. Facts: 3 years SQL, dashboards."
    )}])
    check("fact verifier: a fabricated credential is refined out", _content(ev) == "Corrected final answer.")

    # Motivation / interest / enthusiasm ALONE (no fabricated facts) must NEVER block
    # or rewrite — it is not a claim that can be true or false.
    _reset()
    motiv = ("I am deeply drawn to your lab's work and eager to develop methods for this "
             "project. The mission resonates with my goals.")
    _chat_queue.append(_chat_content(motiv))
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})  # motivation is never flagged
    _gate_queue.append(True)
    ev = await _collect([{"role": "user", "content": "Write a short statement of interest. Facts: ML applicant."}])
    check("fact verifier: motivation alone does NOT block or alter the writing",
          "deeply drawn to your lab" in _content(ev) and "could not safely finalize" not in _content(ev).lower())

    # ---- Chat-memory recall as an OVERFLOW handler (not a per-turn feature) ----
    # Realistic auditors: a sentinel fact present in the DRAFT but ABSENT from the
    # text the auditor was handed is treated as a fabrication. This makes the
    # tests real instead of rubber-stamps: if recall_context fails to reach the
    # verifier, a correctly recalled fact looks invented and gets stripped, so the
    # positive assertion FAILS — which is exactly the memory-vs-verifier bug.
    _SENTINELS = ["Helios", "March 3rd", "$5 million", "$9,000"]
    _orig_fact, _orig_recall = agent._fact_audit, agent._memory_recall
    _orig_store = agent._memory_store
    _audit_inputs = []  # every request+source the unified verifier actually receives

    async def _realistic_fact(full_request, source, candidate, *, session=None):
        seen = full_request + "\n" + (source or "")
        _audit_inputs.append(seen)
        bad = [f for f in _SENTINELS if f in candidate and f not in seen]
        return {"unsupported": bad, "verdict": "FABRICATION" if bad else "CLEAN"}

    _recall_calls, _recall_return = [], []

    async def _fake_recall(chat_id, query, session=None):
        _recall_calls.append((chat_id, query))
        return list(_recall_return)

    async def _noop_store(chat_id, role, content, session=None):
        return True

    agent._fact_audit = _realistic_fact
    agent._memory_recall = _fake_recall
    agent._memory_store = _noop_store

    # Size filler to the configured budget so the test triggers overflow no matter
    # what the threshold is set to (~1.3x budget per block -> history >> budget).
    _BIG = "Filler discussion of unrelated topics. " * (config.MEMORY_CONTEXT_BUDGET_CHARS // 30)

    def _overflow_history(final_q):
        # Helios is stated FIRST, then enough filler to push it out of the kept
        # tail, so only recall can carry it into the answer + verifier.
        return [
            {"role": "user", "content": "Earlier note: my project codename is Helios and we launch March 3rd."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": _BIG},
            {"role": "assistant", "content": _BIG},
            {"role": "user", "content": final_q},
        ]

    # A) Overflow: a recalled fact survives verification (is NOT stripped).
    _reset(); _recall_calls.clear()
    _recall_return[:] = [("user", "my project codename is Helios and we launch March 3rd")]
    _chat_queue.append(_chat_content("Your project codename is Helios and the launch date is March 3rd."))
    _gate_queue.append(True)
    ev = await _collect(
        _overflow_history("What project codename and launch date did I mention earlier?"),
        request_headers={"x-openwebui-chat-id": "long1"},
    )
    out = _content(ev)
    check("memory/overflow: recall fires on a long chat", len(_recall_calls) == 1)
    # Recall must query on the CURRENT question, not be drowned out by the big
    # filler turn (the query is clipped to 2000 chars).
    _q = _recall_calls[0][1] if _recall_calls else ""
    check("memory/overflow: recall queries on the current question, not filler",
          "codename" in _q and "Filler discussion" not in _q)
    check("memory/overflow: recalled fact survives verification", "Helios" in out and "March 3rd" in out)

    # B) Overflow is NOT a fabrication bypass. A fact absent from recall must (1)
    #    never appear in the text handed to the auditors (proving recall_context
    #    carries ONLY what recall returned, not the draft), and (2) be stripped
    #    from the output. Asserting on the auditor INPUTS makes the recall plumbing
    #    the load-bearing thing under test, not the two independent backstops.
    _reset(); _audit_inputs.clear()
    _recall_return[:] = [("user", "my project codename is Helios and we launch March 3rd")]
    _chat_queue.append(_chat_content("Your codename is Helios and your budget is $5 million."))
    _gate_queue.append(True)
    ev = await _collect(
        _overflow_history("Remind me of my codename and budget?"),
        request_headers={"x-openwebui-chat-id": "long2"},
    )
    check("memory/overflow: un-recalled fact never reaches the auditors as source",
          "$5 million" not in " ".join(_audit_inputs))
    check("memory/overflow: recall is not a blanket fabrication bypass", "$5 million" not in _content(ev))

    # D) Self-grounding guard (regresses the HIGH review finding): a recalled
    #    ASSISTANT claim must NOT be laundered into grounding/established-fact
    #    context — otherwise the verifier rubber-stamps the model's own earlier
    #    output. Only USER turns become recall_context. Here recall returns an
    #    assistant claim ($9,000) and a user fact (Helios); the assistant claim
    #    must never reach the auditors and must be stripped from the answer.
    _reset(); _audit_inputs.clear()
    _recall_return[:] = [
        ("assistant", "your account balance is $9,000"),
        ("user", "my project codename is Helios and we launch March 3rd"),
    ]
    _chat_queue.append(_chat_content("Your account balance is $9,000 and your codename is Helios."))
    _gate_queue.append(True)
    ev = await _collect(
        _overflow_history("remind me of my balance and codename"),
        request_headers={"x-openwebui-chat-id": "long-d"},
    )
    check("memory/overflow: recalled ASSISTANT claim is not grounding context",
          "$9,000" not in " ".join(_audit_inputs))
    check("memory/overflow: un-grounded assistant claim is stripped, not laundered",
          "$9,000" not in _content(ev))

    # C) Normal-length chat: recall does NOT fire — native history already covers
    #    it, so custom recall would be pure redundancy.
    _reset(); _recall_calls.clear()
    _chat_queue.append(_chat_content("Hello there."))
    _gate_queue.append(False)
    await _collect(
        [{"role": "user", "content": "hi, short chat"}],
        request_headers={"x-openwebui-chat-id": "short1"},
    )
    check("memory/normal: no recall on a short chat (native history used)", _recall_calls == [])

    agent._fact_audit, agent._memory_recall = _orig_fact, _orig_recall
    agent._memory_store = _orig_store

    # ---- Request de-duplication (idempotency on retries) -----------------------
    dedup._results.clear(); dedup._inflight.clear()
    msgs = [{"role": "user", "content": "what is 2+2?"}]
    key = dedup.make_key(msgs, "PrismAI", "user-1")
    check("dedup: identical request -> same key",
          key == dedup.make_key(list(msgs), "PrismAI", "user-1"))
    check("dedup: different user -> different key",
          key != dedup.make_key(msgs, "PrismAI", "user-2"))

    # first request leads; an identical one arriving mid-flight follows the SAME future
    mode, fut = dedup.begin(key)
    check("dedup: first request is the lead", mode == "lead")
    mode2, fut2 = dedup.begin(key)
    check("dedup: concurrent identical request follows the lead", mode2 == "follow" and fut2 is fut)
    dedup.resolve(key, fut, answer="4")
    check("dedup: follower receives the lead's answer (no second run)", (await fut2) == "4")
    mode3, payload3 = dedup.begin(key)
    check("dedup: later identical request hits the completed cache", mode3 == "cached" and payload3 == "4")

    # a failed request is NOT cached: the next identical one re-runs
    dedup._results.clear(); dedup._inflight.clear()
    kerr = dedup.make_key([{"role": "user", "content": "boom"}], "PrismAI", "user-1")
    _, ferr = dedup.begin(kerr)
    dedup.resolve(kerr, ferr, exc=RuntimeError("boom"))
    try:
        ferr.exception()  # retrieve so it isn't an unhandled-exception warning
    except Exception:
        pass
    check("dedup: a failed request is not cached", dedup.get_cached(kerr) is None)
    check("dedup: after a failure the next identical request re-runs (lead)",
          dedup.begin(kerr)[0] == "lead")

    # expired entries are not served
    _saved_ttl = config.DEDUP_TTL_SECONDS
    config.DEDUP_TTL_SECONDS = -1
    ktl = dedup.make_key([{"role": "user", "content": "stale"}], "PrismAI", "user-1")
    dedup.store(ktl, "old")
    check("dedup: expired entry is not returned", dedup.get_cached(ktl) is None)
    config.DEDUP_TTL_SECONDS = _saved_ttl
    dedup._results.clear(); dedup._inflight.clear()

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        raise SystemExit(1)
    print("all orchestrator contract tests passed")


if __name__ == "__main__":
    asyncio.run(_run_tests())
