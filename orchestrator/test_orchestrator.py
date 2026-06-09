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
}
_chat_queue = []
_gate_queue = []
_tool_gate_queue = []
_app_audit_queue = []
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
    if "honesty auditor" in sys:
        if _honesty_queue:
            return json.dumps(_honesty_queue.pop(0))
        return json.dumps({"unsupported": [], "verdict": "CLEAN"})
    if "calibrated application-writing claim auditor" in sys:
        if _app_audit_queue:
            return json.dumps(_app_audit_queue.pop(0))
        return json.dumps({
            "unsupported_candidate_claims": [],
            "unsupported_company_claims": [],
            "fake_motivation_or_fit": [],
            "acceptable_framing": [],
            "verdict": "CLEAN",
        })
    if "Revise the draft" in sys or "Revise the application-writing draft" in sys:
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
    _app_audit_queue.clear()
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
    check("stream: no verifier ran on plain chat", _calls["verify"] == [])
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
    _gate_queue.append(True)  # needs verification; no source -> blocked
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
    _verify_queue.append({"grounded": True, "unsupported_claims": ""})
    ev = await _collect([{"role": "user", "content": "what is the latest mars news?"}])
    body = _content(ev)
    check("agent: executed web_search tool", _calls["search"] == [("mars news", 2)])
    check("agent: final answer returned after verification", body == "Grounded answer [1].")
    check("agent: switched to grounded model after source", _calls["chat_models"] == [
        config.AGENT_MODEL,
        config.GROUNDED_MODEL,
    ])
    check("agent: verify saw tool source", "Source One" in _calls["verify"][0][0])
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
    _verify_queue.append({"grounded": True, "unsupported_claims": ""})
    await _collect([{"role": "user", "content": "latest open model news with sources"}])
    last_context = json.dumps(_calls["chat_messages"][-1])
    check("search: uncited summary hidden from model-visible tool result", "uncited summary claim" not in last_context)
    check("search: uncited summary excluded from verification source", "uncited summary claim" not in _calls["verify"][0][0])
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

    _reset()
    _saved_repair = config.GROUNDING_REPAIR_STEPS
    config.GROUNDING_REPAIR_STEPS = 0  # surface the block now instead of re-prompting for a repair
    _chat_queue.append(_chat_content("I led a 50-person team for 12 years."))
    _gate_queue.append(True)  # gate flags claims about the user -> audit runs
    _honesty_queue.append({"unsupported": ["50-person team for 12 years"], "verdict": "FABRICATION"})
    _honesty_queue.append({"unsupported": ["50-person team for 12 years"], "verdict": "FABRICATION"})  # persists
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
    _verify_queue.append({"grounded": True, "unsupported_claims": ""})
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
    _verify_queue.extend([
        {"grounded": False, "unsupported_claims": "1. managed 25 people"},
        {"grounded": True, "unsupported_claims": ""},
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

    # Application drafts may be persuasive, but fake candidate/motivation claims
    # are refined out before display.
    _reset()
    _chat_queue.append(_chat_content(
        "Dear Stripe Team,\n\nI am drawn to your mission because it deeply resonates with me. "
        "I led analytics dashboards that transformed executive decision-making."
    ))
    _app_audit_queue.extend([
        {
            "unsupported_candidate_claims": ["led analytics dashboards that transformed executive decision-making"],
            "unsupported_company_claims": [],
            "fake_motivation_or_fit": ["deeply resonates with me"],
            "acceptable_framing": ["Stripe Team"],
            "verdict": "UNSUPPORTED",
        },
        {
            "unsupported_candidate_claims": [],
            "unsupported_company_claims": [],
            "fake_motivation_or_fit": [],
            "acceptable_framing": ["grounded role framing"],
            "verdict": "CLEAN",
        },
    ])
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": (
        "Write a short cover letter for a data analyst role at Stripe. "
        "Candidate facts: 3 years SQL, Tableau dashboards, A/B testing."
    )}])
    check("application audit: strips fake candidate/motivation claims", _content(ev) == "Corrected final answer.")

    # ---- Chat-memory recall as an OVERFLOW handler (not a per-turn feature) ----
    # Realistic auditors: a sentinel fact present in the DRAFT but ABSENT from the
    # text the auditor was handed is treated as a fabrication. This makes the
    # tests real instead of rubber-stamps: if recall_context fails to reach the
    # verifier, a correctly recalled fact looks invented and gets stripped, so the
    # positive assertion FAILS — which is exactly the memory-vs-verifier bug.
    _SENTINELS = ["Helios", "March 3rd", "$5 million", "$9,000"]
    _orig_honesty, _orig_recall = agent._honesty_audit, agent._memory_recall
    _orig_store, _orig_verify = agent._memory_store, toolserver.verify_grounding
    _audit_inputs = []  # every full_request / source the auditors actually receive

    async def _realistic_honesty(full_request, candidate, *, session=None):
        _audit_inputs.append(full_request)
        bad = [f for f in _SENTINELS if f in candidate and f not in full_request]
        return {"unsupported": bad, "verdict": "FABRICATION" if bad else "CLEAN"}

    async def _realistic_verify(source, draft, *, session=None):
        _audit_inputs.append(source)
        _calls["verify"].append((source, draft))
        bad = [f for f in _SENTINELS if f in draft and f not in source]
        return {"grounded": not bad, "unsupported_claims": ", ".join(bad)}

    _recall_calls, _recall_return = [], []

    async def _fake_recall(chat_id, query, session=None):
        _recall_calls.append((chat_id, query))
        return list(_recall_return)

    async def _noop_store(chat_id, role, content, session=None):
        return True

    agent._honesty_audit = _realistic_honesty
    agent._memory_recall = _fake_recall
    agent._memory_store = _noop_store
    toolserver.verify_grounding = _realistic_verify

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

    agent._honesty_audit, agent._memory_recall = _orig_honesty, _orig_recall
    agent._memory_store, toolserver.verify_grounding = _orig_store, _orig_verify

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
