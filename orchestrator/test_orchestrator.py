"""Offline contract tests for the agentic orchestrator.

Run from the repo root:
  python -m orchestrator.test_orchestrator

No network and no prod. Fireworks, search, and tool-server calls are monkey
patched so these tests assert harness behavior rather than model quality.
"""
import asyncio
import json

from orchestrator import agent, config, fireworks, pipeline, search, toolserver

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


async def _fake_complete(messages, model, *, max_tokens, temperature=None, session=None):
    _calls["complete_models"].append(model)
    sys = messages[0]["content"] if messages else ""
    if model == config.VISION_MODEL:
        return "VISIBLE TEXT: Apply for this PhD by Friday.\nCONTEXT: screenshot of an application email."
    if "Decide if a draft needs grounding verification" in sys:
        value = _gate_queue.pop(0) if _gate_queue else False
        return json.dumps({"needs_verification": value, "reason": "test"})
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
    yield ("content", "vision answer")


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


async def _run_tests():
    fails = []

    def check(name, cond):
        print(f"{'PASS' if cond else 'FAIL'}: {name}")
        if not cond:
            fails.append(name)

    fireworks.chat = _fake_chat
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
    _honesty_queue.append({"unsupported": ["10 years of leadership", "$5M in revenue"], "verdict": "FABRICATION"})
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})  # recheck after refine
    ev = await _collect([{"role": "user", "content": "Write a one-line professional bio emphasizing my leadership and revenue impact."}])
    check("honesty: fabrication refined out (original claim not shown)",
          "10 years" not in _content(ev) and _content(ev) == "Corrected final answer.")

    _reset()
    _saved_repair = config.GROUNDING_REPAIR_STEPS
    config.GROUNDING_REPAIR_STEPS = 0  # surface the block now instead of re-prompting for a repair
    _chat_queue.append(_chat_content("I led a 50-person team for 12 years."))
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
    _chat_queue.append(_chat_content("Draft letter using only the user's facts."))
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
        {"status": "success", "filename": "draft.md", "mime_type": "text/markdown"},
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
    check("export: final text returned", _content(ev) == "Exported draft.md.")

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
    _SENTINELS = ["Helios", "March 3rd", "$5 million"]
    _orig_honesty, _orig_recall = agent._honesty_audit, agent._memory_recall
    _orig_store, _orig_verify = agent._memory_store, toolserver.verify_grounding

    async def _realistic_honesty(full_request, candidate, *, session=None):
        bad = [f for f in _SENTINELS if f in candidate and f not in full_request]
        return {"unsupported": bad, "verdict": "FABRICATION" if bad else "CLEAN"}

    async def _realistic_verify(source, draft, *, session=None):
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

    # B) Overflow is NOT a fabrication bypass: a fact absent from recall is still
    #    stripped. This guards the dangerous failure mode of the fix.
    _reset()
    _recall_return[:] = [("user", "my project codename is Helios and we launch March 3rd")]
    _chat_queue.append(_chat_content("Your codename is Helios and your budget is $5 million."))
    _gate_queue.append(True)
    ev = await _collect(
        _overflow_history("Remind me of my codename and budget?"),
        request_headers={"x-openwebui-chat-id": "long2"},
    )
    check("memory/overflow: recall is not a blanket fabrication bypass", "$5 million" not in _content(ev))

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

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        raise SystemExit(1)
    print("all orchestrator contract tests passed")


if __name__ == "__main__":
    asyncio.run(_run_tests())
