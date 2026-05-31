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
    "search": [],
    "post": [],
    "verify": [],
}
_chat_queue = []
_gate_queue = []
_tool_gate_queue = []
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
    sys = messages[0]["content"] if messages else ""
    if "Decide if a draft needs grounding verification" in sys:
        value = _gate_queue.pop(0) if _gate_queue else False
        return json.dumps({"needs_verification": value, "reason": "test"})
    if "Decide whether a proposed tool call is necessary" in sys:
        value = _tool_gate_queue.pop(0) if _tool_gate_queue else True
        return json.dumps({"allow": value, "reason": "test"})
    if "Revise the draft" in sys:
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

    # Vision remains a modality route, not a text-task tier.
    _reset()
    ev = await _collect([{"role": "user", "content": [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
    ]}])
    check("vision: uses vision stream path", _content(ev) == "vision answer")
    check("vision: used vision model", _calls["chat_models"] == [config.VISION_MODEL])

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        raise SystemExit(1)
    print("all orchestrator contract tests passed")


if __name__ == "__main__":
    asyncio.run(_run_tests())
