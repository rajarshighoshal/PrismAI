"""Offline contract tests for the agentic orchestrator.

Run from the repo root:
  python -m orchestrator.test_orchestrator

No network and no prod. Fireworks, search, and tool-server calls are monkey
patched so these tests assert harness behavior rather than model quality.
"""
import asyncio
import json
import time

from orchestrator import agent, config, dedup, fireworks, metadata, pipeline, search, toolserver, verifier

# Real Fireworks fns captured BEFORE the per-test monkeypatching, so the provider-chain
# tests can drive the ACTUAL fallback logic (real chat/stream) with a fake HTTP session.
_REAL_FW_COMPLETE = fireworks.complete
_REAL_FW_CHAT = fireworks.chat
_REAL_FW_STREAM = fireworks.stream


class _FakeHTTPResp:
    """Stand-in aiohttp response: status>=400 -> raise_for_status throws; otherwise serves
    a JSON payload (non-stream) or SSE byte-lines (stream)."""
    def __init__(self, status, payload=None, lines=None):
        self.status = status
        self._payload = payload or {}
        self._lines = lines or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self):
        return self._payload

    @property
    def content(self):
        async def _gen():
            for ln in self._lines:
                yield ln.encode()
        return _gen()


class _FakeHTTPSession:
    """Routes each POST to a behavior fn keyed on the URL; records the URLs hit so a test
    can assert provider order and that the fallback fired."""
    def __init__(self, behavior):
        self._behavior = behavior
        self.urls = []

    def post(self, url, **kw):
        self.urls.append(url)
        return self._behavior(url)

    async def close(self):
        pass

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
_edit_intent_queue = []      # responses for the multi-turn edit classifier
_deliverable_holder = []     # the chat's "prior delivered document" (empty = none)


_last_active_holder = []     # unix time of the chat's previous turn (empty = none)
_edit_write_queue = []       # directed-edit writer responses (empty/"" = revision fails)
_patch_queue = []            # SYSTEM_EDIT_PATCH responses (in-place find/replace edits)
_plan_holder = []            # pending chunked-writer plan from _plan_get (empty = none)
_longdoc_queue = []          # SYSTEM_LONGDOC_GATE responses (bool: is it a long doc?)
_outline_queue = []          # SYSTEM_OUTLINE responses (plan dict)
_plan_intent_queue = []      # SYSTEM_PLAN_INTENT responses ({action, revision})
_section_queue = []          # SYSTEM_SECTION_WRITER responses (section markdown)
_vision_queue = []           # vision-model outputs (M3 two-part / fallback caption)


async def _fake_deliverable_get(chat_id, session=None):
    return _deliverable_holder[0] if _deliverable_holder else None


async def _fake_last_active(chat_id, session=None):
    return _last_active_holder[0] if _last_active_holder else None


async def _fake_deliverable_store(chat_id, content, filename="", fmt="", session=None):
    _calls.setdefault("deliverable_store", []).append((chat_id, content, filename, fmt))
    return True


async def _fake_plan_get(chat_id):
    return _plan_holder[0] if _plan_holder else None


async def _fake_plan_store(chat_id, plan):
    _calls.setdefault("plan_store", []).append((chat_id, plan))
    _plan_holder.clear()
    _plan_holder.append(plan)
    return True


async def _fake_plan_clear(chat_id):
    _calls.setdefault("plan_clear", []).append(chat_id)
    _plan_holder.clear()
    return True


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


async def _fake_stream_chat(messages, model, *, max_tokens, temperature=None, session=None, tools=None, tool_choice=None, label=""):
    _calls["chat_models"].append(model)
    _calls["chat_messages"].append(messages)
    item = _chat_queue.pop(0) if _chat_queue else _chat_content("")
    msg = item.get("message", {})
    content = msg.get("content") or ""
    if content:
        yield ("content", content)
    yield ("final", {"content": content, "tool_calls": msg.get("tool_calls") or [],
                     "finish_reason": item.get("finish_reason")})


async def _fc_body(messages, model, *, max_tokens, temperature=None, session=None, label="", reasoning_effort=None):
    _calls["complete_models"].append(model)
    _calls.setdefault("labels", []).append(label)
    sys = messages[0]["content"] if messages else ""
    if label == "gate:progress":
        return "Working on it"
    if label == "gate:adherence":
        return json.dumps({"followed": True, "severity": "none", "misses": []})
    if model in (config.VISION_MODEL, config.VISION_FALLBACK_MODEL):
        _calls.setdefault("vision_models", []).append(model)
        _calls.setdefault("vision_msgs", []).append(messages)
        if _vision_queue:
            return _vision_queue.pop(0)
        return "VISIBLE TEXT: Apply for this PhD by Friday.\nCONTEXT: screenshot of an application email."
    if "editing a document the user already received" in sys:  # SYSTEM_EDIT_PATCH
        return _patch_queue.pop(0) if _patch_queue else json.dumps({"broad": True})
    if "REVISION TASK" in sys:  # directed-edit writer (one plain completion, no tools)
        return _edit_write_queue.pop(0) if _edit_write_queue else ""
    if "relative to that document" in sys:  # SYSTEM_EDIT_INTENT
        return json.dumps(_edit_intent_queue.pop(0) if _edit_intent_queue else {"action": "new"})
    if "needs fact-grounding verification" in sys:  # SYSTEM_GATE
        value = _gate_queue.pop(0) if _gate_queue else False
        return json.dumps({"needs_verification": value, "reason": "test"})
    if "needs tools, external" in sys:
        value = _request_work_queue.pop(0) if _request_work_queue else True
        return json.dumps({"needs_work": value})
    if "proposed web_search is necessary" in sys:
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
    if "WRITE a long, multi-section" in sys:  # SYSTEM_LONGDOC_GATE
        v = _longdoc_queue.pop(0) if _longdoc_queue else False
        return json.dumps({"longdoc": v, "doc_type": "research paper" if v else ""})
    if "planning a long document" in sys:  # SYSTEM_OUTLINE
        plan = _outline_queue.pop(0) if _outline_queue else {
            "title": "Test Document",
            "sections": [{"heading": "Introduction", "intent": "set up the topic"},
                         {"heading": "Body", "intent": "the main argument"}]}
        return json.dumps(plan)
    if "proposed OUTLINE" in sys:  # SYSTEM_PLAN_INTENT
        return json.dumps(_plan_intent_queue.pop(0) if _plan_intent_queue
                          else {"action": "approve", "revision": ""})
    if "writing ONE section" in sys:  # SYSTEM_SECTION_WRITER
        _calls.setdefault("section_writes", []).append(sys)
        return (_section_queue.pop(0) if _section_queue
                else "## Section\n\nSection prose grounded in the provided source.")
    return "completion"


async def _fake_complete(messages, model, *, max_tokens, temperature=None, session=None,
                         label="", reasoning_effort=None, return_finish=False):
    """Wraps _fc_body to support return_finish (the auditor now reads finish_reason) and to
    simulate an UNUSABLE audit verdict: a _honesty_queue entry of '__TRUNCATED__' returns
    ('', 'length') and '__GARBAGE__' returns unparseable content — both must fail closed."""
    sys = messages[0]["content"] if messages else ""
    if "fact-integrity verifier" in sys and _honesty_queue and _honesty_queue[0] in ("__TRUNCATED__", "__GARBAGE__"):
        _calls["complete_models"].append(model)
        _calls.setdefault("labels", []).append(label)
        _calls["fact_audit"].append(messages[1]["content"] if len(messages) > 1 else "")
        sentinel = _honesty_queue.pop(0)
        content, finish = ("", "length") if sentinel == "__TRUNCATED__" else ("looks fine, no issues.", "stop")
        return (content, finish) if return_finish else content
    result = await _fc_body(messages, model, max_tokens=max_tokens, temperature=temperature,
                            session=session, label=label, reasoning_effort=reasoning_effort)
    return (result, "stop") if return_finish else result


async def _fake_stream(messages, model, *, max_tokens, temperature=None, session=None, label=""):
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
    _edit_intent_queue.clear()
    _deliverable_holder.clear()
    _last_active_holder.clear()
    _edit_write_queue.clear()
    _patch_queue.clear()
    _plan_holder.clear()
    _longdoc_queue.clear()
    _outline_queue.clear()
    _plan_intent_queue.clear()
    _section_queue.clear()
    _vision_queue.clear()
    agent._VISION_CACHE.clear()


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

    # --- unit: unwrapping OWUI's RAG template (applied even with bypass on) ---
    # Shape 1 (newer default): the real query is inside <user_query> tags.
    wrapped_uq = [{"role": "user", "content": (
        "### Task:\nRespond to the user query using the provided context.\n"
        '<context>\n<source id="1">resume text</source>\n</context>\n'
        "<user_query>\nupdate the doc, I finished my MS\n</user_query>"
    )}]
    check("unwrap: <user_query> shape yields the real message",
          agent._last_user_text(wrapped_uq) == "update the doc, I finished my MS")
    # Shape 2 (templates without a {{QUERY}} placeholder, like this instance's saved
    # default): OWUI PREPENDS the rendered template, so the real text follows </context>.
    wrapped_prepend = [{"role": "user", "content": (
        "### Task:\nRespond to the user query using the provided context, incorporating "
        "inline citations.\n### Output:\n...\n"
        '<context>\n<source id="1">resume text</source>\n</context>\n\n'
        "cna you add my geometric probes work to the letter?"
    )}]
    check("unwrap: prepended-template shape yields the text after </context>",
          agent._last_user_text(wrapped_prepend) == "cna you add my geometric probes work to the letter?")
    check("unwrap: a plain message passes through unchanged",
          agent._last_user_text([{"role": "user", "content": "hello there"}]) == "hello there")
    title_task = [{"role": "user", "content": (
        "### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing "
        "the chat history.\n\n### Chat History:\nUser: explain JAMOVI regression")}]
    tags_task = [{"role": "user", "content": (
        "### Task:\nGenerate 1-3 broad tags categorizing the main themes of the "
        "chat history.\n\n### Chat History:\nUser: explain JAMOVI regression")}]
    check("owui metadata: title prompt detected", metadata.owui_metadata_task(title_task) == "title")
    check("owui metadata: tag prompt detected", metadata.owui_metadata_task(tags_task) == "tags")
    check("owui metadata: normal user ask not detected",
          metadata.owui_metadata_task([{"role": "user", "content": "give me a short title for my essay"}]) == "")
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
    agent._deliverable_get = _fake_deliverable_get
    agent._deliverable_store = _fake_deliverable_store
    agent._last_active = _fake_last_active
    agent._plan_get = _fake_plan_get
    agent._plan_store = _fake_plan_store
    agent._plan_clear = _fake_plan_clear
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

    # Fast preamble: a heavy turn shows an INSTANT deterministic status line as reasoning
    # (no LLM call) before the real answer, so the user sees activity immediately.
    _reset()
    config.SHOW_WORK = True
    config.STREAM_PREAMBLE = True
    _chat_queue.append(_chat_content("Plain answer."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "write something for me"}])
    reasoning = "".join(t for k, t in ev if k == "reasoning")
    check("preamble: instant status is VISIBLE chat content, not thinking", "Working on it" in _content(ev))
    check("preamble: answer still produced", _content(ev).endswith("Plain answer."))
    config.STREAM_PREAMBLE = False
    config.SHOW_WORK = False

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

    # Parallel tools: several searches fired in ONE response run concurrently and all
    # results come back in a single round, not N sequential trips.
    _reset()
    _chat_queue.extend([
        _chat_tools(_tool_call("web_search", {"query": "alpha facts"}),
                    _tool_call("web_search", {"query": "beta facts"}, "call_2")),
        _chat_content("Combined answer [1]."),
    ])
    _tool_gate_queue.extend([True, True])   # guard allows both
    _gate_queue.append(True)
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})
    ev = await _collect([{"role": "user", "content": "research alpha and beta with sources"}])
    queries = sorted(q for q, _ in _calls["search"])
    check("parallel tools: both searches ran in one round", queries == ["alpha facts", "beta facts"])
    check("parallel tools: answer returned after the batch", _content(ev) == "Combined answer [1].")

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

    # Auto-polish: an EXPORTED deliverable is automatically polished by the premium
    # writer — there is no model-driven 'polish' tool (its bare "Acknowledged" made the
    # model redraft the letter 2-3 times). The model just writes; the orchestrator polishes.
    _reset()
    import orchestrator.openai_client as _oc
    _oc_avail, _oc_complete = _oc.available, _oc.complete
    _oc.available = lambda: True
    async def _fake_prose(messages, model, *, max_tokens, temperature=None, session=None, label=""):
        _calls.setdefault("prose", []).append(model)
        draft = messages[-1]["content"].split("DRAFT TO POLISH:")[-1].strip()
        return "[polished] " + draft  # preserve content so it's recognized as the same doc
    _oc.complete = _fake_prose
    config.ENABLE_OPENAI_PROSE = True
    letter = ("Dear Hiring Team, I am applying for the data analyst role. Over two years I "
              "built dashboards and ran experiments to inform decisions. " * 4)
    _post_queue.append([{"status": "success", "filename": "cl.docx", "download_url": "/api/v1/files/p/content/cl.docx"}])
    _chat_queue.extend([
        _chat_tools(_tool_call("export_docx", {"markdown": letter, "filename": "cl"})),
        _chat_content("Done — your letter is ready."),
    ])
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "Write a cover letter and export as docx. Facts: 2 years as a data analyst."}])
    filed = _calls["post"][0][1]["markdown"]
    check("auto-polish: exported deliverable polished by the premium writer", _calls.get("prose") == [config.OPENAI_PROSE_MODEL_PREMIUM])
    check("auto-polish: file holds the polished deliverable", filed.startswith("[polished]"))
    _oc.available, _oc.complete = _oc_avail, _oc_complete
    config.ENABLE_OPENAI_PROSE = False

    # If verification has to repair a premium-polished deliverable, the repair must stay
    # on the same prose model instead of falling back to the open refine model.
    _reset()
    _oc_avail, _oc_complete = _oc.available, _oc.complete
    _oc.available = lambda: True
    async def _fake_prose_repair(messages, model, *, max_tokens, temperature=None, session=None, label=""):
        if label == "refine":
            _calls.setdefault("prose_refine", []).append(model)
            return "[polished] " + letter
        _calls.setdefault("prose", []).append(model)
        draft = messages[-1]["content"].split("DRAFT TO POLISH:")[-1].strip()
        return "[polished] " + draft + " Invented award."
    _oc.complete = _fake_prose_repair
    config.ENABLE_OPENAI_PROSE = True
    _post_queue.append([{"status": "success", "filename": "cl.docx", "download_url": "/api/v1/files/p/content/cl.docx"}])
    _chat_queue.extend([
        _chat_tools(_tool_call("export_docx", {"markdown": letter, "filename": "cl"})),
        _chat_content("Done — your letter is ready."),
    ])
    _honesty_queue.extend([
        {"unsupported": ["Invented award"], "verdict": "FABRICATION"},
        {"unsupported": [], "verdict": "CLEAN"},
    ])
    await _collect([{"role": "user", "content": "Write a cover letter and export as docx. Facts: 2 years as a data analyst."}])
    filed = _calls["post"][0][1]["markdown"]
    check("auto-polish: verifier repair stays on the premium prose model",
          _calls.get("prose_refine") == [config.OPENAI_PROSE_MODEL_PREMIUM])
    check("auto-polish: repaired file comes from the premium refine pass",
          filed.startswith("[polished]") and "Invented award" not in filed)
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

    # Native vision (M3): two-part emission splits into an audit-grade transcript + a reading.
    _t, _r = agent._split_vision_output("## EVIDENCE TRANSCRIPT\nTable: X = 5 [T1]\n## READING\nX is 5 (from [T1]).")
    check("vision: two-part output splits into transcript + reading",
          "Table: X = 5 [T1]" in _t and "EVIDENCE TRANSCRIPT" not in _t and "X is 5" in _r)
    _t2, _r2 = agent._split_vision_output("Just a flat caption, no headers.")
    check("vision: a flat caption (no two-part) is used as BOTH source and reading",
          _t2 == _r2 == "Just a flat caption, no headers.")

    # The EVIDENCE TRANSCRIPT reaches the honesty auditor as SOURCE — so image claims are grounded.
    _reset()
    _vision_queue.append("## EVIDENCE TRANSCRIPT\nRevenue Q1 was $1.2M. [T1]\n\n## READING\n"
                         "The image shows Q1 revenue of $1.2M (from [T1]).")
    _chat_queue.append(_chat_content("Q1 revenue was $1.2M."))
    _gate_queue.append(True)                 # this turn IS a factual deliverable -> audit runs
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})
    ev = await _collect([{"role": "user", "content": [
        {"type": "text", "text": "summarize the revenue figure"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]}])
    check("vision: the evidence transcript reaches the auditor as grounding source",
          "Revenue Q1 was $1.2M" in "".join(_calls.get("fact_audit") or []))

    # M3 failure/empty read degrades to the fallback reader (kimi), order preserved.
    _reset()
    _vision_queue.extend(["", "FALLBACK CAPTION from the backup reader."])
    _chat_queue.append(_chat_content("Here's what it shows."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]}])
    check("vision: an empty M3 read degrades to the fallback model",
          _calls.get("vision_models") == [config.VISION_MODEL, config.VISION_FALLBACK_MODEL]
          and "FALLBACK CAPTION" in json.dumps(_calls["chat_messages"][0]))

    # detail:"high" is pinned on the image part so the provider tiles at full res (no downscale
    # confabulation — A/B-proven: default misreads small text, high reads it exactly).
    _reset()
    _vision_queue.append("## EVIDENCE TRANSCRIPT\nTitle [T1]\n## READING\nShows the title (from [T1]).")
    _chat_queue.append(_chat_content("It shows a title."))
    _gate_queue.append(False)
    await _collect([{"role": "user", "content": [
        {"type": "text", "text": "read this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,zzz"}}]}])
    check("vision: detail:high is pinned on the image sent to the reader",
          '"detail": "high"' in json.dumps(_calls.get("vision_msgs") or []))

    # The PRIMARY read gets detail:high; the FALLBACK runs WITHOUT it (fast degrade, so it
    # actually returns when the primary stalled on a huge tiled image).
    _reset()
    _vision_queue.extend(["", "FALLBACK CAPTION."])
    _chat_queue.append(_chat_content("ok"))
    _gate_queue.append(False)
    await _collect([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]}])
    _vm = _calls.get("vision_msgs") or []
    check("vision: fallback read drops detail:high (lighter/faster degrade)",
          len(_vm) == 2 and '"detail": "high"' in json.dumps(_vm[0]) and '"detail"' not in json.dumps(_vm[1]))

    # TOTAL read failure must NOT silently drop the image (the 'I don't see any images' bug):
    # the agent's context keeps an explicit 'image attached but unreadable' note.
    _reset()
    _vision_queue.extend(["", ""])           # primary AND fallback both fail/empty
    _chat_queue.append(_chat_content("I had trouble reading that image — could you re-send it?"))
    _gate_queue.append(False)
    await _collect([{"role": "user", "content": [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]}])
    check("vision: a total read failure keeps an 'image attached but unreadable' note (no silent drop)",
          "could not process it" in json.dumps(_calls.get("chat_messages") or []))

    # Vision cache: the same image is read ONCE; a re-sent image (multi-turn) reuses the
    # transcript with no second M3 call — the big multi-turn-image latency win.
    _reset()
    _vision_queue.append("## EVIDENCE TRANSCRIPT\nCached title [T1]\n## READING\nShows it (from [T1]).")
    _chat_queue.extend([_chat_content("first answer"), _chat_content("second answer")])
    _gate_queue.extend([False, False])
    _imgmsg = {"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,CACHEDIMG"}}]}
    await _collect([_imgmsg])
    _reads_after_first = len(_calls.get("vision_models") or [])
    await _collect([_imgmsg, {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "and what else is in it?"}])
    check("vision: a re-sent image is served from cache (read once, not per turn)",
          _reads_after_first == 1 and len(_calls.get("vision_models") or []) == 1)

    # #2: an image-derived answer is FORCE-audited even when the gate says 'no verify' (image
    # Q&A is otherwise classified 'answering about a file' so the transcript-as-source
    # guarantee silently never fires). With the fix, the audit runs against the transcript.
    _reset()
    _vision_queue.append("## EVIDENCE TRANSCRIPT\nRevenue was $1.2M [T1]\n## READING\n$1.2M (from [T1]).")
    _chat_queue.append(_chat_content("The image shows revenue of $1.2M."))
    _gate_queue.append(False)   # the gate would SKIP verification for image Q&A
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})
    await _collect([{"role": "user", "content": [
        {"type": "text", "text": "what's the revenue?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,REVNUM"}}]}])
    check("vision: image answer is force-audited even when the gate says no-verify (#2 honesty)",
          len(_calls.get("fact_audit") or []) >= 1)

    # review-fix #2: grounding now includes the READING's cited inference (a breed/landmark),
    # not just the literal transcript — so a legitimate visual ID isn't over-blocked.
    _reset()
    _vision_queue.append("## EVIDENCE TRANSCRIPT\nA golden long-haired dog outdoors [T1]\n"
                         "## READING\nIt's a golden retriever (from [T1]).")
    _chat_queue.append(_chat_content("It's a golden retriever."))
    _gate_queue.append(False)
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})
    await _collect([{"role": "user", "content": [
        {"type": "text", "text": "what breed is the dog?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,DOG"}}]}])
    check("vision: grounding includes the reading's cited inference, not transcript-only (#2)",
          "golden retriever" in "".join(_calls.get("fact_audit") or []))

    # review-fix #6: the vision cache is scoped by user — a DIFFERENT user re-reads the same
    # bytes (no shared audit-grade grounding source across users).
    _reset()
    _vision_queue.extend(["## EVIDENCE TRANSCRIPT\nX [T1]\n## READING\nx (from [T1]).",
                          "## EVIDENCE TRANSCRIPT\nX [T1]\n## READING\nx (from [T1])."])
    _chat_queue.extend([_chat_content("a1"), _chat_content("b1")])
    _gate_queue.extend([False, False])
    _uimg = {"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,USERSCOPED"}}]}
    await _collect([_uimg], user_id="alice")
    await _collect([_uimg], user_id="bob")
    check("vision: cache is user-scoped (a different user re-reads, no cross-user share)",
          len(_calls.get("vision_models") or []) == 2)

    # ── Auditor FAILS CLOSED: an unusable verdict blocks, never silently 'clean' (task #30) ──
    _reset()
    _honesty_queue.append("__TRUNCATED__")           # verdict truncated (finish=length)
    status, text = await verifier._verified_or_blocked(
        [{"role": "user", "content": "write about revenue"}],
        "Revenue grew 50% last year.", "Some source material.", force=True, session=None)
    check("audit fail-closed: a TRUNCATED verdict blocks (not read as grounded)",
          status == "audit_unavailable" and "honesty check" in text.lower())

    _reset()
    _honesty_queue.extend(["__GARBAGE__", "__GARBAGE__"])   # unparseable on both attempts
    status, _t = await verifier._verified_or_blocked(
        [{"role": "user", "content": "x"}], "A factual claim.", "Source.", force=True, session=None)
    check("audit fail-closed: an UNPARSEABLE verdict (after retry) blocks", status == "audit_unavailable")

    _reset()
    _honesty_queue.extend(["__GARBAGE__", {"unsupported": [], "verdict": "CLEAN"}])  # garble then real verdict
    status, _t = await verifier._verified_or_blocked(
        [{"role": "user", "content": "x"}], "A factual claim.", "Source.", force=True, session=None)
    check("audit fail-closed: a TRANSIENT garble recovers on retry (not over-blocked)", status == "ok")

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

    # REGRESSION — the over-strip bug: the auditor mis-flags GROUNDED credentials whose
    # exact phrase is right there in the source (a lossy distilled list used to cause
    # this). The verbatim backstop recognizes each as literally present and KEEPS it — a
    # real credential is never stripped just because an LLM mis-flagged it. Every flag
    # here appears verbatim, so nothing genuine remains -> NO refine call at all.
    _reset()
    grounded_letter = (
        "At Clover Health I built a retrieval system delivering a 16,631x speedup, "
        "work accepted to ICLR 2026, building on prior roles at Microsoft and IBM."
    )
    _chat_queue.append(_chat_content(grounded_letter))
    _honesty_queue.append({
        "unsupported": ["16,631x speedup", "accepted to ICLR 2026", "Microsoft and IBM"],
        "verdict": "FABRICATION",
    })
    _gate_queue.append(True)
    ev = await _collect([{"role": "user", "content": (
        '<source id="1" name="resume.docx">Clover Health: built a retrieval system with a '
        "16,631x speedup, accepted to ICLR 2026. Prior roles at Microsoft and IBM.</source>\n"
        "Write a cover letter from my resume."
    )}])
    body = _content(ev)
    check("backstop: a verbatim-grounded credential the auditor mis-flagged is NOT stripped",
          "16,631x" in body and "ICLR 2026" in body and "Microsoft and IBM" in body)
    check("backstop: all flags verbatim in source -> no wasted refine cycle",
          not _calls["refine_prompts"])

    # The OTHER direction — the honesty guarantee: a same-vocabulary INFLATION must NOT
    # be rescued by the backstop. The source says 'collaborated'; the draft claims 'led'.
    # The flagged phrase reuses source words but is NOT a contiguous span of the source,
    # so the verbatim check returns False, the auditor's flag stands, and it is refined
    # out. (A loose word-overlap score would have wrongly kept this lie.)
    _reset()
    _chat_queue.append(_chat_content("I led the product architecture team that rebuilt the system."))
    _honesty_queue.extend([
        {"unsupported": ["led the product architecture team"], "verdict": "FABRICATION"},
        {"unsupported": [], "verdict": "CLEAN"},  # recheck after refine
    ])
    _gate_queue.append(True)
    ev = await _collect([{"role": "user", "content": (
        '<source id="1" name="resume.docx">Collaborated with the product team on system '
        "architecture improvements.</source>\nWrite a cover letter from my resume."
    )}])
    check("honesty: a same-vocabulary inflation ('led' for 'collaborated') is still stripped",
          _content(ev) == "Corrected final answer." and _calls["refine_prompts"])

    # USER'S OWN WORDS ground their facts: a claim the user STATED in chat (not in the
    # uploaded files) must NOT be stripped as unsupported — the exact bug that 'corrected
    # out' the user's real geometric-probes research because it wasn't in the .tex source.
    _reset()
    _chat_queue.append(_chat_content("I currently work on geometric probes for AI safety."))
    _honesty_queue.append({"unsupported": ["geometric probes for AI safety"], "verdict": "FABRICATION"})
    _gate_queue.append(True)
    ev = await _collect([{"role": "user", "content": (
        '<source id="1" name="cv.docx">Rajarshi Ghoshal — ML engineer, Georgia Tech.</source>\n'
        "Add that I currently work on geometric probes for AI safety to my cover letter."
    )}])
    check("grounding: a fact the USER stated in chat is grounded, not stripped",
          "geometric probes for AI safety" in _content(ev) and not _calls["refine_prompts"])
    # The auditor must also know today's date — the writer is told it (system prompt), so
    # a dated letterhead is established context, not a fabrication to strip into "[Date]".
    check("grounding: the auditor is told the current date",
          _calls["fact_audit"] and "current date" in _calls["fact_audit"][0].lower())

    # SELECTIVE VERIFICATION: a casual turn that merely HAS an attachment (assessing it,
    # asking about it) is not a deliverable. The gate says no and there is no export, so
    # the honesty pass must NOT run and must NOT strip a reasonable aside ("like Keybr").
    # (The old "any source present -> always audit" rule fired on every such chat.)
    _reset()
    _chat_queue.append(_chat_content(
        "Yeah, solid typing tutor — for a free tool it covers the fundamentals well, "
        "though it won't have the adaptive drills of paid apps like Keybr."
    ))
    _gate_queue.append(False)  # classifier: opinion about an attachment, not a deliverable
    ev = await _collect([{"role": "user", "content": (
        '<source id="1" name="screenshot">A web-based touch-typing tutor showing the home row.'
        "</source>\nis this helpful to type better?"
    )}])
    check("selective: casual Q&A about an attachment is NOT fact-audited",
          not _calls["fact_audit"] and "Keybr" in _content(ev))

    # ...but an EXPORTED file is always a deliverable: verify even if the gate flakes to
    # 'no', because the user will rely on the document.
    _reset()
    _chat_queue.extend([
        _chat_tools(_tool_call("export_docx", {"markdown": "I have 12 years at Meta.", "filename": "bio"})),
        _chat_content("Done — file ready."),
    ])
    _post_queue.append([{"status": "success", "filename": "bio.docx", "download_url": "/api/v1/files/a/content/bio.docx"}])
    _gate_queue.append(False)  # gate flakes to 'no'...
    _honesty_queue.append({"unsupported": ["12 years at Meta"], "verdict": "FABRICATION"})  # ...but export forces the audit
    _honesty_queue.append({"unsupported": [], "verdict": "CLEAN"})
    ev = await _collect([{"role": "user", "content": "Make a short bio and export it as docx."}])
    check("selective: an exported file is verified even when the gate says no",
          _calls["fact_audit"] and _calls["refine_prompts"])

    # ── Multi-turn edit engine: the action is PROPORTIONAL to the request ───────────
    # RENAME: re-package the already-verified bytes under a new name — NO writer, NO
    # verifier (nothing changed to re-check). The "just rename it" cut.
    _reset()
    _deliverable_holder[:] = [{"content": "Dear Committee,\n\nverified letter body.", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "rename", "filename": "KTH_Cover_Letter", "format": ""})
    _post_queue.append([{"status": "success", "filename": "KTH_Cover_Letter.docx",
                         "download_url": "/api/v1/files/a/content/KTH_Cover_Letter.docx"}])
    ev = await _collect([{"role": "user", "content": "rename it to KTH_Cover_Letter"}],
                        request_headers={"x-openwebui-chat-id": "edit1"})
    body = _content(ev)
    check("edit/rename: re-exports under the new name with a download link",
          "Renamed" in body and "KTH_Cover_Letter.docx" in body)
    check("edit/rename: the writer never runs (pure re-package)", not _calls["chat_models"])
    check("edit/rename: no honesty audit (bytes unchanged)", not _calls["fact_audit"])

    # REFORMAT: same content, new file type — also mechanical, no writer.
    _reset()
    _deliverable_holder[:] = [{"content": "verified body text here", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "reformat", "filename": "", "format": "pdf"})
    _post_queue.append([{"status": "success", "filename": "letter.pdf",
                         "download_url": "/api/v1/files/a/content/letter.pdf"}])
    ev = await _collect([{"role": "user", "content": "give me a pdf version"}],
                        request_headers={"x-openwebui-chat-id": "edit2"})
    body = _content(ev)
    check("edit/reformat: re-exports in the new format, no writer",
          "PDF" in body and "letter.pdf" in body and not _calls["chat_models"])

    # CONTENT EDIT — the DIRECTED pipeline: one writer call on the REAL prior document,
    # verify, then the HARNESS exports with the stored filename/format. No tool-call hope
    # anywhere (live smoke proved the model skips export ~half the time when asked to).
    _reset()
    _deliverable_holder[:] = [{"content": "Dear Committee,\n\nI finish my MS in May 2026.", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "edit", "filename": "", "format": ""})
    _edit_write_queue.append("Dear Committee,\n\nI finished my MS in May 2026.")
    _post_queue.append([{"status": "success", "filename": "letter.docx", "download_url": "/api/v1/files/d/content/letter.docx"}])
    ev = await _collect([{"role": "user", "content": "change 'I finish' to 'I finished'"}],
                        request_headers={"x-openwebui-chat-id": "edit3"})
    check("edit/content: directed pipeline ships the revised file with no agent loop",
          not _calls["chat_models"] and _calls["post"]
          and "I finished my MS" in _calls["post"][0][1]["markdown"]
          and "letter.docx" in _content(ev))

    # IN-PLACE patch (Canvas/Artifact way): a surgical change is applied as a find→replace
    # to the stored document — no full re-emit (the doc is never regenerated).
    _reset()
    _deliverable_holder[:] = [{"content": "Dear Committee, I am applying for the doctoral role at KTH.", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "edit", "filename": "", "format": ""})
    _patch_queue.append(json.dumps({"broad": False, "edits": [{"find": "doctoral role", "replace": "doctoral position"}]}))
    _post_queue.append([{"status": "success", "filename": "letter.docx", "download_url": "/api/v1/files/p/content/letter.docx"}])
    ev = await _collect([{"role": "user", "content": "change 'doctoral role' to 'doctoral position'"}],
                        request_headers={"x-openwebui-chat-id": "patch1"})
    filed = _calls["post"][0][1]["markdown"] if _calls["post"] else ""
    check("edit/patch: a surgical edit is applied in-place, NOT re-emitted",
          "doctoral position" in filed and "doctoral role" not in filed
          and "edit:patch" in _calls.get("labels", []) and "edit:write" not in _calls.get("labels", []))

    # BROAD edit -> the patch model signals 'broad', and we fall back to a full re-emit.
    _reset()
    base = "Dear Committee, I have built ML systems and data science tools for years. " * 3
    _deliverable_holder[:] = [{"content": base, "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "edit", "filename": "", "format": ""})
    _patch_queue.append(json.dumps({"broad": True}))
    _edit_write_queue.append(base.replace("data science", "applied machine learning"))
    _post_queue.append([{"status": "success", "filename": "letter.docx", "download_url": "/api/v1/files/b/content/letter.docx"}])
    ev = await _collect([{"role": "user", "content": "rework it to emphasize applied ML throughout"}],
                        request_headers={"x-openwebui-chat-id": "patch2"})
    check("edit/patch: a broad edit falls back to a full re-emit and still ships",
          "edit:write" in _calls.get("labels", []) and _calls["post"]
          and "applied machine learning" in _calls["post"][0][1]["markdown"])

    # A surgical edit must NOT re-polish / re-voice the whole document: v1 was already
    # polished and voiced; re-running both wastes ~40s and rewrites text the user didn't
    # ask to change. (A NEW export still polishes — the auto-polish test above.)
    _reset()
    import orchestrator.openai_client as _oc2
    _oc2_avail, _oc2_complete = _oc2.available, _oc2.complete
    _oc2.available = lambda: True
    async def _fake_prose2(messages, model, *, max_tokens, temperature=None, session=None, label=""):
        _calls.setdefault("prose", []).append(model)
        return "[re-polished] should not happen"
    _oc2.complete = _fake_prose2
    config.ENABLE_OPENAI_PROSE = True
    prior_letter = ("Dear Committee, I am applying for the doctoral role. My work spans ML "
                    "engineering and parallel algorithms across industry and research. " * 4)
    _deliverable_holder[:] = [{"content": prior_letter, "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "edit", "filename": "", "format": ""})
    edited_letter = prior_letter.replace("doctoral role", "doctoral position")
    _post_queue.append([{"status": "success", "filename": "letter.docx", "download_url": "/api/v1/files/e/content/letter.docx"}])
    _edit_write_queue.append(edited_letter)
    ev = await _collect([{"role": "user", "content": "change 'doctoral role' to 'doctoral position' in the letter"}],
                        request_headers={"x-openwebui-chat-id": "edit6"})
    check("edit/no-repolish: a surgical edit is NOT re-polished or re-voiced",
          not _calls.get("prose"))
    check("edit/no-repolish: the edited file still ships",
          _calls["post"] and "doctoral position" in _calls["post"][0][1]["markdown"])

    # COLLABORATE: on an ambiguous edit the writer may ASK instead of guess — the
    # question ships straight to the user (no export, no reconstruction), and the chat
    # keeps the document stored for their answer.
    _reset()
    _deliverable_holder[:] = [{"content": "Dear Committee, I have built ML systems for years.", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "edit", "filename": "", "format": ""})
    _edit_write_queue.append("Do you want 'data science' replaced everywhere, or only in the experience section?")
    ev = await _collect([{"role": "user", "content": "make the wording consistent with the thing I said"}],
                        request_headers={"x-openwebui-chat-id": "edit9"})
    check("edit/collab: an ambiguous edit yields a clarifying question, not a guess",
          "replaced everywhere" in _content(ev) and not _calls["post"])

    # TEXTUAL TOOL CALL: the model writes "<tool_calls>…" as plain text (DeepSeek leak —
    # live report: raw markup shipped as the answer, nothing executed). The harness
    # nudges once; the model re-issues it properly and the real search runs.
    _reset()
    _chat_queue.extend([
        _chat_content('<tool_calls> <tool_call name="web_search"> {"queries": ["fable 5 benchmarks"]} </tool_call> </tool_calls>'),
        _chat_tools(_tool_call("web_search", {"query": "fable 5 benchmarks"})),
        _chat_content("Fable 5 scores ~80% on Magenta Code [1]."),
    ])
    _tool_gate_queue.append(True)
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "pull the benchmark details"}])
    body = _content(ev)
    check("textual-tool: leaked text call is nudged into a REAL call and the answer ships",
          _calls["search"] and "Magenta Code" in body and "<tool_call" not in body)

    # DOUBLE-VOTE: a flaky 'new' verdict must win twice — the second vote rescues an
    # edit (live smoke caught the classifier flaking ~1 in 4 on typo-heavy edit messages,
    # silently dropping the document). First vote new, second edit -> directed pipeline.
    _reset()
    _deliverable_holder[:] = [{"content": "Dear Committee,\n\nI study probes in ML systems today.", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.extend([{"action": "new"}, {"action": "edit", "filename": "", "format": ""}])
    _edit_write_queue.append("Dear Committee,\n\nI study geometric probes in ML systems today.")
    _post_queue.append([{"status": "success", "filename": "letter.docx", "download_url": "/api/v1/files/v/content/letter.docx"}])
    ev = await _collect([{"role": "user", "content": "also add taht I work on geormetric probes"}],
                        request_headers={"x-openwebui-chat-id": "edit8"})
    check("edit/double-vote: a flaky 'new' is overruled by the second vote and the edit ships",
          _calls["post"] and "geometric probes" in _calls["post"][0][1]["markdown"])

    # FALLBACK: if the directed writer fails (empty / not the document), the turn falls
    # through to the agent loop with the injected doc — and if the model 'finishes' there
    # WITHOUT re-exporting, the harness nudges once and the revised file still ships.
    _reset()
    _deliverable_holder[:] = [{"content": "Dear Committee, I expect to finish my MS in May 2026.", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "edit", "filename": "", "format": ""})
    _edit_write_queue.append("")  # directed revision fails -> fall through to the loop
    _post_queue.append([{"status": "success", "filename": "letter.docx", "download_url": "/api/v1/files/n/content/letter.docx"}])
    _chat_queue.extend([
        _chat_content("Done — I've updated that line for you."),  # no export call!
        _chat_tools(_tool_call("export_docx", {"markdown": "Dear Committee, I finished my MS in May 2026.", "filename": "letter"})),
        _chat_content("Updated file ready."),
    ])
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "I already finished my MS — update the doc"}],
                        request_headers={"x-openwebui-chat-id": "edit7"})
    check("edit/enforce-export: a no-export edit is nudged and the revised file ships",
          _calls["post"] and "I finished my MS" in _calls["post"][0][1]["markdown"]
          and "letter.docx" in _content(ev))
    _oc2.available, _oc2.complete = _oc2_avail, _oc2_complete
    config.ENABLE_OPENAI_PROSE = False

    # NEW: an unrelated follow-up must NOT be hijacked into a revision.
    _reset()
    _deliverable_holder[:] = [{"content": "prior letter", "filename": "letter", "fmt": "docx"}]
    _edit_intent_queue.append({"action": "new"})
    _chat_queue.append(_chat_content("Here is a fresh recommendation letter."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "write a recommendation letter instead"}],
                        request_headers={"x-openwebui-chat-id": "edit4"})
    check("edit/new: an unrelated request is not hijacked into a revision",
          "REVISION TASK" not in json.dumps(_calls["chat_messages"])
          and _content(ev) == "Here is a fresh recommendation letter.")

    # Resume-after-gap: a day+ gap injects a one-line note so the model knows time passed;
    # a same-day follow-up stays clean (no marker on a continuous session).
    _reset()
    _last_active_holder[:] = [time.time() - 3 * 86400]  # previous message ~3 days ago
    _chat_queue.append(_chat_content("Welcome back — where were we?"))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "ok where were we?"}],
                        request_headers={"x-openwebui-chat-id": "gap1"})
    check("gap: a multi-day gap injects a 'resuming after a gap' note",
          "resuming this conversation after a gap" in json.dumps(_calls["chat_messages"]))

    _reset()
    _last_active_holder[:] = [time.time()]  # same instant -> same day
    _chat_queue.append(_chat_content("Sure."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "one more thing"}],
                        request_headers={"x-openwebui-chat-id": "gap2"})
    check("gap: a same-day follow-up gets no gap note",
          "resuming this conversation after a gap" not in json.dumps(_calls["chat_messages"]))


    # Streaming guard: a mid-stream crash becomes a graceful message, never a broken
    # chunked response (which OWUI surfaces as a raw TransferEncodingError).
    async def _boom(messages, **kw):
        yield "content", "partial answer"
        raise RuntimeError("upstream died")
        yield  # pragma: no cover (marks this an async generator)
    _orig_agent_run = pipeline._agent_run
    pipeline._agent_run = _boom
    try:
        out = [(k, t) async for k, t in pipeline.run([{"role": "user", "content": "hi"}])]
    finally:
        pipeline._agent_run = _orig_agent_run
    guard_body = "".join(t for k, t in out if k == "content")
    check("guard: a mid-stream crash yields a graceful retry message, not a broken stream",
          "partial answer" in guard_body and "try again" in guard_body)

    # PrismAI is the only advertised orchestrator alias. Image handling is automatic
    # from attachments, so there is no separate vision model.
    async def _stub_agent_run(messages, **kw):
        yield "content", "agent"

    async def _stub_raw_chat(messages, model, *, session=None):
        yield "content", f"raw:{model}"

    _orig_agent_run = pipeline._agent_run
    _orig_raw_chat = pipeline._raw_chat
    pipeline._agent_run = _stub_agent_run
    pipeline._raw_chat = _stub_raw_chat
    try:
        chat_ev = await _collect([{"role": "user", "content": "hi"}],
                                 user_model=config.ADVERTISED_CHAT_ID)
        raw_model = "accounts/fireworks/models/kimi-k2p6"
        raw_ev = await _collect([{"role": "user", "content": "hi"}], user_model=raw_model)
    finally:
        pipeline._agent_run = _orig_agent_run
        pipeline._raw_chat = _orig_raw_chat
    check("routing: advertised chat model uses the orchestrator path",
          _content(chat_ev) == "agent")
    check("routing: real provider model uses raw pass-through",
          _content(raw_ev) == f"raw:{raw_model}")

    # ---- Chat-memory recall as an OVERFLOW handler (not a per-turn feature) ----
    # Realistic auditors: a sentinel fact present in the DRAFT but ABSENT from the
    # text the auditor was handed is treated as a fabrication. This makes the
    # tests real instead of rubber-stamps: if recall_context fails to reach the
    # verifier, a correctly recalled fact looks invented and gets stripped, so the
    # positive assertion FAILS — which is exactly the memory-vs-verifier bug.
    _SENTINELS = ["Helios", "March 3rd", "$5 million", "$9,000"]
    _orig_fact, _orig_recall = verifier._fact_audit, agent._memory_recall
    _orig_store = agent._memory_store
    _audit_inputs = []  # every request+source the unified verifier actually receives

    async def _realistic_fact(full_request, source, candidate, *, session=None, raw_source=None):
        seen = full_request + "\n" + (raw_source or source or "")
        _audit_inputs.append(seen)
        bad = [f for f in _SENTINELS if f in candidate and f not in seen]
        return {"unsupported": bad, "verdict": "FABRICATION" if bad else "CLEAN"}

    _recall_calls, _recall_return = [], []

    async def _fake_recall(chat_id, query, session=None):
        _recall_calls.append((chat_id, query))
        return list(_recall_return)

    async def _noop_store(chat_id, role, content, session=None):
        return True

    verifier._fact_audit = _realistic_fact
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

    verifier._fact_audit, agent._memory_recall = _orig_fact, _orig_recall
    agent._memory_store = _orig_store

    # ---- Request de-duplication (idempotency on retries) -----------------------
    dedup._results.clear(); dedup._inflight.clear()
    msgs = [{"role": "user", "content": "what is 2+2?"}]
    key = dedup.make_key(msgs, "PrismAI", "user-1")
    check("dedup: identical request -> same key",
          key == dedup.make_key(list(msgs), "PrismAI", "user-1"))
    check("dedup: different user -> different key",
          key != dedup.make_key(msgs, "PrismAI", "user-2"))
    # Identical text in a DIFFERENT chat must run fresh: its side effects (deliverable
    # store, memory) belong to that chat — a cached answer would leave it with no
    # document to edit (caught live by smoke round 6's 0-second cache hit).
    check("dedup: different chat -> different key",
          dedup.make_key(msgs, "PrismAI", "user-1", "chat-a")
          != dedup.make_key(msgs, "PrismAI", "user-1", "chat-b"))

    # ── Provider chain: DeepSeek-direct primary, Fireworks fallback (deepseek-only) ──
    _dsk, _fwk = config.DEEPSEEK_API_KEY, config.FIREWORKS_API_KEY
    config.FIREWORKS_API_KEY = "fwkey"
    config.DEEPSEEK_API_KEY = ""
    check("provider: no DeepSeek key -> Fireworks only",
          [p[3] for p in fireworks._providers("accounts/fireworks/models/deepseek-v4-pro")] == [False])
    config.DEEPSEEK_API_KEY = "dskey"
    _chain = fireworks._providers("accounts/fireworks/models/deepseek-v4-pro")
    check("provider: deepseek model -> DeepSeek-direct THEN Fireworks, prefix stripped",
          [p[3] for p in _chain] == [True, False] and _chain[0][2] == "deepseek-v4-pro")
    check("provider: non-deepseek model -> Fireworks only even with a DeepSeek key",
          [p[3] for p in fireworks._providers("accounts/fireworks/models/kimi-k2p6")] == [False])

    # ── Reasoning policy: MAX for substantive roles, fast for classifiers (label-based) ──
    _saved_re = config.REASONING_EFFORT
    config.REASONING_EFFORT = "max"
    try:
        check("effort: substantive label -> config max", fireworks._effort_for("agent", None) == "max")
        check("effort: gate:* classifier -> none", fireworks._effort_for("gate:tool", None) == "none")
        check("effort: summarize classifier -> none", fireworks._effort_for("summarize", None) == "none")
        check("effort: explicit value always wins", fireworks._effort_for("gate:tool", "low") == "low")
        check("effort: casual chat -> CHAT_REASONING_EFFORT (fast tier, not max)",
              fireworks._effort_for("chat", None) == config.CHAT_REASONING_EFFORT
              and config.CHAT_REASONING_EFFORT != "max")

        def _pay(model, effort, is_ds):
            return fireworks._chat_payload(model, messages=[], max_tokens=8, temperature=0.0,
                                           session_id="s", effort=effort, is_ds=is_ds)
        # DeepSeek-direct: substantive -> thinking enabled + reasoning_effort; classifier -> disabled.
        _dp = _pay("deepseek-v4-pro", "max", True)
        check("payload: DeepSeek-direct pro max -> thinking enabled + max",
              _dp.get("reasoning_effort") == "max" and _dp.get("thinking") == {"type": "enabled"})
        _df = _pay("deepseek-v4-flash", "none", True)
        check("payload: DeepSeek-direct classifier -> thinking disabled, no reasoning_effort",
              _df.get("thinking") == {"type": "disabled"} and "reasoning_effort" not in _df)
        # Fireworks fallback: flash max -> high (its top); none stays none; pro left at default.
        _ff = _pay("accounts/fireworks/models/deepseek-v4-flash", "max", False)
        check("payload: Fireworks flash fallback max -> high", _ff.get("reasoning_effort") == "high")
        _fn = _pay("accounts/fireworks/models/deepseek-v4-flash", "none", False)
        check("payload: Fireworks flash fallback none -> none", _fn.get("reasoning_effort") == "none")
        _fp = _pay("accounts/fireworks/models/deepseek-v4-pro", "max", False)
        check("payload: Fireworks pro fallback -> provider default (no reasoning params)",
              "reasoning_effort" not in _fp and "thinking" not in _fp)
    finally:
        config.REASONING_EFFORT = _saved_re

    # ── Compaction trigger is model-aware: floor = smallest window in the binding set ──
    check("ctx: deepseek-v4 window is 1M (DeepSeek-direct/Fireworks both serve ~1M)",
          config.context_window("accounts/fireworks/models/deepseek-v4-pro") == 1_000_000)
    check("ctx: bare name matches too (DeepSeek-direct strips the prefix)",
          config.context_window("deepseek-v4-flash") == 1_000_000)
    check("ctx: unknown model -> conservative 128k floor (under-estimate room)",
          config.context_window("accounts/fireworks/models/brand-new-thing") == 128_000)
    check("ctx: binding floor follows deepseek now, NOT glm's retired 200k",
          config.MODEL_CONTEXT_TOKENS == min(config.context_window(m) for m in config._BINDING_MODELS)
          and config.MODEL_CONTEXT_TOKENS == 1_000_000)
    check("ctx: vision model (kimi) is NOT in the binding set",
          config.VISION_MODEL not in config._BINDING_MODELS)
    check("fugu guard: private relay URLs are caught precisely",
          config._private_or_test_relay_url("http://172.18.0.1:8080/v1")
          and config._private_or_test_relay_url("http://localhost:8080/v1")
          and not config._private_or_test_relay_url("https://api.sakana.ai/v1"))

    # e2e fallback driving the REAL chat/stream with a fake HTTP session.
    _fakes = (fireworks.complete, fireworks.chat, fireworks.stream)
    fireworks.complete, fireworks.chat, fireworks.stream = _REAL_FW_COMPLETE, _REAL_FW_CHAT, _REAL_FW_STREAM
    try:
        DS = "accounts/fireworks/models/deepseek-v4-pro"
        OK = {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}], "usage": {}}
        s1 = _FakeHTTPSession(lambda url: _FakeHTTPResp(200, OK))
        out1 = await fireworks.complete([{"role": "user", "content": "hi"}], DS, max_tokens=8, session=s1)
        check("provider e2e: DeepSeek success answers WITHOUT touching Fireworks",
              out1 == "answer" and any("deepseek.com" in u for u in s1.urls)
              and not any("fireworks.ai" in u for u in s1.urls))

        def _ovl(url):
            return _FakeHTTPResp(503) if "deepseek.com" in url else _FakeHTTPResp(
                200, {"choices": [{"message": {"content": "fw-fallback"}, "finish_reason": "stop"}], "usage": {}})
        s2 = _FakeHTTPSession(_ovl)
        out2 = await fireworks.complete([{"role": "user", "content": "hi"}], DS, max_tokens=8, session=s2)
        check("provider e2e: DeepSeek overload (503) auto-falls-back to Fireworks",
              out2 == "fw-fallback" and s2.urls[0].startswith("https://api.deepseek.com")
              and any("fireworks.ai" in u for u in s2.urls))

        _sse = ['data: {"choices":[{"delta":{"content":"streamed ok"}}]}', 'data: [DONE]']
        def _ovl_stream(url):
            return _FakeHTTPResp(503) if "deepseek.com" in url else _FakeHTTPResp(200, lines=_sse)
        s3 = _FakeHTTPSession(_ovl_stream)
        _parts = []
        async for kind, text in fireworks.stream([{"role": "user", "content": "hi"}], DS, max_tokens=8, session=s3):
            if kind == "content":
                _parts.append(text)
        check("provider e2e (stream): DeepSeek overload falls back to Fireworks pre-token",
              "".join(_parts) == "streamed ok" and any("fireworks.ai" in u for u in s3.urls))
    finally:
        fireworks.complete, fireworks.chat, fireworks.stream = _fakes
        config.DEEPSEEK_API_KEY, config.FIREWORKS_API_KEY = _dsk, _fwk

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

    # ── Chunked section-writer: outline-first long documents ───────────────────────
    CW_HDRS = {"x-openwebui-chat-id": "chat-cw"}

    # Unit: outline parse + render, plan-intent default.
    plan0 = await agent._generate_outline("write a research paper on sleep", "")
    check("chunked: outline parses into title + sections",
          plan0 and plan0["title"] and len(plan0["sections"]) >= 2)
    rendered = agent._render_outline(plan0)
    check("chunked: rendered outline lists the headings + asks to write/adjust",
          "Introduction" in rendered and "write it" in rendered.lower())
    check("chunked: longdoc gate defaults False on a plain short ask",
          (await agent._classify_longdoc([{"role": "user", "content": "hi there"}]))["longdoc"] is False)

    # Hook B — a NEW long-doc request proposes an outline and stores the plan (does NOT build).
    _reset()
    _longdoc_queue.append(True)
    ev = await _collect([{"role": "user", "content": "write me a comprehensive research paper on sleep"}],
                        request_headers=CW_HDRS)
    out = _content(ev)
    check("chunked: long-doc request shows the outline for approval",
          "Introduction" in out and "Body" in out)
    check("chunked: the outline is persisted as a pending plan", bool(_calls.get("plan_store")))
    check("chunked: nothing is written or filed yet (outline turn only)",
          not _calls.get("section_writes") and not _calls.get("deliverable_store"))

    # Detection gated: a heavy NON-long-doc request never enters the outline flow.
    _reset()
    _longdoc_queue.append(False)
    _chat_queue.append(_chat_content("Here is your cover letter."))
    ev = await _collect([{"role": "user", "content": "write a cover letter"}], request_headers=CW_HDRS)
    check("chunked: non-long-doc heavy turn skips the outline flow (no plan stored)",
          not _calls.get("plan_store") and "cover letter" in _content(ev).lower())

    # Hook A — pending plan + approval BUILDS section-by-section, verifies, exports, clears plan.
    _reset()
    _plan_holder.append({
        "title": "My Paper",
        "sections": [{"heading": "Intro", "intent": "set up"},
                     {"heading": "Methods", "intent": "how"}],
        "source": "The study found X. The data shows Y.",
        "request": "write a paper", "filename": "my_paper", "fmt": "docx"})
    _plan_intent_queue.append({"action": "approve", "revision": ""})
    _section_queue.extend(["## Intro\n\nGrounded intro from source.",
                           "## Methods\n\nGrounded methods from source."])
    _post_queue.append([{"filename": "my_paper.docx", "download_url": "/files/my_paper.docx"}])
    ev = await _collect([{"role": "user", "content": "looks good, write it"}], request_headers=CW_HDRS)
    out = _content(ev)
    if agent._BG_TASKS:  # the deliverable/memory stores are fire-and-forget create_tasks
        await asyncio.gather(*list(agent._BG_TASKS), return_exceptions=True)
    check("chunked: approval writes EVERY section (one focused call each)",
          len(_calls.get("section_writes") or []) == 2)
    check("chunked: built doc is exported + delivered with a download link",
          "ready" in out.lower() and "Download" in out)
    stored = _calls.get("deliverable_store") or []
    check("chunked: assembled doc (all sections) persisted as the deliverable",
          bool(stored) and "Grounded intro" in stored[-1][1] and "Grounded methods" in stored[-1][1])
    check("chunked: pending plan is cleared after a successful build", bool(_calls.get("plan_clear")))

    # Hook A — a blocked honesty check on the assembly does NOT ship + KEEPS the plan.
    _reset()
    _plan_holder.append({"title": "P", "sections": [{"heading": "S", "intent": "i"}],
                         "source": "Real source.", "request": "r", "filename": "p", "fmt": "docx"})
    _plan_intent_queue.append({"action": "approve", "revision": ""})
    _section_queue.append("## S\n\nClaims not in the source.")
    # The auditor is consulted up to 3x (initial -> after surgical refine -> after rewrite);
    # all must stay FABRICATION for the assembly to hard-block.
    for _ in range(3):
        _honesty_queue.append({"unsupported": ["fabricated metric"], "verdict": "FABRICATION"})
    ev = await _collect([{"role": "user", "content": "go"}], request_headers=CW_HDRS)
    if agent._BG_TASKS:
        await asyncio.gather(*list(agent._BG_TASKS), return_exceptions=True)
    check("chunked: a blocked assembly is not exported and the plan is kept",
          not _calls.get("deliverable_store") and not _calls.get("plan_clear"))

    # Hook A — 'revise' re-outlines (stores a new plan) without building.
    _reset()
    _plan_holder.append({"title": "Doc", "sections": [{"heading": "A", "intent": "a"}],
                         "source": "", "request": "write a doc", "filename": "doc", "fmt": "docx"})
    _plan_intent_queue.append({"action": "revise", "revision": "add a Conclusion"})
    _outline_queue.append({"title": "Doc", "sections": [
        {"heading": "A", "intent": "a"}, {"heading": "Conclusion", "intent": "wrap up"}]})
    ev = await _collect([{"role": "user", "content": "add a conclusion section"}], request_headers=CW_HDRS)
    check("chunked: 'revise' re-shows an updated outline, no build",
          "Conclusion" in _content(ev) and bool(_calls.get("plan_store"))
          and not _calls.get("section_writes"))

    # Hook A — 'abandon' clears the plan and falls through to the normal flow.
    _reset()
    _plan_holder.append({"title": "Old", "sections": [{"heading": "A", "intent": "a"}],
                         "source": "", "request": "r", "filename": "old", "fmt": "docx"})
    _plan_intent_queue.append({"action": "abandon", "revision": ""})
    # STREAM_SIMPLE_CHAT is off in this suite, so the post-abandon turn runs the heavy loop.
    _longdoc_queue.append(False)             # not a new long doc either
    _chat_queue.append(_chat_content("Sure, hello!"))
    _gate_queue.append(False)                # no fact-grounding needed
    ev = await _collect([{"role": "user", "content": "never mind, just say hi"}], request_headers=CW_HDRS)
    check("chunked: 'abandon' clears the plan and answers normally",
          bool(_calls.get("plan_clear")) and "hello" in _content(ev).lower()
          and not _calls.get("section_writes"))

    # Per-section verification (post-review fix): each section is audited on its OWN, so a
    # fabricating section is caught and named, the whole build is held, and the plan is kept.
    _reset()
    _plan_holder.append({"title": "Doc", "sections": [{"heading": "Intro", "intent": "x"},
                         {"heading": "Bad", "intent": "y"}], "source": "Real source material.",
                         "request": "r", "filename": "doc", "fmt": "docx"})
    _plan_intent_queue.append({"action": "approve", "revision": ""})
    _section_queue.extend(["## Intro\n\nGrounded.", "## Bad\n\nFabricated claim."])
    _CLEAN = {"unsupported": [], "verdict": "CLEAN"}
    _FAB = {"unsupported": ["fabricated claim"], "verdict": "FABRICATION"}
    _honesty_queue.extend([_CLEAN, _FAB, _FAB, _FAB])   # Intro clean; Bad fails all 3 audit rounds
    ev = await _collect([{"role": "user", "content": "write it"}], request_headers=CW_HDRS)
    if agent._BG_TASKS:
        await asyncio.gather(*list(agent._BG_TASKS), return_exceptions=True)
    out = _content(ev)
    check("chunked: per-section audit isolates + names the failing section, holds the build",
          "§2" in out and "Bad" in out and not _calls.get("deliverable_store")
          and not _calls.get("plan_clear") and len(_calls.get("section_writes") or []) == 2)

    # A section that can't be generated (empty after one retry) -> NO placeholder ships, build held.
    _reset()
    _plan_holder.append({"title": "D", "sections": [{"heading": "S", "intent": "i"}],
                         "source": "src", "request": "r", "filename": "d", "fmt": "docx"})
    _plan_intent_queue.append({"action": "approve", "revision": ""})
    _section_queue.extend(["", ""])           # write + retry both empty
    ev = await _collect([{"role": "user", "content": "go"}], request_headers=CW_HDRS)
    check("chunked: an ungenerable section holds the build (no placeholder shipped)",
          "could not be generated".lower() not in _content(ev).lower()  # the old placeholder text is gone
          and not _calls.get("deliverable_store") and not _calls.get("plan_clear"))

    # Citation-without-source: a NO-source section that cites [N] is blocked (the dead global
    # guard, restored source-aware for this path).
    _reset()
    _plan_holder.append({"title": "P", "sections": [{"heading": "Lit", "intent": "review"}],
                         "source": "", "request": "r", "filename": "p", "fmt": "docx"})
    _plan_intent_queue.append({"action": "approve", "revision": ""})
    _section_queue.append("## Lit\n\nPrior work shows much [1]. References: [1] Smith 2020.")
    ev = await _collect([{"role": "user", "content": "go"}], request_headers=CW_HDRS)
    check("chunked: a no-source section that cites [N] is held back, not shipped",
          not _calls.get("deliverable_store") and not _calls.get("plan_clear"))

    # Cost prefilter (#8): a short, cue-less heavy request never spends the flash longdoc gate.
    check("chunked: _maybe_longdoc prefilter — short cue-less ask is skipped",
          agent._maybe_longdoc([{"role": "user", "content": "summarize this"}]) is False
          and agent._maybe_longdoc([{"role": "user", "content": "write a research paper on X"}]) is True)
    _reset()
    _longdoc_queue.append(True)               # would say longdoc IF the gate were even called
    _chat_queue.append(_chat_content("A short summary."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "summarize this"}], request_headers=CW_HDRS)
    check("chunked: prefilter skips the longdoc classifier on a short request (no gate call)",
          "gate:longdoc" not in (_calls.get("labels") or []) and not _calls.get("plan_store"))

    # Plan escape hatch (#4): a plan past the revise cap is dropped without re-classifying.
    _reset()
    config_max = config.CHUNKED_MAX_REVISES
    _plan_holder.append({"title": "Stuck", "sections": [{"heading": "A", "intent": "a"}],
                         "source": "", "request": "r", "filename": "s", "fmt": "docx",
                         "created_at": 9_999_999_999.0, "revise_count": config_max + 1})  # fresh -> only the cap triggers
    _longdoc_queue.append(False)
    _chat_queue.append(_chat_content("Fresh answer."))
    _gate_queue.append(False)
    ev = await _collect([{"role": "user", "content": "hello again"}], request_headers=CW_HDRS)
    check("chunked: a plan past the revise cap is auto-dropped (no plan-intent classify)",
          bool(_calls.get("plan_clear")) and "gate:plan" not in (_calls.get("labels") or []))

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        raise SystemExit(1)
    print("all orchestrator contract tests passed")


if __name__ == "__main__":
    asyncio.run(_run_tests())
