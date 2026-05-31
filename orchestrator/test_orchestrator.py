"""Offline contract tests for the orchestrator — no network, no prod.

Run from the repo root:  python -m orchestrator.test_orchestrator

Monkeypatches the Fireworks client and the tool-server client so we can assert
the pipeline's control flow, model selection, and verification footer without
any API calls.
"""
import asyncio

from orchestrator import config, fireworks, pipeline, search, toolserver
from orchestrator.depth import classify_depth, CHAT, GROUNDED, DELIVERABLE

_calls = {"models": []}


async def _fake_stream(messages, model, *, max_tokens, temperature=None, session=None):
    """Records the model it was asked to run and yields a canned draft."""
    _calls["models"].append(model)
    for piece in ["Hello ", "from ", "the model."]:
        yield ("content", piece)


async def _fake_verify_grounded(source, draft, *, session=None):
    return {"grounded": True, "unsupported_claims": ""}


async def _fake_verify_ungrounded(source, draft, *, session=None):
    return {"grounded": False, "unsupported_claims": "1. invented metric"}


async def _fake_complete(messages, model, *, max_tokens, temperature=None, session=None):
    sys = messages[0]["content"] if messages else ""
    if "search query" in sys:
        return "mars latest news"
    if "Revise the draft" in sys:
        return "CORRECTED DOCUMENT BODY"
    return "completion"


async def _fake_search(query, *, max_results=None, session=None):
    return [
        {"title": "T1", "url": "https://a.com", "snippet": "snippet one"},
        {"title": "T2", "url": "https://b.com", "snippet": "snippet two"},
    ]


async def _fake_search_empty(query, *, max_results=None, session=None):
    return []


async def _collect(messages, **kw):
    out = []
    async for kind, text in pipeline.run(messages, **kw):
        out.append((kind, text))
    return out


def _content(events):
    return "".join(t for k, t in events if k == "content")


async def _run_tests():
    fails = []

    def check(name, cond):
        print(f"{'PASS' if cond else 'FAIL'}: {name}")
        if not cond:
            fails.append(name)

    # --- depth classifier (the cases that previously regressed) ---
    check("depth: greeting -> CHAT", classify_depth("hey there").tier == CHAT)
    check("depth: 'what's the latest news' -> GROUNDED",
          classify_depth("what's the latest news on mars").tier == GROUNDED)
    check("depth: 'write me a cover letter' -> DELIVERABLE",
          classify_depth("write me a cover letter for a data role").tier == DELIVERABLE)
    check("depth: 'turn these notes into 3 resume bullets' -> DELIVERABLE",
          classify_depth("turn these notes into 3 resume bullets").tier == DELIVERABLE)
    check("depth: 'write this up as a research paper' -> DELIVERABLE",
          classify_depth("actually, write this up as a research paper").tier == DELIVERABLE)
    check("depth: export intent sets wants_export",
          classify_depth("export this as a docx").wants_export is True)

    # patch ALL network-touching pieces up front so no test hits the wire
    fireworks.stream = _fake_stream
    fireworks.complete = _fake_complete
    config.ENABLE_VERIFICATION = True
    config.ENABLE_REFINE = False  # warn-only first; the refine test flips this on
    config.MIN_SOURCE_CHARS = 20

    # --- CHAT: one streamed call on the CHAT model, content passes through ---
    _calls["models"].clear()
    ev = await _collect([{"role": "user", "content": "hey, how are you?"}])
    check("chat: content streamed through", _content(ev) == "Hello from the model.")
    check("chat: used CHAT_MODEL", _calls["models"] == [config.CHAT_MODEL])

    # --- VISION: routes to the vision model ---
    _calls["models"].clear()
    vmsg = [{"role": "user", "content": [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
    ]}]
    await _collect(vmsg)
    check("vision: used VISION_MODEL", _calls["models"] == [config.VISION_MODEL])

    # --- DELIVERABLE + grounded source -> verified footer ---
    _calls["models"].clear()
    toolserver.verify_grounding = _fake_verify_grounded
    src = "Rajarshi knows Python and C++ and built a RAG email triage system."
    deliv = [
        {"role": "user", "content": src},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "now write this up as a cover letter"},
    ]
    ev = await _collect(deliv)
    body = _content(ev)
    check("deliverable: used DRAFT_MODEL", _calls["models"] == [config.DRAFT_MODEL])
    check("deliverable: draft streamed", body.startswith("Hello from the model."))
    check("deliverable: grounded footer appended", "✓ Checked against your source" in body)

    # --- DELIVERABLE + ungrounded -> warning footer with the claim ---
    toolserver.verify_grounding = _fake_verify_ungrounded
    ev = await _collect(deliv)
    body = _content(ev)
    check("deliverable: warning footer appended", "⚠ Verification" in body)
    check("deliverable: lists the unsupported claim", "invented metric" in body)

    # --- DELIVERABLE with no source -> no footer, no fake check ---
    toolserver.verify_grounding = _fake_verify_ungrounded
    ev = await _collect([{"role": "user", "content": "write me an essay about the ocean"}])
    body = _content(ev)
    check("deliverable: no source -> no verification footer",
          "Verification" not in body and "✓" not in body)

    # --- DELIVERABLE ungrounded + refine -> corrected version appended ---
    fireworks.complete = _fake_complete
    config.ENABLE_REFINE = True
    toolserver.verify_grounding = _fake_verify_ungrounded
    ev = await _collect(deliv)
    body = _content(ev)
    check("refine: corrected version appended", "Corrected version below" in body)
    check("refine: corrected body present", "CORRECTED DOCUMENT BODY" in body)
    check("refine: original flagged claim listed", "invented metric" in body)

    # --- same-message source (paste + instruction in one turn) is verified ---
    toolserver.verify_grounding = _fake_verify_grounded
    one_msg = [{"role": "user", "content":
        "write me a cover letter from these notes.\n\n"
        "Notes: Rajarshi is a mechanical engineer who taught himself Python and "
        "built an internal RAG email-triage tool at REAL Brokerage that cut "
        "response time substantially."}]
    ev = await _collect(one_msg)
    check("same-message source: verification fires", "✓ Checked against your source" in _content(ev))

    # --- GROUNDED with search results -> answer + cited sources footer ---
    _calls["models"].clear()
    search.search = _fake_search
    fireworks.complete = _fake_complete
    config.ENABLE_WEB_SEARCH = True
    config.ENABLE_GROUNDED_VERIFY = False
    ev = await _collect([{"role": "user", "content": "what's the latest news on mars"}])
    body = _content(ev)
    check("grounded: answer streamed", body.startswith("Hello from the model."))
    check("grounded: sources footer present", "**Sources**" in body and "a.com" in body)
    check("grounded: answered on CHAT_MODEL", _calls["models"] == [config.CHAT_MODEL])

    # --- GROUNDED with no results -> careful answer, NO sources footer ---
    search.search = _fake_search_empty
    ev = await _collect([{"role": "user", "content": "what's the latest news on mars"}])
    body = _content(ev)
    check("grounded: no results -> no sources footer", "**Sources**" not in body)
    check("grounded: no results -> still answers", body == "Hello from the model.")

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        raise SystemExit(1)
    print("all orchestrator contract tests passed")


if __name__ == "__main__":
    asyncio.run(_run_tests())
