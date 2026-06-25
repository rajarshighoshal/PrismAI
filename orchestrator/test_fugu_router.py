"""Offline unit tests for the Fugu routing gate. The only network call (the hardness
classifier) is monkeypatched, so these run with no API. Covers:
  _maybe_fugu_candidate() — the static cue prefilter
  should_escalate()       — the post-block retry decision (pure logic)
  decide()                — fugu vs deepseek: fast-rejects + classifier-driven routing,
                            including the cue-bypasses-threshold path of the refactored gate.

  python -m orchestrator.test_fugu_router
"""
import asyncio

from orchestrator import config, fugu_router

_fails = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    if not cond:
        _fails.append(name)


def _msgs(text):
    return [{"role": "user", "content": text}]


LONG = "please help me with this task " + "x" * 80  # clears the 80-char fast-reject, no cue

# Snapshot the config we mutate, restored at the end.
_CFG = {k: getattr(config, k) for k in
        ("ENABLE_FUGU", "FUGU_API_KEY", "FUGU_ESCALATE_ON_BLOCK",
         "GROUNDING_REPAIR_STEPS", "FUGU_HARDNESS_THRESHOLD")}


def _decide(messages, **kw):
    return asyncio.run(fugu_router.decide(messages, **kw))


def _decide_with(classifier_result, text=LONG):
    """decide() with the network hardness-classifier stubbed to a canned result."""
    orig = fugu_router._classify_hardness

    async def fake(messages, *, session=None):
        return classifier_result

    fugu_router._classify_hardness = fake
    try:
        return _decide(_msgs(text))
    finally:
        fugu_router._classify_hardness = orig


# ── _maybe_fugu_candidate: static cue prefilter ──
check("cue: 'research paper' is a candidate",
      fugu_router._maybe_fugu_candidate(_msgs("please write a research paper on X")))
check("cue: a casual ask is not a candidate",
      not fugu_router._maybe_fugu_candidate(_msgs("what's the weather today")))

# ── should_escalate: pure logic ──
config.ENABLE_FUGU, config.FUGU_API_KEY, config.GROUNDING_REPAIR_STEPS = True, "k", 2


def _esc(status="unsupported_claims", steps=0):
    return asyncio.run(fugu_router.should_escalate(status, steps))


config.FUGU_ESCALATE_ON_BLOCK = True
check("escalate: yes when enabled + unsupported_claims + repairs left", _esc() is True)
check("escalate: no on a non-claim verify status", _esc(status="ok") is False)
check("escalate: no once repairs are exhausted", _esc(steps=2) is False)
config.FUGU_ESCALATE_ON_BLOCK = False
check("escalate: no when the feature is off", _esc() is False)
config.FUGU_ESCALATE_ON_BLOCK, config.FUGU_API_KEY = True, ""
check("escalate: no without a Fugu key", _esc() is False)

# ── decide: fast-reject paths (no network) ──
config.ENABLE_FUGU, config.FUGU_API_KEY = False, "k"
check("decide: deepseek when Fugu disabled", _decide(_msgs(LONG)) == "deepseek")
config.ENABLE_FUGU, config.FUGU_API_KEY = True, ""
check("decide: deepseek with no key", _decide(_msgs(LONG)) == "deepseek")
config.FUGU_API_KEY = "k"
check("decide: deepseek for an edit turn", _decide(_msgs(LONG), is_edit=True) == "deepseek")
check("decide: deepseek for a user-chosen model", _decide(_msgs(LONG), is_user_model=True) == "deepseek")
check("decide: deepseek when the request is too short", _decide(_msgs("hi")) == "deepseek")

# ── decide: classifier-driven (stubbed result) ──
config.FUGU_HARDNESS_THRESHOLD = 0.65
check("decide: fugu when benefits + confidence >= threshold",
      _decide_with({"benefits_from_multi_model": True, "confidence": 0.9, "why": ""}) == "fugu")
check("decide: deepseek when benefits but confidence < threshold (no cue)",
      _decide_with({"benefits_from_multi_model": True, "confidence": 0.3, "why": ""}) == "deepseek")
check("decide: deepseek when the classifier sees no benefit",
      _decide_with({"benefits_from_multi_model": False, "confidence": 0.99, "why": ""}) == "deepseek")
check("decide: fugu on a static cue + benefit even BELOW threshold",
      _decide_with({"benefits_from_multi_model": True, "confidence": 0.1, "why": ""},
                   text="write a literature review " + LONG) == "fugu")
check("decide: deepseek when the classifier fails (None)", _decide_with(None) == "deepseek")

for _k, _v in _CFG.items():
    setattr(config, _k, _v)

print(f"\n{'all fugu-router tests passed' if not _fails else f'{len(_fails)} FAILED: {_fails}'}")
import sys
sys.exit(1 if _fails else 0)
