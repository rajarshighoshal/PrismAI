"""Fugu routing: decide whether a task should use multi-model orchestration.

Three outcomes:
  fugu      — route to Fugu upfront (task genuinely benefits from multi-model reasoning)
  deepseek  — use DeepSeek v4-pro (default single-model agent loop)
  escalate  — use DeepSeek first, but escalate to Fugu if the verifier catches issues

The router is the genuinely hard part. Fugu Ultra costs ~$30/M output vs DeepSeek's
~$1.10/M — ~27× more. So the classifier must be conservative: default to DeepSeek and
only spend on Fugu when the evidence is strong that a committee of models would produce
a materially better answer.

Key design: we route on TASK STRUCTURE, not task difficulty. A hard math problem might
be solved equally well by one strong model with CoT. A cover letter that requires
creative writing + factual grounding + cross-verification is where Fugu shines.
"""
import json
import logging
import re
from typing import Optional

from . import config, fireworks
from .owui import _last_user_text, _all_user_text

log = logging.getLogger(__name__)

SYSTEM_HARDNESS = (
    "You are a task-hardness classifier. Given a user request, decide whether this "
    "task would BENEFIT from multi-model collaboration — where different models handle "
    "different cognitive roles (planning, execution, verification) and cross-check each "
    "other's work — rather than a single strong model handling everything.\n\n"
    "A task benefits from multi-model orchestration when it has these STRUCTURAL signs:\n"
    "- MULTI-STAGE: the request naturally decomposes into distinct phases (research plan "
    "→ gather sources → draft → verify → polish) where different models could specialize\n"
    "- CROSS-VERIFICATION VALUE: errors in one part of the answer compound, so having "
    "separate models check each other's work catches mistakes one model would miss\n"
    "- MIXED COGNITIVE MODES: the task requires both creative/generative work AND precise "
    "factual/deterministic work (e.g., write a compelling personal statement that also "
    "accurately cites specific achievements from a provided CV)\n"
    "- MULTIPLE PERSPECTIVES: the task benefits from exploring different approaches or "
    "viewpoints before converging (research synthesis, strategy documents, complex analysis)\n"
    "- HIGH STAKES + AMBIGUITY: the deliverable matters AND the source material is "
    "ambiguous, conflicting, or requires judgment calls a single model might get wrong\n\n"
    "A task does NOT benefit from multi-model orchestration when:\n"
    "- It's a single-cognitive-mode task: pure code, pure Q&A, pure editing, pure formatting\n"
    "- The user has provided clear, complete source material and one competent model can "
    "produce the right answer directly\n"
    "- The answer is short (< ~200 words expected output)\n"
    "- It's casual conversation, opinion, brainstorming, or explanation\n"
    "- It's a follow-up edit to an existing document (the verifier already checked it)\n\n"
    "Return JSON only: {\"benefits_from_multi_model\": boolean, \"confidence\": float, "
    "\"why\": string}\n"
    "confidence: 0.0-1.0, where 0.0 = definitely not, 1.0 = definitely yes. The confidence "
    "determines whether we route to Fugu upfront or keep DeepSeek with escalation."
)


async def _classify_hardness(messages, *, session=None) -> Optional[dict]:
    """Classify whether this task benefits from multi-model orchestration.

    Returns {"benefits_from_multi_model": bool, "confidence": float, "why": str}
    or None on parse failure (-> treat as DeepSeek-only, safe default).
    """
    q = _all_user_text(messages).strip()[:6000]
    if not q:
        return None

    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_HARDNESS},
             {"role": "user", "content": q}],
            config.GROUNDING_GATE_MODEL,
            max_tokens=200,
            temperature=0.0,
            session=session,
            label="gate:fugu",
        )
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0) if m else raw)
        return {
            "benefits_from_multi_model": bool(data.get("benefits_from_multi_model", False)),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            "why": str(data.get("why", "")),
        }
    except Exception as e:
        log.warning(f"[fugu-router] hardness classification failed: {e}")
        return None


# Statically obvious Fugu candidates: high-signal phrases that almost always indicate
# a structured deliverable. These bypass the classifier just like _maybe_longdoc().
_FUGU_CUES = (
    "research paper", "literature review", "thesis", "dissertation", "white paper",
    "whitepaper", "strategic plan", "competitive analysis", "due diligence",
    "grant proposal", "manuscript", "systematic review", "meta-analysis",
    "patent landscape", "technical report", "policy paper", "legal brief",
    "multi-source", "cross-reference", "compare and contrast these papers",
    "synthesize the following", "analyze these documents together",
)


def _maybe_fugu_candidate(messages) -> bool:
    """Cheap pre-filter: does the user's request look like a Fugu-worthy task?"""
    t = _last_user_text(messages).strip().lower()
    return any(c in t for c in _FUGU_CUES)


async def decide(
    messages,
    *,
    user_source: str = "",
    is_edit: bool = False,
    is_user_model: bool = False,
    session=None,
) -> str:
    """Decide routing for this turn. Returns 'fugu' | 'deepseek'."""
    if not config.ENABLE_FUGU or not config.FUGU_API_KEY:
        return "deepseek"
    if is_user_model or is_edit:
        return "deepseek"

    q = _last_user_text(messages).strip()

    # Fast rejects — too short for multi-model to matter
    if len(q) < 80:
        return "deepseek"

    # Classify hardness ONCE. A statically high-signal cue (research paper, thesis, …) routes
    # on benefit alone; everything else also needs the confidence threshold to clear.
    result = await _classify_hardness(messages, session=session)
    if not result or not result["benefits_from_multi_model"]:
        return "deepseek"
    if _maybe_fugu_candidate(messages) or result["confidence"] >= config.FUGU_HARDNESS_THRESHOLD:
        log.info(f"[fugu-router] routing to fugu: confidence={result['confidence']:.2f} "
                 f"why={result['why'][:100]}")
        return "fugu"

    return "deepseek"


async def should_escalate(
    verify_status: str,
    repair_steps: int,
    *,
    session=None,
) -> bool:
    """Decide whether to escalate a blocked DeepSeek answer to Fugu.

    Only escalate when:
    - Escalation is enabled
    - The verifier actually caught unsupported claims (not a transient error)
    - We haven't already exhausted repair steps
    - Fugu is available and configured
    """
    if not config.FUGU_ESCALATE_ON_BLOCK:
        return False
    if not config.ENABLE_FUGU or not config.FUGU_API_KEY:
        return False
    if verify_status != "unsupported_claims":
        return False
    if repair_steps >= config.GROUNDING_REPAIR_STEPS:
        return False
    return True
