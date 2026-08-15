"""Task-structure classifier for escalation gating.

Salvaged from the retired Fugu router. The reusable idea: decide escalation on TASK
STRUCTURE, not raw difficulty. A hard math problem may be solved fine by one strong
model with chain-of-thought; a task that mixes creative generation with precise factual
grounding and cross-verification is where a stronger/committee solver earns its cost.

This module ONLY classifies. The escalation gate (DeliveryPipeline, later step) combines
this signal with the run's failure state, and must escalate only when ALL of these hold:
  - the task is genuinely hard/structured (this classifier), AND
  - the current model still fails the honesty verifier AFTER the repair budget is
    exhausted (a real capability shortfall), AND
  - the failure is NOT a transient provider error/timeout/empty (those retry or fall
    back within the SAME tier, never a cost-escalation).
Never escalate on the first block, and never on a transient error.
"""
import json
import logging
import re
from typing import Optional

from . import config, fireworks
from .owui import _last_user_text, _all_user_text

log = logging.getLogger(__name__)

SYSTEM_TASK_STRUCTURE = (
    "You are a task-structure classifier. Given a user request, decide whether this "
    "task would BENEFIT from escalating to a stronger (or multi-model) solver — where "
    "the extra cost buys planning, execution, and verification cross-checking each "
    "other — rather than a single default model handling everything.\n\n"
    "A task benefits when it has these STRUCTURAL signs:\n"
    "- MULTI-STAGE: it decomposes into distinct phases (plan -> gather -> draft -> "
    "verify -> polish) where specialization helps\n"
    "- CROSS-VERIFICATION VALUE: errors compound, so independent checking catches "
    "mistakes one pass would miss\n"
    "- MIXED COGNITIVE MODES: it needs both creative/generative AND precise "
    "factual/deterministic work (e.g. a compelling statement that also cites specific "
    "achievements from a provided CV)\n"
    "- MULTIPLE PERSPECTIVES: it benefits from exploring approaches before converging "
    "(research synthesis, strategy, complex analysis)\n"
    "- HIGH STAKES + AMBIGUITY: the deliverable matters AND the source is ambiguous, "
    "conflicting, or needs judgment a single pass might get wrong\n\n"
    "A task does NOT benefit when:\n"
    "- It is single-cognitive-mode: pure code, pure Q&A, pure editing, pure formatting\n"
    "- Clear, complete source is provided and one competent model answers directly\n"
    "- The answer is short (< ~200 words expected)\n"
    "- It is casual conversation, opinion, brainstorming, or explanation\n"
    "- It is a follow-up edit to an existing document\n\n"
    'Return JSON only: {"benefits_from_escalation": boolean, "confidence": float, '
    '"why": string}\n'
    "confidence: 0.0-1.0, where 0.0 = definitely not, 1.0 = definitely yes."
)

# High-signal phrases that almost always indicate a structured deliverable; these
# bypass the classifier the way _maybe_longdoc() does.
_STRUCTURED_CUES = (
    "research paper", "literature review", "thesis", "dissertation", "white paper",
    "whitepaper", "strategic plan", "competitive analysis", "due diligence",
    "grant proposal", "manuscript", "systematic review", "meta-analysis",
    "patent landscape", "technical report", "policy paper", "legal brief",
    "multi-source", "cross-reference", "compare and contrast these papers",
    "synthesize the following", "analyze these documents together",
)


def obvious_candidate(messages) -> bool:
    """Cheap pre-filter: does the request look structurally hard on its face?"""
    t = _last_user_text(messages).strip().lower()
    return any(c in t for c in _STRUCTURED_CUES)


async def classify(messages, *, session=None) -> Optional[dict]:
    """Classify whether this task benefits from escalation.

    Returns {"benefits_from_escalation": bool, "confidence": float, "why": str}
    or None on failure (caller treats None as 'do not escalate', the safe default).
    """
    q = _all_user_text(messages).strip()[:6000]
    if not q:
        return None
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_TASK_STRUCTURE},
             {"role": "user", "content": q}],
            config.GROUNDING_GATE_MODEL,
            max_tokens=200,
            temperature=0.0,
            session=session,
            label="gate:hardness",
        )
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0) if m else raw)
        return {
            "benefits_from_escalation": bool(data.get("benefits_from_escalation", False)),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            "why": str(data.get("why", "")),
        }
    except Exception as e:
        log.warning(f"[hardness] task-structure classification failed: {e}")
        return None
