"""Escalation gate: decide when a verifier BLOCK should be retried on a stronger solver.

Replaces the retired Fugu escalation with a corrected, provider-neutral policy. Escalate
ONLY when ALL of these hold:
  - the block is a capability shortfall (status 'unsupported_claims'), not a transient error;
  - the model has genuinely failed: the in-loop repair budget is already exhausted;
  - the task is genuinely hard/structured (hardness classifier);
  - an escalation target model is configured (config.ESCALATION_MODEL; empty -> never).
Never on the first block, never on a transient error (those retry/fall back in the same tier).

This module only DECIDES. The host (agent loop) performs the re-solve + re-verify.
"""
import logging

from . import config, hardness

log = logging.getLogger(__name__)


async def should_escalate(status: str, repair_steps: int, *, messages, session=None) -> bool:
    """True only when escalating to config.ESCALATION_MODEL is warranted (see module docstring)."""
    if not config.ESCALATION_MODEL:
        return False
    if status != "unsupported_claims":
        return False
    if repair_steps < config.GROUNDING_REPAIR_STEPS:  # repair budget not yet exhausted
        return False
    if hardness.obvious_candidate(messages):
        return True
    result = await hardness.classify(messages, session=session)
    return bool(
        result
        and result.get("benefits_from_escalation")
        and result.get("confidence", 0.0) >= config.ESCALATION_HARDNESS_THRESHOLD
    )
