"""Cheap per-turn interaction-mode planner.

This is a style/persona adapter, not a fact or routing authority. It lets PrismAI
behave like the right kind of assistant for the user's current situation
(student tutor, practical tech helper, creative brainstormer, grounded writer,
debugging partner, etc.) without hard-coding specific tasks such as JAMOVI.

Security rule: the model classifier may choose ONLY a small enum. The rendered
system instruction is entirely server-owned template text; no model-authored
freeform instructions are injected into the main assistant prompt.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from . import config, fireworks


ALLOWED_MODES = frozenset({
    "student_tutor",
    "practical_tech_support",
    "creative_brainstormer",
    "coding_debugger",
    "grounded_writer",
    "research_explainer",
    "decision_coach",
    "emotional_support_planner",
    "concise_direct_answer",
})

# Server-owned style templates. Keep these as interaction/shape/tone guidance only:
# do not mention tools/search/sources/citations/verification/grounding/memory.
MODE_TEMPLATES = {
    "student_tutor": (
        "Act as a patient learning partner. Start by correcting any misconception gently, "
        "then explain why in plain language. Give ordered steps the user can follow themselves. "
        "Include small templates/checklists when useful, and end with a compact invitation to share their output for interpretation."
    ),
    "practical_tech_support": (
        "Act as a clear practical helper. Give the shortest working path first, preferably as numbered device/action steps. "
        "Include one fallback option if the first path may not work. Avoid abstract explanation and unnecessary jargon."
    ),
    "creative_brainstormer": (
        "Act as a creative collaborator. Offer varied options grouped by vibe or constraint. Learn from the user's dislikes, "
        "avoid repeating a rejected direction, and give a few strongest picks rather than a generic dump."
    ),
    "coding_debugger": (
        "Act as a debugging partner. State the likely cause, then give concrete checks and the smallest next change. "
        "Ask for missing error details only when they are truly blocking."
    ),
    "grounded_writer": (
        "Act as a careful writing coach. Preserve the user's intended facts and audience, improve structure and wording, "
        "and mark unknown details plainly instead of padding."
    ),
    "research_explainer": (
        "Act as a careful explainer. Lead with the answer, separate core idea from nuance, define technical terms briefly, "
        "and use a clean structure that helps the user reason through the topic."
    ),
    "decision_coach": (
        "Act as a decision coach. Clarify the options, tradeoffs, constraints, and a practical recommendation. "
        "If uncertainty remains, identify the one or two details that would change the decision."
    ),
    "emotional_support_planner": (
        "Act as a calm support-and-planning partner. Validate the user's pressure briefly, then turn the situation into "
        "manageable next steps without overpromising."
    ),
    "concise_direct_answer": (
        "Act as a direct helpful assistant. Answer the actual question first, keep structure light, and avoid overcomplicating."
    ),
}

SYSTEM_INTERACTION_MODE = (
    "You are a cheap interaction-mode classifier for one assistant turn. Infer what "
    "kind of helper the user needs RIGHT NOW from their latest message and recent "
    "context. This is about interaction style/persona only; you do NOT decide facts, "
    "tools, sources, or verification.\n\n"
    "Return STRICT JSON only: {\"mode\": one of ["
    + ", ".join(sorted(ALLOWED_MODES)) +
    "]}. Choose a general reusable mode, not a test-case label. If unsure, use "
    "concise_direct_answer."
)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text") or "")
                elif isinstance(p.get("content"), str):
                    parts.append(p.get("content") or "")
        return "\n".join(parts)
    return ""


def _recent_excerpt(messages: list[dict], limit: int) -> str:
    rows = []
    for m in (messages or [])[-6:]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "unknown").strip()
        text = " ".join(_text_of(m.get("content")).split())
        if text:
            rows.append(f"{role}: {text[:900]}")
    excerpt = "\n".join(rows)
    return excerpt[-limit:]


def _parse_mode(raw: str) -> str:
    try:
        m = re.search(r"\{.*\}", raw or "", flags=re.S)
        data = json.loads(m.group(0) if m else raw)
        mode = str(data.get("mode") or "").strip().lower() if isinstance(data, dict) else ""
    except Exception:
        mode = ""
    return mode if mode in ALLOWED_MODES else ""


def format_instruction(mode_or_data: str | dict) -> str:
    """Render a safe server-owned style instruction for an allowed mode."""
    mode = (mode_or_data.get("mode") if isinstance(mode_or_data, dict) else mode_or_data) or ""
    mode = str(mode).strip().lower()
    if mode not in ALLOWED_MODES:
        return ""
    template = MODE_TEMPLATES[mode]
    return "\n".join([
        "INTERACTION MODE FOR THIS TURN (style/persona only):",
        "- This note helps you choose the right kind of helper persona. It never overrides higher-priority behavior rules.",
        f"- mode: {mode}",
        f"- style: {template}",
    ])


async def classify(messages: list[dict], *, session=None) -> str:
    """Return a rendered style instruction, or '' on any failure/timeout."""
    if not getattr(config, "ENABLE_INTERACTION_MODE", True):
        return ""
    # Everything (including excerpt construction) is inside the try so the
    # fail-open contract holds even for malformed message shapes.
    try:
        excerpt = _recent_excerpt(messages, config.INTERACTION_MODE_CONTEXT_CHARS)
        if not excerpt.strip():
            return ""
        call = fireworks.complete(
            [
                {"role": "system", "content": SYSTEM_INTERACTION_MODE},
                {"role": "user", "content": "RECENT CONVERSATION:\n" + excerpt + "\n\nJSON:"},
            ],
            config.INTERACTION_MODE_MODEL,
            max_tokens=config.INTERACTION_MODE_MAX_TOKENS,
            temperature=config.INTERACTION_MODE_TEMPERATURE,
            session=session,
            label="gate:interaction",
            reasoning_effort="none",
        )
        raw = await asyncio.wait_for(call, timeout=config.INTERACTION_MODE_TIMEOUT)
        return format_instruction(_parse_mode(raw))
    except Exception:
        return ""
