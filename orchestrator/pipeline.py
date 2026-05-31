"""The depth-routed orchestration pipeline — the harness.

Every turn: classify depth, then do the minimum machinery that tier deserves.
This is the control flow that replaces router_fn. Prompts are kept short and
plain on purpose (the prompt is just the prior); the value is in the grounding,
verification, and per-task model selection — not in reworded instructions.

- CHAT       : stream a single completion. 1 call. The common, cheap path.
- GROUNDED   : stream a single completion with a be-careful-with-facts prior.
               (Web search is a planned add; not wired in this MVP.)
- DELIVERABLE: stream the draft live (good UX), then — if the user supplied
               source material — verify the draft against it with the
               tool-server auditor and append an honest verification footer.
               This is the "can't lie" guarantee made visible.

Yields (kind, text) tuples; kind in {"content", "reasoning"}. The app forwards
content as delta.content and reasoning as delta.reasoning_content.
"""
from . import config, fireworks, style, toolserver
from .depth import classify_depth, CHAT, GROUNDED, DELIVERABLE

SYSTEM_CHAT = (
    "You are a helpful, knowledgeable assistant. Answer directly and concretely. "
    "If you are not sure about something, say so plainly rather than guessing."
)

SYSTEM_GROUNDED = (
    SYSTEM_CHAT
    + " Separate what you actually know from what you are inferring, and flag "
    "uncertainty explicitly. Do not invent specific facts, numbers, dates, names, "
    "or citations."
)

SYSTEM_DELIVERABLE = (
    "You are writing a finished document for the user. Produce clean, "
    "well-structured text in exactly the format requested, ready to use. Write "
    "natural, specific prose — no filler, no clichés, and never leave placeholder "
    "brackets like [Your Name]. Do not invent facts, credentials, metrics, "
    "institutions, or citations that are not given or true; if a needed detail is "
    "unknown, leave a single clearly-marked blank the user can fill in."
)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _text_of(m.get("content")).strip()
    return ""


def _has_images(messages) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    return True
    return False


def _extract_source(messages) -> str:
    """Source material to verify a deliverable against = everything the USER
    supplied except the final instruction. Captures the common 'here are my
    notes … now write it up' flow across turns. We deliberately do NOT use
    assistant turns as ground truth (model output isn't a source).

    MVP limitation: if the source is pasted in the SAME message as the
    instruction, it is not separated out here — verification is skipped rather
    than risk a false check. Improving this (attachment/quote detection) is a
    planned follow-up.
    """
    user_texts = [
        _text_of(m.get("content")).strip()
        for m in messages
        if m.get("role") == "user"
    ]
    if len(user_texts) <= 1:
        return ""
    return "\n\n".join(t for t in user_texts[:-1] if t).strip()


def _with_system(messages, extra):
    """Merge `extra` into an existing system message, or insert one at the top.
    Leaves all user/assistant content (including vision parts) untouched."""
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            base = _text_of(m.get("content"))
            m["content"] = (base + "\n\n" + extra).strip() if base else extra
            return out
    return [{"role": "system", "content": extra}] + out


async def run(messages, *, user_id="", session=None):
    """Drive one chat turn. Async generator of (kind, text)."""
    if not messages:
        yield ("content", "")
        return

    text = _last_user_text(messages)
    depth = classify_depth(text)
    vision = _has_images(messages)

    # Vision turns always go to the vision model as a single streamed pass.
    if vision:
        msgs = _with_system(messages, SYSTEM_CHAT)
        async for kind, t in fireworks.stream(
            msgs, config.VISION_MODEL, max_tokens=config.CHAT_MAX_TOKENS, session=session
        ):
            yield (kind, t)
        return

    # CHAT / GROUNDED → one streamed completion.
    if depth.tier != DELIVERABLE:
        sys = SYSTEM_GROUNDED if depth.tier == GROUNDED else SYSTEM_CHAT
        msgs = _with_system(messages, sys)
        async for kind, t in fireworks.stream(
            msgs, config.CHAT_MODEL, max_tokens=config.CHAT_MAX_TOKENS, session=session
        ):
            yield (kind, t)
        return

    # DELIVERABLE → stream the draft live, then verify against source material.
    sys = SYSTEM_DELIVERABLE
    if depth.needs_style:
        profile = style.get_style_profile(user_id)
        if profile:
            sys += (
                "\n\nThe user's writing voice (emulate tone, rhythm, and "
                "structure — NOT facts):\n" + profile
            )
    msgs = _with_system(messages, sys)

    draft_parts = []
    async for kind, t in fireworks.stream(
        msgs, config.DRAFT_MODEL, max_tokens=config.DRAFT_MAX_TOKENS, session=session
    ):
        if kind == "content":
            draft_parts.append(t)
        yield (kind, t)
    draft = "".join(draft_parts).strip()

    if not config.ENABLE_VERIFICATION or not draft:
        return
    source = _extract_source(messages)
    if len(source) < config.MIN_SOURCE_CHARS:
        return  # nothing to verify against — don't fake a check

    res = await toolserver.verify_grounding(source, draft, session=session)
    if res is None:
        return  # auditor offline — stay silent rather than block the answer
    if res.get("grounded"):
        yield (
            "content",
            "\n\n---\n*✓ Checked against your source material — no unsupported "
            "claims found.*",
        )
    else:
        claims = (res.get("unsupported_claims") or "").strip()
        yield (
            "content",
            "\n\n---\n*⚠ Verification — these statements were not supported by "
            "your source material and may be inaccurate. Please confirm before "
            "relying on them:*\n\n" + claims,
        )
