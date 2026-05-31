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
import re

from . import config, fireworks, search, style, toolserver
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

SYSTEM_GROUNDED_SEARCH = (
    "Answer using the web search results provided below. Base the answer on them, "
    "cite sources inline as [1], [2] matching the numbered results, and use the "
    "specific facts and numbers they contain. If the results don't cover part of "
    "the question, say so explicitly rather than guessing. Do not invent sources "
    "or facts beyond what the results support."
)

SYSTEM_QUERY = (
    "Turn the user's message into ONE concise web search query that would surface "
    "the facts needed to answer it. Output only the query text — no quotes, no "
    "preamble, no explanation."
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


def _same_message_source(text: str) -> str:
    """Pull pasted source material out of a single instruction message —
    fenced blocks, blockquotes, text after a source:/notes:/context: marker,
    and substantial non-instruction paragraphs. Lets 'write a cover letter from
    this: <paste>' get verified even when it's all one message."""
    t = (text or "").strip()
    if not t:
        return ""
    parts = []
    for m in re.findall(r"```[^\n]*\n(.*?)```", t, flags=re.S):
        parts.append(m.strip())
    for m in re.findall(r"(?m)((?:^>.*(?:\n|$))+)", t):
        parts.append(re.sub(r"(?m)^>\s?", "", m).strip())
    for m in re.findall(
        r"(?is)(?:^|\n)\s*(?:sources?|notes?|context|references?)\s*[:\-]\s*(.+?)(?:\n\s*\n|$)",
        t,
    ):
        parts.append(m.strip())
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(paras) > 1:
        for p in paras:
            if len(p) >= 80 and classify_depth(p).tier != DELIVERABLE:
                parts.append(p)
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n\n".join(out).strip()


def _extract_source(messages) -> str:
    """Source material to verify a deliverable against = everything the USER
    supplied except the final instruction. Combines the cross-turn flow ('here
    are my notes' … 'now write it up') with same-message pasted source. We
    deliberately do NOT use assistant turns as ground truth (model output isn't
    a source)."""
    user_texts = [
        _text_of(m.get("content")).strip()
        for m in messages
        if m.get("role") == "user"
    ]
    if not user_texts:
        return ""
    prior = "\n\n".join(t for t in user_texts[:-1] if t).strip()
    same = _same_message_source(user_texts[-1])
    return "\n\n".join(p for p in (prior, same) if p).strip()


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


async def _make_search_query(text, session) -> str:
    """Compress the turn into a single <=400-char search query (cheap call)."""
    try:
        q = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_QUERY},
             {"role": "user", "content": text}],
            config.QUERY_MODEL,
            max_tokens=64,
            temperature=0.0,
            session=session,
        )
    except Exception:
        q = ""
    q = " ".join((q or "").split()) or text
    return q[: config.QUERY_MAX_CHARS]


def _sources_footer(results) -> str:
    lines = ["\n\n---\n**Sources**"]
    for i, r in enumerate(results, 1):
        label = (r.get("title") or r.get("url") or "").strip()
        url = (r.get("url") or "").strip()
        lines.append(f"{i}. [{label}]({url})" if url else f"{i}. {label}")
    return "\n".join(lines)


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

    # GROUNDED → search the web, ground the answer in results, cite sources.
    if depth.tier == GROUNDED:
        results = []
        if config.ENABLE_WEB_SEARCH:
            query = await _make_search_query(text, session)
            results = await search.search(query, session=session)
        if results:
            ctx = search.format_context(results)
            msgs = _with_system(messages, SYSTEM_GROUNDED_SEARCH + "\n\nWeb search results:\n" + ctx)
            answer_parts = []
            async for kind, t in fireworks.stream(
                msgs, config.CHAT_MODEL, max_tokens=config.CHAT_MAX_TOKENS, session=session
            ):
                if kind == "content":
                    answer_parts.append(t)
                yield (kind, t)
            if config.ENABLE_GROUNDED_VERIFY:
                answer = "".join(answer_parts).strip()
                res = await toolserver.verify_grounding(ctx, answer, session=session)
                if res is not None and not res.get("grounded"):
                    claims = (res.get("unsupported_claims") or "").strip()
                    yield ("content", "\n\n*⚠ Not supported by the cited sources:*\n\n" + claims)
            yield ("content", _sources_footer(results))
            return
        # No results (search off/empty/failed) → answer with a careful prior.
        msgs = _with_system(messages, SYSTEM_GROUNDED)
        async for kind, t in fireworks.stream(
            msgs, config.CHAT_MODEL, max_tokens=config.CHAT_MAX_TOKENS, session=session
        ):
            yield (kind, t)
        return

    # CHAT → one streamed completion. The common, cheap path.
    if depth.tier != DELIVERABLE:
        msgs = _with_system(messages, SYSTEM_CHAT)
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
        return

    claims = (res.get("unsupported_claims") or "").strip()
    if config.ENABLE_REFINE:
        corrected = await _refine(source, draft, claims, session)
        if corrected:
            yield (
                "content",
                "\n\n---\n*⚠ The draft above contained claims not supported by your "
                "source material. Corrected version below (unsupported claims removed "
                "or fixed):*\n\n" + corrected
                + "\n\n*Flagged in the original draft:*\n\n" + claims,
            )
            return
    # Refine disabled or failed → warn-only footer.
    yield (
        "content",
        "\n\n---\n*⚠ Verification — these statements were not supported by your "
        "source material and may be inaccurate. Please confirm before relying on "
        "them:*\n\n" + claims,
    )


async def _refine(source, draft, claims, session) -> str:
    """One grounded rewrite that removes/fixes only the unsupported claims."""
    refine_sys = (
        "Revise the draft so every claim is supported by the SOURCE. Remove or "
        "correct ONLY the listed unsupported claims; keep all supported content, "
        "structure, tone, and format unchanged. Output only the revised document, "
        "no commentary."
    )
    refine_user = (
        f"SOURCE:\n{source}\n\nUNSUPPORTED CLAIMS TO FIX:\n{claims}\n\nDRAFT:\n{draft}"
    )
    try:
        return await fireworks.complete(
            [{"role": "system", "content": refine_sys},
             {"role": "user", "content": refine_user}],
            config.REFINE_MODEL,
            max_tokens=config.DRAFT_MAX_TOKENS,
            session=session,
        )
    except Exception:
        return ""
