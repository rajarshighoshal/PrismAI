"""The can't-lie gate: fact verification for every deliverable.

One fact-integrity check for ANY kind of writing — no document-type branching.
The auditor sees the request, the FULL source, and the draft side by side and
flags only unsupported FACTS (motivation/opinion/framing always pass). A flag
whose exact words sit verbatim in the source is a false positive and is kept
(_claim_verbatim_in_source — the deterministic backstop). Genuine flags get a
surgical refine, a recheck, one rewrite, and only then a refuse-AND-help block.

Grounding = uploaded sources + recalled facts + the USER'S OWN statements
(typo-tolerant; their current word beats an older file) + today's date.
"""
import json
import logging
import re

from . import config, fireworks
from .owui import _SOURCE_BLOCK_RE, _all_user_text, _last_user_text
from .timectx import _now_line
from .prompts import SYSTEM_FACT_AUDIT, SYSTEM_GATE, SYSTEM_CHANGE_SUMMARY

log = logging.getLogger(__name__)


def _has_citation_markers(text: str) -> bool:
    return bool(
        re.search(r"\[[1-9][0-9]*\]", text or "")
        or re.search(r"(?im)^\s*(?:sources?|references?)\s*:", text or "")
    )


async def _needs_verification(messages, candidate: str, source: str, *, session=None) -> bool:
    if not config.ENABLE_GROUNDING_GATE:
        return bool(source)
    if not candidate.strip():
        return False
    payload = {
        "latest_user": _last_user_text(messages),
        "source_available": bool(source.strip()),
        "draft": candidate[:6000],
    }
    try:
        raw = await fireworks.complete(
            [
                {"role": "system", "content": SYSTEM_GATE},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            config.GROUNDING_GATE_MODEL,
            max_tokens=120,
            temperature=0.0,
            session=session,
            label="gate:verify",
        )
        match = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(match.group(0) if match else raw)
        return bool(data.get("needs_verification"))
    except Exception:
        return bool(source)


async def _refine_complete(prompt: str, user: str, *, prose=None, session=None) -> str:
    """Run a refine pass on the SAME prose model that wrote the draft when one was
    used (so paid Opus/GPT prose isn't silently reverted to the open model on a
    fix); otherwise use the default open REFINE_MODEL."""
    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    try:
        if prose is not None:
            client, model = prose
            return await client.complete(
                msgs, model, max_tokens=config.DRAFT_MAX_TOKENS,
                temperature=config.WRITER_TEMPERATURE, session=session,
                label="refine",
            )
        return await fireworks.complete(
            msgs, config.REFINE_MODEL,
            max_tokens=config.DRAFT_MAX_TOKENS,
            temperature=config.WRITER_TEMPERATURE, session=session,
            label="refine",
        )
    except Exception:
        return ""


# Corrections are SURGICAL by default — the draft was already shown to the user, so
# the verifier patches only the flagged spans and leaves the rest untouched. Only
# when a surgical patch can't clear the audit (the draft is too wrong to fix in
# place) do we escalate to a full rewrite-with-feedback by the upstream writer.
_SURGICAL_DIRECTIVE = (
    "Make the SMALLEST edit that fixes the flagged problems: change ONLY the specific "
    "words or sentences that are unsupported, and keep every other sentence, the "
    "wording, structure, tone, and length EXACTLY as written. Do not re-style, "
    "re-order, expand, or rewrite anything you are not explicitly fixing."
)


_REWRITE_DIRECTIVE = (
    "The draft has too many unsupported claims to patch in place. Write it again from "
    "the user's request and the SOURCE, fully addressing the feedback. Use ONLY real, "
    "supported facts — invent nothing — and keep the requested format and length."
)


def _edit_directive(rewrite: bool) -> str:
    return _REWRITE_DIRECTIVE if rewrite else _SURGICAL_DIRECTIVE


_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm_token_str(text) -> str:
    """Lowercase, reduce to [a-z0-9] tokens joined by single spaces and wrapped in
    spaces, so a substring test matches only on whole-token boundaries."""
    return " " + " ".join(_WORD_RE.findall(str(text).lower())) + " "


def _claim_verbatim_in_source(phrase, source_norm: str) -> bool:
    """True ONLY when a flagged phrase appears near-verbatim and CONTIGUOUS in the
    source — the narrow case where the auditor flagged something literally present, so
    it's a clear false positive to keep. Unlike a word-overlap score this never rescues
    a semantic inflation ('led' for a sourced 'collaborated', '$1.5M/year' for '/month',
    'p<0.001' for 'p<0.05'): those are not contiguous spans of the source, so the
    auditor's flag stands and the claim is stripped. A tight backstop BEHIND the
    full-source auditor, which makes the real semantic judgement."""
    toks = _WORD_RE.findall(str(phrase).lower())
    if len(toks) < 2:  # too short for a reliable verbatim match — defer to the auditor
        return False
    return (" " + " ".join(toks) + " ") in source_norm


def _fit_audit_source(source: str, draft: str, budget: int) -> str:
    """Fit `source` into `budget` chars for the auditor WITHOUT head-truncating.

    Head-truncation was the over-strip bug: a long job posting in front pushed the
    résumé past the cut, so grounded credentials looked unsupported. Real documents
    (résumé + posting ~11k) fit the budget whole and return unchanged. Only a
    genuinely oversized source is trimmed — and then by *relevance to the draft*
    (keep the paragraphs whose vocabulary the draft actually uses, in original
    order), so the spans a claim is grounded in survive instead of the tail being
    silently dropped. For sources far beyond one model's window the correct path is
    chunk-and-audit (union of supported); this helper covers the realistic middle.
    """
    source = source.strip()
    if len(source) <= budget:
        return source
    draft_words = set(_WORD_RE.findall(draft.lower()))
    paras = [p for p in re.split(r"\n\s*\n", source) if p.strip()]
    scored = [
        (i, len(draft_words & set(_WORD_RE.findall(p.lower()))), p)
        for i, p in enumerate(paras)
    ]
    kept, used = [], 0
    for i, _score, p in sorted(scored, key=lambda t: t[1], reverse=True):
        if used + len(p) > budget:
            continue
        kept.append((i, p))
        used += len(p) + 2
    kept.sort()
    return "\n\n".join(p for _i, p in kept) or source[:budget]


# The auditor could NOT return a usable verdict (call failed / empty / unparseable /
# truncated). This is NOT 'clean' — the can't-lie layer FAILS CLOSED on it.
_AUDIT_ERROR = "ERROR"


class _AuditUnavailable(Exception):
    """Raised when the honesty auditor can't produce a usable verdict, so the draft must
    not be certified. The verify flow catches it and blocks (fail closed)."""


async def _fact_audit(full_request: str, source: str, candidate: str, *, session=None, raw_source=None):
    """The single fact-integrity verifier for ANY written deliverable. Sees the USER
    REQUEST, the full SOURCE MATERIAL, and the DRAFT side by side and flags only
    unsupported FACTUAL claims (against the user's stated facts + SOURCE + common
    knowledge); motivation, opinion, and framing pass. `raw_source` (if given) mirrors
    `source` and is used only for the diagnostic.
    Returns {unsupported, verdict:FABRICATION|CLEAN} on a real verdict, or {verdict:ERROR}
    when it could NOT get a usable verdict after a retry — a FAIL-CLOSED signal, never a
    silent CLEAN. An empty draft has no claims -> CLEAN."""
    if not candidate.strip():
        return {"unsupported": [], "verdict": "CLEAN"}
    fitted = _fit_audit_source(source, candidate, config.AUDIT_SOURCE_BUDGET)
    # De-dup only: the source lives in SOURCE MATERIAL, so drop the identical <source>
    # blocks from the request (no information lost). NO truncation — the auditor sees
    # the whole request, the whole source, and the whole draft.
    request = _SOURCE_BLOCK_RE.sub("", full_request).strip()
    user = (
        f"USER REQUEST (instructions; the FACTS are in SOURCE MATERIAL):\n{request}\n\n"
        f"SOURCE MATERIAL:\n{fitted if fitted else '(none)'}\n\n"
        f"DRAFT:\n{candidate}"
    )
    for attempt in range(2):  # one retry for a transient hiccup / formatting fluke
        try:
            raw, finish = await fireworks.complete(
                [{"role": "system", "content": SYSTEM_FACT_AUDIT},
                 {"role": "user", "content": user}],
                config.HONESTY_MODEL,
                max_tokens=config.AUDIT_MAX_TOKENS,
                temperature=0.0,
                reasoning_effort=config.AUDIT_REASONING_EFFORT,
                session=session,
                label="audit",
                return_finish=True,
            )
            match = re.search(r"\{.*\}", raw, flags=re.S)
            data = None
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None  # truncated mid-object / malformed
            if isinstance(data, dict) and data.get("verdict"):
                if config.LOG_SOURCE_DIAG:
                    source_norm = _norm_token_str(raw_source or fitted)
                    flagged = data.get("unsupported") or []
                    false_pos = sum(1 for f in flagged if _claim_verbatim_in_source(f, source_norm))
                    log.info(f"[audit-diag] reasoning={config.AUDIT_REASONING_EFFORT} audit_src_chars={len(fitted)} "
                             f"verdict={data.get('verdict')} flagged={len(flagged)} verbatim_false_pos={false_pos}")
                return data
            # No usable verdict. If truncated, a retry won't help (same input, same cap) ->
            # fail closed now; otherwise retry once for a transient garble.
            if finish == "length":
                log.warning("[audit] verdict truncated/unparseable (finish=length) -> FAIL CLOSED")
                return {"verdict": _AUDIT_ERROR, "reason": "truncated"}
            log.warning(f"[audit] no parseable verdict (attempt {attempt + 1}/2)")
        except Exception as e:
            log.warning(f"[audit] call failed (attempt {attempt + 1}/2): {type(e).__name__}: {e}")
    log.warning("[audit] no usable verdict after retry -> FAIL CLOSED")
    return {"verdict": _AUDIT_ERROR, "reason": "no_verdict"}


async def _refine_facts(full_request: str, source: str, candidate: str, unsupported,
                        *, rewrite=False, prose=None, user_said: str = "", session=None) -> str:
    """Remove the unsupported FACTS the verifier flagged, keeping everything else —
    motivation, framing, tone, structure — exactly. Surgical by default; full rewrite
    only when the draft is too wrong to patch."""
    listed = "\n".join(f"- {u}" for u in unsupported) if unsupported else "(unsupported factual claims)"
    prompt = (
        _edit_directive(rewrite) + " "
        "Remove or neutrally rephrase each listed claim so the draft asserts no fact "
        "the user or SOURCE did not support. Do NOT invent replacements. If only PART of "
        "a flagged claim is unsupported (an invented elaboration attached to a fact the "
        "user did state), remove ONLY the unsupported part — never delete a fact the user "
        "themselves stated or the source contains. KEEP all motivation, interest, framing, "
        "tone, and structure exactly — only the unsupported FACTS change. If removing a "
        "claim leaves the draft thinner, that is fine; do not pad with invented detail. "
        "Output only the revised deliverable."
    )
    # The user's own statements are a HARD keep-list: when a flagged sentence mixes an
    # invented elaboration with a fact the user stated, only the elaboration goes — the
    # live smoke run caught the refiner deleting the whole sentence about half the time
    # when this was mere prompt nuance instead of an explicit constraint.
    keep = (f"FACTS THE USER THEMSELVES STATED — these are established; NEVER remove them, "
            f"even if a flagged claim contains one (trim only the unsupported part). The "
            f"user types with typos: match by MEANING, not spelling, and their current "
            f"statements override older uploaded documents:\n"
            f"{user_said}\n\n") if user_said.strip() else ""
    user = (
        f"USER REQUEST:\n{full_request}\n\n"
        f"SOURCE MATERIAL:\n{source if source.strip() else '(none)'}\n\n"
        + keep +
        f"UNSUPPORTED CLAIMS TO FIX:\n{listed}\n\n"
        f"DRAFT:\n{candidate}"
    )
    return await _refine_complete(prompt, user, prose=prose, session=session)


def _facts_block_msg(unsupported) -> str:
    """Refuse AND help — a bare refusal is honest but useless (the A/B lost honesty-trap
    cases on helpfulness, not honesty). Name what's blocked, why, and the concrete ways
    forward, so the user can act immediately."""
    listed = "\n".join(f"- {u}" for u in unsupported) if unsupported else ""
    return (
        "I can't present those claims as true — they aren't in your sources or anything "
        "you've told me, and inventing them could genuinely hurt you if challenged:\n\n"
        + listed
        + "\n\nThree ways forward, pick any:\n"
        "1. **Give me the real details** for the points above and I'll include them accurately.\n"
        "2. **I write the strongest truthful version now**, using only what you've given me — "
        "often the honest framing reads better than the inflated one.\n"
        "3. **Point me at a source** (a file, a link, or just tell me the facts) and I'll "
        "verify and work it in."
    )


async def _summarize_correction(before: str, after: str, *, session=None) -> str:
    """One cheap call describing WHAT the verifier changed, so the chat can show the
    correction in a sentence or two instead of re-dumping the whole deliverable."""
    if not (before.strip() and after.strip()):
        return ""
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_CHANGE_SUMMARY},
             {"role": "user", "content": f"BEFORE:\n{before[:8000]}\n\nAFTER:\n{after[:8000]}"}],
            config.HONESTY_MODEL, max_tokens=200, temperature=0.0, session=session,
            label="summarize",
        )
        return (raw or "").strip()
    except Exception:
        return ""


async def _verified_or_blocked(messages, candidate: str, source: str, *, recall_context: str = "", prose=None, force: bool = False, session=None):
    """ONE fact-integrity check for any kind of writing — email, resume, letter,
    report, research, chat. No document-type branching: flag only unsupported FACTS,
    leave motivation / opinion / framing untouched, surgically correct, escalate to a
    rewrite if a patch can't clear it, and block only if facts still can't be made
    truthful."""
    if not config.ENABLE_VERIFICATION:
        return "ok", candidate

    # Recalled facts are the user's OWN earlier statements, surfaced only when a long
    # chat overflowed the context budget (see run()). They are established context, so
    # the verifier must see them — otherwise a correctly-recalled fact gets flagged as
    # a fabrication and stripped. For normal chats recall_context is "" (no-op).
    _rc = (recall_context or "").strip()
    _recall_extra = ("\n\nEARLIER IN THIS CONVERSATION (the user already stated):\n" + _rc) if _rc else ""
    # The USER is the authority on their OWN facts. Anything they state in the conversation
    # ("I've now finished my MS", "I work on geometric probes") grounds their own claims —
    # alongside the uploaded source and recalled facts. Without this the verifier strips
    # facts the user ADDED in chat just because they aren't in the original files (the
    # exact bug that 'corrected out' the user's real geometric-probes research). The
    # honesty guarantee still holds: the MODEL can't invent anything the user didn't give.
    _user_said = _all_user_text(messages)
    # Today's date joins the SOURCE (not just the request): the audit prompt tells the
    # auditor facts live in SOURCE MATERIAL, and in-source the verbatim backstop protects
    # a dated letterhead mechanically (flash at low reasoning flaked on it otherwise).
    grounding_source = "\n\n".join(p for p in (_now_line(), source, _rc, _user_said) if p and p.strip())

    # Verify only a FACTUAL DELIVERABLE — the cheap classifier decides, and an exported
    # file always counts (a document the user will rely on). The mere PRESENCE of a
    # source does not force a verification: that blanket rule made the honesty pass fire
    # on every casual turn that happened to have pasted text or an image, and strip
    # reasonable asides ("software like Keybr or TypingClub") from an opinion. Opinion,
    # assessment, and Q&A about an attachment are not deliverables and skip straight through.
    needs = force or await _needs_verification(messages, candidate, grounding_source, session=session)
    if not needs:
        return "ok", candidate

    # The auditor and refiner must know today's date too — the writer is TOLD the current
    # date (system prompt), so a dated letterhead is established context, not a fabrication
    # to strip (the bug that "corrected" a real date into a [Date] placeholder).
    full_request = _now_line() + "\n\n" + _all_user_text(messages) + _recall_extra

    if config.ENABLE_HONESTY_AUDIT:
        # Hand the auditor the FULL source (relevance-fitted inside _fact_audit), the
        # request, and the draft — side by side — and let it make the SEMANTIC call. No
        # distilled fact list in between: a lossy summary dropped real credentials (so the
        # auditor flagged them) and can't preserve verbatim figures anyway. With the whole
        # source in view it can tell 'led' is unsupported when the source says
        # 'collaborated'; a tight verbatim backstop then keeps only the flags whose exact
        # phrase is literally in the source (the auditor mis-flagging a quote).
        source_norm = _norm_token_str(grounding_source)

        async def _flags(text):
            """Genuinely-unsupported claims in `text`: the auditor's flags MINUS any whose
            exact phrase appears contiguously in the source (a literal false positive). The
            auditor judges meaning; the backstop only rescues verbatim quotes — it never
            keeps a same-words inflation. [] means clean. Raises _AuditUnavailable when the
            auditor returned NO usable verdict — so the flow fails closed instead of reading
            an unavailable audit as 'clean'."""
            a = await _fact_audit(full_request, grounding_source, text, session=session, raw_source=grounding_source)
            if a and str(a.get("verdict", "")).upper() == _AUDIT_ERROR:
                raise _AuditUnavailable()
            if not (a and str(a.get("verdict", "")).upper().startswith("FAB")):
                return []
            return [u for u in (a.get("unsupported") or []) if not _claim_verbatim_in_source(u, source_norm)]

        try:
            unsupported = await _flags(candidate)
            if unsupported:
                refined = await _refine_facts(full_request, grounding_source, candidate, unsupported, prose=prose, user_said=_user_said, session=session)
                remaining = (await _flags(refined)) if refined else unsupported
                if remaining:  # surgical patch left genuine fabrications — one full-rewrite attempt
                    refined = await _refine_facts(full_request, grounding_source, candidate, unsupported, rewrite=True, prose=prose, user_said=_user_said, session=session)
                    remaining = (await _flags(refined)) if refined else unsupported
                if refined and not remaining:
                    candidate = refined
                else:
                    return "unsupported_claims", _facts_block_msg(unsupported)
        except _AuditUnavailable:
            # The honesty check could not run (truncated/unavailable verdict). For a can't-lie
            # product the only safe move is to fail CLOSED — never present an unverified draft.
            log.warning("[verify] auditor unavailable -> failing closed (not certifying draft)")
            return ("audit_unavailable",
                    "I couldn't complete the honesty check on this draft (the verifier didn't "
                    "return a usable result), so I won't present it as verified. Please try again.")

    # Citations must rest on real retrieved sources.
    if not grounding_source.strip() and _has_citation_markers(candidate):
        return (
            "citation_without_source",
            "The previous draft included citations or source labels, but no sources were actually supplied or retrieved.",
        )
    return "ok", candidate
