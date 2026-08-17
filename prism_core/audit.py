"""The fact-integrity audit — the reusable heart of PrismAI's can't-lie gate.

Provider-injected: the model call arrives as a callable, so this module never imports
a provider, a config, or any OpenWebUI code. Stdlib + prism_core.verifier only.

Host contract for `complete_fn` (matches PrismAI's provider clients):

    await complete_fn(messages, model, *, max_tokens=int, temperature=float,
                      reasoning_effort=str | None, session=..., label=str,
                      return_finish=True) -> (raw_text: str, finish_reason: str)

Behavior (identical to the orchestrator's built-in audit): the auditor sees the full
REQUEST, the relevance-fitted SOURCE, and the DRAFT side by side; it flags only
unsupported FACTS. A de-dup strips <source> blocks from the request (the same text
already lives in SOURCE MATERIAL) — never truncation. One retry on a transient
garble; a truncated or unparseable verdict FAILS CLOSED ({verdict: AUDIT_ERROR}).
"""
import json
import logging
import re

from .verifier import (
    AUDIT_ERROR,
    SOURCE_BLOCK_RE,
    claim_verbatim_in_source,
    fit_audit_source,
    norm_token_str,
)

log = logging.getLogger(__name__)

SYSTEM_FACT_AUDIT = (
    "You are a fact-integrity verifier for an assistant's written DRAFT. The writing can "
    "be anything — a message, email, resume, letter, bio, summary, or a research, "
    "project, or class report. The TYPE does not matter; you check only that it invents "
    "no FACTS.\n"
    "The user's own statements are authoritative for their own facts, two rules follow: "
    "(1) users type with TYPOS — a draft claim that is a cleaned-up spelling/grammar of "
    "something the user stated is SUPPORTED (match meaning, not spelling); (2) when the "
    "user's CURRENT statement conflicts with an older uploaded document, the user's "
    "statement WINS — people's situations change after their files were written.\n"
    "You are given the USER REQUEST (which mixes facts the user states with instructions "
    "about what to write), SOURCE MATERIAL (uploaded documents, retrieved sources, prior "
    "context), and the DRAFT.\n"
    "Flag every VERIFIABLE FACTUAL claim in the DRAFT not supported by the user's stated "
    "facts, the SOURCE, or genuine common knowledge:\n"
    "- about the user: credentials, employers, titles, years of experience, education, "
    "metrics, revenue, team sizes, awards, or specific past projects/events/experiences "
    "asserted as having happened;\n"
    "- about the world: statistics, dates, names, quantities, citations, study findings, "
    "technical or historical facts;\n"
    "- any invented backstory or event presented as real.\n"
    "CRITICAL: an INSTRUCTION to include or 'emphasize' something is NOT evidence it is "
    "true — 'emphasize my 8 years of leadership' does not make '8 years of leadership' a "
    "supported fact.\n"
    "NEVER flag content that cannot be true or false: motivation, interest, enthusiasm, "
    "intent ('eager to', 'drawn to', 'committed to learning', 'hope to contribute'), "
    "opinions, framing, aspirations, tone, structure, and hedged or forward-looking "
    "statements. Generic, plausible interest in a role, topic, field, or collaboration "
    "is fine even if unstated. Genuine common knowledge needs no source. Facts the user "
    "DID give (and reasonable paraphrase) are supported — never flag them.\n"
    "Output strict JSON only: {\"unsupported\": [\"exact phrase\", ...], \"verdict\": "
    "\"FABRICATION\" if any unsupported factual claim exists, else \"CLEAN\"}."
)


async def fact_audit(full_request: str, source: str, candidate: str, *,
                     complete_fn, model: str, max_tokens: int,
                     reasoning_effort: str = "max", source_budget: int = 60000,
                     session=None, raw_source=None, diag: bool = False) -> dict:
    """Audit `candidate` against `source` (+ the user's own request text) for unsupported
    facts. Returns {"unsupported": [...], "verdict": "FABRICATION"|"CLEAN"} or
    {"verdict": AUDIT_ERROR} when the auditor produces no usable verdict — FAILS CLOSED."""
    if not candidate.strip():
        return {"unsupported": [], "verdict": "CLEAN"}
    fitted = fit_audit_source(source, candidate, source_budget)
    # De-dup only: the source lives in SOURCE MATERIAL, so drop the identical <source>
    # blocks from the request (no information lost). NO truncation — the auditor sees
    # the whole request, the whole source, and the whole draft.
    request = SOURCE_BLOCK_RE.sub("", full_request).strip()
    user = (
        f"USER REQUEST (instructions; the FACTS are in SOURCE MATERIAL):\n{request}\n\n"
        f"SOURCE MATERIAL:\n{fitted if fitted else '(none)'}\n\n"
        f"DRAFT:\n{candidate}"
    )
    for attempt in range(2):  # one retry for a transient hiccup / formatting fluke
        try:
            raw, finish = await complete_fn(
                [{"role": "system", "content": SYSTEM_FACT_AUDIT},
                 {"role": "user", "content": user}],
                model,
                max_tokens=max_tokens,
                temperature=0.0,
                reasoning_effort=reasoning_effort,
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
                if diag:
                    source_norm = norm_token_str(raw_source or fitted)
                    flagged = data.get("unsupported") or []
                    false_pos = sum(1 for f in flagged if claim_verbatim_in_source(f, source_norm))
                    log.info(f"[audit-diag] reasoning={reasoning_effort} audit_src_chars={len(fitted)} "
                             f"verdict={data.get('verdict')} flagged={len(flagged)} verbatim_false_pos={false_pos}")
                return data
            # No usable verdict. If truncated, a retry won't help (same input, same cap) ->
            # fail closed now; otherwise retry once for a transient garble.
            if finish == "length":
                log.warning("[audit] verdict truncated/unparseable (finish=length) -> FAIL CLOSED")
                return {"verdict": AUDIT_ERROR, "reason": "truncated"}
            log.warning(f"[audit] no parseable verdict (attempt {attempt + 1}/2)")
        except Exception as e:
            log.warning(f"[audit] call failed (attempt {attempt + 1}/2): {type(e).__name__}: {e}")
    log.warning("[audit] no usable verdict after retry -> FAIL CLOSED")
    return {"verdict": AUDIT_ERROR, "reason": "no_verdict"}
