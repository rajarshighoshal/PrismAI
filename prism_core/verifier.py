"""PrismAI honesty verifier — pure, provider-agnostic core.

The reusable heart of the "can't-lie" gate, with NO dependency on OpenWebUI, the model
provider, or the orchestrator. It holds the deterministic pieces the honesty guarantee
rests on:
  - token normalization + the verbatim backstop (rescue a mis-flagged real quote, never
    a semantic inflation),
  - relevance-fit of an over-long source into an audit budget (never head-truncate),
  - citation-marker detection,
  - the fail-closed sentinel/exception used when the auditor returns no usable verdict.

The audit MODEL CALL is injected by the host, so this module never imports a provider.
Publish-ready: depends only on the standard library.
"""
import re

# Whole-token matcher shared by normalization and the verbatim backstop.
WORD_RE = re.compile(r"[a-z0-9]+")

# PrismAI's source-block delimiter convention: attached/retrieved material travels
# inside <source ...>...</source> blocks. The channel adapter (e.g. the OpenWebUI
# adapter) parses them out; the audit strips their duplicates from the request.
SOURCE_BLOCK_RE = re.compile(r"<source\b[^>]*>(.*?)</source>", re.S | re.I)

# The auditor could NOT return a usable verdict (call failed / empty / unparseable /
# truncated). This is NOT 'clean' — the can't-lie layer FAILS CLOSED on it.
AUDIT_ERROR = "ERROR"


class AuditUnavailable(Exception):
    """Raised when the honesty auditor can't produce a usable verdict, so the draft must
    not be certified. The host's verify flow catches it and blocks (fail closed)."""


def has_citation_markers(text: str) -> bool:
    return bool(
        re.search(r"\[[1-9][0-9]*\]", text or "")
        or re.search(r"(?im)^\s*(?:sources?|references?)\s*:", text or "")
    )


def norm_token_str(text) -> str:
    """Lowercase, reduce to [a-z0-9] tokens joined by single spaces and wrapped in
    spaces, so a substring test matches only on whole-token boundaries."""
    return " " + " ".join(WORD_RE.findall(str(text).lower())) + " "


def claim_verbatim_in_source(phrase, source_norm: str) -> bool:
    """True ONLY when a flagged phrase appears near-verbatim and contiguous in the source
    — a deterministic backstop for false-positive flags. Never rescues semantic inflations."""
    toks = WORD_RE.findall(str(phrase).lower())
    if len(toks) < 2:  # too short for a reliable verbatim match — defer to the auditor
        return False
    return (" " + " ".join(toks) + " ") in source_norm


def fit_audit_source(source: str, draft: str, budget: int) -> str:
    """Fit `source` into `budget` chars by relevance to the draft, not head-truncation —
    head-truncation was the bug where a long prefix pushed critical content past the cut."""
    source = source.strip()
    if len(source) <= budget:
        return source
    draft_words = set(WORD_RE.findall(draft.lower()))
    paras = [p for p in re.split(r"\n\s*\n", source) if p.strip()]
    scored = [
        (i, len(draft_words & set(WORD_RE.findall(p.lower()))), p)
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
