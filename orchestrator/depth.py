"""Agentic depth classifier — decide how much machinery a turn deserves.

Core principle (user's, firm): cost must be task-dependent, not a fixed
multiplier. Most turns are plain chat and should cost ONE model call. Only when
the user actually wants a grounded deliverable do we spend on
search/verification/style. Orchestration is CONTROL FLOW, not prompting.

Pure + dependency-free so it can be unit-tested offline. Returns a Depth
decision; the pipeline acts on it.

Depth tiers
-----------
CHAT       : plain conversational turn. 1 call. No search, no verify, no style.
GROUNDED   : a factual/research ask. Add web grounding + (if available) the
             citation/verification path. ~1 call + grounding.
DELIVERABLE: produce a document (cover letter, paper, report, resume, email).
             Add style-memory + grounding + post-hoc verify_grounding. The
             expensive tier — only entered on a real document request.

Intent-shift handling
---------------------
A turn is classified on its own text, so "actually, write this up as a research
paper" naturally lands in DELIVERABLE mid-conversation without any mode toggle —
exactly the "flows into deliverable mode" behavior we want.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Verbs/nouns that signal the user wants a produced artifact, not just an answer.
_DELIVERABLE_RE = re.compile(
    r"\b("
    r"write\s+(me\s+)?(a|an|my|the)\s+"
    r"(cover\s*letter|letter|essay|paper|report|proposal|statement|bio|"
    r"article|blog|post|memo|summary|draft|review|abstract|readme|"
    r"resume|cv|cover|email|response|reply)"
    r"|draft\s+(a|an|my|the)\b"
    r"|compose\s+(a|an|my|the)\b"
    r"|turn\s+(this|these|it|those|my\s+notes|these\s+notes)\s+into\b"
    r"|rewrite\s+(this|it|the)\b"
    r"|polish\s+(this|my)\b"
    r"|make\s+(this|it)\s+(into\s+)?(a|an)\s+\w+"
    r"|export\s+(this|it|as)\b"
    r"|write\s+(this|it|that)\s+(up\s+)?(as|into)\b"
    r")",
    re.IGNORECASE,
)

# Explicit "produce a file" intent — strongest deliverable signal.
_EXPORT_RE = re.compile(
    r"\b(export|download|\.docx|\.pdf|word\s+doc|pdf|as\s+a\s+(docx|pdf|file))\b",
    re.IGNORECASE,
)

# Factual / research signals — wants grounding, may not be a document.
_GROUNDED_RE = re.compile(
    r"\b("
    r"search|look\s+up|google|find\s+(out|online|sources?)|cite|citation|"
    r"reference|doi|according\s+to|latest|current|recent|news|"
    r"what'?s\s+the\s+latest|as\s+of\s+(today|now|\d{4})|"
    r"summari[sz]e\s+(the|this)\s+(paper|article|study|source|url|https?://)|"
    r"fact[-\s]?check|verify|is\s+it\s+true"
    r")\b",
    re.IGNORECASE,
)

# Pure-chat tells — short, conversational, no production ask. Used only as a
# tie-breaker; absence of deliverable/grounded signals already implies CHAT.
_CHAT_HINT_RE = re.compile(
    r"^\s*(hi|hey|hello|thanks|thank\s+you|ok(ay)?|cool|nice|lol|"
    r"what\s+do\s+you\s+think|how\s+are\s+you|got\s+it)\b",
    re.IGNORECASE,
)

CHAT = "CHAT"
GROUNDED = "GROUNDED"
DELIVERABLE = "DELIVERABLE"


@dataclass
class Depth:
    tier: str            # CHAT | GROUNDED | DELIVERABLE
    wants_export: bool   # explicit file output requested
    reason: str          # short human-readable why (for status emit / logs)

    @property
    def needs_grounding(self) -> bool:
        return self.tier in (GROUNDED, DELIVERABLE)

    @property
    def needs_style(self) -> bool:
        return self.tier == DELIVERABLE

    @property
    def needs_verification(self) -> bool:
        # Verify documents and grounded factual answers; never plain chat.
        return self.tier in (GROUNDED, DELIVERABLE)


def classify_depth(text: str) -> Depth:
    """Classify one user message into a Depth tier. Pure function."""
    t = (text or "").strip()
    if not t:
        return Depth(CHAT, False, "empty")

    wants_export = bool(_EXPORT_RE.search(t))
    is_deliverable = bool(_DELIVERABLE_RE.search(t)) or wants_export
    is_grounded = bool(_GROUNDED_RE.search(t))

    if is_deliverable:
        return Depth(
            DELIVERABLE,
            wants_export,
            "document request" + (" + export" if wants_export else ""),
        )
    if is_grounded:
        return Depth(GROUNDED, False, "factual/research request")
    # Default: cheap chat. (A short greeting just makes this more certain.)
    why = "casual chat" if _CHAT_HINT_RE.search(t) else "no deliverable/grounding signal"
    return Depth(CHAT, False, why)
