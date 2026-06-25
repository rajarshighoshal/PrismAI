"""Agent loop state — replaces the boolean-flag spaghetti in agent.run()."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    """All mutable state for one agent loop turn.

    Every flag that was previously a scattered local variable in run()
    is tracked here with clear names and initialization. The loop reads
    and updates this state through well-defined transitions.
    """

    # ── Tool execution counters ──────────────────────────────────────
    tool_call_count: int = 0
    web_search_count: int = 0
    repair_steps: int = 0

    # ── Collected outputs ────────────────────────────────────────────
    tool_sources: list[str] = field(default_factory=list)
    export_links: list[tuple[str, str]] = field(default_factory=list)  # (filename, url)
    pending_exports: list[dict] = field(default_factory=list)

    # ── Polish configuration ─────────────────────────────────────────
    polish_voice: Optional[str] = None       # e.g. "gpt-5.5", "opus", "sonnet"
    polish_voice_pass: Optional[str] = None  # e.g. "warm", "formal", "none"

    # ── Flow control flags (one-shot nudges) ─────────────────────────
    budget_note_added: bool = False   # tool budget exhausted note injected?
    edit_nudged: bool = False         # edit re-export nudge injected?
    textual_tool_nudged: bool = False # DSML text-tool-call recovery nudge injected?

    # ── Delivery tracking ────────────────────────────────────────────
    streamed_live: bool = False       # was content already sent to user?
    filed_deliverable: bool = False   # does the file carry the deliverable body?
    fugu_escalated: bool = False     # already tried Fugu escalation this turn?
