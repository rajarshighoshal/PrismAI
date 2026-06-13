"""
Citation verification for the PrismAI router outlet.

Extracted from router_fn.py as an independent module. Audits model responses
against search results: checks presence of citations, validity of cited URLs,
and accuracy of claim-to-source attribution.

Dependencies are injected at construction — no circular import on router_fn.py.
"""

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Audits model responses against search results for citation accuracy.

    Checks three things in order:
    1. PRESENCE — every factual claim has an inline citation
    2. VALIDITY — every cited URL appears verbatim in the search results
    3. ATTRIBUTION ACCURACY — each citation actually supports the claim it's attached to
    """

    def __init__(
        self,
        call_llm: Callable,       # async (prompt, model, max_tokens, fallback_chain, log_role, log_chat_id) → str
        emit_status: Callable,    # async (event_emitter, description, done=True) → None
        emit_replace: Callable,   # async (event_emitter, content) → None
        valves,
    ):
        self.call_llm = call_llm
        self.emit_status = emit_status
        self.emit_replace = emit_replace
        self.valves = valves

    # ── Main verification entry point ──────────────────────────────────

    async def verify_citations(
        self,
        response: str,
        search_context: str,
        verifier_fallback_chain: list[str],
    ) -> tuple[bool, str]:
        """Audit the RESPONSE against the SEARCH_RESULTS.

        Returns (passed: bool, reason: str). Fail-open on LLM unavailability
        — the user gets the answer with a warning, not a broken reply.
        """
        # Give the full response and search context to the verifier LLM —
        # don't truncate. The old [:4000] clip on response silently passed
        # fabricated claims in the tail of long answers.
        prompt = (
            "Audit the RESPONSE against the SEARCH_RESULTS below.\n\n"
            "Check THREE things in order:\n"
            "1. PRESENCE — every factual claim (numbers, dates, names, statistics) has an "
            "inline citation: [N] numbered ref, [Source: <url>], or bare URL.\n"
            "2. VALIDITY — every cited URL appears verbatim in SEARCH_RESULTS. "
            "Invented URLs = hallucination = FAIL.\n"
            "3. ATTRIBUTION ACCURACY — each citation actually supports the specific claim "
            "it is attached to. A citation is wrong if the source says X but the claim "
            "says Y, or if the source doesn't address the claim at all. "
            "Misattributed citations = FAIL even if the URL is real.\n\n"
            "A ref mapping to 'Tavily AI Summary' is valid when SEARCH_RESULTS has a "
            "'Tavily AI Summary' section.\n\n"
            "OUTPUT — exactly ONE line:\n"
            "  PASS: <one-sentence reason>\n"
            "  FAIL: <one-sentence reason — name the failing check (presence/validity/attribution)>\n\n"
            "Do NOT reason aloud. Your ENTIRE output is the verdict line.\n\n"
            f"SEARCH_RESULTS:\n{search_context}\n\n"
            f"RESPONSE:\n{response}\n\n"
            "Verdict:"
        )
        verdict = await self.call_llm(
            prompt=prompt,
            model=self.valves.VERIFIER_MODEL,
            max_tokens=200,
            fallback_chain=verifier_fallback_chain,
            log_role="verifier",
            log_chat_id="",
        )
        if not verdict:
            return True, "verifier LLM unavailable — fail-open"

        return self._parse_verdict(verdict)

    def _parse_verdict(self, raw_verdict: str) -> tuple[bool, str]:
        """Parse the verifier LLM's output into (passed, reason).

        Tries structured PASS/FAIL markers first, then falls back to keyword
        analysis for chatty/reasoning models that ignore format instructions.
        """
        # Strip thinking blocks (DeepSeek/Kimi reasoning leakage)
        cleaned = re.sub(r"<think>.*?</think>|<tool_call>.*?</tool_call>",
                         "", raw_verdict, flags=re.DOTALL).strip()

        # Structured markers
        pass_match = re.search(r"\bPASS\b\s*:?\s*([^\n]*)", cleaned, re.IGNORECASE)
        fail_match = re.search(r"\bFAIL\b\s*:?\s*([^\n]*)", cleaned, re.IGNORECASE)

        if pass_match and (not fail_match or pass_match.start() < fail_match.start()):
            reason = pass_match.group(1).strip() or "citations verified"
            return True, reason
        if fail_match:
            reason = fail_match.group(1).strip() or "unspecified"
            return False, reason

        # Keyword fallback for models that emit prose instead of structured verdicts.
        # Word-boundary matching only — "unsupported" must not match "supported".
        lower = cleaned.lower()
        fail_re = re.compile(
            r"\b(hallucinat\w*|fabricat\w*|invent(?:ed|s|ing)?|incorrect|"
            r"unsupported|uncited|mismatch\w*|wrong|fail(?:ed|s|ing)?)\b"
            r"|\bnot\s+supported\b|\bdoes(?:n'?t|\s+not)\s+match\b"
            r"|\bmissing\s+citation\b"
        )
        pass_re = re.compile(
            r"\b(accurate|correct|verified|matches|valid|legitimate|genuine)\b"
        )
        has_fail = bool(fail_re.search(lower))
        has_pass = bool(pass_re.search(lower))

        if has_fail and not has_pass:
            return False, f"keyword-fallback FAIL: {cleaned[:120]}"
        if has_pass and not has_fail:
            return True, f"keyword-fallback PASS: {cleaned[:120]}"

        logger.warning("Verifier verdict unparseable — fail-open. Raw: %s", cleaned[:200])
        return True, "verdict unparseable — fail-open"

    # ── Event emission helpers ─────────────────────────────────────────

    async def emit_status(
        self,
        event_emitter,
        description: str,
        done: bool = True,
    ) -> None:
        """Emit a status event to the OpenWebUI conversation surface."""
        if event_emitter is None or not self.valves.EMIT_STATUS_EVENTS:
            return
        try:
            await event_emitter({
                "type": "status",
                "data": {"description": description, "done": done},
            })
        except Exception as e:
            logger.warning("Event emit failed: %s", e)

    async def emit_replace(
        self,
        event_emitter,
        content: str,
    ) -> None:
        """Emit a replace event to update the displayed message content."""
        if event_emitter is None:
            return
        try:
            await event_emitter({
                "type": "replace",
                "data": {"content": content},
            })
        except Exception as e:
            logger.warning("Replace emit failed: %s", e)
