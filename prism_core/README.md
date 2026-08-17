# prism-core

**A fail-closed honesty gate for LLM answers grounded in user-provided sources.**

The audit checks a draft against the source material (plus the user's own statements)
and flags only unsupported *facts* — invented credentials, metrics, dates, scope —
while leaving motivation, tone, framing, and hedged language alone. Deterministic
backstops keep it honest in both directions:

- **fail-closed** — if the auditor can't produce a usable verdict (truncated, malformed,
  transport error), the draft is *not* certified. No silent pass.
- **verbatim backstop** — a flagged claim whose exact phrase appears contiguously in the
  source is rescued (auditor false positive); a same-meaning inflation is never rescued.
- **no truncation games** — an over-long source is fitted by *relevance to the draft*,
  not head-cut (head-cutting was the bug where critical content slid past the cut).

Standard library only. No provider, framework, or chat-UI dependency: **the model call
is injected by you.**

## Usage

```python
import asyncio
from prism_core.audit import fact_audit

async def my_complete(messages, model, *, max_tokens, temperature,
                      reasoning_effort, session, label, return_finish):
    # call whatever provider you like; return (raw_text, finish_reason)
    ...

result = asyncio.run(fact_audit(
    full_request="Write my resume summary. I mentored two interns.",
    source="Mentored two interns on an internal tool.",
    candidate="Senior engineer who led a team of 50 engineers.",
    complete_fn=my_complete,
    model="your-model-id",
    max_tokens=2000,
    reasoning_effort="max",
))
# -> {"unsupported": ["led a team of 50 engineers"], "verdict": "FABRICATION"}
# or {"verdict": "ERROR", ...} when no usable verdict arrives (fail closed)
```

## Contents

| module | what it holds |
|---|---|
| `prism_core.audit` | `fact_audit()` + the audit prompt (`SYSTEM_FACT_AUDIT`) |
| `prism_core.verifier` | token normalization, verbatim backstop, relevance-fit, fail-closed sentinel, `<source>` delimiter |
| `prism_core.messages` | channel-neutral chat-message helpers (content extraction, image split, inline-source heuristic) |

## License

MIT (this directory). The parent project, [PrismAI](https://github.com/rajarshighoshal/PrismAI),
is AGPL-3.0 — only `prism_core/` is MIT.

## Status

This is what I use in production, extracted so my own code has a real boundary.
Published without promises: no roadmap, PRs may not get merged.
