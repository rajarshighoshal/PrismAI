# PrismAI real-use basket

This is the benchmark PrismAI should optimize for: the user's actual OpenWebUI
distribution, not a generic model leaderboard.

Axes:

- research-with-sources
- MSc psychology / academic writeup
- resume rewrite
- cover letter
- email polish
- normal chat
- honesty trap
- image/table question
- edit existing document

`run.py --selftest` validates the case schema without model calls. A live runner can
be added on top of the same JSONL shape once cases are filled out with stable expected
checks.

Metrics to report for every live run:

- pass/fail checks
- latency seconds
- model-call count
- token usage / estimated cost
- verifier result
- export/file correctness
