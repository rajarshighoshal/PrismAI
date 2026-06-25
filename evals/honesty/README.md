# Honesty eval

Turns PrismAI's can't-lie guarantee from an *assertion* into *measured numbers* — the
TRINITY-style ablation table the verifier currently lacks (its decisions live in code
comments: flash-vs-pro at `config.py:172`, max-reasoning audit, the verbatim backstop).

## What it measures

The guarantee has two failure modes that pull against each other; a real benchmark needs
both:

| Failure mode | Metric | The bug it catches |
|---|---|---|
| **Miss** — an invented fact slips through | `catch (TPR)` | the lie reaches the user |
| **Over-strip** — a real, supported fact is flagged | `over-strip` | the verifier deletes the user's true credential |

`precision` and `F1` combine them. A model that flags *everything* scores TPR 100% and
over-strip 100% — useless, and the table makes that obvious at a glance.

## How it works

For each labeled case (`cases.jsonl`) the harness calls the **real** auditor
(`verifier._fact_audit`) with grounding built exactly as `verifier._verified_or_blocked`
builds it — SOURCE material **plus the user's own request text**, so a fact the *user*
stated counts as grounded (the geometric-probes / user-authority axis). It optionally
applies the **real** verbatim backstop, then scores by marker presence in the flagged
claims.

Ablation grid (each row = a config comment made measurable):

- `flash · max · backstop` — the live default
- `flash · max · NO backstop` — what the verbatim backstop is worth (expect over-strip ↑)
- `flash · low` / `flash · none` — is max-reasoning audit actually buying accuracy?
- `pro · max` — is the heavier auditor worth 5× the cost? (validates the flash default)

## First results (2026-06-25, `deepseek-v4` auditor, 26 cases)

```
config                     catch(TPR)  over-strip    F1
flash · max · backstop        94.7%       ~0%*     94.7%   <- production default
flash · low · backstop        94.7%        0%      97.3%
flash · none (reasoning off)  84.2%       ~0%*     88.9%
pro · max · backstop         100.0%        0%     100.0%
```
`*` over-strip rows differ by <=1 case across configs — auditor non-determinism at temp 0; treat +/-2% as noise.

Findings on this set:
- The production default (flash·max) measurably holds the guarantee: ~95% of planted fabrications caught, ~0% of real facts stripped. Asserted -> measured.
- `pro·max` is perfect (100%), but the entire gap over flash is ONE subtle case — likely not worth 5x cost except on high-stakes turns.
- Reasoning OFF drops catch to 84%, and all 3 misses are NUMERIC/DATE fabrications — this validates keeping `AUDIT_REASONING_EFFORT` off `none`.
- LOW reasoning ties MAX (both 94.7%) — a possible cost/latency win, pending a larger/harder set.
- flash's consistent blind spot: subtle scale inflation ("training over 200 students" beside a real fact), caught only by `pro`.

## Run

```bash
python -m evals.honesty.harness --selftest   # scoring math + dataset labels, NO API/keys
python -m evals.honesty.harness --quick      # base config only, cheapest live signal
python -m evals.honesty.harness --limit 6    # smoke a few cases live
python -m evals.honesty.harness              # full ablation grid
```

Live runs need an auditor key in env (`FIREWORKS_API_KEY` and/or `DEEPSEEK_API_KEY`).
Easiest on the server where keys + deps already live:

```bash
docker cp evals owui-orchestrator:/app/evals
docker exec owui-orchestrator python -m evals.honesty.harness
```

## Cost

The auditor is `deepseek-v4-flash` (~$0.27 in / $1.10 out per 1M). One audit per case per
config; the synthetic set (~27 cases × 5 configs ≈ 135 calls) is roughly **<$1**. The
`pro · max` arm costs ~5× that arm alone. Cheap enough to gate every verifier change.

## Known limits (honest)

- **Marker matching is token-contiguous.** A correct flag phrased so it drops the marker's
  tokens reads as a miss; an over-broad flag that happens to contain a keep marker's tokens
  reads as an over-strip. It mirrors the verifier's own matching, but it under-counts subtle
  paraphrase. The diagnostics print every miss/over-strip so you can eyeball false signals.
- **Synthetic, small, and ours.** ~27 hand-authored cases covering the resume/cover-letter
  failure modes PrismAI actually hits. It validates *PrismAI's specific axis*, not generic
  RAG faithfulness.
- **Audit-flag level, not full end-to-end.** It measures what the auditor *flags*, not the
  final refined output. Flag-level is what the ablations target; an end-to-end pass
  (`_verified_or_blocked`) is the natural next layer.

## Extending

- **More cases:** append JSONL lines. `--selftest` enforces label integrity (every marker
  present in its draft; fabrications ungrounded; keeps grounded) so a mislabeled case fails
  loudly before it can poison a number.
- **RAGTruth:** for a standard, externally-comparable grounding number, adapt a subset of
  [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (18k span-labeled hallucination
  examples) into the same `{request, source, draft, fabrication_markers, keep_markers}`
  shape. It covers generic faithfulness; the synthetic set covers the user-authority axis
  RAGTruth explicitly excludes.
