# Case study: score final and stored state, preserve the earlier stream

This case study covers an evaluation defect in PrismAI's real-use basket. It is
about release-gate design and the evaluation boundary, not a claim that this
basket is already enforced by CI or a production metric.

## Problem

PrismAI has more than one output surface. It can stream text into the chat and
store the source content used to generate a formal deliverable separately. Its
verifier can also append a corrected version or warning after an optimistically
streamed draft.

The original basket runner joined content events into one transcript and ran
both required-text and forbidden-text checks against that transcript. This
created two evaluation blind spots:

1. For an export task, the chat can contain a correction summary and download
   link while the canonical document source lives in the deliverable store. A
   transcript-only check cannot inspect that content.
2. The obvious partial repair, scoring only the final corrected segment, can
   hide unsupported content that was shown earlier in the stream.

The defect was therefore in the evaluation boundary. "What did the agent
produce?" could not be represented by one chat string.

## Diagnosis

The release gate needed three distinct views of a run:

- the concatenated user-visible `content` events, for exposure detection;
- text after the exact `*Corrected version:*` marker, when that marker appears;
- the stored document source, for the content used to generate the artifact.

A final-state assertion and a streamed-exposure signal answer different
questions. Combining them into one pass/fail string either penalizes a successful
correction or erases an unsafe intermediate output.

## Fix

[Commit `ecd7968`](https://github.com/rajarshighoshal/PrismAI/commit/ecd796882d81e2170e1446848f836cb623bd0f28)
changed the runner in four places:

1. Each case receives a timestamped chat ID so its stored document source can be
   fetched after the pipeline finishes.
2. The runner retrieves that source content instead of inferring document
   correctness from the download link.
3. Required and forbidden assertions are evaluated against text after the exact
   correction marker, when present, plus the stored document source.
4. If forbidden content is absent from final state but present in the full
   stream, the result keeps a non-blocking `streamed-before-correction` design
   flag.

The relevant implementation is visible in the
[artifact and state scoring logic](https://github.com/rajarshighoshal/PrismAI/blob/ecd796882d81e2170e1446848f836cb623bd0f28/evals/prismai_basket/run.py#L142-L199).
The public
[case basket](https://github.com/rajarshighoshal/PrismAI/blob/ecd796882d81e2170e1446848f836cb623bd0f28/evals/prismai_basket/cases.jsonl)
supplies synthetic requests, optional source blocks, mechanical checks, and
explicit forbidden literals to the runner. It is not a deterministic unit test
of the three-state scoring logic.

## Verification

The change is inspectable in the commit diff:

- before the change, `includes` and `must_not_include` checks used the lowercased
  full transcript;
- after the change, those checks use final state plus stored source content;
- a forbidden literal found only in the earlier stream is retained as a design
  note instead of disappearing.

Separate public contract tests verify the two product behaviors behind the
diagnosis: an optimistic answer can
[stream before a visible verifier response](https://github.com/rajarshighoshal/PrismAI/blob/1b25c375726de89dcc2ca37422e5155e988776f8/orchestrator/test_orchestrator.py#L548-L565),
and an exported document body can be
[absent from chat while remaining the file source](https://github.com/rajarshighoshal/PrismAI/blob/1b25c375726de89dcc2ca37422e5155e988776f8/orchestrator/test_orchestrator.py#L819-L839).

The basket's `--selftest` command validates only case IDs, kinds, requests, and
check field shapes without calling a model. Live runs execute the production
pipeline and report pass/fail checks, manual checks, latency, model calls, token
counts, and design notes.

## Remaining limitations

This commit fixes the scoring boundary, but it does not claim more than the code
does:

- stored source content is checked, not rendered DOCX or PDF bytes;
- streamed-before-correction exposure is diagnostic and does not make `ok` false;
- deliverable retrieval happens immediately, errors are swallowed, and missing
  content does not produce a dedicated failure, so an artifact-required release
  policy must await or poll storage and fail when content is absent;
- the commit does not add a deterministic scorer unit test;
- the refusal heuristic is global to the transcript, so an honesty marker in one
  sentence can downgrade a forbidden literal elsewhere;
- the timestamp uses whole seconds, so concurrent runs of the same case can share
  a chat ID;
- the runner supplies a non-empty chat ID and has no cleanup step. Evaluation
  inputs should therefore be synthetic or redacted until retention and deletion
  are explicit.

## General lesson

An agent release gate should follow the product's delivery semantics, not the
shape that is easiest to log. Final-state correctness, stored-source correctness,
rendered-artifact correctness, and unsafe intermediate exposure are separate
invariants. The evaluator should preserve each signal, and the release policy
should state which signals block deployment.
