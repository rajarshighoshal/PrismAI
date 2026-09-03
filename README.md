# PrismAI

An agentic orchestrator for OpenWebUI. It drives a model-directed tool-calling
loop over open-weight models, verifies formal deliverables before export, and can
openly correct streamed answers when later verification finds a problem. It also
polishes formal deliverables with the writer model best suited to each task.

The name: a *prism* splits one input into the right paths. Each task is handled by
the model strongest at it — reasoning, grounding, perception, and writing — rather
than one model doing everything.

## What it does

- **Agentic tool loop** — an open-weight model (DeepSeek / GLM / Kimi via Fireworks)
  decides which tools to call (web search, URL fetch, citation lookup, grounding
  verification, file export) and chains them until it can answer.
- **Verification at delivery boundaries.** The delivery path applies an honesty
  audit and source-grounding check to formal deliverables before export. With
  optimistic chat streaming enabled, provisional text may appear before
  verification; any later correction is explicit and the evaluation basket can
  retain that earlier exposure as a separate diagnostic.
- **Model-selected prose polish** — for writing that matters (cover letters,
  statements, research prose, important email), the agent picks the writer model
  that fits the piece, with an optional final voice pass. Polished output still
  passes verification.
- **Per-chat memory** — conversation turns are stored with embeddings and recalled
  via hybrid BM25 + cosine retrieval; older turns are summarized into compact memory
  notes as a chat grows.
- **Vision** — attached images are transcribed up front so the agent can reason over
  their content.
- **Untrusted-content handling** — text returned by tools (web pages, search results)
  is treated as data, not instructions, to resist prompt injection.

## Engineering case study

The real-use evaluation basket once scored only the assistant transcript. That
missed the stored source content used to generate a document and made a simple
post-correction score blind to unsupported text that had already streamed. The
[release-gate case study](evals/prismai_basket/CASE_STUDY_RELEASE_GATE.md)
explains how the runner was changed to score the exact post-correction segment
plus stored source content while preserving earlier user-visible exposure as a
separate, non-blocking signal.

## Components

| Component | Role |
|---|---|
| `orchestrator/` | The OpenAI-compatible agentic service OpenWebUI connects to as a model. |
| `tool-server/` | Grounding verification, citation lookup, file exports (DOCX / PDF / CSV / Markdown), and the per-chat memory store. |
| `router_fn.py` | A thin OpenWebUI filter that wires the chat UI to the orchestrator. |

## Configuration

All behavior is environment-driven — see `orchestrator/config.py` for the full set
of options and defaults. Secrets (API keys) are supplied via the environment / an
untracked env file and are never committed.

## License

AGPL-3.0 — free to use, modify, and self-host. If you serve a modified version to
users, you must share your modifications under the same license.
