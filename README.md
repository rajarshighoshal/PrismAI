# PrismAI

An agentic orchestrator for OpenWebUI — an OpenAI-compatible service that drives a
model-driven tool-calling loop over open-weight models, verifies its own output
before showing it, and polishes formal deliverables with the writer model best
suited to each task.

The name: a *prism* splits one input into the right paths. Each task is handled by
the model strongest at it — reasoning, grounding, perception, and writing — rather
than one model doing everything.

## What it does

- **Agentic tool loop** — an open-weight model (DeepSeek / GLM / Kimi via Fireworks)
  decides which tools to call (web search, URL fetch, citation lookup, grounding
  verification, file export) and chains them until it can answer.
- **Verification before display** — every deliverable passes an honesty audit (which
  catches claims about the user they never actually made) and a source-grounding
  check before it is shown. Unsupported drafts are revised or withheld, never
  surfaced as confident fact.
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
