# OWUI Hybrid Router

An OpenWebUI filter function that adds semantic routing, citation verification, vision proxy, and per-chat memory to any LLM.

## What it does

Intercepts every message in OpenWebUI and adds:

- **Semantic routing** — classifies queries into FACTUAL / REASONING / CODING / RESEARCH / CASUAL using embedding similarity + LLM fallback, injects category-specific system prompts
- **Multi-provider fallback** — Groq primary, Fireworks fallback. If one provider is down, the next picks up transparently
- **Web search + citation enforcement** — Tavily search for FACTUAL/RESEARCH queries, hybrid regex + LLM verification that cited URLs actually appear in search results and support the claims they're attached to
- **Vision proxy** — non-vision models (GLM, DeepSeek) get image descriptions via a vision model, so they can "see" images in the conversation
- **Per-chat semantic memory** — stores conversation turns with embeddings, recalls relevant prior context on long chats via hybrid BM25 + cosine retrieval
- **Memory compression** — summarizes oldest turns when chats grow past a threshold, keeping long-horizon context without unbounded growth
- **Deleted-chat cleanup** — referential sweep on every turn removes memory for chats the user deleted in the UI
- **Usage logging** — per-call analytics (model, tokens, latency, fallback status) to a local SQLite table
- **Name addressing** — uses the logged-in user's name instead of "the user"
- **Embedding circuit breaker** — fails fast when the embedding provider is down instead of blocking for 45 seconds

## Setup

1. Deploy OpenWebUI (Docker or bare metal)
2. In the admin UI → Functions → Add Function → paste `router_fn.py`
3. Set Valves: `FIREWORKS_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`
4. Enable as a global filter

## Deployment

```bash
# Update function on a running server (no restart needed)
./update.sh
```

`update.sh` pulls from git, copies the file into the container, and updates the function row in `webui.db`. OpenWebUI hot-reloads.

## Configuration

All behavior is controlled via Valves (OpenWebUI's per-function config):

| Valve | Default | What it does |
|---|---|---|
| `CLASSIFIER_MODEL` | `groq/llama-3.1-8b-instant` | Routing classifier (Groq primary) |
| `MAIN_MODEL` | `accounts/fireworks/models/glm-5p1` | Primary chat model |
| `VERIFIER_MODEL` | `groq/llama-3.3-70b-versatile` | Citation auditor (Groq primary) |
| `ENABLE_CHAT_MEMORY` | `true` | Per-chat semantic memory |
| `CHAT_MEMORY_TOP_K` | `8` | Recalled turns per query |
| `ENABLE_HYBRID_RETRIEVAL` | `true` | BM25 + cosine combined scoring |
| `ENABLE_QUERY_REWRITE` | `true` | Rewrite follow-up queries for better recall |
| `ENABLE_CHAT_MEMORY_COMPRESSION` | `true` | Summarize oldest turns on long chats |
| `ADDRESS_USER_BY_NAME` | `true` | Use logged-in user's name |

See `router_fn.py` → `class Valves` for the full list with descriptions.

## Architecture

```
inlet (before model)
  ├── classify query (embedding → LLM fallback)
  ├── image routing (caption for classifier)
  ├── web search (FACTUAL / RESEARCH)
  ├── chat memory recall (hybrid BM25 + cosine)
  ├── vision proxy (caption images for non-vision models)
  └── inject system prompt (category-specific + search + memory)

outlet (after model)
  ├── strip thinking blocks + route tags
  ├── citation verification (presence + validity + attribution)
  ├── store turn to chat memory
  ├── referential cleanup (deleted chats)
  └── background compression (long chats)
```

## License

AGPL-3.0 — free to use, modify, and self-host. If you serve a modified version to users, you must share your modifications under the same license.
