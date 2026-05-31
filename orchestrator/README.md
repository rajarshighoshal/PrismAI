# Orchestrator

A standalone **OpenAI-compatible** service that sits behind OpenWebUI and
replaces the `router_fn` filter. OWUI becomes a pure chat UI; this service makes
the decisions — classify turn depth, pick the best model per task, draft, verify
grounding, and stream the result.

Founding reason: closed assistants make factual mistakes and present them with
confidence and **no verification**. This puts a verification step into the turn
itself, so deliverables grounded in the user's own material get checked before
they're trusted.

## Design (harness, not prompting)

Prompts are short and plain on purpose — the prompt is just the prior. The value
is in the control flow:

| Tier | Trigger | Machinery | Cost |
|------|---------|-----------|------|
| **CHAT** | default | one streamed completion (`deepseek-v4-pro`) | 1 call |
| **GROUNDED** | factual/research ask | one streamed completion with a careful-with-facts prior (web search planned) | 1 call |
| **DELIVERABLE** | "write a …", "turn this into …", export intent | stream the draft live → if the user supplied source material, audit it with the tool-server and append an honest verification footer | 1 call + 1 audit |

Cost is **task-dependent**, not a fixed multiplier: plain chat stays at one call;
only real document turns pay for verification, and only when there's actual
source material to check against (no source → no fake check).

Vision turns (messages with images) route to the vision model
(`kimi-k2p6`) as a single streamed pass.

## Files

- `app.py` — OpenAI wire format: `GET /v1/models`, `POST /v1/chat/completions`
  (streaming + non-streaming), `/health`.
- `pipeline.py` — the depth-routed control flow (the harness).
- `depth.py` — pure, tested turn-depth classifier.
- `fireworks.py` — streaming/non-streaming Fireworks client. Reads
  `message.content` for the answer; forwards `reasoning_content` only as the
  collapsible thinking UI, never as final output.
- `toolserver.py` — client for `verify_grounding` (fails soft).
- `style.py` — reads per-user writing-style profiles from `webui.db`
  (populated by the weekly `consolidate_style` job; read-only here).
- `config.py` — all env-driven.
- `deploy.sh` — build + run the container on OWUI's docker network.
- `test_orchestrator.py` — offline contract tests (no network).

## Configure (env / `orchestrator.env`)

`orchestrator.env` is git-ignored and holds secrets. Minimum:

```
FIREWORKS_API_KEY=...
```

Other knobs (with defaults) live in `config.py`: `CHAT_MODEL`, `VISION_MODEL`,
`DRAFT_MODEL`, `TOOL_SERVER_URL`, `ENABLE_VERIFICATION`, `ENABLE_STYLE_MEMORY`,
`MIN_SOURCE_CHARS`, `ORCH_API_KEY`.

## Test

```
python -m orchestrator.test_orchestrator
```

## Deploy

```
./orchestrator/deploy.sh
```

Then in OWUI: **Admin → Settings → Connections → add an OpenAI connection** with
base URL `http://owui-orchestrator:8002/v1` and any API key (set `ORCH_API_KEY`
to enforce one). For per-user style memory, set
`ENABLE_FORWARD_USER_INFO_HEADERS=true` on the `open-webui` container so the
logged-in user's id is forwarded.

## Status

MVP: CHAT + GROUNDED + DELIVERABLE-with-verification all wired and streaming.
Planned follow-ups: web search on the GROUNDED tier, same-message source
detection, optional draft→verify→**refine** loop for a clean grounded final, and
export (docx/pdf) hand-off on export intent.
