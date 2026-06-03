# Orchestrator

A standalone **OpenAI-compatible** service that sits behind OpenWebUI. OWUI
becomes the chat UI; this service runs a model-driven tool loop, picks models
per step, and enforces verification before source-bound factual output is shown.

The goal is not prompt engineering. The harness gives the model the tools and
keeps the hard guarantees in code:

- the model decides whether to call tools and which tools to chain
- all existing server tools are exposed: search, fetch, exports, citation lookup,
  citation search, and grounding verification
- source-bearing answers are checked before display
- web citations are limited to URL-backed retrieved sources; search summaries
  are not treated as citable evidence
- unsupported drafts are withheld, repaired, or blocked
- costs scale with the task: plain chat can finish with one model call; research
  and exports pay for the extra tool and verification work

## Model Roles

Defaults are env-driven in `config.py`:

- `AGENT_MODEL`: tool-use controller, default `deepseek-v4-pro`
- `GROUNDED_MODEL`: final model after source-gathering tools, default `glm-5p1`
- `GROUNDING_GATE_MODEL`: cheap final gate, default `gpt-oss-120b`
- auditor model: lives in the tool-server `verify_grounding` endpoint

Vision turns still route to `VISION_MODEL` because image input support is a
modality constraint, not a task-flow router.

## Files

- `app.py` - OpenAI wire format: `GET /v1/models`, `POST /v1/chat/completions`
  (streaming + non-streaming), `/health`.
- `agent.py` - model-driven tool-call loop and verification gate.
- `pipeline.py` - compatibility wrapper exposing `agent.run`.
- `fireworks.py` - Fireworks chat client, including OpenAI-compatible tool calls.
- `search.py` - provider-pluggable web search; Tavily path keeps advanced search
  and AI summary parity with the proven `router_fn.py` behavior.
- `toolserver.py` - async client for the existing tool-server endpoints.
- `style.py` - reads per-user writing-style profiles from `webui.db`, style only.
- `ab_eval.py` - Claude-vs-agent A/B harness with judge output.
- `test_orchestrator.py` - offline contract tests, no network.

## Configure

`orchestrator.env` is git-ignored and holds secrets. Minimum:

```env
FIREWORKS_API_KEY=...
```

Common knobs:

```env
TOOL_SERVER_URL=http://owui-tool-server:8001
TAVILY_API_KEY=...
AGENT_MAX_STEPS=8
GROUNDING_REPAIR_STEPS=2
ENABLE_VERIFICATION=true
ENABLE_GROUNDING_GATE=true
```

For export attachment, the tool-server still needs OpenWebUI chat/message
headers or its own `OPENWEBUI_API_KEY`. The orchestrator forwards the relevant
headers when OWUI provides them.

## Test

```bash
python -m orchestrator.test_orchestrator
python -m py_compile orchestrator/*.py
```

## A/B Eval

```bash
python -m orchestrator.ab_eval --out ab_results.json
```

This calls `claude -p <prompt>` by default, runs the same prompt through the
local agent, then asks a judge model to compare intent, prose, grounding, and
verification honesty. Override the Claude command with `CLAUDE_CMD` or
`--claude-cmd`. Override the judge with `JUDGE_MODEL`.

## Deploy

```bash
./orchestrator/deploy.sh
```

Then in OWUI: **Admin -> Settings -> Connections -> add an OpenAI connection**
with base URL `http://owui-orchestrator:8002/v1`. Prod deploy remains
user-gated.

## Borrowed ideas & attribution

`prompt_security.py` (untrusted-context wrapper + prompt-injection policy) is
adapted from the **Odysseus** project
(https://github.com/pewdiepie-archdaemon/odysseus, MIT License, © 2025 Odysseus
Contributors). All external/tool-gathered content is wrapped as untrusted data so
the model treats it as reference material, never as instructions — which directly
reinforces this service's anti-fabrication / verification goal. See the
repo-root [`CREDITS.md`](../CREDITS.md) for full attribution and other ideas we
studied from that project.
