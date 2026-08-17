# PrismAI Orchestrator — How a Turn Actually Flows

One page to make the system legible. Two services, one promise: **grounded writing that
can't lie, that you can edit across turns like a document — not re-roll like a slot machine.**

```
OpenWebUI  ──OpenAI-compatible──▶  orchestrator (:8002)  ──HTTP──▶  tool-server (:8001)
   │                                   │                              │
   │ injects files as <source> blocks  │ agent loop, gates, verifier  │ exports (pandoc),
   │ + wraps msgs in its RAG template  │ edit engine, polish/voice    │ chat memory (sqlite),
   │   (even with bypass on — bug)     │                              │ deliverable store, search
```

## The lifecycle of one heavy turn (`agent.run()`)

```
 1. UNWRAP      OWUI wraps the user's message in its RAG template even with bypass on.
                _unwrap_owui() recovers the real text: <user_query>…</user_query> if
                present, else everything after the last </context>.

 2. STARTUP I/O (parallel) style profile ─ prior deliverable ─ last-active time
                → _gap_note(): one line if the user is resuming after a day+ gap.

 3. VISION      image turns → kimi transcribes (all images concurrently); the agent and
                verifier see the transcription framed as the assistant's own sight.

 4. EDIT ENGINE if this chat already delivered a document, _classify_edit() (reasoning-on
                flash) routes by MINIMAL ACTION:
                  rename/reformat → re-export the stored verified bytes. No writer,
                                    no verifier, no polish. Done in seconds.
                  edit            → inject the REAL prior document ("REVISION TASK"),
                                    writer changes only what was asked.
                  new             → normal flow.

 5. GROUNDING   _user_source() = every <source> block (attached files, any role) + pasted
                source-like text. For an edit, the prior document joins the source.

 6. AGENT LOOP  deepseek-v4-pro picks tools (web_search batched + parallel, exports
                deferred), SYSTEM_TOOL_GUARD vetoes needless searches, budgets capped.

 7. POLISH+VOICE (new exported documents only — never surgical edits)
                gpt-5.5 polish → voice register classifier → sonnet voice pass.

 8. VERIFY      _verified_or_blocked(): the can't-lie gate. Verifies FACTUAL DELIVERABLES
                (classifier-decided; an export always counts). Grounding = files + recall
                + the USER'S OWN chat statements + today's date. The auditor (flash,
                reasoning-on) sees request + FULL source + draft side by side; flags only
                unsupported FACTS. A flag whose exact words sit verbatim in the source is
                a false positive and is kept. Genuine flags → surgical refine → recheck →
                (if needed) one rewrite → else block. Motivation/opinion never blocked.

 9. DELIVER     file built FROM the verified text (export-arg never trusted); chat gets a
                what-changed note + download link — never a second copy of the document.

10. PERSIST     deliverable stored (versioned, ≤30/chat) for the next turn's edit;
                user+assistant turns embedded into chat memory (overflow recall only).
```

A mid-stream crash anywhere yields a clean "try again" (pipeline.py guard), never a broken
chunked response.

## Files

The repo is split into a channel-neutral core and the OWUI-facing service:

| File | Role |
|---|---|
| `prism_core/` | channel-neutral, provider-agnostic package (stdlib only; MIT-licensed, own `pyproject.toml`, publish-ready): `audit.py` (the provider-injected fact audit + its prompt), `verifier.py` (verbatim backstop, source-fit, fail-closed sentinel, `<source>` delimiter), `messages.py` (generic message helpers) |
| `orchestrator/` | the OWUI channel adapter + turn engine (one of two deployables) |
| `orchestrator/app.py` | FastAPI shell, OpenAI-compatible endpoints |
| `orchestrator/pipeline.py` | entry + streaming crash guard |
| `orchestrator/agent.py` | turn engine only: phase sequencing + the agent loop (carved from the old 1850-line god module) |
| `orchestrator/vision.py` | image transcription phase (cache, describe, split) |
| `orchestrator/prose.py` | export polish + voice pass (provider per vibe) |
| `orchestrator/tools.py` | tool layer: schemas, guard gate, execution, result shaping |
| `orchestrator/delivery.py` | persist-turn, background-task tracking, re-export, `_same_doc` |
| `orchestrator/editing.py` | multi-turn edit: intent classify, patch-in-place, re-deliver |
| `orchestrator/longdoc.py` | chunked long-doc writer: outline plan → per-section build |
| `orchestrator/hardness.py` / `escalation.py` | task-structure classifier + the gated stronger-model escalation (off by default: `ESCALATION_MODEL=""`) |
| `orchestrator/ports.py` | structural Protocols for the provider/tool/memory/style/artifact/event seams (additive; wired in later) |
| `orchestrator/owui.py` | OWUI adapter parsing only: RAG-template unwrap, `<source>` blocks, user source |
| `orchestrator/memory_client.py` | tool-server HTTP: chat memory, deliverables, last-active |
| `orchestrator/timectx.py` | current-time line + resume-after-gap note |
| `orchestrator/verifier.py` | the can't-lie gate: audit, refine, block (pure pieces live in `prism_core/verifier.py`) |
| `orchestrator/prompts.py` | every system prompt + tool schemas |
| `orchestrator/config.py` | env-tunable knobs (models, budgets, timezone) |
| `orchestrator/fireworks.py` / `openai_client.py` / `anthropic_client.py` / `gemini.py` | model clients, all traced (`[trace] label=… ttft=… ttlt=…`) |
| `orchestrator/style.py` | per-user voice profile (read-only, off-thread) |
| `orchestrator/dedup.py` | identical concurrent requests share one run |
| `tool-server/main.py` | exports, search, scrape, deliverable + memory endpoints |
| `tool-server/memory.py` | sqlite (single worker thread, WAL): chat turns, FTS, deliverables |
| `owui-patches/` + `router_fn.py` | OWUI-v0.9.5 shims owned by the adapter boundary: patch OWUI's `routers/openai.py` + `utils/middleware.py` at deploy time (re-applied by `update.sh` after every container recreate — they do not survive one) |

## Model routing (who does what)

| Job | Model |
|---|---|
| writer (grounded + chat) | deepseek-v4-pro |
| gates/classifiers (verify? voice? work?) + honesty audit | deepseek-v4-flash |
| edit-intent gate (a wrong verdict drops the user's doc) | deepseek-v4-pro, 'new' must win twice |
| vision | kimi-k2p6 |
| export polish / voice pass | model-driven: any served model ID routed by shape (`accounts/…` → Fireworks, `gpt-*` → OpenAI, `claude-*` → Anthropic) — env-pinned (`AUTO_POLISH_MODEL`, `OPENAI_PROSE_MODEL_PREMIUM`, `VOICE_MODEL`), so the prose tiers can ride the single Fireworks bill |
| refine (fix flagged facts) | same model that wrote the prose |

## Debugging a live turn

`docker logs owui-orchestrator` and grep, in order of usefulness:
- `[trace]` — every model call: label, in/out tokens, TTFT, TTLT
- `[edit-intent]` — what the edit classifier decided and on what text
- `[audit-diag]` — verifier verdict, flag count, verbatim false-positives
- `[source-diag]` — how much grounding source arrived, by role

## Standing invariants (don't break these)

1. The file is built from the **verified** text; chat never duplicates the document body.
2. The verifier strips only unsupported **facts** — never motivation, opinion, framing.
3. The **user's own statements ground their own facts**; the model still can't invent.
4. Action proportional to request: a rename must never wake the writer.
5. No sync sqlite on an event loop; every model call carries a `label=` for tracing.
