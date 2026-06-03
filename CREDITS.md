# Credits & Attribution

This project stands on the shoulders of excellent open-source work. Where we have
borrowed ideas or adapted code, we credit it here generously and explicitly.

## Odysseus — https://github.com/pewdiepie-archdaemon/odysseus

A self-hosted, local-first AI workspace (the self-hosted answer to the
ChatGPT/Claude UI experience) by Felix ("PewDiePie") and the Odysseus
contributors. Licensed MIT, Copyright (c) 2025 Odysseus Contributors.

It's a genuinely impressive, mature codebase — provider-agnostic LLM core,
multi-round agent loop, ChromaDB memory, MCP tooling, an IterResearch-style deep
research loop, and a "Cookbook" that does VRAM-aware hardware fitting and serves
local models over SSH. We learned a lot reading it. Thank you.

### Adapted code

- **Untrusted-context / prompt-injection hardening** —
  `orchestrator/prompt_security.py` is adapted from Odysseus'
  `src/prompt_security.py`. The pattern: treat all external/tool-gathered content
  (web results, fetched pages, citation lookups) as **data, not instructions**,
  by wrapping it in an explicit untrusted-source envelope and injecting a
  prompt-safety policy into the system prompt. This complements our verification
  layer — prompt-injection and fabrication are the same trust-boundary problem.
  (MIT License; copyright notice preserved in that file.)

### Ideas studied (not copied, but informed our design)

- Hybrid memory retrieval (semantic + BM25 + recency blend with category boosts).
- Provider failover via host canonicalization + consecutive-failure cooldown.
- Prompt caching via a stable system prefix + cache breakpoint.
- Context compaction at a token-budget threshold via self-summarization.
- Blind A/B comparison mode as a first-class evaluation surface.

If you maintain Odysseus and would like any attribution changed, expanded, or
removed, please open an issue — we're happy to adjust.
