"""Orchestrator configuration — all env-driven, with sane defaults.

Secrets (FIREWORKS_API_KEY) come from the environment via the deploy --env-file,
never hard-coded. Defaults mirror the values already proven in router_fn so the
orchestrator behaves identically out of the box.
"""
import os


def _flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off", ""}


FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
).rstrip("/")

# Google Gemini API (legacy, kept for fallback).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ENABLE_GEMINI_PROSE = _flag("ENABLE_GEMINI_PROSE", "false")
GEMINI_PROSE_MODEL = os.getenv("GEMINI_PROSE_MODEL", "gemini-3.1-pro-preview")

# OpenAI API for high-value prose (cover letters, resumes, research papers).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENABLE_OPENAI_PROSE = _flag("ENABLE_OPENAI_PROSE", "true")
OPENAI_PROSE_MODEL = os.getenv("OPENAI_PROSE_MODEL", "gpt-5.5")
# gpt-5.5, not -pro (-pro is Responses-API only, 404s on chat/completions)
OPENAI_PROSE_MODEL_PREMIUM = os.getenv("OPENAI_PROSE_MODEL_PREMIUM", "gpt-5.5")

# Anthropic API for quality-tier prose (research papers, executive briefs).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ENABLE_ANTHROPIC_PROSE = _flag("ENABLE_ANTHROPIC_PROSE", "true")
ANTHROPIC_PROSE_MODEL = os.getenv("ANTHROPIC_PROSE_MODEL", "claude-opus-4-8")
# Standard-tier prose: Sonnet. Benchmarks (EQ-Bench creative writing) rank Sonnet
# above GPT-4o for prose, and it runs on the working Anthropic key, so the
# standard tier is Sonnet by default; GPT-4o remains an OpenAI fallback only.
ANTHROPIC_STANDARD_MODEL = os.getenv("ANTHROPIC_STANDARD_MODEL", "claude-sonnet-4-6")

# tool-server (same docker network) — verification + export primitives.
TOOL_SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://owui-tool-server:8001").rstrip("/")

# Per-task model selection. Defaults mirror router_fn.
CHAT_MODEL = os.getenv("CHAT_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
VISION_MODEL = os.getenv("VISION_MODEL", "accounts/fireworks/models/kimi-k2p6")
DRAFT_MODEL = os.getenv("DRAFT_MODEL", CHAT_MODEL)          # deliverable first draft
REFINE_MODEL = os.getenv("REFINE_MODEL", CHAT_MODEL)        # grounding fix pass
# Agentic harness model roles. The controller decides tool use; the final model
# shifts to GLM after source-bearing tools because it measured stronger there.
AGENT_MODEL = os.getenv("AGENT_MODEL", CHAT_MODEL)
GROUNDED_MODEL = os.getenv("GROUNDED_MODEL", "accounts/fireworks/models/glm-5p1")
GROUNDING_GATE_MODEL = os.getenv("GROUNDING_GATE_MODEL", "accounts/fireworks/models/deepseek-v4-flash")
# The auditor model lives in the tool-server (gpt-oss-120b); we just call it.

# Advertised model ids — what OWUI shows in this connection's model list.
ADVERTISED_CHAT_ID = os.getenv("ADVERTISED_CHAT_ID", "PrismAI")
ADVERTISED_VISION_ID = os.getenv("ADVERTISED_VISION_ID", "PrismAI Vision")

# Generation knobs.
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "4096"))
DRAFT_MAX_TOKENS = int(os.getenv("DRAFT_MAX_TOKENS", "8192"))
# Split temperature by job instead of one compromise value:
# - TOOL_TEMPERATURE: turns where the model decides/chains tools. Low = reliable
#   tool selection and tight instruction-following (no "Here's a..." preamble).
# - WRITER_TEMPERATURE: generating the final written artifact (and refine
#   rewrites). Higher = natural, non-templated prose. The decision/gate calls
#   stay hard 0.0 (classification, never creative).
# TEMPERATURE kept as a back-compat default for any remaining shared call.
TOOL_TEMPERATURE = float(os.getenv("TOOL_TEMPERATURE", "0.2"))
WRITER_TEMPERATURE = float(os.getenv("WRITER_TEMPERATURE", "0.55"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))

# Networking.
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "180"))
STREAM_IDLE_TIMEOUT = float(os.getenv("STREAM_IDLE_TIMEOUT", "90"))

# Optional bearer token OWUI must present (the connection's API key).
# Empty = accept any (service is localhost-bound on a private docker net).
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "")

# Per-user style profiles, read-only, from OWUI's sqlite db.
STYLE_DB_PATH = os.getenv("STYLE_DB_PATH", "/app/backend/data/webui.db")
ENABLE_STYLE_MEMORY = _flag("ENABLE_STYLE_MEMORY", "true")

# Verification: run verify_grounding on deliverables that have source material.
ENABLE_VERIFICATION = _flag("ENABLE_VERIFICATION", "true")
ENABLE_GROUNDING_GATE = _flag("ENABLE_GROUNDING_GATE", "true")

# Honesty audit: catch claims about the USER (experience, seniority, credentials,
# metrics, revenue) that the user never actually stated — an INSTRUCTION to assert
# X is not evidence X is true. Runs on the final draft regardless of the grounding
# gate (which wrongly waves through "creative writing" that inflates a resume).
# Bake-off (2026-05-31) showed deepseek-v4-pro / glm-5p1 / gpt-oss-120b / gemini
# all catch it; kimi leaks chain-of-thought. Default to the strong model; set
# HONESTY_MODEL=accounts/fireworks/models/deepseek-v4-flash for a cheaper auditor.
ENABLE_HONESTY_AUDIT = _flag("ENABLE_HONESTY_AUDIT", "true")
ENABLE_APPLICATION_CLAIM_AUDIT = _flag("ENABLE_APPLICATION_CLAIM_AUDIT", "true")
HONESTY_MODEL = os.getenv("HONESTY_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
GROUNDING_REPAIR_STEPS = int(os.getenv("GROUNDING_REPAIR_STEPS", "2"))
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "10"))
MAX_WEB_SEARCHES_PER_TURN = int(os.getenv("MAX_WEB_SEARCHES_PER_TURN", "4"))

# Show-your-work: stream tool-step narration ("Searching… Reading… Verifying…")
# to the UI as reasoning_content so the chat visibly acts agentic, like claude.ai.
SHOW_WORK = _flag("SHOW_WORK", "true")

# Minimum source length (chars) before a deliverable is worth verifying.
MIN_SOURCE_CHARS = int(os.getenv("MIN_SOURCE_CHARS", "200"))

# Chat-memory recall is an OVERFLOW/compaction handler, not a per-turn feature.
# OWUI sends the full native conversation every request; recall only kicks in when
# the conversation grows past a fraction of the binding model's context window —
# then the recent tail is kept verbatim and older relevant facts are recalled to
# stand in for the compacted head.
#
# The cap is a fraction of the SMALLEST generation window (glm-5p1, ~200k tokens),
# NOT deepseek's 1M: a grounded turn runs on glm, so the conversation must never be
# allowed to fill more than its share of glm's window. Capping the conversation at
# 20% leaves ~80% for the system prompt, tool schemas, accumulated tool results,
# and the answer. ~3.5 chars/token for prose.
MODEL_CONTEXT_TOKENS = int(os.getenv("MODEL_CONTEXT_TOKENS", "200000"))      # glm-5p1 floor
MEMORY_COMPACT_FRACTION = float(os.getenv("MEMORY_COMPACT_FRACTION", "0.20"))
MEMORY_CONTEXT_BUDGET_CHARS = int(
    os.getenv("MEMORY_CONTEXT_BUDGET_CHARS",
              str(int(MODEL_CONTEXT_TOKENS * MEMORY_COMPACT_FRACTION * 3.5)))  # ~140,000 chars
)

# Prose polish (premium Opus/Sonnet) is slow + costly. Only spend it on substantial
# prose: a short factual / numeric / conversational answer must NOT trigger an Opus
# rewrite (that was burning ~6-10s on one-line answers). The second "voice pass" is
# a SECOND premium call, reserved for genuinely long-form prose.
POLISH_MIN_CHARS = int(os.getenv("POLISH_MIN_CHARS", "320"))
POLISH_VOICE_MIN_CHARS = int(os.getenv("POLISH_VOICE_MIN_CHARS", "1200"))

# Prose tier classifier model — cheap, fast model to determine if a request is
# high-value formal prose (→ Gemini) or casual conversation (→ GLM).
PROSE_CLASSIFIER_MODEL = os.getenv("PROSE_CLASSIFIER_MODEL", "accounts/fireworks/models/deepseek-v4-flash")

# On an ungrounded deliverable, run one refine pass that strips/fixes the
# unsupported claims and append the corrected version. Off -> warn-only footer.
ENABLE_REFINE = _flag("ENABLE_REFINE", "true")

# --- GROUNDED-tier web search (free-first, provider-pluggable) ---
ENABLE_WEB_SEARCH = _flag("ENABLE_WEB_SEARCH", "true")
# auto = first configured of searxng -> tavily -> duckduckgo. Or pin one.
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "auto").lower()
SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")  # self-hosted, OSS (preferred)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")        # hosted fallback
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "20"))
# Cheap, no-CoT-leak model to compress the turn into a <=400-char search query.
QUERY_MODEL = os.getenv("QUERY_MODEL", "accounts/fireworks/models/deepseek-v4-flash")
QUERY_MAX_CHARS = int(os.getenv("QUERY_MAX_CHARS", "400"))
# Also audit GROUNDED answers against retrieved snippets (extra call). Default off
# to keep GROUNDED cheap; verification is reserved for deliverables.
ENABLE_GROUNDED_VERIFY = _flag("ENABLE_GROUNDED_VERIFY", "false")
