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

# DeepSeek's own API as the PRIMARY provider for deepseek models, with Fireworks (above)
# as the automatic fallback. DeepSeek's API is OpenAI-compatible AND uses the SAME model
# names (deepseek-v4-pro / deepseek-v4-flash) and the same reasoning_effort param — so the
# only difference is base_url + key + dropping the "accounts/fireworks/models/" prefix.
# Inert until DEEPSEEK_API_KEY is set: with no key, deepseek calls go to Fireworks as before.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
ENABLE_DEEPSEEK_DIRECT = _flag("ENABLE_DEEPSEEK_DIRECT", "true")

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
# Auto-polish voice for exported deliverables — gpt-5.5 (calibrated academic/formal
# substance) by default; opus for bolder corporate persuasion. The model no longer
# picks this via a tool (that confused it); the orchestrator just applies it.
AUTO_POLISH_MODEL = os.getenv("AUTO_POLISH_MODEL", "gpt-5.5")

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
GROUNDED_MODEL = os.getenv("GROUNDED_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
GROUNDING_GATE_MODEL = os.getenv("GROUNDING_GATE_MODEL", "accounts/fireworks/models/deepseek-v4-flash")
# The auditor model lives in the tool-server (deepseek-v4-flash, thinking off).

# Advertised model ids — what OWUI shows in this connection's model list.
ADVERTISED_CHAT_ID = os.getenv("ADVERTISED_CHAT_ID", "PrismAI")
ADVERTISED_VISION_ID = os.getenv("ADVERTISED_VISION_ID", "PrismAI Vision")

# Generation knobs — max_tokens caps OUTPUT (reasoning + answer TOGETHER, since MAX
# reasoning is now default for substantive deepseek calls; see REASONING_EFFORT).
#
# WHY cap at all instead of just using the provider max? Billing is on ACTUAL output
# tokens, so a high ceiling is FREE for a normal answer — the cap is NOT a cost lever,
# it's a GUARDRAIL: (a) it bounds a runaway/looping generation (worst-case wall-clock
# ~= ceiling / output-rate, so a stuck stream fails fast instead of hanging for minutes),
# and (b) the provider hard-caps deepseek-v4 output at ~131k on Fireworks (our fallback)
# anyway, so "all possible" lands there regardless. Rule: substantive generations get a
# GENEROUS ceiling (well above any realistic thinking+answer — nothing truncates) but
# stay a few x under 131k for runaway safety; classifiers/gates keep a TIGHT cap on
# purpose (output is one label; a tight cap enforces terseness + catches misbehavior
# instantly). Two substantive ceilings, split by what the call EMITS:
#   GENERATION_MAX_TOKENS (32k) — a chat REPLY / agent turn / audit verdict: generous for any
#     reply + max-reasoning thinking, well under the ~131k provider output cap.
#   DRAFT_MAX_TOKENS (64k) — a DELIVERABLE / full whole-document REWRITE. When you drop a paper
#     and say "rewrite it", the model re-emits the ENTIRE document in one reply, so this path
#     needs ~2x the reply budget (~40k words out + thinking): a paper (6-10k words) or an MSc
#     thesis (15-20k words) fits with room. For a full PhD-thesis single-shot rewrite, bump
#     toward 120k via env (~40min worst-case stream). Input is NEVER the limit — any paper fits
#     the 1M context window many times over; only the written-back output is bounded here.
GENERATION_MAX_TOKENS = int(os.getenv("GENERATION_MAX_TOKENS", "32000"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", str(GENERATION_MAX_TOKENS)))
DRAFT_MAX_TOKENS = int(os.getenv("DRAFT_MAX_TOKENS", "64000"))
# Vision transcribes the image VERBATIM ("quote visible text exactly") for the
# downstream text agent. 1024 (~700 words) truncated dense inputs — a full paper page,
# a wide table, a psych-study figure, a long screenshot — silently dropping the bottom
# of the image. 8192 covers a dense page-or-two; it's a ceiling (bills on actual tokens),
# and vision runs on kimi (no reasoning/thinking overhead), so this is pure transcription.
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "8192"))
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

# Networking. Per-purpose timeouts (seconds). The old single 180s blanket let one
# stalled upstream hang a whole turn for 3 minutes with no user feedback; each
# call now carries a budget matched to what it actually does. All env-overridable.
# HTTP_TIMEOUT stays as a back-compat catch-all default for any caller without a
# more specific budget.
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "90"))
STREAM_IDLE_TIMEOUT = float(os.getenv("STREAM_IDLE_TIMEOUT", "90"))
# Fireworks non-streaming completion: agent steps, gates, query compression.
GEN_TIMEOUT = float(os.getenv("GEN_TIMEOUT", "90"))
# tool-server calls: verify_grounding (LLM auditor), web_search, fetch_url, export.
TOOL_SERVER_TIMEOUT = float(os.getenv("TOOL_SERVER_TIMEOUT", "60"))
# Per-chat memory recall/store — must be quick. If the memory service is slow,
# degrade (skip recall / drop the async store) rather than hang the turn.
MEMORY_TIMEOUT = float(os.getenv("MEMORY_TIMEOUT", "15"))
# Premium prose polish (Opus/Sonnet/GPT-5.5/Gemini). Long-form generation is
# legitimately slow, so this stays generous — but still bounded, not 3 minutes.
PROSE_TIMEOUT = float(os.getenv("PROSE_TIMEOUT", "120"))

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
# all catch it; kimi leaks chain-of-thought. Default to flash: it runs on EVERY
# turn (~2.2s on pro vs ~1s on flash), and an adversarial vet (2026-06-07) showed
# flash flags the same fabrications — overt and subtle ("team of 50 engineers")
# — and stays CLEAN on the user's real stated facts. Set HONESTY_MODEL=...-pro to
# revert to the heavier auditor.
ENABLE_HONESTY_AUDIT = _flag("ENABLE_HONESTY_AUDIT", "true")
HONESTY_MODEL = os.getenv("HONESTY_MODEL", "accounts/fireworks/models/deepseek-v4-flash")
# The honesty audit is a careful grounding task, not a snap classifier: run flash WITH
# chain-of-thought so it actually locates each credential in the source instead of
# guessing "unsupported" and over-stripping. Empty string = no reasoning (snap mode).
# User policy (global, 2026-06-11): MAX reasoning on every substantive model call; only
# classifier roles (gates, summaries) stay fast. The provider layer pins this by LABEL —
# substantive labels get REASONING_EFFORT, classifier labels (gate:*, summarize) get "none".
# Pinned explicitly because DeepSeek defaults to "high", not "max".
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "max")
AUDIT_REASONING_EFFORT = os.getenv("AUDIT_REASONING_EFFORT", "max") or None
# The auditor reads the FULL source and now THINKS at MAX reasoning before its JSON
# verdict; max_tokens bounds thinking + verdict together. Generous on purpose — a
# truncated verdict silently fail-softs (didn't actually verify) = honesty hole. Shares
# the generation ceiling (was 900 once -> truncated; 8000 pre-thinking).
AUDIT_MAX_TOKENS = int(os.getenv("AUDIT_MAX_TOKENS", str(GENERATION_MAX_TOKENS)))
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
# chat / agent / voice answers — shares the budget with MAX-reasoning thinking.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", str(GENERATION_MAX_TOKENS)))
GROUNDING_REPAIR_STEPS = int(os.getenv("GROUNDING_REPAIR_STEPS", "2"))
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "10"))
MAX_WEB_SEARCHES_PER_TURN = int(os.getenv("MAX_WEB_SEARCHES_PER_TURN", "4"))

# Show-your-work: stream tool-step narration ("Searching… Reading… Verifying…")
# to the UI as reasoning_content so the chat visibly acts agentic, like claude.ai.
SHOW_WORK = _flag("SHOW_WORK", "true")

# The current date/time injected into the agent prompt is formatted in this local
# timezone (the request itself carries none). Default = IST (+05:30), no DST.
LOCAL_TZ_OFFSET_MINUTES = int(os.getenv("LOCAL_TZ_OFFSET_MINUTES", "330"))
LOCAL_TZ_LABEL = os.getenv("LOCAL_TZ_LABEL", "IST")

# Plain-chat live streaming: when a cheap classifier says the turn needs no tools,
# sources, or verification, stream the answer token-by-token instead of running the
# buffered agentic loop. Critical turns (facts/source/application writing) still go
# through the verify-first loop.
STREAM_SIMPLE_CHAT = _flag("STREAM_SIMPLE_CHAT", "true")

# On a heavy turn (deliverable/source), stream a one-line acknowledgment from the
# fast model into the thinking panel immediately, so the user sees tokens flowing
# in ~0.5s instead of a blank wait while the first heavy generation runs.
STREAM_PREAMBLE = _flag("STREAM_PREAMBLE", "true")

# Optimistic answer streaming: stream the open-model answer live (token-by-token,
# interleaved with the thinking breadcrumbs), then verify the finished artifact and
# openly self-correct if a claim was unsupported — instead of holding the whole
# answer until it's verified. The deliverable a model is about to polish is NOT
# streamed (the polished version is); user-chosen-model regens stream their model.
STREAM_ANSWER = _flag("STREAM_ANSWER", "true")

# Minimum source length (chars) before a deliverable is worth verifying.
MIN_SOURCE_CHARS = int(os.getenv("MIN_SOURCE_CHARS", "200"))

# Chat-memory recall is an OVERFLOW/compaction handler, not a per-turn feature.
# OWUI sends the full native conversation every request; recall only kicks in when
# the conversation grows past a fraction of the binding model's context window —
# then the recent tail is kept verbatim and older relevant facts are recalled to
# stand in for the compacted head.
#
# The budget is a fraction of the SMALLEST window among the models that bind a
# FULL-conversation turn (_BINDING_MODELS) — a long chat must never overflow whichever
# model serves the turn, INCLUDING the Fireworks fallback. We derive the floor from the
# actually-configured models instead of a hardcoded number, so swapping a model retunes
# the trigger automatically.
#
# deepseek-v4 (our main; pro for chat/agent/grounded, flash for the gate) serves ~1M on
# BOTH providers: DeepSeek-direct 1,048,576 / Fireworks 1,000,000 -> take the smaller,
# 1,000,000 (covers the fallback). This REPLACES glm-5p1's 200k floor (glm bound grounded
# turns and is now retired). Vision (kimi) is NOT a binder: it only ever sees the single
# image-bearing message, never the conversation, so its window doesn't constrain the budget.
CONTEXT_WINDOWS = {
    "deepseek-v4-pro": 1_000_000,    # DeepSeek-direct 1,048,576 / Fireworks 1,000,000
    "deepseek-v4-flash": 1_000_000,
    "kimi-k2p6": 256_000,
    "gemini-3.1-pro": 1_000_000,
    "gpt-5.5": 400_000,
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
}


def context_window(model: str) -> int:
    """Input-token context window for a model id (substring match on the bare name).
    Unknown models fall back to a conservative 128k floor so we under- rather than
    over-estimate a new model's room."""
    name = (model or "").split("/")[-1]
    best = max((win for key, win in CONTEXT_WINDOWS.items() if key in name), default=0)
    return best or 128_000


# Models that receive the FULL (compacted) conversation each turn. The budget must fit
# the SMALLEST of these. QUERY / prose-classifier models only see a single turn, and
# premium prose models polish a deliverable (not the raw conversation), so neither binds.
_BINDING_MODELS = [CHAT_MODEL, AGENT_MODEL, GROUNDED_MODEL, DRAFT_MODEL, REFINE_MODEL,
                   GROUNDING_GATE_MODEL]
# Compact at 80% of that window, leaving ~20% for the system prompt, tool schemas,
# accumulated tool results, and the answer. At 1M that's ~200k tokens of headroom —
# comfortable even for a heavily tool-using grounded turn. ~3.5 chars/token for prose.
MODEL_CONTEXT_TOKENS = int(os.getenv(
    "MODEL_CONTEXT_TOKENS", str(min(context_window(m) for m in _BINDING_MODELS))))
MEMORY_COMPACT_FRACTION = float(os.getenv("MEMORY_COMPACT_FRACTION", "0.80"))
# ~3.5 chars/token is Latin-prose-tuned; dense scripts (e.g. CJK) run ~1-2
# chars/token, so this under-counts tokens there — tolerable because the budget
# sits far under the model window. Clamp to a floor so a stray MEMORY_COMPACT_
# FRACTION=0 / MODEL_CONTEXT_TOKENS=0 can't collapse it to 0 and overflow every
# chat on turn 1.
MEMORY_CONTEXT_BUDGET_CHARS = max(20000, int(
    os.getenv("MEMORY_CONTEXT_BUDGET_CHARS",
              str(int(MODEL_CONTEXT_TOKENS * MEMORY_COMPACT_FRACTION * 3.5)))  # ~560,000 chars
))

# Prose polish (premium Opus/Sonnet) is slow + costly. Only spend it on substantial
# prose: a short factual / numeric / conversational answer must NOT trigger an Opus
# rewrite (that was burning ~6-10s on one-line answers). The second "voice pass" is
# a SECOND premium call, reserved for genuinely long-form prose.
POLISH_MIN_CHARS = int(os.getenv("POLISH_MIN_CHARS", "320"))
POLISH_VOICE_MIN_CHARS = int(os.getenv("POLISH_VOICE_MIN_CHARS", "1200"))

# Request de-duplication / idempotency. A byte-identical request (same messages +
# model + user) that arrives again within the window must not re-run the whole
# pipeline: it replays the first one's answer (completed cache) or attaches to it
# while still in flight (single-flight). Window is short — a retry happens within
# seconds; a genuinely new ask of the same question later re-runs for a fresh answer.
ENABLE_DEDUP = _flag("ENABLE_DEDUP", "true")
DEDUP_TTL_SECONDS = float(os.getenv("DEDUP_TTL_SECONDS", "120"))
# A follower attached to an in-flight identical request waits at most this long
# for the lead's answer before falling back to running its own — and the fallback
# is LOGGED, not silently swallowed. Keep < DEDUP_TTL so a slow lead still
# populates the cache the follower can use on its own re-run.
DEDUP_WAIT_TIMEOUT = float(os.getenv("DEDUP_WAIT_TIMEOUT", "90"))

# Prose tier classifier model — cheap, fast model to determine if a request is
# high-value formal prose (→ Gemini) or casual conversation (→ GLM).
PROSE_CLASSIFIER_MODEL = os.getenv("PROSE_CLASSIFIER_MODEL", "accounts/fireworks/models/deepseek-v4-flash")

# The honesty / application-claim auditors must see the WHOLE source they judge a
# draft against. The old hard [:6000] clip silently dropped the tail of a long
# source — e.g. a job posting (~6k chars) concatenated before a résumé pushed the
# résumé's later roles past the cut, so real, grounded credentials looked
# "unsupported" and got stripped. The auditor model has a large context; these
# budgets only bound pathological inputs, not real documents. If a source ever
# exceeds this, trim by relevance to the draft (see agent._fit_audit_source), never
# by head.
AUDIT_SOURCE_BUDGET = int(os.getenv("AUDIT_SOURCE_BUDGET", "60000"))
AUDIT_DRAFT_BUDGET = int(os.getenv("AUDIT_DRAFT_BUDGET", "16000"))

# One-line diagnostic: how big the extracted grounding source is vs. the raw chars
# per role. Decisive for "did the uploaded file even reach `source`?" — if a large
# system/context block exists but user_source stays small, OWUI is RAG-injecting the
# file where the user-only source extractor can't see it. Logs COUNTS only, never
# content; safe to leave on.
LOG_SOURCE_DIAG = _flag("LOG_SOURCE_DIAG", "true")

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

# Where the spend panel lives (the tool-server's /usage page). Set the real
# browser-reachable URL in orchestrator.env; shows on the OWUI model card.
USAGE_PANEL_URL = os.getenv("USAGE_PANEL_URL", "http://localhost:8001/usage")
