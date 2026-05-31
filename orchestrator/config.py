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
GROUNDING_GATE_MODEL = os.getenv("GROUNDING_GATE_MODEL", "accounts/fireworks/models/gpt-oss-120b")
# The auditor model lives in the tool-server (gpt-oss-120b); we just call it.

# Advertised model ids — what OWUI shows in this connection's model list.
ADVERTISED_CHAT_ID = os.getenv("ADVERTISED_CHAT_ID", "assistant")
ADVERTISED_VISION_ID = os.getenv("ADVERTISED_VISION_ID", "assistant-vision")

# Generation knobs.
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "4096"))
DRAFT_MAX_TOKENS = int(os.getenv("DRAFT_MAX_TOKENS", "8192"))
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
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
GROUNDING_REPAIR_STEPS = int(os.getenv("GROUNDING_REPAIR_STEPS", "2"))
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "6"))
MAX_WEB_SEARCHES_PER_TURN = int(os.getenv("MAX_WEB_SEARCHES_PER_TURN", "2"))

# Minimum source length (chars) before a deliverable is worth verifying.
MIN_SOURCE_CHARS = int(os.getenv("MIN_SOURCE_CHARS", "200"))

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
QUERY_MODEL = os.getenv("QUERY_MODEL", "accounts/fireworks/models/gpt-oss-120b")
QUERY_MAX_CHARS = int(os.getenv("QUERY_MAX_CHARS", "400"))
# Also audit GROUNDED answers against retrieved snippets (extra call). Default off
# to keep GROUNDED cheap; verification is reserved for deliverables.
ENABLE_GROUNDED_VERIFY = _flag("ENABLE_GROUNDED_VERIFY", "false")
