"""Premium prose polish + voice pass — provider selection and the polish/voice model calls.

Split out of the agent god-module (part of the in-progress prism_core). Chooses a prose
provider (OpenAI / Anthropic / Gemini) with graceful fallback, builds the polish request,
classifies a voice register, and runs the optional voice-only pass. No agent-runtime deps.
"""
import json
import logging
import re

from . import config, fireworks, gemini, openai_client, anthropic_client, prompt_security
from .owui import _all_user_text
from .timectx import _now_line
from .prompts import SYSTEM_VOICE_REGISTER, _PROSE_POLISH_SYS, _VOICE_REGISTER, _VOICE_PASS_SYS

log = logging.getLogger(__name__)


def _client_for_model(model: str):
    """Route a configured prose model to its provider client by ID shape. A Fireworks
    path (accounts/…) goes to the Fireworks client (the single-bill path: deepseek /
    glm / kimi / qwen / …); gpt-* to OpenAI, claude-* to Anthropic, gemini-* to Gemini —
    the latter three only when that provider is usable, else None so callers fall back."""
    m = (model or "").strip().lower()
    if not m:
        return None
    if m.startswith("accounts/"):
        return fireworks if config.FIREWORKS_API_KEY else None
    if m.startswith("gpt-"):
        return openai_client if openai_client.available() else None
    if m.startswith("claude-"):
        return anthropic_client if anthropic_client.available() else None
    if m.startswith("gemini-"):
        return gemini if gemini.available() else None
    return None


# Legacy voice aliases (existing envs, traces, and the agent's learned choices use them).
_VOICE_ALIASES = {
    "gpt-5.5": lambda: config.OPENAI_PROSE_MODEL_PREMIUM,
    "opus": lambda: config.ANTHROPIC_PROSE_MODEL,
    "sonnet": lambda: config.ANTHROPIC_STANDARD_MODEL,
}


def _prose_provider(voice):
    """Map the polish voice — an alias OR a full model ID — to (client, model), honoring
    availability with graceful fallback. None if no provider is usable (stay on the open draft)."""
    alias = _VOICE_ALIASES.get(voice)
    requested = alias() if alias else voice
    client = _client_for_model(requested)
    if client is not None:
        return client, requested
    # requested provider unusable — fall back to any usable prose model
    for cand in (config.ANTHROPIC_PROSE_MODEL, config.OPENAI_PROSE_MODEL_PREMIUM):
        client = _client_for_model(cand)
        if client is not None:
            return client, cand
    if gemini.available():
        return gemini, config.GEMINI_PROSE_MODEL
    return None


def _prose_polish_messages(messages, candidate, source):
    """Build the polish request: the user's ask + (optional) source as untrusted
    reference + the open model's draft to rewrite. No tool-role messages /
    tool_calls (OpenAI-compat endpoints, esp. Gemini, choke on those)."""
    user_req = _all_user_text(messages)
    parts = [f"USER REQUEST:\n{user_req}"]
    if source.strip():
        parts.append(prompt_security.wrap_untrusted("gathered source material", source[:12000]))
    parts.append(f"DRAFT TO POLISH:\n{candidate}")
    # The polisher gets today's date too — without it, gpt-5.5 letterheads a formal
    # document with a "[Date]" placeholder (it can't know the date, so it blanks it).
    return [
        {"role": "system", "content": _PROSE_POLISH_SYS + "\n\n" + _now_line()},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


async def _classify_voice_register(request, candidate, *, session=None) -> str:
    """Pick the voice-pass register (warm/formal/none) for an exported deliverable from the document itself."""
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_VOICE_REGISTER},
             {"role": "user", "content": f"REQUEST:\n{request[:1500]}\n\nDELIVERABLE (excerpt):\n{candidate[:1500]}"}],
            config.GROUNDING_GATE_MODEL, max_tokens=30, temperature=0.0,
            session=session, label="gate:voice")
        m = re.search(r"\{.*\}", raw, flags=re.S)
        reg = str(json.loads(m.group(0) if m else raw).get("register", "none")).lower()
        return reg if reg in ("warm", "formal") else "none"
    except Exception:
        return "none"


async def _voice_pass(candidate, register, *, session=None):
    """Optional voice-only pass at a register (warm/formal). Never alters facts.
    The provider follows config.VOICE_MODEL — any served model (Fireworks path,
    gpt-*, claude-*), so the voice tier no longer depends on one vendor's account."""
    client = _client_for_model(config.VOICE_MODEL)
    if client is None:
        return candidate
    sys = _VOICE_PASS_SYS.replace("{register}", _VOICE_REGISTER.get(register, _VOICE_REGISTER["formal"]))
    try:
        out = await client.complete(
            [{"role": "system", "content": sys}, {"role": "user", "content": f"DRAFT:\n{candidate}"}],
            config.VOICE_MODEL,
            max_tokens=config.AGENT_MAX_TOKENS, temperature=config.WRITER_TEMPERATURE, session=session,
            label="voice")
        return out.strip() if (out and out.strip()) else candidate
    except Exception as e:
        log.warning(f"[voice_pass] {register} failed, keeping draft: {e}")
        return candidate
