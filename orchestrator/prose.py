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


def _prose_provider(voice):
    """Map the agent-chosen polish voice to (client, model), honoring availability
    with graceful fallback. None if no provider is usable (stay on the open draft)."""
    if voice == "gpt-5.5" and openai_client.available():
        return openai_client, config.OPENAI_PROSE_MODEL_PREMIUM
    if voice == "opus" and anthropic_client.available():
        return anthropic_client, config.ANTHROPIC_PROSE_MODEL
    if voice == "sonnet" and anthropic_client.available():
        return anthropic_client, config.ANTHROPIC_STANDARD_MODEL
    # requested provider unavailable — fall back to any usable prose model
    if anthropic_client.available():
        return anthropic_client, config.ANTHROPIC_PROSE_MODEL
    if openai_client.available():
        return openai_client, config.OPENAI_PROSE_MODEL_PREMIUM
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
    """Optional sonnet voice-only pass at a register (warm/formal). Never alters facts."""
    if not anthropic_client.available():
        return candidate
    sys = _VOICE_PASS_SYS.replace("{register}", _VOICE_REGISTER.get(register, _VOICE_REGISTER["formal"]))
    try:
        out = await anthropic_client.complete(
            [{"role": "system", "content": sys}, {"role": "user", "content": f"DRAFT:\n{candidate}"}],
            config.ANTHROPIC_STANDARD_MODEL,
            max_tokens=config.AGENT_MAX_TOKENS, temperature=config.WRITER_TEMPERATURE, session=session,
            label="voice")
        return out.strip() if (out and out.strip()) else candidate
    except Exception as e:
        log.warning(f"[voice_pass] {register} failed, keeping draft: {e}")
        return candidate
