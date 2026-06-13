"""Compatibility wrapper for the model-driven agent harness.

Raw models pass through transparently (stream only, no harness).
PrismAI/assistant get the full pipeline (tools, verification, polish).
"""

import logging
import os
from . import config, perf
from .agent import run as _agent_run
from . import fireworks

log = logging.getLogger(__name__)

def _configured_internal_names():
    names = {
        config.ADVERTISED_CHAT_ID,
        "PrismAI",
        "assistant",
    }
    names.update(os.getenv("INTERNAL_MODEL_IDS", "").split(","))
    return frozenset(n.strip() for n in names if n and n.strip())


_INTERNAL_NAMES = _configured_internal_names()


async def _raw_chat(messages, model, *, session):
    """Pass-through: stream raw model response, no harness. Use same provider
    dispatch as the harness (DeepSeek-direct → Fireworks fallback)."""
    async for kind, text in fireworks.stream(
        messages, model,
        max_tokens=config.AGENT_MAX_TOKENS,
        temperature=config.WRITER_TEMPERATURE,
        session=session,
        label="raw",
    ):
        yield kind, text


async def run(messages, *, user_id="", session=None, request_headers=None, user_model=""):
    perf.set_user(user_id)
    user_model = (user_model or "").strip()
    forward = user_model if (user_model and user_model not in _INTERNAL_NAMES) else ""
    started = False
    try:
        if forward:
            async for kind, text in _raw_chat(messages, forward, session=session):
                if kind == "content" and text:
                    started = True
                yield kind, text
        else:
            async for kind, text in _agent_run(
                messages, user_id=user_id, session=session,
                request_headers=request_headers, user_model="",
            ):
                if kind == "content" and text:
                    started = True
                yield kind, text
    except Exception:
        log.exception("pipeline failed mid-stream")
        yield "content", (
            "\n\n— sorry, something went wrong while finishing that. Please try again."
            if started else
            "Sorry, something went wrong on my end. Please try again."
        )
