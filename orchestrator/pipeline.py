"""Compatibility wrapper for the model-driven agent harness."""

import logging
import os
from .agent import run as _agent_run

log = logging.getLogger(__name__)

_INTERNAL_NAMES = frozenset(
    n.strip() for n in os.getenv("INTERNAL_MODEL_IDS", "PrismAI,assistant,assistant-vision").split(",") if n.strip()
)


async def run(messages, *, user_id="", session=None, request_headers=None, user_model=""):
    user_model = (user_model or "").strip()
    forward = user_model if (user_model and user_model not in _INTERNAL_NAMES) else ""
    started = False
    try:
        async for kind, text in _agent_run(
            messages, user_id=user_id, session=session,
            request_headers=request_headers, user_model=forward,
        ):
            if kind == "content" and text:
                started = True
            yield kind, text
    except Exception:
        # A crash mid-generation must NOT break the chunked HTTP response — that surfaces
        # to OWUI as a raw "TransferEncodingError / payload not completed". Convert it into
        # a graceful message so the turn ends cleanly and the user can simply retry.
        log.exception("agent.run failed mid-stream")
        yield "content", (
            "\n\n— sorry, something went wrong while finishing that. Please try again."
            if started else
            "Sorry, something went wrong on my end. Please try again."
        )
