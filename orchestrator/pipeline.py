"""Compatibility wrapper for the model-driven agent harness."""

import os
from .agent import run as _agent_run

_INTERNAL_NAMES = frozenset(
    n.strip() for n in os.getenv("INTERNAL_MODEL_IDS", "PrismAI,assistant,assistant-vision").split(",") if n.strip()
)


async def run(messages, *, user_id="", session=None, request_headers=None, user_model=""):
    user_model = (user_model or "").strip()
    if user_model and user_model not in _INTERNAL_NAMES:
        async for kind, text in _agent_run(
            messages, user_id=user_id, session=session,
            request_headers=request_headers, user_model=user_model,
        ):
            yield kind, text
    else:
        async for kind, text in _agent_run(
            messages, user_id=user_id, session=session,
            request_headers=request_headers,
        ):
            yield kind, text
