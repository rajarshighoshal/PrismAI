"""
title: PrismAI
description: Thin OWUI filter — pure pass-through to the PrismAI orchestrator.
  All logic (tools, verification, vision, memory, search) lives in the orchestrator.
author: open-webui-community
version: 11.0
"""

from typing import Optional, Any, Awaitable, Callable
from pydantic import BaseModel, Field

EventEmitter = Optional[Callable[[dict], Awaitable[Any]]]

# Models that match one of OWUI's PrismAI connections.
ORCHESTRATOR_MODEL_IDS = frozenset({"PrismAI"})


class Filter:
    class Valves(BaseModel):
        FIREWORKS_API_KEY: str = Field(
            default="",
            description="Fireworks.ai API key (required for all model calls).",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── Inlet: pass-through ────────────────────────────────────────────

    async def inlet(
        self, body: dict, __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        return body

    # ── Outlet: forward to orchestrator ─────────────────────────────────

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> dict:
        return body
