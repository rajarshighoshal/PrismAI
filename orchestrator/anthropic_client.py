"""Anthropic Claude API client for premium prose output.

Mirrors the interface of openai_client.py and fireworks.py.
"""
import json
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

from . import config
from . import perf as _perf

log = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live Anthropic calls")


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }


def available() -> bool:
    return bool(config.ANTHROPIC_API_KEY) and config.ENABLE_ANTHROPIC_PROSE


def _temperature_deprecated(model: str) -> bool:
    """Opus 4.8+ reject an explicit `temperature` param (400). Omit it for them."""
    m = (model or "").lower()
    return "opus-4-8" in m or "opus-4-7" in m


def _convert_messages(messages):
    """Convert OpenAI-style messages to Anthropic format."""
    system_parts = []
    converted = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            text = content if isinstance(content, str) else str(content)
            system_parts.append(text)
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


async def complete(messages, model, *, max_tokens, temperature=None, session=None, label="") -> str:
    result = await chat(
        messages,
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        session=session,
        label=label,
    )
    return (result.get("message", {}).get("content") or "").strip()


async def chat(
    messages,
    model,
    *,
    max_tokens,
    temperature=None,
    session=None,
    label="",
) -> dict:
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        system, converted = _convert_messages(messages)
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": converted,
        }
        # Prompt caching: cache_control attaches to the SYSTEM content block, NOT
        # the top-level payload (a top-level cache_control field is rejected with
        # 400). A structured system block with cache_control={"type":"ephemeral"}
        # caches the stable system prefix for ~5 min, cutting repeat-call cost.
        if system:
            payload["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        # Newer Anthropic models (e.g. Opus 4.8) REJECT an explicit `temperature`
        # ("temperature is deprecated for this model" -> 400). Only send it for
        # models that still accept it; otherwise omit and use the model default.
        temp = config.WRITER_TEMPERATURE if temperature is None else temperature
        if temp is not None and not _temperature_deprecated(model):
            payload["temperature"] = temp

        t0 = _perf.now()
        async with session.post(
            f"{ANTHROPIC_BASE_URL}/messages",
            headers=_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.PROSE_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        usage = data.get("usage", {})
        _perf.trace(label, model, t0=t0,
                    in_tok=usage.get("input_tokens"), out_tok=usage.get("output_tokens"))

        content_blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        )
        return {
            "message": {"role": "assistant", "content": text},
            "finish_reason": data.get("stop_reason"),
        }
    finally:
        if own:
            await session.close()


async def stream(messages, model, *, max_tokens, temperature=None, session=None, label=""):
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        system, converted = _convert_messages(messages)
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": converted,
            "stream": True,
        }
        if system:
            payload["system"] = system  # plain string is valid; caching omitted on stream
        temp = config.WRITER_TEMPERATURE if temperature is None else temperature
        if temp is not None and not _temperature_deprecated(model):
            payload["temperature"] = temp

        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        t0 = _perf.now()
        ttft = None
        in_tok = out_tok = None
        async with session.post(
            f"{ANTHROPIC_BASE_URL}/messages",
            headers=_headers(),
            json=payload,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    in_tok = ((event.get("message") or {}).get("usage") or {}).get("input_tokens")
                elif etype == "message_delta":
                    out_tok = (event.get("usage") or {}).get("output_tokens", out_tok)
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    text = delta.get("text")
                    if text:
                        if ttft is None:
                            ttft = _perf.now()
                        yield ("content", text)
        _perf.trace(label, model, t0=t0, ttft=ttft, in_tok=in_tok, out_tok=out_tok)
    finally:
        if own:
            await session.close()
