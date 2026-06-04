"""OpenAI API client for high-value prose output.

Mirrors the interface of fireworks.py and gemini.py for easy swapping.
"""
import json
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

from . import config

log = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live OpenAI calls")


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
    }


def available() -> bool:
    return bool(config.OPENAI_API_KEY) and config.ENABLE_OPENAI_PROSE


async def complete(messages, model, *, max_tokens, temperature=None, session=None) -> str:
    result = await chat(
        messages,
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        session=session,
    )
    return (result.get("message", {}).get("content") or "").strip()


async def chat(
    messages,
    model,
    *,
    max_tokens,
    temperature=None,
    session=None,
) -> dict:
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": config.WRITER_TEMPERATURE if temperature is None else temperature,
        }
        log.info(f"[openai] model={model} messages={len(messages)} tokens={max_tokens}")
        async with session.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.HTTP_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        usage = data.get("usage", {})
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        log.info(f"[openai] completed cached_tokens={cached} total_tokens={usage.get('total_tokens', 0)}")
        choice = data["choices"][0]
        return {
            "message": choice.get("message") or {},
            "finish_reason": choice.get("finish_reason"),
        }
    finally:
        if own:
            await session.close()


async def stream(messages, model, *, max_tokens, temperature=None, session=None):
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": config.WRITER_TEMPERATURE if temperature is None else temperature,
            "stream": True,
        }
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        async with session.post(
            f"{OPENAI_BASE_URL}/chat/completions",
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
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content")
                if c:
                    yield ("content", c)
    finally:
        if own:
            await session.close()
