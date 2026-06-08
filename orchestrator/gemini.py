"""Google Gemini API client for high-value prose output.

Uses Google's OpenAI-compatible endpoint (v1beta) so the interface mirrors
fireworks.py. Falls back gracefully if GOOGLE_API_KEY is unset.
"""
import json

try:
    import aiohttp
except ImportError:
    aiohttp = None

from . import config


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live Gemini calls")


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.GOOGLE_API_KEY}",
    }


def available() -> bool:
    """True if Gemini is configured and can be used."""
    return bool(config.GOOGLE_API_KEY) and config.ENABLE_GEMINI_PROSE


async def complete(messages, model, *, max_tokens, temperature=None, session=None) -> str:
    """Non-streaming completion. Returns the final answer text."""
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
    """Non-streaming chat completion.

    Returns {"message": ..., "finish_reason": ...}.
    """
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
        async with session.post(
            f"{GEMINI_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.PROSE_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        choice = data["choices"][0]
        return {
            "message": choice.get("message") or {},
            "finish_reason": choice.get("finish_reason"),
        }
    finally:
        if own:
            await session.close()


async def stream(messages, model, *, max_tokens, temperature=None, session=None):
    """Streaming completion. Yields (kind, text) where kind is 'content'."""
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
            f"{GEMINI_BASE_URL}/chat/completions",
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
