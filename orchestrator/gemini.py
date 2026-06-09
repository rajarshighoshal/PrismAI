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
from . import perf as _perf


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


async def complete(messages, model, *, max_tokens, temperature=None, session=None, label="") -> str:
    """Non-streaming completion. Returns the final answer text."""
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
        t0 = _perf.now()
        async with session.post(
            f"{GEMINI_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.PROSE_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        _u = data.get("usage", {})
        _perf.trace(label, model, t0=t0, in_tok=_u.get("prompt_tokens"), out_tok=_u.get("completion_tokens"))
        choice = data["choices"][0]
        return {
            "message": choice.get("message") or {},
            "finish_reason": choice.get("finish_reason"),
        }
    finally:
        if own:
            await session.close()


async def stream(messages, model, *, max_tokens, temperature=None, session=None, label=""):
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
            "stream_options": {"include_usage": True},
        }
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        t0 = _perf.now()
        ttft = None
        usage = None
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
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                if ttft is None:
                    ttft = _perf.now()
                delta = choices[0].get("delta") or {}
                c = delta.get("content")
                if c:
                    yield ("content", c)
        _u = usage or {}
        _perf.trace(label, model, t0=t0, ttft=ttft,
                    in_tok=_u.get("prompt_tokens"), out_tok=_u.get("completion_tokens"))
    finally:
        if own:
            await session.close()
