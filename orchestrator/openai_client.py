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
from . import perf as _perf

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


def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))


def _token_payload(model, max_tokens, temperature) -> dict:
    # gpt-5.x/o-series: max_completion_tokens, no custom temperature
    if _is_reasoning_model(model):
        return {"max_completion_tokens": max_tokens}
    return {
        "max_tokens": max_tokens,
        "temperature": config.WRITER_TEMPERATURE if temperature is None else temperature,
    }


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
        payload = {"model": model, "messages": messages,
                   **_token_payload(model, max_tokens, temperature)}
        t0 = _perf.now()
        async with session.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.PROSE_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        usage = data.get("usage", {})
        _perf.trace(label, model, t0=t0,
                    in_tok=usage.get("prompt_tokens"), out_tok=usage.get("completion_tokens"))
        choice = data["choices"][0]
        return {
            "message": choice.get("message") or {},
            "finish_reason": choice.get("finish_reason"),
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
        payload = {"model": model, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True},
                   **_token_payload(model, max_tokens, temperature)}
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        t0 = _perf.now()
        ttft = None
        usage = None
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
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content")
                if c:
                    if ttft is None:
                        ttft = _perf.now()
                    yield ("content", c)
        u = usage or {}
        _perf.trace(label, model, t0=t0, ttft=ttft,
                    in_tok=u.get("prompt_tokens"), out_tok=u.get("completion_tokens"))
    finally:
        if own:
            await session.close()
