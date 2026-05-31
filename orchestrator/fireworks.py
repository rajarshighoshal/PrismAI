"""Fireworks chat client: streaming + non-streaming.

Two hard-won rules baked in:
- The ANSWER is message.content. DeepSeek V4 / Kimi put chain-of-thought in
  `reasoning_content`; pipeline logic must never treat that as final output.
- For user-facing streaming we DO forward reasoning_content separately so OWUI
  can render it as collapsible "thinking" — but accumulation for the verify
  loop only ever uses content.
"""
import json

try:
    import aiohttp
except ImportError:  # lets offline tests run without installed service deps
    aiohttp = None

from . import config


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live Fireworks calls")


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if config.FIREWORKS_API_KEY:
        h["Authorization"] = f"Bearer {config.FIREWORKS_API_KEY}"
    return h


async def complete(messages, model, *, max_tokens, temperature=None, session=None) -> str:
    """Non-streaming completion. Returns the final answer text (content only)."""
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
    tools=None,
    tool_choice=None,
) -> dict:
    """Non-streaming chat completion.

    Returns {"message": ..., "finish_reason": ...}. When `tools` is supplied,
    the returned message may contain OpenAI-compatible `tool_calls`.
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
            "temperature": config.TEMPERATURE if temperature is None else temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        async with session.post(
            f"{config.FIREWORKS_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.HTTP_TIMEOUT),
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
    """Streaming completion. Yields (kind, text) where kind is 'content' or
    'reasoning'. Caller forwards 'content' as delta.content and may forward
    'reasoning' as delta.reasoning_content for the thinking UI.
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
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "stream": True,
        }
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        async with session.post(
            f"{config.FIREWORKS_BASE_URL}/chat/completions",
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
                rc = delta.get("reasoning_content")
                if rc:
                    yield ("reasoning", rc)
                c = delta.get("content")
                if c:
                    yield ("content", c)
    finally:
        if own:
            await session.close()
