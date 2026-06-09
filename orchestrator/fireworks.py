"""Fireworks chat client: streaming + non-streaming.

Two hard-won rules baked in:
- The ANSWER is message.content. DeepSeek V4 / Kimi put chain-of-thought in
  `reasoning_content`; pipeline logic must never treat that as final output.
- For user-facing streaming we DO forward reasoning_content separately so OWUI
  can render it as collapsible "thinking" — but accumulation for the verify
  loop only ever uses content.
"""
import hashlib
import json
import logging
import time

try:
    import aiohttp
except ImportError:
    aiohttp = None

from . import config
from .perf import trace as _perf_trace

log = logging.getLogger(__name__)


def _trace(label, model, *, t0, ttft=None, usage=None):
    u = usage or {}
    _perf_trace(label, model, t0=t0, ttft=ttft,
                in_tok=u.get("prompt_tokens"), out_tok=u.get("completion_tokens"))


def compute_session_id(messages, user_id: str = "") -> str:
    """Compute stable session ID from message context for cache affinity."""
    parts = [user_id]
    for m in messages[:5]:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}:{content[:200]}")
        elif isinstance(content, list):
            text = " ".join(
                p.get("text", "")[:100] for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
            parts.append(f"{role}:{text}")
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live Fireworks calls")


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if config.FIREWORKS_API_KEY:
        h["Authorization"] = f"Bearer {config.FIREWORKS_API_KEY}"
    return h


async def complete(messages, model, *, max_tokens, temperature=None, session=None, user_id=None, label="") -> str:
    """Non-streaming completion. Returns the final answer text (content only)."""
    result = await chat(
        messages,
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        session=session,
        user_id=user_id,
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
    tools=None,
    tool_choice=None,
    user_id=None,
    label="",
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
        session_id = compute_session_id(messages, user_id or "")
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "user": session_id,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        # Flash is only ever used here as a fast classifier/auditor — never let it
        # spend latency on chain-of-thought.
        if "deepseek-v4-flash" in model:
            payload.setdefault("reasoning_effort", "none")
        headers = _headers()
        headers["x-session-affinity"] = session_id
        t0 = time.perf_counter()
        async with session.post(
            f"{config.FIREWORKS_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.GEN_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        usage = data.get("usage") or {}
        _trace(label, model, t0=t0, usage=usage)
        choice = data["choices"][0]
        return {
            "message": choice.get("message") or {},
            "finish_reason": choice.get("finish_reason"),
        }
    finally:
        if own:
            await session.close()


async def stream(messages, model, *, max_tokens, temperature=None, session=None, user_id=None, label=""):
    """Streaming completion. Yields (kind, text) where kind is 'content' or
    'reasoning'. Caller forwards 'content' as delta.content and may forward
    'reasoning' as delta.reasoning_content for the thinking UI.
    """
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        session_id = compute_session_id(messages, user_id or "")
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "user": session_id,
        }
        if "deepseek-v4-flash" in model:
            payload.setdefault("reasoning_effort", "none")
        headers = _headers()
        headers["x-session-affinity"] = session_id
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        t0 = time.perf_counter()
        ttft = None
        usage = None
        async with session.post(
            f"{config.FIREWORKS_BASE_URL}/chat/completions",
            headers=headers,
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
                rc = delta.get("reasoning_content")
                if rc:
                    if ttft is None:
                        ttft = time.perf_counter()
                    yield ("reasoning", rc)
                c = delta.get("content")
                if c:
                    if ttft is None:
                        ttft = time.perf_counter()
                    yield ("content", c)
        _trace(label, model, t0=t0, ttft=ttft, usage=usage)
    finally:
        if own:
            await session.close()


async def stream_chat(messages, model, *, max_tokens, temperature=None, session=None,
                      tools=None, tool_choice=None, user_id=None, label=""):
    """Streaming chat that ALSO surfaces tool calls. Yields ('reasoning', text) and
    ('content', text) as they arrive, then a final ('final', {content, tool_calls,
    finish_reason}) once the stream completes — so the agent loop can stream the
    model's answer live while still acting on any tool calls it made.
    """
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        session_id = compute_session_id(messages, user_id or "")
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "user": session_id,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if "deepseek-v4-flash" in model:
            payload.setdefault("reasoning_effort", "none")
        headers = _headers()
        headers["x-session-affinity"] = session_id
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        content_parts, calls, finish = [], {}, None
        t0 = time.perf_counter()
        ttft = None
        usage = None
        async with session.post(
            f"{config.FIREWORKS_BASE_URL}/chat/completions",
            headers=headers, json=payload, timeout=timeout,
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
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                rc = delta.get("reasoning_content")
                if rc:
                    if ttft is None:
                        ttft = time.perf_counter()
                    yield ("reasoning", rc)
                c = delta.get("content")
                if c:
                    if ttft is None:
                        ttft = time.perf_counter()
                    content_parts.append(c)
                    yield ("content", c)
                for tc in (delta.get("tool_calls") or []):
                    slot = calls.setdefault(
                        tc.get("index", 0),
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
        _trace(label, model, t0=t0, ttft=ttft, usage=usage)
        yield ("final", {
            "content": "".join(content_parts),
            "tool_calls": [calls[i] for i in sorted(calls)],
            "finish_reason": finish,
        })
    finally:
        if own:
            await session.close()
