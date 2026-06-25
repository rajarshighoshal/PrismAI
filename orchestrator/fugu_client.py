"""Sakana Fugu API client — OpenAI-compatible, mirrors openai_client.py pattern.

Fugu is a learned multi-model orchestrator behind one endpoint. Two models:
  fugu                  — balanced, opt-out agents configurable (subset of models)
  fugu-ultra-20260615   — max-quality, full agent pool (the default)

Under Sakana's subscription plans ($20-$200/mo), both models cost the same —
Ultra is strictly better and is the default. Under PAYG ($5/$30 per M for
Ultra, model-rate for standard), standard is cheaper but at that point you're
better off on subscription anyway at any real volume.

IMPORTANT: The base URL is NOT a public constant — it's shown in your Sakana
console (console.sakana.ai) and varies per account. Set FUGU_BASE_URL in
orchestrator.env from the console. The default below is a guess and WILL fail.

Ref: https://sakana.ai/fugu/
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

FUGU_BASE_URL = config.FUGU_BASE_URL.rstrip("/")


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live Fugu calls")


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.FUGU_API_KEY}",
    }


def available() -> bool:
    return bool(config.FUGU_API_KEY) and config.ENABLE_FUGU


def _fugu_model(ultra: bool = False) -> str:
    return "fugu-ultra-20260615" if ultra else config.FUGU_MODEL


async def complete(
    messages,
    *,
    max_tokens,
    temperature=None,
    session=None,
    label="fugu",
    ultra: bool = False,
) -> str:
    """Non-streaming completion. Returns the answer text."""
    result = await chat(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        session=session,
        label=label,
        ultra=ultra,
    )
    return (result.get("message", {}).get("content") or "").strip()


async def chat(
    messages,
    *,
    max_tokens,
    temperature=None,
    session=None,
    label="fugu",
    ultra: bool = False,
) -> dict:
    """Non-streaming chat completion. Returns {"message": ..., "finish_reason": ..., "usage": ...}."""
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        model = _fugu_model(ultra)
        temp = config.WRITER_TEMPERATURE if temperature is None else temperature
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
        }
        t0 = _perf.now()
        async with session.post(
            f"{FUGU_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.FUGU_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        usage = data.get("usage", {})
        _perf.trace(label, model, t0=t0,
                    in_tok=usage.get("input_tokens"),
                    out_tok=usage.get("completion_tokens"))

        choice = data["choices"][0]
        return {
            "message": choice.get("message") or {},
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
        }
    finally:
        if own:
            await session.close()


async def stream(
    messages,
    *,
    max_tokens,
    temperature=None,
    session=None,
    label="fugu",
    ultra: bool = False,
):
    """Streaming completion. Yields (kind, text) where kind is 'content'."""
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        model = _fugu_model(ultra)
        temp = config.WRITER_TEMPERATURE if temperature is None else temperature
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        t0 = _perf.now()
        ttft = None
        usage = None
        async with session.post(
            f"{FUGU_BASE_URL}/chat/completions",
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
                    in_tok=u.get("prompt_tokens"),
                    out_tok=u.get("completion_tokens"))
    finally:
        if own:
            await session.close()


async def answer(
    messages,
    source: str,
    *,
    session=None,
    ultra: bool = True,
) -> str:
    """One-shot: send the conversation + grounding source to Fugu, get the answer back.

    Fugu's own coordinator decides which agents to invoke and how to decompose the task.
    We don't run a tool loop — Fugu handles that internally.
    """
    system = (
        "The user has provided source material (uploaded files, pasted text). "
        "Answer their request using these sources. The user's own statements about "
        "themselves are authoritative facts. Do not invent credentials, experience, "
        "or facts the user did not state.\n\n"
        "SOURCE MATERIAL:\n" + (source or "(none provided)")
    )
    fugu_messages = [{"role": "system", "content": system}]
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            content = m.get("content") or ""
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)
            fugu_messages.append({"role": m["role"], "content": content})

    return await complete(
        fugu_messages,
        max_tokens=config.DRAFT_MAX_TOKENS if source else config.AGENT_MAX_TOKENS,
        session=session,
        label="fugu:answer",
        ultra=ultra,
    )
