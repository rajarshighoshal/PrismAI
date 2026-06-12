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


def _provider_headers(key: str, session_id: str) -> dict:
    h = {"Content-Type": "application/json", "x-session-affinity": session_id}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _providers(model: str):
    """Ordered (base_url, api_key, provider_model, is_deepseek_direct) to try. For deepseek
    models with a DeepSeek key configured: DeepSeek-direct first (same model name, just
    drop the 'accounts/fireworks/models/' prefix), Fireworks as the fallback. Everything
    else — and the no-key case — is Fireworks only, identical to before."""
    fw = (config.FIREWORKS_BASE_URL, config.FIREWORKS_API_KEY, model, False)
    if (config.ENABLE_DEEPSEEK_DIRECT and config.DEEPSEEK_API_KEY
            and "deepseek" in (model or "")):
        return [(config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY, model.split("/")[-1], True), fw]
    return [fw]


def _effort_for(label: str, explicit) -> str:
    """Resolve reasoning effort by ROLE. An explicit value wins. Classifier labels
    (gate:*, summarize) -> 'none'. CASUAL CHAT (label 'chat', the plain-chat fast path
    already gated as 'no work needed') -> CHAT_REASONING_EFFORT (default 'none'): measured
    ~2x faster (1.4s vs 3.0s) on a conversational turn AND it stops max-thinking from eating
    the token budget and truncating a short reply. Everything SUBSTANTIVE (agent loop,
    deliverables, audit, grounded, vision, edits) -> config.REASONING_EFFORT ('max')."""
    if explicit is not None:
        return explicit
    lbl = label or ""
    if lbl.startswith("gate:") or lbl == "summarize":
        return "none"
    if lbl == "chat":
        return config.CHAT_REASONING_EFFORT
    return config.REASONING_EFFORT


def _chat_payload(provider_model, *, messages, max_tokens, temperature, session_id,
                  tools=None, tool_choice=None, stream=False, effort=None, is_ds=False):
    p = {
        "model": provider_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "user": session_id,
    }
    if stream:
        p["stream"] = True
        p["stream_options"] = {"include_usage": True}
    if tools is not None:
        p["tools"] = tools
    if tool_choice is not None:
        p["tool_choice"] = tool_choice
    # Reasoning pinned by the resolved effort. DeepSeek-direct: thinking:enabled +
    # reasoning_effort (max/high/…) for substantive work; thinking:disabled to pin a
    # classifier fast (it ignores Fireworks' "none" and would otherwise default to "high").
    # Fireworks flash (fallback path): reasoning_effort directly, with "max" -> "high"
    # (its top); Fireworks pro keeps its own default on that rare path.
    if "deepseek" in (provider_model or "") and effort:
        if is_ds:
            if effort == "none":
                p["thinking"] = {"type": "disabled"}
            else:
                p["reasoning_effort"] = effort
                p["thinking"] = {"type": "enabled"}
        elif "deepseek-v4-flash" in provider_model:
            p.setdefault("reasoning_effort", "high" if effort == "max" else effort)
    return p


async def complete(messages, model, *, max_tokens, temperature=None, session=None, user_id=None, label="", reasoning_effort=None, return_finish=False):
    """Non-streaming completion. Returns the final answer text (content only) — or, when
    return_finish=True, the tuple (content, finish_reason) so a caller can tell a complete
    answer from a TRUNCATED one (finish_reason == 'length'). The honesty auditor needs that:
    a verdict cut off mid-emit must fail closed, not read as 'clean'."""
    result = await chat(
        messages,
        model,
        max_tokens=max_tokens,
        temperature=temperature,
        session=session,
        user_id=user_id,
        label=label,
        reasoning_effort=reasoning_effort,
    )
    content = (result.get("message", {}).get("content") or "").strip()
    if return_finish:
        return content, result.get("finish_reason")
    return content


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
    reasoning_effort=None,
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
        effort = _effort_for(label, reasoning_effort)
        providers = _providers(model)
        last_exc = None
        for idx, (base, key, pmodel, is_ds) in enumerate(providers):
            payload = _chat_payload(
                pmodel, messages=messages, max_tokens=max_tokens, temperature=temperature,
                session_id=session_id, tools=tools, tool_choice=tool_choice,
                effort=effort, is_ds=is_ds)
            t0 = time.perf_counter()
            try:
                async with session.post(
                    f"{base}/chat/completions",
                    headers=_provider_headers(key, session_id),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.GEN_TIMEOUT),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                _trace(label, pmodel, t0=t0, usage=data.get("usage") or {})
                choice = data["choices"][0]
                return {
                    "message": choice.get("message") or {},
                    "finish_reason": choice.get("finish_reason"),
                }
            except Exception as e:
                last_exc = e
                if idx + 1 < len(providers):
                    log.warning(f"[provider] {'deepseek-direct' if is_ds else 'fireworks'} "
                                f"chat failed for {label or pmodel} ({type(e).__name__}: {e}); "
                                f"falling back")
        raise last_exc
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
        effort = _effort_for(label, None)
        providers = _providers(model)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        last_exc = None
        for idx, (base, key, pmodel, is_ds) in enumerate(providers):
            payload = _chat_payload(pmodel, messages=messages, max_tokens=max_tokens,
                                    temperature=temperature, session_id=session_id, stream=True,
                                    effort=effort, is_ds=is_ds)
            t0 = time.perf_counter()
            ttft = None
            usage = None
            yielded = False
            try:
                async with session.post(
                    f"{base}/chat/completions",
                    headers=_provider_headers(key, session_id),
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
                            yielded = True
                            yield ("reasoning", rc)
                        c = delta.get("content")
                        if c:
                            if ttft is None:
                                ttft = time.perf_counter()
                            yielded = True
                            yield ("content", c)
                _trace(label, pmodel, t0=t0, ttft=ttft, usage=usage)
                return
            except Exception as e:
                last_exc = e
                # Can only fall back if nothing was streamed yet (no half-answer to the user).
                if yielded or idx + 1 >= len(providers):
                    raise
                log.warning(f"[provider] deepseek-direct stream failed pre-token "
                            f"({type(e).__name__}); falling back to fireworks")
        if last_exc:
            raise last_exc
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
        effort = _effort_for(label, None)
        providers = _providers(model)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=config.STREAM_IDLE_TIMEOUT)
        last_exc = None
        for idx, (base, key, pmodel, is_ds) in enumerate(providers):
            payload = _chat_payload(pmodel, messages=messages, max_tokens=max_tokens,
                                    temperature=temperature, session_id=session_id,
                                    tools=tools, tool_choice=tool_choice, stream=True,
                                    effort=effort, is_ds=is_ds)
            content_parts, calls, finish = [], {}, None
            t0 = time.perf_counter()
            ttft = None
            usage = None
            yielded = False
            try:
                async with session.post(
                    f"{base}/chat/completions",
                    headers=_provider_headers(key, session_id), json=payload, timeout=timeout,
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
                            yielded = True
                            yield ("reasoning", rc)
                        c = delta.get("content")
                        if c:
                            if ttft is None:
                                ttft = time.perf_counter()
                            yielded = True
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
                _trace(label, pmodel, t0=t0, ttft=ttft, usage=usage)
                yield ("final", {
                    "content": "".join(content_parts),
                    "tool_calls": [calls[i] for i in sorted(calls)],
                    "finish_reason": finish,
                })
                return
            except Exception as e:
                last_exc = e
                if yielded or idx + 1 >= len(providers):
                    raise
                log.warning(f"[provider] deepseek-direct stream_chat failed pre-token "
                            f"({type(e).__name__}); falling back to fireworks")
        if last_exc:
            raise last_exc
    finally:
        if own:
            await session.close()
