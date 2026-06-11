"""Tool-server LLM chat with a provider chain.

DeepSeek-direct is PRIMARY for deepseek models (when a DeepSeek key is set), Fireworks is
the automatic FALLBACK; non-deepseek and the no-key case are Fireworks-only. DeepSeek's API
is OpenAI-compatible with the SAME model names (just drop the 'accounts/fireworks/models/'
prefix), so the only per-provider difference is base_url + key. Mirrors the orchestrator's
fireworks.py chain so the auditor and memory compression survive a DeepSeek outage.
"""
import logging
import os

import httpx

logger = logging.getLogger("tool-server")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
ENABLE_DEEPSEEK_DIRECT = os.getenv("ENABLE_DEEPSEEK_DIRECT", "true").lower() not in {
    "0", "false", "no", "off", ""}


def _providers(model: str):
    """(base_url, api_key, provider_model) to try in order: DeepSeek-direct first for
    deepseek models when keyed, then Fireworks. Non-deepseek / no key -> Fireworks only."""
    fw = (FIREWORKS_BASE, FIREWORKS_API_KEY, model)
    if ENABLE_DEEPSEEK_DIRECT and DEEPSEEK_API_KEY and "deepseek" in (model or ""):
        return [(DEEPSEEK_BASE, DEEPSEEK_API_KEY, model.split("/")[-1]), fw]
    return [fw]


async def chat(model, messages, *, max_tokens=1500, temperature=0.0,
               reasoning_effort=None, timeout=60.0) -> str:
    """Non-streaming chat with the DeepSeek-direct -> Fireworks chain. Returns content text.
    Raises the last provider's exception if every provider fails."""
    providers = _providers(model)
    last = None
    for idx, (base, key, pmodel) in enumerate(providers):
        is_ds = base == DEEPSEEK_BASE
        payload = {"model": pmodel, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        # Reasoning pinned by reasoning_effort (user policy: MAX for substantive work, "none"
        # for classifiers). DeepSeek-direct: thinking:enabled + reasoning_effort for real
        # effort, thinking:disabled to pin fast (it ignores Fireworks' "none" and defaults to
        # "high"). Fireworks flash (fallback): reasoning_effort directly, "max" -> "high"
        # (its top); Fireworks pro keeps its default on that rare path.
        if reasoning_effort and "deepseek" in pmodel:
            if is_ds:
                if reasoning_effort == "none":
                    payload["thinking"] = {"type": "disabled"}
                else:
                    payload["reasoning_effort"] = reasoning_effort
                    payload["thinking"] = {"type": "enabled"}
            elif "deepseek-v4-flash" in pmodel:
                payload["reasoning_effort"] = "high" if reasoning_effort == "max" else reasoning_effort
        try:
            async with httpx.AsyncClient(timeout=timeout) as cl:
                r = await cl.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                )
                r.raise_for_status()
                return (r.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            last = e
            if idx + 1 < len(providers):
                logger.warning(f"[provider] deepseek-direct failed ({type(e).__name__}); "
                               f"falling back to fireworks")
    raise last
