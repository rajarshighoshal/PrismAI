"""One structured perf line per model call, across every provider.

[trace] label=<step> model=<name> in=<prompt_tokens> out=<completion_tokens>
        ttft=<first-token s> ttlt=<last-token s>

Grep `[trace]` in the orchestrator logs to reconstruct a turn's full model chain
with real token counts and latency. TTFT==TTLT for non-streaming calls (the whole
response arrives at once).
"""
import asyncio
import logging
import time

try:
    import aiohttp
except ImportError:
    aiohttp = None

from . import config

log = logging.getLogger("perf")

_BG: set = set()  # strong refs: fire-and-forget tasks must not be GC'd mid-flight


async def _post_usage(model, label, in_tok, out_tok):
    """Persist the call's tokens to the tool-server usage ledger (the spend panel).
    Fails soft — accounting must never break a turn."""
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(
                f"{config.TOOL_SERVER_URL}/usage/log",
                json={"model": str(model).split("/")[-1], "label": str(label or "?"),
                      "in_tok": int(in_tok or 0), "out_tok": int(out_tok or 0)},
                timeout=aiohttp.ClientTimeout(total=5),
            )
    except Exception:
        pass


def now() -> float:
    return time.perf_counter()


def trace(label, model, *, t0, ttft=None, in_tok=None, out_tok=None):
    ttlt = time.perf_counter() - t0
    ttft_s = f"{ttft - t0:.2f}" if ttft else f"{ttlt:.2f}"
    log.info(
        f"[trace] label={label or '?'} model={str(model).split('/')[-1]} "
        f"in={in_tok if in_tok is not None else '?'} "
        f"out={out_tok if out_tok is not None else '?'} "
        f"ttft={ttft_s}s ttlt={ttlt:.2f}s"
    )
    if aiohttp is not None and (in_tok or out_tok):
        try:
            t = asyncio.get_running_loop().create_task(_post_usage(model, label, in_tok, out_tok))
            _BG.add(t)
            t.add_done_callback(_BG.discard)
        except RuntimeError:
            pass  # no running loop (sync/test context) — accounting is best-effort
