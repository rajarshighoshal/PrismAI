"""One structured perf line per model call, across every provider.

[trace] label=<step> model=<name> in=<prompt_tokens> out=<completion_tokens>
        ttft=<first-token s> ttlt=<last-token s>

Grep `[trace]` in the orchestrator logs to reconstruct a turn's full model chain
with real token counts and latency. TTFT==TTLT for non-streaming calls (the whole
response arrives at once).
"""
import logging
import time

log = logging.getLogger("perf")


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
