"""Request de-duplication / idempotency.

When the exact same request arrives again within a short window — a client retry
after a network hiccup, a double-submit, or a hung connection — we must not run
the whole agent + LLM pipeline a second time. Two layers:

- completed-result cache: an identical request that arrives AFTER the first one
  finished replays the stored answer (no LLM call).
- single-flight: an identical request that arrives WHILE the first is still
  running attaches to the first one's result instead of starting a parallel run.

Keyed on the request CONTENT (messages + model + user), so only a byte-identical
question coalesces. Short TTL: a retry happens within seconds; a genuinely new
ask of the same question after the window re-runs and gets a fresh answer.

In-process only (a dict): de-duping retries against the same worker, which is
where client/network retries land. Not a distributed cache.
"""
import asyncio
import hashlib
import json
import logging
import time

from . import config

log = logging.getLogger(__name__)

_results: dict = {}    # key -> (answer, expiry_monotonic)
_inflight: dict = {}   # key -> the lead request's Future


def enabled() -> bool:
    return getattr(config, "ENABLE_DEDUP", True)


def make_key(messages, model: str, user_id: str) -> str:
    raw = json.dumps([messages, model, user_id], sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _sweep(now: float) -> None:
    # Opportunistic cleanup so the dict can't grow unbounded under load.
    if len(_results) <= 1024:
        return
    for k in [k for k, (_, exp) in _results.items() if exp <= now]:
        _results.pop(k, None)


def get_cached(key: str):
    hit = _results.get(key)
    if not hit:
        return None
    answer, exp = hit
    if exp <= time.monotonic():
        _results.pop(key, None)
        return None
    return answer


def store(key: str, answer: str) -> None:
    if answer and answer.strip():
        now = time.monotonic()
        _results[key] = (answer, now + config.DEDUP_TTL_SECONDS)
        _sweep(now)


def begin(key: str):
    """Classify an incoming request. Returns (mode, payload):

      ("cached", answer)  -> replay the stored answer; do not run the pipeline.
      ("follow", future)  -> an identical request is in flight; await its result.
      ("lead", future)    -> you are the original; run the pipeline, then call
                             resolve() with the answer (or the failure).
    """
    cached = get_cached(key)
    if cached is not None:
        return "cached", cached
    fut = _inflight.get(key)
    if fut is not None and not fut.done():
        return "follow", fut
    fut = asyncio.get_event_loop().create_future()
    _inflight[key] = fut
    return "lead", fut


def resolve(key: str, fut: "asyncio.Future", *, answer: str = None, exc: BaseException = None) -> None:
    """The lead publishes its result to any followers and clears the in-flight
    slot. A successful, non-empty answer is also stored in the completed cache.
    On failure (or an empty/aborted answer) followers receive the exception and
    fall back to running their own request."""
    if exc is None and answer is not None and answer.strip():
        store(key, answer)
    if not fut.done():
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(answer or "")
    if _inflight.get(key) is fut:
        _inflight.pop(key, None)
