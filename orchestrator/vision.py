"""Native-vision phase: read attached image(s) into an EVIDENCE TRANSCRIPT + READING.

Split out of the agent god-module as a self-contained capability (part of the in-progress
prism_core). The model call is injected via `fireworks` (M3 primary + fallback); the
per-image cache is question-independent so multi-turn chats about one image never re-read it.
No dependency on the agent runtime.
"""
import asyncio
import hashlib
import logging
import re

from . import config, fireworks
from .owui import _has_images, _split_content_parts
from .prompts import SYSTEM_VISION

log = logging.getLogger(__name__)


def _split_vision_output(text: str):
    """Split vision emission into (evidence_transcript, reading) for audit-grade grounding. Falls back to whole text when the two-part structure is absent."""
    t = (text or "").strip()
    if not t:
        return "", ""
    parts = re.split(r"(?im)^\s*#+\s*reading\b.*$", t, maxsplit=1)
    if len(parts) == 2:
        transcript = re.sub(r"(?im)^\s*#+\s*evidence transcript\b.*$", "", parts[0]).strip()
        reading = parts[1].strip()
        if transcript:
            return transcript, reading
    return t, t  # no clean split -> the whole emission serves as both source and context


# Per-image vision cache: the EVIDENCE TRANSCRIPT is a question-independent, literal read of
# the pixels, so once a given image is read we never pay the 15-50s M3 call again — OWUI
# re-sends the whole conversation every turn, so a multi-turn chat about one image used to
# re-read it on EVERY follow-up. Bounded; keyed by image-content hash.
_VISION_CACHE: dict = {}
_VISION_CACHE_MAX = int(getattr(config, "VISION_CACHE_MAX", 0) or 128)


def _image_hash(image_parts, user_id: str = "") -> str:
    # Scope the key by user so two different users who upload the byte-identical image don't
    # share one cached grounding source (the value is also request-shaped via the reading).
    h = hashlib.sha256()
    h.update((user_id or "").encode("utf-8", "ignore") + b"\x00")
    for p in image_parts:
        h.update(((p.get("image_url") or {}).get("url") or "").encode("utf-8", "ignore"))
    return h.hexdigest()


async def _describe_images_for_agent(messages, *, user_id="", session=None):
    """Replace each image with native-vision text (EVIDENCE TRANSCRIPT + READING). Returns (messages, transcript) for audit routing. Images read concurrently and cached."""
    async def _describe(m):
        content = m.get("content")
        text_parts, image_parts = _split_content_parts(content)
        if not image_parts:
            return dict(m), ""
        user_text = "\n".join(t.strip() for t in text_parts if t.strip())
        ihash = _image_hash(image_parts, user_id)
        cached = _VISION_CACHE.get(ihash)
        if cached:
            # Reuse the cached read (the question-independent transcript drives this; the text
            # reasoner answers THIS turn's question from it). Skips the expensive re-read.
            _VISION_CACHE[ihash] = _VISION_CACHE.pop(ihash)  # LRU promote: a hot image isn't evicted
            new_m = dict(m)
            new_m["content"] = (((user_text + "\n\n") if user_text else "")
                                + "[What you see in the attached image:]\n" + cached)
            return new_m, cached
        prompt = (
            "The user attached image(s) with this message. Read them natively and follow your "
            "two-part contract — EVIDENCE TRANSCRIPT, then READING.\n\n"
            f"USER REQUEST:\n{user_text or '(none)'}"
        )
        detail = (config.VISION_IMAGE_DETAIL or "").strip().lower()

        def _content(use_detail: bool):
            # High image detail makes the provider tile a large/dense image at full resolution
            # instead of downscaling it into a blur the model confabulates from (A/B-proven).
            parts = []
            for p in image_parts:
                if use_detail and detail and detail != "auto" and isinstance(p.get("image_url"), dict):
                    p = {**p, "image_url": {**p["image_url"], "detail": detail}}
                parts.append(p)
            return [{"type": "text", "text": prompt}] + parts

        out = ""
        readers = [m for m in (config.VISION_MODEL, config.VISION_FALLBACK_MODEL) if m]
        for i, model in enumerate(readers):
            # Primary (M3) gets high-detail tiling for faithfulness; the FALLBACK runs WITHOUT
            # it — a lighter, faster degrade that actually returns when the primary stalled on a
            # huge tiled image (both stalling = the image getting silently dropped, the bug).
            try:
                _read = fireworks.complete(
                    [{"role": "system", "content": SYSTEM_VISION},
                     {"role": "user", "content": _content(use_detail=(i == 0))}],
                    model, max_tokens=config.VISION_MAX_TOKENS, temperature=0.0,
                    session=session, label="vision",
                    reasoning_effort=config.VISION_REASONING_EFFORT)
                # Deadline the PRIMARY so a stall hands off to the light fallback promptly.
                if i == 0 and len(readers) > 1 and config.VISION_PRIMARY_TIMEOUT > 0:
                    _read = asyncio.wait_for(_read, timeout=config.VISION_PRIMARY_TIMEOUT)
                out = (await _read).strip()
                if out:
                    break
            except Exception as e:
                log.warning(f"[vision] {model.split('/')[-1]} read failed: {type(e).__name__}: {e}")
        new_m = dict(m)
        if out:
            # Cache + GROUND against the full output (transcript + cited READING), not the
            # literal transcript alone: a legitimate visual identification the reading makes
            # (a breed/landmark/object, cited to a region) is then IN the grounding source, so
            # the forced image audit doesn't flag it as an 'unsupported' world-fact. The
            # transcript is question-independent (durable), so the cache stays valid per image.
            if len(_VISION_CACHE) >= _VISION_CACHE_MAX:
                _VISION_CACHE.pop(next(iter(_VISION_CACHE)), None)  # evict oldest (LRU; hits promote)
            _VISION_CACHE[ihash] = out
            # Frame as the assistant's OWN sight of the image (not user-pasted text), so the
            # reasoner doesn't disclaim "based on the transcription you gave me".
            new_m["content"] = (((user_text + "\n\n") if user_text else "")
                                + "[What you see in the attached image:]\n" + out)
        else:
            # NEVER silently drop the image — without this the agent sees only the question and
            # claims "no image was attached". Keep an explicit signal so it owns the failure.
            new_m["content"] = (((user_text + "\n\n") if user_text else "")
                                + "[An image WAS attached, but the vision reader could not "
                                "process it this time. Tell the user you had trouble reading "
                                "the image and ask them to re-send it — do NOT say no image was "
                                "attached.]")
        return new_m, out   # "" on a failed read -> no grounding source, no forced audit

    pairs = list(await asyncio.gather(*(_describe(m) for m in messages)))
    msgs = [p[0] for p in pairs]
    transcript = "\n\n".join(p[1] for p in pairs if p[1].strip())
    return msgs, transcript


async def _read_images(messages, user_id, session):
    """Transcribe attached images. Returns (messages, transcript)."""
    if not _has_images(messages):
        return messages, ""
    return await _describe_images_for_agent(messages, user_id=user_id, session=session)
