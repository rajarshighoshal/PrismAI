"""
Vision proxy for the PrismAI router — image captioning for non-vision models.

Extracted from router_fn.py as an independent module. Handles:
- Auto-detection of OpenWebUI base URL for resolving internal image paths
- Resolution of internal/relative image URLs to base64 data URIs
- Short routing captions (1-2 sentences for classification)
- Detailed captions (comprehensive for vision proxy — text-only models "see")
- Per-chat image caption caching (no repeat caption calls for re-sent images)

Dependencies are injected at construction, keeping this module importable
without circular dependencies on router_fn.py.
"""

import asyncio
import base64
import hashlib
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class VisionProxy:
    """Image captioning + resolution for the router's vision proxy feature.

    Takes callbacks for model dispatch and LLM/vision model calls so it
    stays independent of the router's Filter class.
    """

    def __init__(
        self,
        dispatch_model: Callable,    # (model_id) → (base_url, api_key, stripped_id, extra_headers)
        get_session: Callable,       # () → aiohttp.ClientSession
        call_vision: Callable,       # async (vision_content, max_tokens, fallback_chain, log_role) → str or None
        emit_status: Callable,       # async (event_emitter, description) → None
        valves,
    ):
        self.dispatch_model = dispatch_model
        self.get_session = get_session
        self.call_vision = call_vision
        self.emit_status = emit_status
        self.valves = valves
        # Cached OWUI base URL (auto-detected once per process)
        self._owui_base_url: Optional[str] = None
        # Per-chat image caption cache: (chat_id, url_hash) → caption
        # Prevents re-captioning the same image on follow-up turns.
        self._caption_cache: dict[str, str] = {}
        self._caption_cache_max = 200

    # ── OWUI base URL detection ────────────────────────────────────────

    async def _detect_owui_base_url(self) -> Optional[str]:
        """Auto-detect the OpenWebUI base URL for fetching internal images.

        Checks: WEBUI_URL env → port env vars → localhost probe → Docker names.
        Result is cached after first successful detection.
        """
        if self._owui_base_url is not None:
            return self._owui_base_url or None

        # 1. WEBUI_URL — OpenWebUI sets this for its own API callbacks
        env_url = os.environ.get("WEBUI_URL", "")
        if env_url and env_url.startswith("http"):
            self._owui_base_url = env_url.rstrip("/")
            logger.info("OWUI base URL from WEBUI_URL: %s", self._owui_base_url)
            return self._owui_base_url

        # 2. Port from env vars
        for env_key in ("OPEN_WEBUI_PORT", "PORT", "SERVER_PORT"):
            port = os.environ.get(env_key, "")
            if port.isdigit():
                self._owui_base_url = f"http://localhost:{port}"
                logger.info("OWUI base URL from %s", env_key)
                return self._owui_base_url

        # 3. Probe common localhost ports (skip 11434 — that's ollama)
        session = await self.get_session()
        for port in (8080, 3000, 80):
            candidate = f"http://localhost:{port}"
            try:
                async with session.head(
                    f"{candidate}/api/v1/auths",
                    timeout=asyncio.TimeoutError(1) if hasattr(asyncio, "TimeoutError") else None,
                ) as probe:
                    if probe.status in (200, 401, 403, 405, 422):
                        self._owui_base_url = candidate
                        logger.info("OWUI detected at %s (probe %d)", candidate, probe.status)
                        return self._owui_base_url
            except Exception:
                continue

        # 4. Docker container names
        for host in ("http://open-webui:8080", "http://openwebui:8080"):
            try:
                async with session.head(f"{host}/api/v1/auths") as probe:
                    if probe.status in (200, 401, 403, 405, 422):
                        self._owui_base_url = host
                        logger.info("OWUI detected at %s", host)
                        return self._owui_base_url
            except Exception:
                continue

        logger.warning("Could not auto-detect OpenWebUI base URL — internal images won't resolve")
        self._owui_base_url = ""  # cache the failure
        return None

    # ── Image URL resolution ───────────────────────────────────────────

    async def _resolve_image_urls(
        self,
        image_parts: list[dict],
        event_emitter=None,
    ) -> list[dict]:
        """Convert internal/relative image URLs to base64 data URIs.

        OpenWebUI sends images as internal URLs (e.g. /api/v1/files/...) that
        external APIs can't fetch. Downloads them locally and rewrites to
        data:image/...;base64,... URIs.
        """
        base = await self._detect_owui_base_url()
        resolved: list[dict] = []
        errors: list[str] = []
        session = await self.get_session()

        for part in image_parts:
            url = (part.get("image_url") or {}).get("url", "")

            # Already a data URI or public URL — pass through
            if url.startswith("data:image/") or (
                url.startswith("https://") and "localhost" not in url and "127.0.0.1" not in url
            ):
                resolved.append(part)
                continue

            # Need base URL to resolve internal paths
            if not base:
                errors.append(f"Internal image URL but OWUI base not detected: {url[:60]}")
                continue

            if url.startswith("/"):
                full_url = f"{base}{url}"
            elif url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
                path = url.split("//", 1)[-1]
                path = "/" + path.split("/", 1)[-1] if "/" in path else url
                full_url = f"{base}{path}"
            else:
                logger.warning("Unclassifiable image URL, passing through: %s", url[:100])
                resolved.append(part)
                continue

            try:
                async with session.get(full_url) as img_resp:
                    if img_resp.status != 200:
                        errors.append(f"Image fetch HTTP {img_resp.status}: {url[:60]}")
                        continue
                    img_bytes = await img_resp.read()
                    content_type = img_resp.headers.get("Content-Type", "image/png")
                    if not content_type.startswith("image/"):
                        content_type = self._guess_image_type(img_bytes)
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    data_uri = f"data:{content_type};base64,{b64}"
                    resolved.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
                    logger.info("Resolved local image → base64 (%d bytes)", len(img_bytes))
            except Exception as e:
                errors.append(f"Image fetch error: {str(e)[:80]}")

        if errors and event_emitter and self.valves.EMIT_STATUS_EVENTS:
            for err in errors[:3]:
                await self.emit_status(event_emitter, f"⚠️ Image resolve: {err}")

        return resolved

    @staticmethod
    def _guess_image_type(data: bytes) -> str:
        """Guess MIME type from magic bytes when Content-Type header is missing/wrong."""
        if data[:4] == b"\x89PNG":
            return "image/png"
        if data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        if data[:4] == b"GIF8":
            return "image/gif"
        return "image/png"

    # ── Caption generation ─────────────────────────────────────────────

    async def caption_for_routing(
        self,
        image_parts: list[dict],
        user_text: str = "",
        event_emitter=None,
    ) -> Optional[str]:
        """Short caption (1-2 sentences) for routing classification.

        Used by the router inlet to understand image content for category
        classification without sending images to the LLM classifier.
        """
        if not image_parts or not self.valves.IMAGE_CAPTION_MODEL:
            return None

        clean_parts = await self._resolve_image_urls(image_parts, event_emitter)
        if not clean_parts:
            await self.emit_status(
                event_emitter,
                "⚠️ Image captioning: no reachable images after resolving URLs.",
            )
            return None

        prompt_text = (
            "Describe what you see in this image in 1-2 short sentences. "
            "Focus on: subject matter (code, math, chart, photo, diagram, etc.), "
            "any visible text or labels, and the overall topic."
        )
        if user_text.strip():
            prompt_text += f" The user's accompanying text is: '{user_text.strip()[:200]}'"

        vision_content: list[dict] = [{"type": "text", "text": prompt_text}]
        vision_content.extend(clean_parts)

        caption = await self.call_vision(
            vision_content,
            self.valves.IMAGE_CAPTION_MAX_TOKENS,
            self._caption_fallback_chain(),
            log_role="caption",
        )
        if caption is None:
            await self.emit_status(
                event_emitter,
                "⚠️ Image captioning failed (all models in fallback chain).",
            )
        return caption

    async def caption_detailed(
        self,
        image_parts: list[dict],
        user_text: str = "",
        event_emitter=None,
    ) -> Optional[str]:
        """Rich, detailed caption for the vision proxy.

        Unlike the short routing caption, this produces a thorough description
        that preserves all key details a text-only model needs: visible text,
        error messages, code snippets, chart values, layout, colors.
        """
        if not image_parts or not self.valves.IMAGE_CAPTION_MODEL:
            return None

        clean_parts = await self._resolve_image_urls(image_parts, event_emitter)
        if not clean_parts:
            return None

        prompt_text = (
            "Provide a thorough, detailed description of this image. "
            "This description will be read by a text-only AI that cannot see the image, "
            "so be as comprehensive as possible. Include:\n"
            "1. TYPE: What kind of image is this? (screenshot, photo, diagram, chart, "
            "document scan, whiteboard, meme, etc.)\n"
            "2. VISIBLE TEXT: Quote ALL visible text exactly as it appears — error messages, "
            "code, labels, titles, axis values, button text, filenames, URLs, numbers. "
            "Do NOT paraphrase or summarize text — reproduce it verbatim.\n"
            "3. LAYOUT & STRUCTURE: Describe the spatial arrangement.\n"
            "4. COLORS & SHAPES: Key visual elements.\n"
            "5. CONTEXT: What is the overall subject?\n\n"
            "Be specific and precise. A developer reading this description should be able "
            "to understand and act on the image content without seeing it."
        )
        if user_text.strip():
            prompt_text += f'\n\nThe user\'s message accompanying this image is: "{user_text.strip()[:300]}"'

        vision_content: list[dict] = [{"type": "text", "text": prompt_text}]
        vision_content.extend(clean_parts)

        caption = await self.call_vision(
            vision_content,
            self.valves.IMAGE_PROXY_MAX_TOKENS,
            self._caption_fallback_chain(),
            log_role="caption_detailed",
        )
        if caption is None:
            await self.emit_status(
                event_emitter,
                "⚠️ Detailed image caption failed (all models in fallback chain).",
            )
        return caption

    def _caption_fallback_chain(self):
        """Build the caption model fallback chain with the primary first."""
        chain = [
            "accounts/fireworks/models/kimi-k2p5",
            "groq/meta-llama/llama-4-scout-17b-16e-instruct",
            "accounts/fireworks/models/kimi-k2p6",
        ]
        if self.valves.IMAGE_CAPTION_MODEL in chain:
            chain.remove(self.valves.IMAGE_CAPTION_MODEL)
        chain.insert(0, self.valves.IMAGE_CAPTION_MODEL)
        if not self.valves.GROQ_API_KEY:
            chain = [m for m in chain if not m.startswith("groq/")]
        return chain

    # ── Vision proxy: replace images with captions in messages ─────────

    async def run_vision_proxy(
        self,
        messages: list[dict],
        query_raw: str,
        chat_id: str,
        event_emitter=None,
    ) -> None:
        """Replace image parts with text captions for non-vision models.

        Mutates `messages` in place: for every message that contains image_url
        parts, replaces them with a text description so the text-only model can
        "see" the image. Captions are cached per (chat_id, url_hash) so
        follow-up turns don't re-caption the same images.
        """
        if not self.valves.ENABLE_VISION_PROXY:
            return

        # Collect all image parts across all messages
        all_image_parts: list[tuple[int, dict]] = []
        for i, m in enumerate(messages):
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    all_image_parts.append((i, part))

        if not all_image_parts:
            return

        # Resolve all unique images (dedup by original URL)
        resolved_pairs = []
        for _, original_part in all_image_parts:
            original_url = (original_part.get("image_url") or {}).get("url", "")
            for resolved_part in await self._resolve_image_urls(
                [original_part], event_emitter=event_emitter
            ):
                resolved_pairs.append((original_url, resolved_part))

        # Build caption map (cached + new)
        chat_cache_id = chat_id or "unknown"
        image_url_to_caption: dict[str, str] = {}
        unique_images: dict[str, dict] = {}
        for original_url, resolved_part in resolved_pairs:
            if original_url and original_url not in unique_images:
                unique_images[original_url] = resolved_part

        new_captions, cached_captions = 0, 0
        for url, resolved_part in unique_images.items():
            url_digest = hashlib.sha256(url.encode()).hexdigest()[:32]
            cache_key = f"{chat_cache_id}:{url_digest}"
            if cache_key in self._caption_cache:
                image_url_to_caption[url] = self._caption_cache[cache_key]
                cached_captions += 1
            else:
                caption = await self.caption_detailed(
                    [resolved_part], user_text=query_raw, event_emitter=event_emitter
                )
                if caption:
                    image_url_to_caption[url] = caption
                    self._caption_cache[cache_key] = caption
                    while len(self._caption_cache) > self._caption_cache_max:
                        self._caption_cache.pop(next(iter(self._caption_cache)))
                    new_captions += 1
                else:
                    image_url_to_caption[url] = "[Image description unavailable]"

        # Apply captions: replace image_url parts with text
        total_replaced = 0
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            new_parts, has_images = [], False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    caption_text = image_url_to_caption.get(url, "[Attached image]")
                    new_parts.append({"type": "text", "text": f"\n[Attached image: {caption_text}]"})
                    has_images = True
                    total_replaced += 1
                else:
                    new_parts.append(part)
            if has_images:
                m["content"] = new_parts

        if new_captions > 0:
            await self.emit_status(
                event_emitter,
                f"🖥️ Vision proxy: captioned {new_captions} new image(s), "
                f"reused {cached_captions} cached, replaced {total_replaced} image part(s).",
            )
