"""Thin async client for the tool-server (verification + export primitives).

Fails soft: on any error returns None / falls back, so a tool-server hiccup
degrades gracefully instead of breaking the chat turn.
"""
import aiohttp

from . import config


async def verify_grounding(source: str, draft: str, *, session=None):
    """Audit draft claims against source. Returns the tool-server's dict
    {grounded: bool, unsupported_claims: str} or None on error."""
    own = session is None
    if own:
        session = aiohttp.ClientSession()
    try:
        async with session.post(
            f"{config.TOOL_SERVER_URL}/verify_grounding",
            json={"source": source, "draft": draft},
            timeout=aiohttp.ClientTimeout(total=config.HTTP_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception:
        return None
    finally:
        if own:
            await session.close()
