"""Thin async client for the tool-server (verification + export primitives).

Fails soft: on any error returns None / falls back, so a tool-server hiccup
degrades gracefully instead of breaking the chat turn.
"""
try:
    import aiohttp
except ImportError:  # lets offline tests run without installed service deps
    aiohttp = None

from . import config


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live tool-server calls")


def _forward_headers(headers=None) -> dict:
    """Forward only headers the tool-server can safely use for file attach."""
    if not headers:
        return {}
    lower = {str(k).lower(): str(v) for k, v in dict(headers).items() if v is not None}
    out = {}
    for name in (
        "authorization",
        "x-open-webui-chat-id",
        "x-open-webui-message-id",
    ):
        if lower.get(name):
            out[name] = lower[name]
    return out


def summarize_result(name: str, result):
    """Keep model-visible tool output compact and avoid base64 context bloat."""
    if name.startswith("export_") and isinstance(result, list):
        meta = {}
        for item in result:
            if isinstance(item, dict):
                meta.update(item)
        if meta:
            return meta
        return {"status": "success", "note": "export completed"}
    return result


async def post(path: str, payload: dict, *, session=None, headers=None):
    """POST to the tool-server. Returns JSON or an error dict."""
    own = session is None
    if own:
        _require_aiohttp()
        session = aiohttp.ClientSession()
    try:
        async with session.post(
            f"{config.TOOL_SERVER_URL}{path}",
            json=payload,
            headers=_forward_headers(headers),
            timeout=aiohttp.ClientTimeout(total=config.TOOL_SERVER_TIMEOUT),
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                return {
                    "error": True,
                    "status": resp.status,
                    "detail": data.get("detail") if isinstance(data, dict) else data,
                }
            return data
    except Exception as e:
        return {"error": True, "detail": f"{type(e).__name__}: {e}"}
    finally:
        if own:
            await session.close()


async def verify_grounding(source: str, draft: str, *, session=None):
    """Audit draft claims against source. Returns the tool-server's dict
    {grounded: bool, unsupported_claims: str} or None on error."""
    res = await post(
        "/verify_grounding",
        {"source": source, "draft": draft},
        session=session,
    )
    if isinstance(res, dict) and res.get("error"):
        return None
    return res
