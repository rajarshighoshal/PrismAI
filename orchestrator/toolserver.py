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
    """Forward the headers the tool-server needs to attach exported files.

    OWUI forwards X-OpenWebUI-Chat-Id / -Message-Id (no hyphen in 'openwebui');
    the tool-server reads the hyphenated x-open-webui-* spelling — so translate.
    """
    if not headers:
        return {}
    lower = {str(k).lower(): str(v) for k, v in dict(headers).items() if v is not None}
    out = {}
    if lower.get("authorization"):
        out["authorization"] = lower["authorization"]
    for owui_name, ts_name in (
        ("x-openwebui-chat-id", "x-open-webui-chat-id"),
        ("x-openwebui-message-id", "x-open-webui-message-id"),
    ):
        value = lower.get(owui_name) or lower.get(ts_name)
        if value:
            out[ts_name] = value
    return out


def summarize_result(name: str, result):
    """Keep model-visible tool output compact and avoid base64 context bloat.

    For exports, the model only needs to know it succeeded — the file delivery
    (upload + download link) is the harness's job. We deliberately do NOT surface
    upload errors to the model, or it tries to "fix" them by re-exporting other
    formats in a loop.
    """
    if name.startswith("export_") and isinstance(result, list):
        meta = {}
        for item in result:
            if isinstance(item, dict):
                meta.update(item)
        fmt = name.replace("export_", "")
        return {
            "status": "success",
            "filename": meta.get("filename"),
            "note": f"The {fmt} file was generated and delivered to the user. Do not export again.",
        }
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
