"""Spend panel — usage ledger ($ by user/model/job) + actual Fireworks bill (by channel).

Mounted on the app via app.include_router(usage.router). Admin-only: every data endpoint
validates the caller's OWUI session token (role=admin) against OWUI's own /api/v1/auths/
— no separate password, the browser's existing OWUI session is the credential. The page
and sidebar-button assets live in static/ as real files, not Python string literals.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

import memory

logger = logging.getLogger("tool-server")
router = APIRouter()

OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://open-webui:8080").rstrip("/")
_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _asset(name: str) -> str:
    with open(os.path.join(_STATIC, name), encoding="utf-8") as f:
        return f.read()


# ── Pricing ($/Mtok in,out) ────────────────────────────────────────────────────────
# Fireworks published serverless rates (fireworks.ai/pricing, verified 2026-06);
# OpenAI/Anthropic for the prose tiers. NOTE: Fireworks bills CACHED input at 50%, so
# actual spend on cache-heavy workloads runs below token*rate here — the authoritative
# dollars are the Fireworks bill (see /usage/fireworks). flash/oss/v3 are estimates.
# Override any of these via the USAGE_PRICES env (JSON); applied at query time.
USAGE_PRICES: dict = {
    "deepseek-v4-pro": (1.74, 3.48), "deepseek-v4-flash": (0.22, 0.88),
    "glm-5p1": (1.40, 4.40), "glm-5": (1.40, 4.40),
    "kimi-k2p6": (0.95, 4.00), "kimi-k2p5": (0.95, 4.00), "kimi-k2-thinking": (0.95, 4.00),
    "gpt-5.5": (1.25, 10.00), "claude-sonnet-4-6": (3.00, 15.00),
    "qwen3-embedding": (0.02, 0.0),
    "deepseek-v3p1": (0.27, 1.10), "deepseek-v3p2": (0.27, 1.10),
    "gpt-oss-120b": (0.15, 0.60), "gpt-oss-20b": (0.05, 0.20),
    "mixtral-8x22b-instruct": (0.90, 0.90), "cogito-671b-v2-p1": (0.90, 0.90),
    "v4-research-writing": (1.74, 3.48),
}

# Live-priced overrides from monthly update_prices.py cron job. Read a JSON file
# of {"model": [input_price, output_price]} from the data volume if present.
# Deployed prices take precedence over hardcoded defaults.
_PRICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage_prices.json")
if os.path.exists(_PRICE_FILE):
    try:
        with open(_PRICE_FILE) as f:
            live = {k: tuple(v) for k, v in json.load(f).items()}
        USAGE_PRICES.update(live)
        logger.info("Loaded %d live prices from %s", len(live), _PRICE_FILE)
    except Exception:
        pass

try:
    USAGE_PRICES.update({k: tuple(v) for k, v in json.loads(os.getenv("USAGE_PRICES", "{}")).items()})
except Exception:
    pass


def _usage_cost(model: str, in_tok: int, out_tok: int) -> float:
    m = (model or "").split("/")[-1]
    for key, (pi, po) in USAGE_PRICES.items():
        if key in m:
            return (in_tok * pi + out_tok * po) / 1_000_000
    return 0.0


# ── Admin auth via the OWUI session ────────────────────────────────────────────────
_admin_cache: dict = {}  # token -> (expiry_monotonic, user) — avoid hammering OWUI


async def _owui_admin(token: str) -> Optional[dict]:
    """Validate an OWUI session token and REQUIRE admin role — verified against OWUI's own
    /api/v1/auths/, so the browser's existing session is the only credential needed."""
    token = (token or "").strip()
    if not token:
        return None
    hit = _admin_cache.get(token)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{OPENWEBUI_BASE_URL}/api/v1/auths/",
                                  headers={"Authorization": f"Bearer {token}"})
        user = r.json() if r.status_code == 200 else {}
        admin = user if user.get("role") == "admin" else None
        _admin_cache[token] = (time.monotonic() + 60, admin)
        return admin
    except Exception:
        return None


def _bearer(request: Request) -> str:
    return (request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or request.query_params.get("token", ""))


def _period_window(period: str):
    """Map a dropdown period to a [since, until) unix window + a label.
    '' / 'all' -> all time; 'last3' -> last 3 calendar months; 'YYYY-MM' -> that month."""
    p = (period or "").strip()
    now = dt.datetime.now(dt.timezone.utc)
    if not p or p == "all":
        return None, None, "all time"
    if p == "last3":
        y, m = now.year, now.month - 2
        y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
        start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        return start.timestamp(), None, "last 3 months"
    try:
        y, m = (int(x) for x in p.split("-"))
        start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        nxt = dt.datetime(ny, nm, 1, tzinfo=dt.timezone.utc)
        return start.timestamp(), nxt.timestamp(), p
    except Exception:
        return None, None, "all time"


# ── Actual Fireworks bill, split by channel (= API key) ────────────────────────────
# billing/summary gives REAL cache-adjusted $ per MODEL (no key split); billingUsage gives
# TOKENS per (key, model) but no $. Combine: apportion each model's real $ to channels by
# their token share of that model. So OpenCode vs PrismAI show their actual billed dollars.
_FW_ACCT: dict = {}


def _fw_money(c: dict) -> float:
    return int(c.get("units") or 0) + int(c.get("nanos") or 0) / 1e9


def _fw_norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower().replace(".", "p"))


async def _fw_account():
    if _FW_ACCT:
        return _FW_ACCT["id"], _FW_ACCT["created"]
    key = os.getenv("FIREWORKS_API_KEY", "")
    async with httpx.AsyncClient(timeout=15) as cl:
        r = await cl.get("https://api.fireworks.ai/v1/accounts",
                         headers={"Authorization": "Bearer " + key})
    a = (r.json().get("accounts") or [{}])[0]
    _FW_ACCT["id"] = a.get("name", "").split("/")[-1]
    _FW_ACCT["created"] = dt.datetime.fromisoformat(
        a.get("createTime", "2026-01-01T00:00:00Z").replace("Z", "+00:00"))
    return _FW_ACCT["id"], _FW_ACCT["created"]


async def _fireworks_billing(since, until) -> dict:
    acct, created = await _fw_account()
    now = dt.datetime.now(dt.timezone.utc)
    s = dt.datetime.fromtimestamp(since, dt.timezone.utc) if since else created
    u = dt.datetime.fromtimestamp(until, dt.timezone.utc) if until else now
    s = max(s, created)
    wins, cur = [], s
    while cur < u:                                  # Fireworks caps windows at 31 days
        nxt = min(cur + dt.timedelta(days=30), u)
        wins.append((cur, nxt)); cur = nxt
    key = os.getenv("FIREWORKS_API_KEY", "")
    hdr = {"Authorization": "Bearer " + key}
    model_usd, key_model_tok, model_tok = {}, {}, {}
    async with httpx.AsyncClient(timeout=30) as cl:
        for a, b in wins:
            qp = f"start_time={a:%Y-%m-%dT%H:%M:%SZ}&end_time={b:%Y-%m-%dT%H:%M:%SZ}"
            base = f"https://api.fireworks.ai/v1/accounts/{acct}"
            r1 = await cl.get(f"{base}/billing/summary?{qp}", headers=hdr)
            for li in r1.json().get("lineItems", []):
                nm = _fw_norm(str(li.get("groupingValue") or ""))
                model_usd[nm] = model_usd.get(nm, 0.0) + _fw_money(li.get("totalCost") or {})
            r2 = await cl.get(f"{base}/billingUsage?{qp}&groupBy=api_key_name", headers=hdr)
            for row in r2.json().get("serverlessCosts", []):
                ch = row.get("group", {}).get("api_key_name") or "(unnamed key)"
                nm = _fw_norm(str(row.get("modelName", "")).split("/")[-1])
                t = int(row.get("promptTokens") or 0) + int(row.get("completionTokens") or 0)
                key_model_tok.setdefault(ch, {})[nm] = key_model_tok.setdefault(ch, {}).get(nm, 0) + t
                model_tok[nm] = model_tok.get(nm, 0) + t
    by_channel = {}
    for ch, mt in key_model_tok.items():
        usd = sum(model_usd.get(nm, 0.0) * (t / model_tok[nm])
                  for nm, t in mt.items() if model_tok.get(nm))
        by_channel[ch] = {"usd": round(usd, 4)}
    by_model = {nm: {"usd": round(v, 4)} for nm, v in model_usd.items() if v > 0}
    return {"by_channel": by_channel, "by_model": by_model,
            "total_usd": round(sum(model_usd.values()), 2)}


# ── Routes ─────────────────────────────────────────────────────────────────────────
class UsageLogRequest(BaseModel):
    model: str
    label: str = ""
    in_tok: int = 0
    out_tok: int = 0
    user_id: str = ""


@router.post("/usage/log", operation_id="usage_log")
async def usage_log(req: UsageLogRequest) -> dict:
    # Called only on the internal docker network by the orchestrator (not proxied out).
    return {"ok": await memory.log_usage(req.model, req.label, req.in_tok, req.out_tok, req.user_id)}


@router.get("/usage/summary", operation_id="usage_summary")
async def usage_summary_api(request: Request, period: str = "") -> dict:
    if not await _owui_admin(_bearer(request)):
        raise HTTPException(status_code=403, detail="admin only")
    await memory.sweep_owui_usage()  # fold in direct-chat usage before reporting
    since, until, label = _period_window(period)
    s = await memory.usage_summary(cost_fn=_usage_cost, since=since, until=until)
    s["period"] = label
    s["multi_month"] = until is None or label == "last 3 months"
    return s


@router.get("/usage/fireworks")
async def usage_fireworks_api(request: Request, period: str = "") -> dict:
    if not await _owui_admin(_bearer(request)):
        raise HTTPException(status_code=403, detail="admin only")
    since, until, label = _period_window(period)
    try:
        d = await _fireworks_billing(since, until)
    except Exception as e:
        logger.warning(f"fireworks billing failed: {e}")
        d = {"by_channel": {}, "by_model": {}, "total_usd": 0, "error": str(e)}
    d["period"] = label
    return d


@router.get("/usage-button.js")
async def usage_button_js():
    """The admin-only sidebar button, injected once into OWUI's page. It clones OWUI's own
    'New Chat' control to match the theme, opens the same-origin /usage panel (no second
    login), and fails silent for non-admins / if the sidebar isn't found."""
    return Response(_asset("usage-button.js"), media_type="application/javascript")


@router.get("/usage")
async def usage_page():
    """Client-rendered, admin-only. Served same-origin as OWUI so its JS reads the existing
    OWUI session token from localStorage — no second login. All $ comes from /usage/summary
    and /usage/fireworks (both admin-gated)."""
    return HTMLResponse(_asset("usage.html"), headers={"Cache-Control": "no-store"})
