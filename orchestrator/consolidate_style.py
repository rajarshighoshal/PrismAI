"""Per-user STYLE memory consolidation — webui.db-native (style/intent ONLY, never facts).

Design constraint (user, firm): distill HOW a user writes — voice, tone, formatting
habits, recurring intents — and NEVER carry FACTS across chats. Cross-chat fact
merging would let the model assert "facts" about the user pulled from unrelated
conversations: the exact fabrication risk this project exists to prevent. Style
generalizes safely; facts do not.

Reads OWUI's own webui.db `chat` table (the live source now that chats route
through the orchestrator, not router_fn) and writes one row per user into
`user_style_profile` in the SAME webui.db — which orchestrator/style.py reads.
Standalone weekly job (cron); does not touch the live request path.

Run:  python -m orchestrator.consolidate_style        (uses STYLE_DB_PATH)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request

DB_PATH = os.getenv("STYLE_DB_PATH", "/app/backend/data/webui.db")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
# Instruction-tuned, no chain-of-thought leak (verified).
CONSOLIDATION_MODEL = os.getenv("STYLE_MODEL", "accounts/fireworks/models/gpt-oss-120b")
MIN_TURNS = int(os.getenv("STYLE_MIN_TURNS", "5"))      # need enough signal
MAX_TURNS = int(os.getenv("STYLE_MAX_TURNS", "80"))     # cap per user
MAX_CHARS = int(os.getenv("STYLE_MAX_CHARS", "12000"))  # cap prompt size

PROFILE_DDL = """
CREATE TABLE IF NOT EXISTS user_style_profile (
    user_id      TEXT PRIMARY KEY,
    profile      TEXT NOT NULL,
    turns_seen   INTEGER NOT NULL,
    updated_at   REAL NOT NULL
)
"""

EXTRACT_SYS = (
    "You analyze a user's own chat messages to build a WRITING-STYLE profile that "
    "helps an assistant match this person's voice when drafting for them. Extract "
    "ONLY style, tone, formatting habits, and recurring task/intent patterns. "
    "Output STRICT JSON with keys: voice (1-2 sentences on tone/register), "
    "formatting (bullets vs prose, typical length, structure habits), vocabulary "
    "(notable word/phrase tendencies), intents (recurring kinds of requests). "
    "ABSOLUTE RULE: include NO FACTS or CLAIMS about the person (no job, skills, "
    "projects, biography, numbers, employers). Only HOW they write, never WHAT is "
    "true about them. If a detail is a fact rather than a style trait, omit it."
)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _user_messages_from_chat(blob: dict) -> list[str]:
    """Pull the USER's own message texts from an OWUI chat JSON blob."""
    msgs = blob.get("messages")
    if isinstance(msgs, dict):
        msgs = list(msgs.values())
    if not isinstance(msgs, list):
        hist = blob.get("history") or {}
        hm = hist.get("messages") if isinstance(hist, dict) else None
        msgs = list(hm.values()) if isinstance(hm, dict) else []
    out = []
    for m in msgs or []:
        if isinstance(m, dict) and m.get("role") == "user":
            t = _text_of(m.get("content")).strip()
            if t:
                out.append(t)
    return out


def _llm(messages: list[dict], budget: int = 700) -> str:
    body = {"model": CONSOLIDATION_MODEL, "messages": messages,
            "max_tokens": budget, "temperature": 0.0}
    req = urllib.request.Request(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "owui-style-consolidator/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return (data["choices"][0]["message"].get("content") or "").strip()


def consolidate() -> dict:
    if not os.path.exists(DB_PATH):
        return {"status": "no-db", "path": DB_PATH}
    if not FIREWORKS_API_KEY:
        return {"status": "no-key", "reason": "FIREWORKS_API_KEY not set"}

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(PROFILE_DDL)
    con.commit()

    # Gather each user's own messages across all their chats.
    by_user: dict[str, list[str]] = {}
    for r in con.execute("SELECT user_id, chat FROM chat WHERE chat IS NOT NULL"):
        uid = r["user_id"]
        if not uid:
            continue
        try:
            blob = json.loads(r["chat"])
        except Exception:
            continue
        msgs = _user_messages_from_chat(blob)
        if msgs:
            by_user.setdefault(uid, []).extend(msgs)

    updated, skipped = 0, 0
    for uid, turns in by_user.items():
        if len(turns) < MIN_TURNS:
            skipped += 1
            continue
        sample = "\n\n---\n\n".join(turns[-MAX_TURNS:])[:MAX_CHARS]
        try:
            profile = _llm([
                {"role": "system", "content": EXTRACT_SYS},
                {"role": "user", "content": f"USER MESSAGES:\n{sample}\n\nStyle profile JSON:"},
            ])
        except Exception as e:
            print(f"  user {uid[:8]}: extraction failed: {e}")
            continue
        con.execute(
            "INSERT INTO user_style_profile (user_id, profile, turns_seen, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
            "profile=excluded.profile, turns_seen=excluded.turns_seen, "
            "updated_at=excluded.updated_at",
            (uid, profile, len(turns), time.time()),
        )
        updated += 1
        print(f"  user {uid[:8]}: profile updated ({len(turns)} turns)")
    con.commit()
    con.close()
    return {"status": "ok", "users_updated": updated,
            "users_skipped_too_few": skipped, "users_total": len(by_user)}


if __name__ == "__main__":
    print(json.dumps(consolidate(), indent=2))
