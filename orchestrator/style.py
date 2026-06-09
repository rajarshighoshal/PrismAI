"""Per-user writing-style profiles, read from OWUI's sqlite db (read-only).

Populated by the weekly consolidate_style job (style-memory branch), which
extracts STYLE/INTENT only — never facts. Read-only here. Fails soft: if the
db/table/column/row is missing, returns "" and the deliverable path just writes
without a personalized voice. The read is offloaded to a worker thread so the
blocking sqlite call never stalls the event loop in the per-request hot path.
"""
import asyncio
import sqlite3

from . import config


def _read_profile(user_id: str) -> str:
    try:
        con = sqlite3.connect(
            f"file:{config.STYLE_DB_PATH}?mode=ro", uri=True, timeout=2.0
        )
        try:
            row = con.execute(
                "SELECT profile FROM user_style_profile WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        finally:
            con.close()
        return (row[0] if row and row[0] else "").strip()
    except Exception:
        return ""


async def get_style_profile(user_id: str) -> str:
    if not (config.ENABLE_STYLE_MEMORY and user_id):
        return ""
    return await asyncio.to_thread(_read_profile, user_id)
