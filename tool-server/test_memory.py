"""Offline unit tests for the chat-memory module.

Temp SQLite DB + mocked embedding/summarize, so these assert memory behavior
(store, recall gate, compression, the embed-failure guard) without network.

Run from the repo root or tool-server/:
  python3 tool-server/test_memory.py
"""
import asyncio
import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["CHAT_MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_mem.db")
os.environ.setdefault("FIREWORKS_API_KEY", "test-key")
os.environ["CHAT_MEMORY_COMPRESS_WHEN_OVER"] = "6"
os.environ["CHAT_MEMORY_COMPRESS_CHUNK"] = "5"

import memory  # noqa: E402


def _vec(text):
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


async def _embed(text):
    return _vec(text)


async def _summarize(turns):
    return "Summary of: " + " | ".join(c for _, c in turns)


memory.get_embedding = _embed
memory.summarize_turns = _summarize


async def _fresh_db():
    if memory._memory_conn is not None:
        memory._memory_conn.close()
    memory._memory_conn = None
    memory._memory_disabled = False
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(memory.DB_PATH + suffix)
        except OSError:
            pass


async def main():
    fails = []

    def check(name, cond):
        print(f"{'PASS' if cond else 'FAIL'}: {name}")
        if not cond:
            fails.append(name)

    # Store + recall round-trip.
    await _fresh_db()
    await memory.store("c1", "user", "My project codename is Helios and we launch March 3rd.")
    await memory.store("c1", "assistant", "Got it — Helios launches March 3rd.")
    out = await memory.recall("c1", "what is my project codename?", top_k=5)
    check("store+recall: stored turn is recalled", any("Helios" in c for _, c in out))

    # Recall gate: an empty chat returns nothing.
    await _fresh_db()
    check("recall gate: empty chat returns nothing", await memory.recall("empty", "anything") == [])

    # Duplicate content is stored once.
    await _fresh_db()
    await memory.store("c4", "user", "A unique sentence about quantum widgets.")
    again = await memory.store("c4", "user", "A unique sentence about quantum widgets.")
    check("store: duplicate content not stored twice", again is False)

    # Compression compacts raw rows into a summary that is still recallable
    # (regression for the MIN_TURNS-counts-only-raw-rows bug).
    await _fresh_db()
    for i in range(8):
        await memory.store("c2", "user", f"Fact {i}: the codename is Helios, detail {i}.")
    compacted = await memory.maybe_compress_chat("c2")
    check("compression: compacted raw rows into a summary", compacted >= 5)
    out = await memory.recall("c2", "codename Helios", top_k=5)
    check("recall after compression: the summary is recallable", any("Helios" in c for _, c in out))

    # Embed-failure guard: a failed summary embed must NOT delete the raw rows.
    await _fresh_db()
    for i in range(8):
        await memory.store("c3", "user", f"Row {i}: content about widgets and gears {i}.")
    memory.get_embedding = lambda text: _empty()
    compacted = await memory.maybe_compress_chat("c3")
    memory.get_embedding = _embed
    check("compression guard: no compaction when embed fails", compacted == 0)
    conn = await memory.get_conn()
    raw = conn.execute("SELECT COUNT(*) FROM chat_turns WHERE chat_id='c3' AND is_summary=0").fetchone()[0]
    check("compression guard: raw rows preserved on embed failure", raw == 8)

    # Deliverable store: latest is returned, and edits append as new versions.
    await _fresh_db()
    check("deliverable: none stored yet -> None", await memory.get_deliverable("d1") is None)
    v1 = await memory.store_deliverable("d1", "Dear Committee, version one.", "letter.docx", "docx")
    check("deliverable: first store is version 1", v1 == 1)
    v2 = await memory.store_deliverable("d1", "Dear Committee, version two (edited).", "letter.docx", "docx")
    check("deliverable: an edit appends as version 2", v2 == 2)
    got = await memory.get_deliverable("d1")
    check("deliverable: get returns the LATEST version", got and got["version"] == 2 and "version two" in got["content"])
    check("deliverable: metadata round-trips", got["filename"] == "letter.docx" and got["fmt"] == "docx")
    check("deliverable: empty content is not stored", await memory.store_deliverable("d1", "   ") == 0)
    check("deliverable: a different chat is isolated", await memory.get_deliverable("d2") is None)

    # Version history is bounded per chat (no unbounded growth on edit-heavy chats).
    await _fresh_db()
    for i in range(memory.DELIVERABLES_MAX_PER_CHAT + 5):
        await memory.store_deliverable("dcap", f"version {i}")
    conn = await memory.get_conn()
    cnt = conn.execute("SELECT COUNT(*) FROM deliverables WHERE chat_id=?", ("dcap",)).fetchone()[0]
    check("deliverable: history is capped per chat", cnt == memory.DELIVERABLES_MAX_PER_CHAT)
    check("deliverable: the latest version survives the cap",
          (await memory.get_deliverable("dcap"))["version"] == memory.DELIVERABLES_MAX_PER_CHAT + 5)

    # Pending plan (chunked-writer outline) round-trips via kv, isolated per chat, cleared on demand.
    await _fresh_db()
    check("plan: none pending -> None", await memory.get_plan("p1") is None)
    sample_plan = {"title": "Sleep & Memory", "sections": [{"heading": "Intro", "intent": "x"}],
                   "source": "study", "fmt": "docx"}
    check("plan: store ok", await memory.store_plan("p1", sample_plan) is True)
    got_plan = await memory.get_plan("p1")
    check("plan: round-trips (title + sections preserved)",
          got_plan and got_plan["title"] == "Sleep & Memory" and got_plan["sections"][0]["heading"] == "Intro")
    check("plan: a different chat is isolated", await memory.get_plan("p2") is None)
    newer = {"title": "Sleep & Memory v2", "sections": []}
    await memory.store_plan("p1", newer)
    check("plan: re-store overwrites (one pending plan per chat)",
          (await memory.get_plan("p1"))["title"] == "Sleep & Memory v2")
    await memory.clear_plan("p1")
    check("plan: cleared -> None again", await memory.get_plan("p1") is None)

    # Usage sweep: direct-OWUI-chat tokens land in the ledger; PrismAI rows are
    # skipped (already ledgered at call time); re-sweep dedups via source_id.
    await _fresh_db()
    import sqlite3 as _sq, json as _json, time as _time
    owui = os.path.join(tempfile.mkdtemp(), "webui.db")
    src = _sq.connect(owui)
    src.execute("CREATE TABLE chat (id TEXT, user_id TEXT, chat TEXT, updated_at REAL)")
    src.execute("CREATE TABLE user (id TEXT, name TEXT, email TEXT)")
    src.execute("INSERT INTO user VALUES ('u-1', 'Rajarshi', 'r@x.com')")
    msgs = {"m1": {"role": "assistant", "model": "accounts/fireworks/models/glm-5p1",
                   "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
            "m2": {"role": "assistant", "model": "PrismAI", "usage": {"prompt_tokens": 9, "completion_tokens": 9}}}
    src.execute("INSERT INTO chat VALUES (?, ?, ?, ?)",
                ("c-x", "u-1", _json.dumps({"history": {"messages": msgs}}), _time.time()))
    src.commit(); src.close()
    memory.OWUI_DB_PATH = owui
    added = await memory.sweep_owui_usage()
    check("sweep: direct-chat usage row added (PrismAI skipped)", added == 1)
    check("sweep: re-sweep dedups", await memory.sweep_owui_usage() == 0)
    conn = await memory.get_conn()
    row = conn.execute("SELECT model, in_tok, out_tok, label, user_id FROM usage WHERE source_id IS NOT NULL").fetchone()
    check("sweep: tokens + label + owner recorded", row == ("glm-5p1", 100, 50, "owui-chat", "u-1"))

    # Per-user, per-row-priced summary: two users, two models, $ correct per bucket.
    await _fresh_db()
    memory.OWUI_DB_PATH = "/nonexistent"  # no name map -> raw ids; isolate this case
    await memory.log_usage("deepseek-v4-pro", "agent", 1_000_000, 0, user_id="u-alice")
    await memory.log_usage("gpt-5.5", "polish", 0, 1_000_000, user_id="u-bob")
    cost = lambda m, i, o: (i * (1.0 if "deepseek" in m else 2.0) + o * (0.0 if "deepseek" in m else 10.0)) / 1_000_000
    s = await memory.usage_summary(cost_fn=cost)  # all time
    check("summary: per-user buckets present", set(s["by_user"]) == {"u-alice", "u-bob"})
    check("summary: per-user $ priced per row", round(s["by_user"]["u-alice"]["usd"], 2) == 1.0
          and round(s["by_user"]["u-bob"]["usd"], 2) == 10.0)
    check("summary: total $ sums all rows", s["total_usd"] == 11.0)
    check("summary: by_month bucket present", len(s["by_month"]) >= 1)
    # since/until window filters rows out
    empty = await memory.usage_summary(cost_fn=cost, since=1.0, until=2.0)
    check("summary: out-of-window range is empty", empty["calls"] == 0 and empty["total_usd"] == 0)

    # ── llm provider chain: reasoning params per provider (max policy) ──
    import llm  # noqa: E402
    llm.DEEPSEEK_API_KEY = "dskey"
    llm.ENABLE_DEEPSEEK_DIRECT = True
    _cap = []

    class _Resp:
        def __init__(self, status, body):
            self.status_code, self._body = status, body
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")
        def json(self):
            return self._body

    def _make_client(post_fn):
        class _C:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, headers=None, json=None):
                _cap.append((url, json))
                return post_fn(url, json)
        return _C

    _real = llm.httpx.AsyncClient
    try:
        # DeepSeek-direct success: substantive -> thinking enabled + max, name prefix stripped.
        llm.httpx.AsyncClient = _make_client(
            lambda u, j: _Resp(200, {"choices": [{"message": {"content": "ok"}}]}))
        _cap.clear()
        await llm.chat("accounts/fireworks/models/deepseek-v4-pro",
                       [{"role": "user", "content": "x"}], reasoning_effort="max")
        _p = _cap[0][1]
        check("llm: DeepSeek-direct max -> thinking enabled + max, name stripped",
              "deepseek.com" in _cap[0][0] and _p["model"] == "deepseek-v4-pro"
              and _p.get("reasoning_effort") == "max" and _p.get("thinking") == {"type": "enabled"})
        _cap.clear()
        await llm.chat("accounts/fireworks/models/deepseek-v4-flash",
                       [{"role": "user", "content": "x"}], reasoning_effort="none")
        _p = _cap[0][1]
        check("llm: DeepSeek-direct classifier -> thinking disabled, no reasoning_effort",
              _p.get("thinking") == {"type": "disabled"} and "reasoning_effort" not in _p)

        # DeepSeek-direct down -> Fireworks fallback, flash max translated to "high".
        llm.httpx.AsyncClient = _make_client(
            lambda u, j: _Resp(503, {}) if "deepseek.com" in u
            else _Resp(200, {"choices": [{"message": {"content": "fw"}}]}))
        _cap.clear()
        out = await llm.chat("accounts/fireworks/models/deepseek-v4-flash",
                             [{"role": "user", "content": "x"}], reasoning_effort="max")
        _fw = [j for (u, j) in _cap if "fireworks.ai" in u][0]
        check("llm: Fireworks flash fallback max -> high",
              out == "fw" and _fw.get("reasoning_effort") == "high")
    finally:
        llm.httpx.AsyncClient = _real

    print("\n" + ("all memory tests passed" if not fails else f"{len(fails)} FAILED: {fails}"))
    return 1 if fails else 0


async def _empty():
    return []


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
