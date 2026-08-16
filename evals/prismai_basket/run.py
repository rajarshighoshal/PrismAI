"""PrismAI real-use basket harness.

The benchmark PrismAI optimizes for: the user's actual OpenWebUI distribution, not a
generic leaderboard. This is the Phase-2 bake-off tool — point it at different model
pins and compare pass rates, latency, and cost on the SAME cases.

  python -m evals.prismai_basket.run --selftest            # validate case shape (no API)
  python -m evals.prismai_basket.run --live --limit 2      # smoke the live runner
  python -m evals.prismai_basket.run --live --agent-model accounts/fireworks/models/glm-5p2

Case schema (JSONL):
  id, kind, request           required
  source                      optional grounding; delivered as an OWUI <source> block
                              (the real attachment channel shape)
  checks                      list; entries "includes:<text>" are scored mechanically
                              (case-insensitive substring of the answer); anything else
                              is reported as MANUAL (a human judges it)
  must_not_include            list of literals that must NOT appear (case-insensitive)
  live                        false = scaffolding, skipped by the live runner

Live runs need provider keys in env (FIREWORKS_API_KEY / DEEPSEEK_API_KEY). Cases that
need file export or deliverable memory also need a reachable tool-server
(TOOL_SERVER_URL) — they SKIP (not fail) when it is down. Persistence is off:
chat_id is empty and style memory is disabled, so a bake-off writes nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

# Run as `python -m evals.prismai_basket.run` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASES = Path(__file__).with_name("cases.jsonl")
KINDS = {
    "research_with_sources", "psychology_writeup", "resume", "cover_letter",
    "email_polish", "normal_chat", "honesty_trap", "image_table", "edit_existing_document",
}
# Kinds whose happy path needs the tool-server (file export / deliverable memory).
_TOOL_KINDS = {"cover_letter", "edit_existing_document"}


def load_cases(path=CASES):
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: invalid JSON: {e}") from e
        out.append(c)
    return out


def selftest():
    cases = load_cases()
    failures = []
    ids = set()
    for c in cases:
        cid = c.get("id")
        if not cid or cid in ids:
            failures.append(f"bad/duplicate id: {cid!r}")
        ids.add(cid)
        if c.get("kind") not in KINDS:
            failures.append(f"{cid}: unknown kind {c.get('kind')!r}")
        if not str(c.get("request") or "").strip():
            failures.append(f"{cid}: missing request")
        for field in ("checks", "must_not_include"):
            if field in c and not isinstance(c[field], list):
                failures.append(f"{cid}: {field} must be a list")
    print(f"cases={len(cases)} kinds={sorted({c.get('kind') for c in cases})}")
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("all basket selftests passed")


# ----------------------------------------------------------------------- live runner
def _messages_for(case: dict) -> list[dict]:
    """Deliver the case source the way OWUI delivers an attachment: a <source> block
    inline in the user message — so the run exercises the real parsing/grounding path.
    A source that already carries <source> blocks (multi-source cases) is used as-is."""
    request = case["request"]
    source = str(case.get("source") or "").strip()
    if source:
        if "<source" not in source:
            source = f'<source id="1" name="case-source.txt">{source}</source>'
        request = f"{source}\n\n{request}"
    return [{"role": "user", "content": request}]


async def _toolserver_up(session) -> bool:
    import aiohttp
    from orchestrator import config
    try:
        async with session.get(f"{config.TOOL_SERVER_URL}/health",
                               timeout=aiohttp.ClientTimeout(total=3)) as r:
            return r.status == 200
    except Exception:
        return False


async def run_case(case: dict, *, session, overrides: dict) -> dict:
    """Drive one case through the REAL pipeline (orchestrator.pipeline.run) and score it."""
    from orchestrator import config, perf, pipeline

    # Tap perf.trace for per-case call count + token usage (restored after the case).
    calls = {"n": 0, "in": 0, "out": 0}
    real_trace = perf.trace

    def tap(label, model, *, t0, ttft=None, in_tok=None, out_tok=None):
        calls["n"] += 1
        calls["in"] += int(in_tok or 0)
        calls["out"] += int(out_tok or 0)
        return real_trace(label, model, t0=t0, ttft=ttft, in_tok=in_tok, out_tok=out_tok)

    saved = {k: getattr(config, k) for k in
             ("CHAT_MODEL", "AGENT_MODEL", "GROUNDED_MODEL", "GROUNDING_GATE_MODEL",
              "SHOW_WORK", "ENABLE_STYLE_MEMORY", "ENABLE_INTERACTION_MODE")}
    for k, v in overrides.items():
        setattr(config, k, v)
    config.SHOW_WORK = False              # progress notes are UX polish, not scored behavior
    config.ENABLE_STYLE_MEMORY = False    # no webui.db off the server; no writes
    config.ENABLE_INTERACTION_MODE = False

    perf.trace = tap
    t0 = time.perf_counter()
    parts, links = [], []
    error = ""
    try:
        async for kind, text in pipeline.run(_messages_for(case), user_id="basket",
                                             session=session, request_headers={}):
            if kind == "content":
                parts.append(text)
    except Exception as e:  # a pipeline crash is a case failure, not a harness crash
        error = repr(e)
    finally:
        perf.trace = real_trace
        for k, v in saved.items():
            setattr(config, k, v)
    latency = time.perf_counter() - t0

    answer = "".join(parts)
    low = answer.lower()
    links = re.findall(r"📎 \[[^\]]+\]\(([^)]+)\)", answer)

    fails, manuals, notes = [], [], []
    # A correct refusal/block on a trap may legitimately QUOTE the fabricated claim
    # ("I can't say you led a team of 50 engineers — the source only mentions two
    # interns"). An honesty marker downgrades a must_not_include hit to a noted pass.
    _HONESTY_MARKERS = ("can't", "cannot", "not in the source", "doesn't say", "does not say",
                        "only mentions", "only says", "unsupported", "not supported", "unable to")
    honest = any(m in low for m in _HONESTY_MARKERS)
    for lit in case.get("must_not_include") or []:
        if lit.lower() in low:
            if honest:
                notes.append(f"quoted-in-refusal (ok): {lit!r}")
            else:
                fails.append(f"must_not_include leaked: {lit!r}")
    for chk in case.get("checks") or []:
        if chk.startswith("includes:"):
            if chk[len("includes:"):].lower() not in low:
                fails.append(f"missing: {chk[len('includes:'):]!r}")
        elif chk == "docx export":  # known mechanical shorthand
            if not any(".docx" in u for u in links):
                fails.append("no .docx export link in answer")
        else:
            manuals.append(chk)
    if error:
        fails.append(f"pipeline error: {error}")
    if not answer.strip() and not error:
        fails.append("empty answer")

    return {
        "id": case["id"], "ok": not fails, "fails": fails, "manual": manuals,
        "notes": notes,
        "latency_s": round(latency, 1), "calls": calls["n"],
        "in_tok": calls["in"], "out_tok": calls["out"],
        "chars": len(answer), "links": links,
    }


async def run_live(cases, *, concurrency: int, overrides: dict):
    try:
        import aiohttp
    except ImportError:
        print("aiohttp required for --live (install it, or use --selftest).", file=sys.stderr)
        sys.exit(2)
    live = [c for c in cases if c.get("live")]
    if not live:
        print("no live cases (every case has live:false — flip the ones ready to run)")
        return
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tools_up = await _toolserver_up(session)
        if not tools_up:
            print(f"tool-server unreachable — {_TOOL_KINDS & {c['kind'] for c in live} or 'tool'} cases will SKIP")

        async def one(case):
            if case["kind"] in _TOOL_KINDS and not tools_up:
                return {"id": case["id"], "ok": None, "fails": [], "manual": [],
                        "latency_s": 0, "calls": 0, "in_tok": 0, "out_tok": 0,
                        "chars": 0, "links": [], "skip": "tool-server down"}
            async with sem:
                return await run_case(case, session=session, overrides=overrides)

        print(f"live cases={len(live)}  concurrency={concurrency}  "
              f"overrides={overrides or 'config defaults'}\n")
        header = f"{'case':<22} {'result':<8} {'lat(s)':>7} {'calls':>6} {'in':>7} {'out':>7} {'chars':>7}"
        print(header)
        print("-" * len(header))
        results = await asyncio.gather(*(one(c) for c in live))
        npass = nfail = nskip = 0
        for r in results:
            if r.get("skip"):
                nskip += 1
                print(f"{r['id']:<22} {'SKIP':<8} {'—':>7} {'—':>6} {'—':>7} {'—':>7} {'—':>7}  ({r['skip']})")
                continue
            ok = r["ok"]
            npass += bool(ok)
            nfail += not ok
            print(f"{r['id']:<22} {'PASS' if ok else 'FAIL':<8} {r['latency_s']:>7} "
                  f"{r['calls']:>6} {r['in_tok']:>7} {r['out_tok']:>7} {r['chars']:>7}")
            for f in r["fails"]:
                print(f"    FAIL: {f}")
            for n in r.get("notes") or []:
                print(f"    note: {n}")
            for m in r["manual"]:
                print(f"    MANUAL (judge by eye): {m}")
        print(f"\n{npass} passed, {nfail} failed, {nskip} skipped "
              f"({len(results)} live cases)")


def main():
    ap = argparse.ArgumentParser(description="PrismAI real-use basket")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true", help="run live:true cases through the real pipeline")
    ap.add_argument("--case", default="", help="run a single case id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--chat-model", default="", help="override config.CHAT_MODEL for the run")
    ap.add_argument("--agent-model", default="", help="override config.AGENT_MODEL for the run")
    ap.add_argument("--grounded-model", default="", help="override config.GROUNDED_MODEL for the run")
    ap.add_argument("--gate-model", default="", help="override config.GROUNDING_GATE_MODEL for the run")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c.get("id") == args.case]
        if not cases:
            raise SystemExit(f"no case with id {args.case!r}")
    if args.limit:
        cases = cases[:args.limit]
    overrides = {}
    if args.chat_model:
        overrides["CHAT_MODEL"] = args.chat_model
    if args.agent_model:
        overrides["AGENT_MODEL"] = args.agent_model
    if args.grounded_model:
        overrides["GROUNDED_MODEL"] = args.grounded_model
    if args.gate_model:
        overrides["GROUNDING_GATE_MODEL"] = args.gate_model

    if not args.live:
        print(json.dumps({
            "status": "ready",
            "cases": len(cases),
            "live_cases": sum(1 for c in cases if c.get("live")),
            "note": "pass --live to run (needs provider keys); flip case live flags as you fill them out.",
        }, indent=2))
        return
    asyncio.run(run_live(cases, concurrency=args.concurrency, overrides=overrides))


if __name__ == "__main__":
    main()
