"""PrismAI real-use basket harness skeleton.

Selftest validates case shape with no network/API calls:
  python -m evals.prismai_basket.run --selftest
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


CASES = Path(__file__).with_name("cases.jsonl")
KINDS = {
    "research_with_sources", "psychology_writeup", "resume", "cover_letter",
    "email_polish", "normal_chat", "honesty_trap", "image_table", "edit_existing_document",
}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    print(json.dumps({
        "status": "skeleton",
        "cases": len(load_cases()),
        "started_at": time.time(),
        "note": "Live runner intentionally not implemented yet; fill cases/checkers first.",
    }, indent=2))


if __name__ == "__main__":
    main()
