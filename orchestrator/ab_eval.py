"""Claude-vs-agent A/B harness.

Runs the local agentic orchestrator path against Claude CLI output, then asks a
judge model to compare the two on intent, prose, grounding, and verification.

Usage:
  python -m orchestrator.ab_eval
  python -m orchestrator.ab_eval --case-file cases.json --out ab_results.json

The Claude side uses `claude -p <prompt>` by default. Override with CLAUDE_CMD
or --claude-cmd if your CLI wrapper differs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import aiohttp

from . import agent, config, fireworks

DEFAULT_CASES = [
    {
        "name": "cover_letter_source_bound",
        "prompt": (
            "Write a concise cover letter for a data analyst role using only these facts.\n\n"
            "Notes: Rajarshi has mechanical engineering training, taught himself Python, "
            "built an internal RAG email-triage tool at REAL Brokerage, and cares about "
            "verification and useful automation. Do not invent employers, degrees, metrics, "
            "or tools not listed here."
        ),
    },
    {
        "name": "research_grounding",
        "prompt": (
            "Give me a current, source-grounded summary of the latest major development "
            "in open-weight reasoning models. Cite sources and separate verified facts "
            "from uncertainty."
        ),
    },
    {
        "name": "normal_chat",
        "prompt": "Explain why prompt engineering alone is not enough to make a model reliable, in plain language.",
    },
]

JUDGE_SYSTEM = (
    "You are a strict A/B judge for assistant outputs. Compare RESPONSE A and "
    "RESPONSE B against the user's prompt. Score each 1-10 for intent match, "
    "prose quality, factual grounding, and verification honesty. Penalize "
    "fabricated specifics, fake citations, unsupported claims, hedging that "
    "avoids the task, and tool/process chatter. Return JSON only with keys: "
    "winner, scores, reasons, and improvements_for_agent."
)


async def run_agent(prompt: str) -> str:
    events = []
    async with aiohttp.ClientSession() as session:
        async for kind, text in agent.run(
            [{"role": "user", "content": prompt}],
            user_id="",
            session=session,
            request_headers={},
        ):
            if kind == "content":
                events.append(text)
    return "".join(events).strip()


def run_claude(prompt: str, claude_cmd: str, timeout: int) -> str:
    exe = shutil.which(claude_cmd)
    if not exe:
        return f"[SKIPPED: Claude CLI command not found: {claude_cmd}]"
    try:
        proc = subprocess.run(
            [exe, "-p", prompt],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as e:
        return f"[ERROR running Claude CLI: {type(e).__name__}: {e}]"
    if proc.returncode != 0:
        return "[ERROR running Claude CLI]\n" + (proc.stderr or proc.stdout).strip()
    return proc.stdout.strip()


async def judge(prompt: str, agent_text: str, claude_text: str) -> dict:
    items = [
        ("agent", agent_text),
        ("claude", claude_text),
    ]
    random.shuffle(items)
    label_map = {"A": items[0][0], "B": items[1][0]}
    user = {
        "prompt": prompt,
        "response_a": items[0][1],
        "response_b": items[1][1],
    }
    raw = await fireworks.complete(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
        ],
        os.getenv("JUDGE_MODEL", config.GROUNDING_GATE_MODEL),
        max_tokens=1600,
        temperature=0.0,
    )
    try:
        data = json.loads(raw)
    except Exception:
        data = {"raw_judgment": raw}
    data["label_map"] = label_map
    if isinstance(data.get("winner"), str) and data["winner"] in label_map:
        data["winner_name"] = label_map[data["winner"]]
    return data


def load_cases(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_CASES
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("case file must be a JSON list")
    return data


async def main_async(args) -> int:
    if args.prompt:
        cases = [{"name": args.name or "ad_hoc", "prompt": args.prompt}]
    else:
        cases = load_cases(args.case_file)
    results = []
    for case in cases:
        name = case.get("name") or "unnamed"
        prompt = case["prompt"]
        started = time.time()
        agent_text = ""
        claude_text = ""
        judgment = {}
        if args.mode in {"both", "agent"}:
            agent_text = await run_agent(prompt)
        if args.mode in {"both", "claude"}:
            claude_text = run_claude(prompt, args.claude_cmd, args.timeout)
        if args.mode == "both" and not args.skip_judge:
            judgment = await judge(prompt, agent_text, claude_text)
        results.append(
            {
                "name": name,
                "prompt": prompt,
                "agent": agent_text,
                "claude": claude_text,
                "judgment": judgment,
                "seconds": round(time.time() - started, 2),
            }
        )
        if judgment:
            print(f"{name}: winner={judgment.get('winner_name') or judgment.get('winner')}")
        else:
            print(f"{name}: collected {args.mode}")

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", default="")
    parser.add_argument("--out", default="ab_results.json")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--claude-cmd", default=os.getenv("CLAUDE_CMD", "claude"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("AB_TIMEOUT", "180")))
    parser.add_argument("--mode", choices=["both", "agent", "claude"], default="both")
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
