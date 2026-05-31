"""Combined A/B report across judges, joined to the private key.

Reads ab_key.json + ab_cases.json + each available judge verdict file, and
prints per-judge tallies, per-category breakdown, axis averages, and where the
judges DISAGREE (the real signal when one judge is home-field biased).

Judges (use whichever files exist):
  - Claude (blind, home-field biased)        : ab_claude_judge.json
  - Gemini 3 Pro (blind, neutral/external)   : ab_gemini_judge.json
"""
import json
import os
import statistics as st

AXES = ("intent_match", "prose", "grounding", "honesty")
JUDGES = [
    ("Claude (home-field)", "orchestrator/ab_claude_judge.json"),
    ("Gemini-3-Pro (neutral)", "orchestrator/ab_gemini_judge.json"),
]


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _winner_system(v, mp):
    w = v.get("winner")
    return mp.get(w, "tie") if w in ("A", "B") else "tie"


def _sys_axis(v, mp):
    """Return {'agent':{axes}, 'claude':{axes}} from a verdict + mapping."""
    out = {"agent": {}, "claude": {}}
    for letter in ("a", "b"):
        sysname = mp.get(letter.upper())
        sc = v.get(letter, {})
        if sysname in out and isinstance(sc, dict):
            out[sysname] = sc
    return out


def main():
    key = _load("orchestrator/ab_key.json") or {}
    cases = {c["id"]: c for c in (_load("orchestrator/ab_cases.json") or [])}

    judges = [(name, _load(path)) for name, path in JUDGES]
    judges = [(n, d) for n, d in judges if d]
    if not judges:
        print("no judge verdict files found")
        return

    for name, verdicts in judges:
        tally = {"agent": 0, "claude": 0, "tie": 0}
        cat = {}
        axis = {"agent": {a: [] for a in AXES}, "claude": {a: [] for a in AXES}}
        scored = 0
        for cid, v in verdicts.items():
            mp = key.get(cid)
            if not mp or "error" in v:
                continue
            scored += 1
            who = _winner_system(v, mp)
            tally[who] += 1
            c = cases.get(cid, {}).get("category", "?")
            cat.setdefault(c, {"agent": 0, "claude": 0, "tie": 0})[who] += 1
            sa = _sys_axis(v, mp)
            for s in ("agent", "claude"):
                for ax in AXES:
                    if isinstance(sa[s].get(ax), (int, float)):
                        axis[s][ax].append(sa[s][ax])

        print(f"\n===== {name} — {scored} cases =====")
        print(f"head-to-head: agent={tally['agent']}  claude={tally['claude']}  tie={tally['tie']}")
        print("by category: " + ", ".join(
            f"{k}({v['agent']}-{v['claude']}-{v['tie']})" for k, v in sorted(cat.items())))
        ag = [x for ax in AXES for x in axis['agent'][ax]]
        cl = [x for ax in AXES for x in axis['claude'][ax]]
        print("axis avg (agent vs claude): " + ", ".join(
            f"{ax} {st.mean(axis['agent'][ax]):.2f}/{st.mean(axis['claude'][ax]):.2f}"
            for ax in AXES if axis['agent'][ax] and axis['claude'][ax]))
        if ag and cl:
            print(f"OVERALL: agent {st.mean(ag):.2f}  vs  claude {st.mean(cl):.2f}")

    # disagreement table (only if >=2 judges)
    if len(judges) >= 2:
        print("\n===== per-case winners across judges =====")
        names = [n for n, _ in judges]
        print(f"{'case':22s} {'category':12s} " + " ".join(f"{n[:14]:14s}" for n in names))
        print("-" * (36 + 15 * len(names)))
        for cid in key:
            row = []
            for _, v in judges:
                ver = v.get(cid)
                row.append(_winner_system(ver, key[cid]) if ver and "error" not in ver else "-")
            cat = cases.get(cid, {}).get("category", "?")
            flag = "  <-- DISAGREE" if len(set(r for r in row if r != "-")) > 1 else ""
            print(f"{cid:22s} {cat:12s} " + " ".join(f"{r:14s}" for r in row) + flag)


if __name__ == "__main__":
    main()
