"""Join a blind judge's verdicts with the private A/B key and report results.

Usage: python -m orchestrator.ab_score <judge_verdict.json> <label>
The verdict file maps id -> {a:{4 axes}, b:{4 axes}, winner, why}.
ab_key.json maps id -> {"A":"agent"|"claude","B":...}.
"""
import json
import sys
import statistics as st

AXES = ("intent_match", "prose", "grounding", "honesty")


def main():
    verdict_path = sys.argv[1] if len(sys.argv) > 1 else "orchestrator/ab_claude_judge.json"
    label = sys.argv[2] if len(sys.argv) > 2 else "judge"
    verdicts = json.load(open(verdict_path))
    key = json.load(open("orchestrator/ab_key.json"))
    cases = {c["id"]: c for c in json.load(open("orchestrator/ab_cases.json"))}

    tally = {"agent": 0, "claude": 0, "tie": 0}
    axis = {"agent": {a: [] for a in AXES}, "claude": {a: [] for a in AXES}}
    cat_tally = {}
    rows = []

    for cid, v in verdicts.items():
        mp = key.get(cid)
        if not mp:
            continue
        # which letter is which system
        a_sys = mp["A"]  # 'agent' or 'claude'
        b_sys = mp["B"]
        a_scores = v.get("a", {})
        b_scores = v.get("b", {})
        sys_scores = {a_sys: a_scores, b_sys: b_scores}
        for sysname in ("agent", "claude"):
            sc = sys_scores.get(sysname, {})
            for ax in AXES:
                if isinstance(sc.get(ax), (int, float)):
                    axis[sysname][ax].append(sc[ax])
        w = v.get("winner")
        who = mp.get(w, "tie") if w in ("A", "B") else "tie"
        tally[who] = tally.get(who, 0) + 1
        cat = cases.get(cid, {}).get("category", "?")
        cat_tally.setdefault(cat, {"agent": 0, "claude": 0, "tie": 0})
        cat_tally[cat][who] += 1
        a_tot = sum(a_scores.get(ax, 0) for ax in AXES)
        b_tot = sum(b_scores.get(ax, 0) for ax in AXES)
        ag_tot = a_tot if a_sys == "agent" else b_tot
        cl_tot = a_tot if a_sys == "claude" else b_tot
        rows.append((cid, cat, who, ag_tot, cl_tot, v.get("why", "")))

    print(f"\n===== {label} (blind) =====")
    print(f"{'case':22s} {'cat':12s} {'winner':7s} {'agent/40':8s} {'claude/40':9s}")
    print("-" * 80)
    for cid, cat, who, ag, cl, why in rows:
        print(f"{cid:22s} {cat:12s} {who:7s} {ag:<8d} {cl:<9d}")
    print("-" * 80)
    print(f"HEAD-TO-HEAD  agent={tally['agent']}  claude={tally['claude']}  tie={tally['tie']}")
    print("\nby category:")
    for cat, t in sorted(cat_tally.items()):
        print(f"  {cat:14s} agent={t['agent']} claude={t['claude']} tie={t['tie']}")
    print("\naxis averages (1-10):")
    print(f"  {'axis':14s} {'agent':6s} {'claude':6s}")
    for ax in AXES:
        a = st.mean(axis['agent'][ax]) if axis['agent'][ax] else 0
        c = st.mean(axis['claude'][ax]) if axis['claude'][ax] else 0
        print(f"  {ax:14s} {a:<6.2f} {c:<6.2f}")
    ag_overall = st.mean([x for ax in AXES for x in axis['agent'][ax]]) if any(axis['agent'].values()) else 0
    cl_overall = st.mean([x for ax in AXES for x in axis['claude'][ax]]) if any(axis['claude'].values()) else 0
    print(f"  {'OVERALL':14s} {ag_overall:<6.2f} {cl_overall:<6.2f}")


if __name__ == "__main__":
    main()
