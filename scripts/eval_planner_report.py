#!/usr/bin/env python3
"""Compare a planner eval run against the stored champion + floor.

Usage: eval_planner_report.py <run-tag> [--baseline evals/planner-champion.json]
Reads ~/.cache/tinygpt/runs/h2-combined-<run-tag>/{ambig,oos,destructive}.json
"""
import argparse
import json
import sys
from pathlib import Path

SUITES = ["ambig", "oos", "destructive"]
REPO = Path(__file__).resolve().parent.parent


def pct(passed, total):
    return 100.0 * passed / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag")
    ap.add_argument("--baseline", default=str(REPO / "evals" / "planner-champion.json"))
    args = ap.parse_args()

    base = json.loads(Path(args.baseline).read_text())
    run_dir = Path.home() / ".cache" / "tinygpt" / "runs" / f"h2-combined-{args.run_tag}"

    candidate = {}
    patterns = {}
    for suite in SUITES:
        f = run_dir / f"{suite}.json"
        if not f.exists():
            sys.exit(f"missing result: {f} — did the eval finish?")
        d = json.loads(f.read_text())
        candidate[suite] = (d["passed"], d["total"])
        patterns[suite] = d.get("failure_patterns", [])

    champ, floor = base["champion"], base["floor"]
    name_w = max(len(args.run_tag), len(champ["model"]), len(floor["model"]), 5) + 2

    print(f"\n=== eval-planner: {args.run_tag} vs champion ===\n")
    header = f"{'model':<{name_w}}" + "".join(f"{s:>14}" for s in SUITES)
    print(header)
    print("-" * len(header))
    for label, rec in (("floor", floor), ("champion", champ)):
        cells = "".join(
            f"{rec[s][0]:>3}/{rec[s][1]:<3}({pct(*rec[s]):3.0f}%)" for s in SUITES
        )
        print(f"{rec['model']:<{name_w}}{cells}  [{label}]")
    cells = "".join(
        f"{candidate[s][0]:>3}/{candidate[s][1]:<3}({pct(*candidate[s]):3.0f}%)" for s in SUITES
    )
    print(f"{args.run_tag:<{name_w}}{cells}  [candidate]")

    print()
    wins = []
    for suite in SUITES:
        delta = pct(*candidate[suite]) - pct(*champ[suite])
        mark = "+" if delta > 0 else ""
        verdict = "WIN" if delta > 0 else ("tie" if delta == 0 else "loss")
        if delta >= 0:
            wins.append(suite)
        print(f"  {suite:<12} {mark}{delta:5.1f}pp vs champion  ({verdict})")

    # Triage: don't make the reader mine 130 raw rows — lead with the
    # grouped failure modes (written by eval_pace_unhappy.py since 2026-06-12;
    # absent on older runs).
    if any(patterns.values()):
        print("--- candidate failure patterns (top 3 per suite) ---")
        for suite in SUITES:
            for p in patterns[suite][:3]:
                print(f"  {suite:<12} {p['count']:>3}×  {p['pattern']}"
                      f"  (e.g. {p['fixtures'][0]})")
            rest = patterns[suite][3:]
            if rest:
                print(f"  {suite:<12}      … and {sum(p['count'] for p in rest)}"
                      f" more across {len(rest)} patterns")

    print()
    if len(wins) == len(SUITES) and any(
        pct(*candidate[s]) > pct(*champ[s]) for s in SUITES
    ):
        print("VERDICT: NEW CHAMPION CANDIDATE — beats or ties the champion on all dims.")
        print("  Next: update evals/planner-champion.json and Pace's")
        print("  Info.plist:LocalPlannerModelIdentifier (see docs/DRILLDOWN.md precedent).")
        print("  Caveat: n=130; dims at n=40 carry ~±15pp CI — re-run before swapping.")
    else:
        losses = [s for s in SUITES if s not in wins]
        if losses:
            print(f"VERDICT: champion stands ({champ['model']}). Candidate loses on: {', '.join(losses)}.")
        else:
            print(f"VERDICT: champion stands ({champ['model']}). Candidate ties on every dim — no reason to swap.")
    print()


if __name__ == "__main__":
    main()
