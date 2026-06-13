#!/usr/bin/env python3
"""Compare a planner eval run against the stored champion + floor.

Usage:
  eval_planner_report.py <run-tag> [--baseline evals/planner-champion.json]
                         [--candidate-b <run-tag-b>]

With --candidate-b, renders an A/B view: floor / champion / candidate-A /
candidate-B, plus an A-vs-B delta per suite and a failure-pattern diff so
patterns that shrink under B get a ↓ marker and patterns new in B get a ⚠.

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


def load_run(tag: str):
    run_dir = Path.home() / ".cache" / "tinygpt" / "runs" / f"h2-combined-{tag}"
    score = {}
    patterns = {}
    for suite in SUITES:
        f = run_dir / f"{suite}.json"
        if not f.exists():
            sys.exit(f"missing result: {f} — did the eval finish?")
        d = json.loads(f.read_text())
        score[suite] = (d["passed"], d["total"])
        patterns[suite] = d.get("failure_patterns", [])
    return score, patterns


def cell(rec, suite):
    return f"{rec[suite][0]:>3}/{rec[suite][1]:<3}({pct(*rec[suite]):3.0f}%)"


def print_table(rows, name_w):
    header = f"{'model':<{name_w}}" + "".join(f"{s:>14}" for s in SUITES)
    print(header)
    print("-" * len(header))
    for label, name, rec in rows:
        cells = "".join(cell(rec, s) for s in SUITES)
        print(f"{name:<{name_w}}{cells}  [{label}]")


def print_patterns(label, patterns):
    if not any(patterns.values()):
        return
    print(f"--- {label} failure patterns (top 3 per suite) ---")
    for suite in SUITES:
        for p in patterns[suite][:3]:
            print(f"  {suite:<12} {p['count']:>3}×  {p['pattern']}"
                  f"  (e.g. {p['fixtures'][0]})")
        rest = patterns[suite][3:]
        if rest:
            print(f"  {suite:<12}      … and {sum(p['count'] for p in rest)}"
                  f" more across {len(rest)} patterns")


def print_pattern_diff(pat_a, pat_b):
    """↓ = pattern shrank under B (fewer failures of that kind).
       ⚠ = pattern appeared only in B (new failure mode introduced by B).
       Patterns identical or only in A are silent — the rule of thumb is
       "tiering helped if ↓ outnumbers ⚠ in the OOS / ambig suites".
    """
    if not any(pat_a.values()) and not any(pat_b.values()):
        return
    print("--- A/B failure-pattern diff (↓ = B shrank vs A; ⚠ = new in B) ---")
    for suite in SUITES:
        a_counts = {p["pattern"]: p["count"] for p in pat_a[suite]}
        b_counts = {p["pattern"]: p["count"] for p in pat_b[suite]}
        for pat, b in sorted(b_counts.items(), key=lambda kv: -kv[1]):
            a = a_counts.get(pat, 0)
            if a == 0:
                print(f"  {suite:<12} ⚠   0→{b}  {pat}")
            elif b < a:
                print(f"  {suite:<12} ↓ {a:>3}→{b:<3}  {pat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag")
    ap.add_argument("--baseline",
                    default=str(REPO / "evals" / "planner-champion.json"))
    ap.add_argument("--candidate-b", default=None,
                    help="optional second run-tag for A/B comparison; "
                         "tag B is shown alongside A with an A-vs-B delta")
    args = ap.parse_args()

    base = json.loads(Path(args.baseline).read_text())
    champ, floor = base["champion"], base["floor"]
    candidate, patterns = load_run(args.run_tag)
    if args.candidate_b:
        candidate_b, patterns_b = load_run(args.candidate_b)
    else:
        candidate_b, patterns_b = None, None

    names = [args.run_tag, champ["model"], floor["model"]]
    if args.candidate_b:
        names.append(args.candidate_b)
    name_w = max(len(n) for n in names + ["model"]) + 2

    title = f"\n=== eval-planner: {args.run_tag}"
    if args.candidate_b:
        title += f" vs {args.candidate_b} (A/B)"
    title += f" vs champion ===\n"
    print(title)

    rows = [
        ("floor", floor["model"], floor),
        ("champion", champ["model"], champ),
        ("candidate-A" if args.candidate_b else "candidate",
         args.run_tag, candidate),
    ]
    if args.candidate_b:
        rows.append(("candidate-B", args.candidate_b, candidate_b))
    print_table(rows, name_w)

    print()
    wins = []
    for suite in SUITES:
        delta_champ = pct(*candidate[suite]) - pct(*champ[suite])
        mark = "+" if delta_champ > 0 else ""
        verdict = "WIN" if delta_champ > 0 else ("tie" if delta_champ == 0 else "loss")
        if delta_champ >= 0:
            wins.append(suite)
        line = f"  {suite:<12} {mark}{delta_champ:5.1f}pp A vs champion  ({verdict})"
        if args.candidate_b:
            delta_b_champ = pct(*candidate_b[suite]) - pct(*champ[suite])
            delta_ab = pct(*candidate_b[suite]) - pct(*candidate[suite])
            mark_ab = "+" if delta_ab > 0 else ""
            line += (f" | B {('+' if delta_b_champ > 0 else '')}"
                     f"{delta_b_champ:5.1f}pp vs champ"
                     f" | B {mark_ab}{delta_ab:5.1f}pp vs A")
        print(line)

    if args.candidate_b:
        # E9 ship gate: ≥5pp gain on either ambig or oos without ≥3pp
        # regression on action/destructive. Surfaced inline so the reader
        # doesn't have to do arithmetic from the deltas above.
        ambig_oos_gain = max(
            pct(*candidate_b["ambig"]) - pct(*candidate["ambig"]),
            pct(*candidate_b["oos"]) - pct(*candidate["oos"]),
        )
        destr_reg = pct(*candidate["destructive"]) - pct(*candidate_b["destructive"])
        print()
        if ambig_oos_gain >= 5.0 and destr_reg < 3.0:
            print(f"  E9 GATE: PASS — B beats A by {ambig_oos_gain:.1f}pp on "
                  f"unhappy-path dims, destructive regressed {destr_reg:.1f}pp.")
            print("  Action: tiered prompts become the default — bump v11 to "
                  "v11-compact in Pace's Info.plist and eval_planner.sh.")
        else:
            print(f"  E9 GATE: HOLD — best unhappy-dim gain {ambig_oos_gain:.1f}pp "
                  f"(needs ≥5pp), destructive regression {destr_reg:.1f}pp "
                  f"(needs <3pp).")

    print()
    if args.candidate_b:
        print_pattern_diff(patterns, patterns_b)
    else:
        print_patterns("candidate", patterns)

    print()
    if not args.candidate_b:
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
                print(f"VERDICT: champion stands ({champ['model']}). "
                      f"Candidate loses on: {', '.join(losses)}.")
            else:
                print(f"VERDICT: champion stands ({champ['model']}). "
                      f"Candidate ties on every dim — no reason to swap.")
    print()


if __name__ == "__main__":
    main()
