#!/usr/bin/env python3
"""T0 of pace-task-loop-v1: per-step planner latency on a multi-step task.

Replays an ordered fixture sequence (a task trace — each step's prompt
carries prior progress, see evals/task-latency-download-v1/) against a
running tinygpt serve and reports per-step latency + correctness, then
extrapolates to the full quaternius scenario (71 packs).

Latency here is the DECISION cost only (planner round-trip). AX snapshot +
dispatch (~0.1-0.2s/step) get added as a constant in the extrapolation; the
true end-to-end number comes from T2's live run.

Usage:
  tinygpt serve <baked-hf> --quantize int8 --grammar <schema> --port 8765 &
  python3 scripts/bench_task_latency.py \
    --fixtures-dir evals/task-latency-download-v1 \
    --serve-url http://127.0.0.1:8765/v1/chat/completions \
    --sys-prompt grammars/pace-system-prompt-v11.txt \
    --runs 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from fake_pace import parse_fixture, score_response

DISPATCH_OVERHEAD_S = 0.15  # AX snapshot + AX.press dispatch, measured-ish constant
QUATERNIUS_STEPS = 85       # 71 Drive packs + navigation steps


def format_user(fx: dict) -> str:
    parts: list[str] = []
    if fx["elements"]:
        parts.append("on-screen elements:")
        for el in fx["elements"]:
            parts.append(f"[{el['id']}] {el['role']}|{el['pos']}|{el['label']}|{el['text']}")
        parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


def query(url: str, sys_prompt: str, fx: dict, timeout: int = 120) -> tuple[str, float]:
    body = {
        "model": "tinygpt",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": format_user(fx)},
        ],
        "temperature": 0.0, "max_tokens": 250, "stream": False,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    dt = time.time() - t0
    return json.loads(raw)["choices"][0]["message"]["content"], dt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures-dir", type=Path, required=True)
    p.add_argument("--serve-url", required=True)
    p.add_argument("--sys-prompt", type=Path, required=True)
    p.add_argument("--runs", type=int, default=3, help="timed repeats per step (after 1 warmup)")
    p.add_argument("--json", dest="json_out", action="store_true")
    args = p.parse_args()

    sys_prompt = args.sys_prompt.read_text().strip()
    steps = sorted(args.fixtures_dir.glob("step*.txt"))
    if not steps:
        print(f"no step*.txt in {args.fixtures_dir}", file=sys.stderr)
        return 2

    print(f"=== task-latency bench: {args.fixtures_dir.name} "
          f"({len(steps)} steps, best of {args.runs}) ===\n")
    print(f"{'step':<10} | {'decision ms':>11} | {'correct':>7} | response")
    print("-" * 78)

    results = []
    total_best = 0.0
    all_correct = True
    for fp in steps:
        fx = parse_fixture(fp.read_text())
        query(args.serve_url, sys_prompt, fx)  # warmup (prompt-cache, lazy init)
        times, last_resp = [], ""
        for _ in range(args.runs):
            last_resp, dt = query(args.serve_url, sys_prompt, fx)
            times.append(dt)
        ok, fails = score_response(last_resp, fx["expects"], fx["elements"], fx["free_text"])
        best = min(times)
        total_best += best
        all_correct &= ok
        results.append({"step": fp.stem, "best_s": best,
                        "median_s": statistics.median(times),
                        "correct": ok, "fails": fails})
        print(f"{fp.stem:<10} | {best * 1000:>11.0f} | {'PASS' if ok else 'FAIL':>7} | "
              f"{last_resp[:60]!r}")
        if not ok:
            for f in fails:
                print(f"{'':<10}   ✗ {f}")

    per_step = total_best / len(steps)
    extrapolated = QUATERNIUS_STEPS * (per_step + DISPATCH_OVERHEAD_S)
    print()
    print(f"per-step decision (best): {per_step * 1000:.0f} ms")
    print(f"per-step incl. dispatch overhead (+{DISPATCH_OVERHEAD_S * 1000:.0f} ms): "
          f"{(per_step + DISPATCH_OVERHEAD_S) * 1000:.0f} ms")
    print(f"extrapolated quaternius scenario ({QUATERNIUS_STEPS} steps): "
          f"{extrapolated / 60:.1f} min of agent time (download time excluded)")
    print(f"correctness: {'ALL PASS' if all_correct else 'FAILURES — latency numbers void, fix accuracy first'}")

    if args.json_out:
        print(json.dumps({"steps": results, "per_step_best_s": per_step,
                          "extrapolated_task_s": extrapolated,
                          "all_correct": all_correct}, indent=1))
    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
