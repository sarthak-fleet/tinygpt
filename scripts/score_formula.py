#!/usr/bin/env python3
"""score_formula — measure (speed × accuracy) / cost for a tinygpt model.

The user's North Star (2026-06-09): result = (speed × accuracy) / cost-to-run.
Higher = better. This script computes that score for any model+adapter setup,
emitting a single comparable number plus the per-axis breakdown.

Speed   = harmonic mean of TTFW (ms⁻¹ scaled) and steady tok/s, normalized.
Accuracy = pass rate on the target eval suite (default: fm-fixtures-v2).
Cost    = disk size (GB) + peak RSS during serve (GB), summed in GB.

Usage:

  python3 scripts/score_formula.py \\
    --serve-args "--lora /path/to.lora --grammar /path/to/schema.json" \\
    --model-dir /path/to/baked-hf \\
    --label "v9-LoRA-fp16" \\
    --fixtures /Users/sarthak/Desktop/fleet/pace/evals/fm-fixtures-v2

Then run against another candidate (quantized, distilled, alternate LoRA)
with the same flags and compare formula scores.

The script does NOT start serve — it expects you to have a serve running
at --serve-url. This keeps it composable with whatever process management
you prefer (the README suggests `tinygpt serve` directly).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent


# ----- Speed -----


def time_request(url: str, sys_prompt: str, user: str,
                  max_tokens: int = 80, stream: bool = True) -> dict:
    body = {
        "model": "tinygpt",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    t0 = time.time()
    t_first = None
    n_tokens = 0
    if stream:
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.decode("utf-8").strip()
                if line.startswith("data:") and not line.endswith("[DONE]"):
                    try:
                        d = json.loads(line[5:].strip())
                        delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            if t_first is None:
                                t_first = time.time()
                            n_tokens += 1
                    except json.JSONDecodeError:
                        pass
        t_end = time.time()
        # Tok/s = generated tokens / decode time (after first token)
        decode_s = max(t_end - t_first, 1e-6) if t_first else 0
        return {
            "ttfw_ms": (t_first - t0) * 1000 if t_first else None,
            "total_ms": (t_end - t0) * 1000,
            "n_tokens": n_tokens,
            "tok_per_s": (n_tokens - 1) / decode_s if t_first and n_tokens > 1 else None,
        }
    else:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        return {
            "ttfw_ms": None,
            "total_ms": (time.time() - t0) * 1000,
            "n_tokens": resp.get("usage", {}).get("completion_tokens", 0),
            "tok_per_s": None,
        }


def measure_speed(serve_url: str, sys_prompt: str, runs: int = 5) -> dict:
    """5 calibration requests. Returns median TTFW + tok/s + jitter."""
    USERS = [
        "on-screen elements:\n[0] dock_icon|24,1000|Mail|Inbox\n\nuser said: open my email",
        "on-screen elements:\n[0] dock_icon|24,1000|Safari\n\nuser said: what's html",
        "on-screen elements:\n[0] dock_icon|24,1000|Mail\n\nuser said: send a message to mom",
        "on-screen elements:\n[0] dock_icon|24,1000|Mail|Inbox\n[1] dock_icon|72,1000|Calendar\n\nuser said: schedule a meeting",
        "on-screen elements:\n[0] dock_icon|24,1000|Notes\n\nuser said: write a note about dinner ideas",
    ]
    # Warmup (don't count)
    time_request(serve_url, sys_prompt, USERS[0])

    ttfws = []
    tps = []
    for i in range(runs):
        r = time_request(serve_url, sys_prompt, USERS[i % len(USERS)])
        if r["ttfw_ms"] is not None:
            ttfws.append(r["ttfw_ms"])
        if r["tok_per_s"] is not None:
            tps.append(r["tok_per_s"])

    return {
        "ttfw_ms_median": statistics.median(ttfws) if ttfws else None,
        "ttfw_ms_min":    min(ttfws) if ttfws else None,
        "ttfw_ms_max":    max(ttfws) if ttfws else None,
        "tok_per_s_median": statistics.median(tps) if tps else None,
        "n_calibration_runs": runs,
    }


# ----- Accuracy -----


def measure_accuracy(fixtures_dir: Path, serve_url: str,
                     sys_prompt_path: Path | None = None) -> dict:
    """Reuse eval_pace_v2 — run it as a subprocess and parse the score."""
    cmd = [
        sys.executable, str(HERE / "eval_pace_v2.py"),
        "--fixtures-dir", str(fixtures_dir),
        "--serve-url", serve_url,
    ]
    if sys_prompt_path:
        cmd += ["--sys-prompt", str(sys_prompt_path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    text = out.stdout
    # Parse "Model score:        X/Y  (Z%)" line
    pass_n = total_n = None
    for line in text.splitlines():
        if "Model score:" in line:
            parts = line.split()
            # e.g. ['Model', 'score:', '5/15', '(33.3%)']
            for p in parts:
                if "/" in p and p.replace("/", "").isdigit():
                    pass_n, total_n = (int(x) for x in p.split("/"))
                    break
            break
    return {
        "fixtures_dir": fixtures_dir.name,
        "pass_n": pass_n,
        "total_n": total_n,
        "pass_rate": pass_n / total_n if pass_n is not None and total_n else 0.0,
        "raw_tail": "\n".join(text.splitlines()[-12:]),
    }


# ----- Cost -----


def measure_cost(model_dir: Path, serve_pid: int | None = None) -> dict:
    """Disk size of model dir + peak RSS of serve process if PID provided."""
    disk_bytes = 0
    if model_dir.exists():
        for root, _, files in os.walk(model_dir):
            for f in files:
                disk_bytes += os.path.getsize(os.path.join(root, f))

    rss_gb = None
    if serve_pid:
        # macOS `ps -o rss` gives KB
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(serve_pid)],
                                  capture_output=True, text=True, timeout=5)
            rss_kb = int(out.stdout.strip() or 0)
            rss_gb = rss_kb / (1024 * 1024)
        except (ValueError, subprocess.TimeoutExpired):
            pass

    return {
        "model_disk_gb": disk_bytes / (1024**3),
        "serve_rss_gb": rss_gb,
        "cost_gb": (disk_bytes / (1024**3)) + (rss_gb or 0),
    }


# ----- Formula -----


def compute_formula(speed: dict, accuracy: dict, cost: dict) -> dict:
    """result = (speed × accuracy) / cost. Normalized so a baseline of
    (TTFW=120ms, tok/s=50, accuracy=33%, cost=2GB) → score = 1.0.
    """
    # speed term: combine TTFW (ms, lower-is-better) and tok/s (higher-is-better)
    # Normalized: faster TTFW gives higher score, faster decode gives higher score.
    ttfw_ms = speed.get("ttfw_ms_median") or 1000.0
    tok_s = speed.get("tok_per_s_median") or 1.0
    # Baseline: TTFW=120ms, tok/s=50. Score components both normalized to 1.0 at baseline.
    speed_score = (120.0 / ttfw_ms) * (tok_s / 50.0)

    # accuracy term: pass-rate fraction
    acc_score = accuracy.get("pass_rate", 0.0)

    # cost term: 1/(cost in GB). Lower cost = higher score.
    cost_gb = cost.get("cost_gb", 1.0) or 1.0
    cost_score = 2.0 / cost_gb  # baseline 2GB → 1.0

    formula = (speed_score * acc_score) * cost_score

    return {
        "speed_score":    speed_score,
        "accuracy_score": acc_score,
        "cost_score":     cost_score,
        "formula":        formula,
        "note": "speed=baseline(120ms,50tok/s) gives 1.0; accuracy=pass_rate; cost=2GB baseline gives 1.0",
    }


# ----- CLI -----


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="human-readable label for this measurement")
    p.add_argument("--model-dir", type=Path, required=True,
                     help="path to the model dir on disk (for cost measurement)")
    p.add_argument("--serve-url", default="http://127.0.0.1:8765/v1/chat/completions")
    p.add_argument("--serve-pid", type=int, default=None,
                     help="serve process pid for RSS measurement (optional)")
    p.add_argument("--fixtures", type=Path,
                     default=Path("/Users/sarthak/Desktop/fleet/pace/evals/fm-fixtures-v2"))
    p.add_argument("--sys-prompt", type=Path, default=None,
                     help="system prompt path (default: eval_pace_v2's default)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--out", type=Path, default=None,
                     help="JSON output path (default: ~/.cache/tinygpt/scores/<label>-<ts>.json)")
    args = p.parse_args()

    print(f"=== formula score: {args.label} ===")
    print(f"  model-dir: {args.model_dir}")
    print(f"  serve-url: {args.serve_url}")
    print()

    # System prompt for speed calibration
    sys_prompt_path = args.sys_prompt or Path(
        "/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v6-label.txt"
    )
    sys_prompt = sys_prompt_path.read_text().strip()

    print("[1/3] speed...")
    speed = measure_speed(args.serve_url, sys_prompt, runs=args.runs)
    print(f"  TTFW median: {speed['ttfw_ms_median']:.0f} ms  (min {speed['ttfw_ms_min']:.0f}, max {speed['ttfw_ms_max']:.0f})")
    print(f"  tok/s median: {speed['tok_per_s_median']:.1f}")

    print("\n[2/3] accuracy...")
    accuracy = measure_accuracy(args.fixtures, args.serve_url,
                                 sys_prompt_path=args.sys_prompt)
    print(f"  pass: {accuracy['pass_n']}/{accuracy['total_n']} ({accuracy['pass_rate']*100:.1f}%)")

    print("\n[3/3] cost...")
    cost = measure_cost(args.model_dir, serve_pid=args.serve_pid)
    print(f"  disk: {cost['model_disk_gb']:.2f} GB")
    if cost['serve_rss_gb']:
        print(f"  serve RSS: {cost['serve_rss_gb']:.2f} GB")
    print(f"  cost total: {cost['cost_gb']:.2f} GB")

    formula = compute_formula(speed, accuracy, cost)
    print()
    print(f"=== FORMULA SCORE: {formula['formula']:.3f} ===")
    print(f"  speed_score:    {formula['speed_score']:.3f}")
    print(f"  accuracy_score: {formula['accuracy_score']:.3f}")
    print(f"  cost_score:     {formula['cost_score']:.3f}")
    print(f"  (baseline 1.0 = TTFW 120ms, tok/s 50, cost 2GB, accuracy at pass_rate)")

    result = {
        "label": args.label,
        "timestamp": int(time.time()),
        "model_dir": str(args.model_dir),
        "serve_url": args.serve_url,
        # Harness config — without this, baselines measured under the wrong
        # prompt are indistinguishable from real scores (see v11-baselines
        # 2026-06-10 addendum: v9 "33.3%" was v6-prompt drift; truth was 60%).
        "sys_prompt": str(sys_prompt_path),
        "sys_prompt_was_default": args.sys_prompt is None,
        "speed": speed,
        "accuracy": accuracy,
        "cost": cost,
        "formula": formula,
    }
    out_path = args.out or Path.home() / ".cache/tinygpt/scores" / f"{args.label}-{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
