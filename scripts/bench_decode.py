#!/usr/bin/env python3
"""Decode-speed bench against any OpenAI-compatible endpoint.

Measures TTFT (time to first token), per-token ITL, and steady-state
decode tok/s by streaming /v1/chat/completions. Designed for the
"Gemma decode tok/s" row in docs/research/mac_decode_baseline_m5pro.md
but generic over any model on any OpenAI-compatible server (LM Studio,
ollama, tinygpt serve, vLLM, …).

Reuses the SSE / per-token-timing pattern from scripts/score_formula.py
(`time_request`) instead of duplicating it — the only delta here is
that we capture per-token latencies (not just first-token) so we can
report ITL quantiles, and we do an --n-run sweep to get medians.

Output: JSON to stdout. With --jsonl, also appends a row matching the
EvalHarnessSupport.Row schema (task="decode") so it slots into the
shared eval-compare leaderboard.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SYS_PROMPT = (
    "You are a helpful assistant. Answer concisely."
)
DEFAULT_USER_PROMPT = (
    "Write three sentences about why bridges have keystones. "
    "Avoid bullet points; flow them together."
)


def quantiles(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)

    def at(p: float) -> float:
        # nearest-rank quantile. for n>=4 it matches the numpy default
        # closely enough that the mac_decode_baseline doc's "p95 / p99"
        # columns stay readable.
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs[k]
    return {
        "median": statistics.median(xs),
        "p95": at(0.95),
        "p99": at(0.99),
        "min": xs[0],
        "max": xs[-1],
        "n": n,
    }


def one_run(url: str, model: str, sys_prompt: str, user_prompt: str,
            max_tokens: int, timeout: int) -> dict:
    """One streamed chat completion. Returns per-token timings + counts."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.monotonic()
    t_first = None
    t_prev = None
    itls_ms: list[float] = []
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not (line.startswith("data:") and not line.endswith("[DONE]")):
                continue
            try:
                d = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            delta = (d.get("choices", [{}])[0]
                       .get("delta", {})
                       .get("content", ""))
            if not delta:
                continue
            now = time.monotonic()
            if t_first is None:
                t_first = now
            else:
                itls_ms.append((now - t_prev) * 1000.0)
            t_prev = now
            n_tokens += 1
    t_end = time.monotonic()
    if t_first is None:
        # Server returned no token deltas — likely a non-streaming-compat
        # endpoint or a model load failure. Raise so the caller can flag.
        raise RuntimeError("no tokens streamed; is --url an OpenAI-compat "
                           "/v1/chat/completions and the model loaded?")
    return {
        "ttft_ms": (t_first - t0) * 1000.0,
        "decode_s": max(t_end - t_first, 1e-6),
        "itls_ms": itls_ms,
        "n_tokens": n_tokens,
        "total_ms": (t_end - t0) * 1000.0,
    }


def poll_rss_mb(pid: int) -> float | None:
    """One-shot RSS read via `ps -o rss=`. Returns MB."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], timeout=2
        ).decode().strip()
        return int(out) / 1024.0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",
                    default=os.environ.get(
                        "BENCH_URL",
                        "http://127.0.0.1:1234/v1/chat/completions"),
                    help="OpenAI-compatible /v1/chat/completions endpoint")
    ap.add_argument("--model", required=True,
                    help="model id the server expects (e.g. google/gemma-3-12b-it)")
    ap.add_argument("--n", type=int, default=20,
                    help="measured runs (default 20)")
    ap.add_argument("--warm", type=int, default=3,
                    help="warmup runs that are not measured (default 3)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="tokens to generate per run (default 128)")
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-request timeout in seconds (default 180)")
    ap.add_argument("--sys-prompt", default=DEFAULT_SYS_PROMPT)
    ap.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    ap.add_argument("--rss-pid", type=int, default=None,
                    help="optional PID of the serve process — RSS is "
                         "polled once per measured run via ps")
    ap.add_argument("--jsonl", default=None,
                    help="append EvalHarnessSupport-shaped rows here")
    ap.add_argument("--label", default=None,
                    help="row label (default: --model)")
    args = ap.parse_args()

    label = args.label or args.model

    for _ in range(args.warm):
        try:
            one_run(args.url, args.model, args.sys_prompt, args.user_prompt,
                    args.max_tokens, args.timeout)
        except Exception as e:
            sys.exit(f"warmup failed: {e}")

    ttfts: list[float] = []
    decode_tps: list[float] = []
    all_itls: list[float] = []
    rsses: list[float] = []
    wall_s_total = 0.0
    for i in range(args.n):
        r = one_run(args.url, args.model, args.sys_prompt, args.user_prompt,
                    args.max_tokens, args.timeout)
        ttfts.append(r["ttft_ms"])
        # Steady-state tok/s = (n - 1) / time_after_first.
        # If only one token came back, skip; nothing to report.
        if r["n_tokens"] > 1:
            decode_tps.append((r["n_tokens"] - 1) / r["decode_s"])
        all_itls.extend(r["itls_ms"])
        wall_s_total += r["total_ms"] / 1000.0
        if args.rss_pid is not None:
            mb = poll_rss_mb(args.rss_pid)
            if mb is not None:
                rsses.append(mb)
        # heartbeat so a stalled run is obvious from the terminal
        print(f"  run {i+1}/{args.n}: ttft={r['ttft_ms']:.1f}ms  "
              f"n={r['n_tokens']}  "
              f"tok/s={(r['n_tokens']-1)/r['decode_s']:.1f}",
              file=sys.stderr, flush=True)

    out = {
        "label": label,
        "url": args.url,
        "model": args.model,
        "n_runs": args.n,
        "max_tokens_per_run": args.max_tokens,
        "ttft_ms": quantiles(ttfts),
        "itl_ms": quantiles(all_itls),
        "decode_tok_s": quantiles(decode_tps),
        "peak_rss_mb": quantiles(rsses) if rsses else None,
        "wall_seconds_total": wall_s_total,
    }
    print(json.dumps(out, indent=2))

    if args.jsonl:
        # EvalHarnessSupport.Row schema (snake_case keys, JSONL one per line).
        # One row per metric so eval-compare's --by model / --by task views
        # both work without reshaping.
        path = Path(args.jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        common = {
            "run_id": uuid.uuid4().hex,
            "model_name": label,
            "model_path": args.url,
            "model_step": None,
            "baseline": False,
            "task": "decode",
            "n_examples": args.n,
            "wall_seconds": wall_s_total,
            "harness_version": "bench_decode.py/1",
        }
        rows = [
            {**common, "subtask": "ttft_ms_p99", "metric": "ms",
             "score": out["ttft_ms"]["p99"]},
            {**common, "subtask": "itl_ms_p99", "metric": "ms",
             "score": out["itl_ms"]["p99"]},
            {**common, "subtask": "decode_tok_s_median", "metric": "tok_per_s",
             "score": out["decode_tok_s"]["median"]},
        ]
        if out["peak_rss_mb"]:
            rows.append({**common, "subtask": "peak_rss_mb_p99",
                         "metric": "mb",
                         "score": out["peak_rss_mb"]["p99"]})
        with path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nappended {len(rows)} rows to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
