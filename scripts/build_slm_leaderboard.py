#!/usr/bin/env python3
"""Mac SLM agentic leaderboard v0 — manifest-driven aggregator.

Combines per-model eval artifacts into one publication-shape table:
  * decode tok/s + TTFT p99 + peak RSS (from `scripts/bench_decode.py`)
  * BFCL avg + per-category (from `tinygpt eval-bfcl`'s JSON output)
  * τ-bench retail + airline pass@1 (from `tinygpt eval-tau-bench`'s JSON)
  * Pace unhappy-path ambig/oos/destructive pass rates
    (from `~/.cache/tinygpt/runs/h2-combined-<tag>/*.json`)
  * Composite = mean(accuracy) × speed × cost,
    mirrors the formula in scripts/score_formula.py §3 (no DRY breach).

Reads existing artifacts only — does NOT re-run any eval. The
`scripts/eval_slm_full.sh` wrapper is the thing that produces all four
inputs for one model in one go.

Outputs:
  * docs/research/mac_slm_leaderboard_v0.md   (human view + runbook)
  * docs/research/data/slm_leaderboard_v0.json (machine view)

Manifest schema (YAML or JSON, --manifest path):

    rows:
      - label: gemma-3-12b-it
        params: "12B"
        unhappy_tag: h2-gemma-12b
        bfcl_json: docs/research/data/bfcl-gemma-12b.json
        tau_json:  docs/research/data/tau-gemma-12b.json
        decode_json: docs/research/data/decode-gemma-12b.json

Any of bfcl_json / tau_json / decode_json / unhappy_tag can be omitted —
missing columns render as "—" in the table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
UNHAPPY_DIR = Path.home() / ".cache" / "tinygpt" / "runs"
SUITES = ("ambig", "oos", "destructive")
DEFAULT_OUT_MD = REPO / "docs" / "research" / "mac_slm_leaderboard_v0.md"
DEFAULT_OUT_JSON = REPO / "docs" / "research" / "data" / "slm_leaderboard_v0.json"


def load_manifest(path: Path) -> list[dict]:
    text = path.read_text()
    if not text.strip():
        return []
    try:
        # JSON-only manifest reader — keeps the stdlib-only invariant.
        # Re-encoding a YAML manifest as JSON is a one-liner if needed.
        data = json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"manifest must be valid JSON: {e}")
    rows = data.get("rows", data if isinstance(data, list) else [])
    if not isinstance(rows, list):
        sys.exit("manifest: 'rows' must be a list")
    return rows


def pct(passed: int, total: int) -> float:
    return 100.0 * passed / total if total else 0.0


def load_unhappy(tag: str) -> dict | None:
    """Returns {ambig, oos, destructive, avg} pass rates from the
    h2-combined runs. None if any of the three suite files is missing.
    """
    run_dir = UNHAPPY_DIR / f"h2-combined-{tag}"
    out: dict[str, float] = {}
    for s in SUITES:
        f = run_dir / f"{s}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        out[s] = pct(d.get("passed", 0), d.get("total", 0))
    out["avg"] = statistics.mean(out[s] for s in SUITES)
    return out


def load_decode(p: Path) -> dict | None:
    """Reads the JSON that bench_decode.py prints to stdout."""
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    out: dict[str, Any] = {
        "ttft_p99": d.get("ttft_ms", {}).get("p99"),
        "decode_tok_s": d.get("decode_tok_s", {}).get("median"),
    }
    rss = d.get("peak_rss_mb") or {}
    out["peak_rss_mb"] = rss.get("p99") if rss else None
    return out


def load_bfcl(p: Path) -> dict | None:
    """Pulls the BFCL average from `tinygpt eval-bfcl`'s JSON output.

    The exact shape is: a dict of categories → scores. Average over
    numeric values; bail if the file's shape is unrecognized rather than
    guessing — wrong number is worse than a missing one.
    """
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    numbers = _collect_numbers(d)
    if not numbers:
        return None
    return {"avg": statistics.mean(numbers),
            "n_categories": len(numbers)}


def load_tau(p: Path) -> dict | None:
    """Pulls retail + airline pass@1 from `tinygpt eval-tau-bench`'s JSON.

    Tries the canonical {retail: …, airline: …} shape first; falls back
    to averaging all numerics.
    """
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    out: dict[str, Any] = {}
    for env in ("retail", "airline"):
        v = _extract_number(d, env)
        if v is not None:
            out[env] = v
    if not out:
        nums = _collect_numbers(d)
        if nums:
            out["avg"] = statistics.mean(nums)
        else:
            return None
    else:
        out["avg"] = statistics.mean(out.values())
    return out


def _extract_number(d: Any, key: str) -> float | None:
    """Best-effort: walk nested dicts looking for `key` → float."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    n = _collect_numbers(v)
                    if n:
                        return statistics.mean(n)
            else:
                got = _extract_number(v, key)
                if got is not None:
                    return got
    elif isinstance(d, list):
        for v in d:
            got = _extract_number(v, key)
            if got is not None:
                return got
    return None


def _collect_numbers(d: Any) -> list[float]:
    out: list[float] = []
    if isinstance(d, dict):
        for v in d.values():
            out.extend(_collect_numbers(v))
    elif isinstance(d, list):
        for v in d:
            out.extend(_collect_numbers(v))
    elif isinstance(d, bool):
        pass  # bool is a numeric subtype in Python — don't count it
    elif isinstance(d, (int, float)):
        out.append(float(d))
    return out


def composite(accuracy: float | None, tok_s: float | None,
              rss_mb: float | None) -> float | None:
    """Mirrors scripts/score_formula.py §3 (speed × accuracy × cost).
    Returns None when any input is missing — composite is meaningless
    until all three legs are measured.
    """
    if accuracy is None or tok_s is None:
        return None
    speed = tok_s / 50.0  # 50 tok/s is the realtime floor
    cost = 2.0 / max(rss_mb / 1024.0, 0.5) if rss_mb else 1.0
    return accuracy / 100.0 * speed * cost


def build_table(rows: list[dict]) -> list[dict]:
    table: list[dict] = []
    for r in rows:
        label = r.get("label") or r.get("model_id") or "?"
        params = r.get("params", "?")
        unhappy = load_unhappy(r["unhappy_tag"]) if r.get("unhappy_tag") else None
        bfcl = load_bfcl(Path(r["bfcl_json"])) if r.get("bfcl_json") else None
        tau = load_tau(Path(r["tau_json"])) if r.get("tau_json") else None
        decode = load_decode(Path(r["decode_json"])) if r.get("decode_json") else None

        accuracy_parts = []
        if bfcl: accuracy_parts.append(bfcl["avg"])
        if tau: accuracy_parts.append(tau["avg"])
        if unhappy: accuracy_parts.append(unhappy["avg"])
        accuracy = statistics.mean(accuracy_parts) if accuracy_parts else None

        comp = composite(
            accuracy,
            decode["decode_tok_s"] if decode else None,
            decode["peak_rss_mb"] if decode else None,
        )

        table.append({
            "label": label,
            "params": params,
            "unhappy": unhappy,
            "bfcl": bfcl,
            "tau": tau,
            "decode": decode,
            "accuracy": accuracy,
            "composite": comp,
        })
    table.sort(key=lambda r: (-(r["composite"] or -1),
                              -(r["accuracy"] or -1),
                              r["label"]))
    return table


def _fmt(x: float | None, fmt: str = "{:.1f}", dash: str = "—") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) else dash


def render_markdown(table: list[dict]) -> str:
    head = ("| rank | model | params | decode tok/s | TTFT p99 (ms) "
            "| RSS p99 (MB) | BFCL avg | τ-bench avg | "
            "unhappy avg | composite |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for i, r in enumerate(table, start=1):
        decode = r["decode"] or {}
        bfcl_avg = (r["bfcl"] or {}).get("avg")
        tau_avg = (r["tau"] or {}).get("avg")
        unhappy_avg = (r["unhappy"] or {}).get("avg")
        lines.append(
            f"| {i} | {r['label']} | {r['params']} | "
            f"{_fmt(decode.get('decode_tok_s'))} | "
            f"{_fmt(decode.get('ttft_p99'))} | "
            f"{_fmt(decode.get('peak_rss_mb'), fmt='{:.0f}')} | "
            f"{_fmt(bfcl_avg, fmt='{:.1f}')} | "
            f"{_fmt(tau_avg, fmt='{:.1f}')} | "
            f"{_fmt(unhappy_avg, fmt='{:.1f}')} | "
            f"{_fmt(r['composite'], fmt='{:.3f}')} |"
        )
    return "\n".join(lines)


def write_doc(out_md: Path, table_md: str, n_rows: int):
    """Replace the auto-generated table block in the leaderboard page.

    The page is mostly hand-written (runbook, formula citation, what
    each suite measures). We only rewrite the block between
    <!-- TABLE START --> and <!-- TABLE END --> markers so re-running
    this script doesn't blow away the prose.
    """
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if out_md.exists():
        existing = out_md.read_text()
    else:
        existing = LEADERBOARD_STUB
    start = "<!-- TABLE START -->"
    end = "<!-- TABLE END -->"
    if start in existing and end in existing:
        i = existing.index(start) + len(start)
        j = existing.index(end)
        new = existing[:i] + "\n\n" + table_md + "\n\n" + existing[j:]
    else:
        # Markers missing — re-stub and embed the table. Don't trash
        # whatever was there; keep an archive comment.
        new = LEADERBOARD_STUB.replace(
            f"{start}\n\n{end}",
            f"{start}\n\n{table_md}\n\n{end}"
        )
        new += f"\n\n<!-- archived prior content:\n{existing}\n-->\n"
    out_md.write_text(new)


LEADERBOARD_STUB = """---
title: Mac SLM agentic leaderboard v0
description: One artifact that cross-cuts decode speed, BFCL, τ-bench, and Pace unhappy-paths — the publication-shape view we're missing.
---

# Mac SLM agentic leaderboard v0

**Status:** scaffolding shipped 2026-06-13; populated as models are
benchmarked locally via `scripts/eval_slm_full.sh`.

**Why it exists.** Each suite already produces its own JSON, but no
single view answers "which Mac-runnable SLM is the best agent for Pace
right now?" That question is what a product-shape leaderboard exists
to answer: rank by composite, then drill into the dimensions that
matter for the specific deployment (e.g. tight RSS for ANE routing
later, high BFCL for tool-calling primary).

**Composite formula.** `accuracy × speed × cost`, where:

- `accuracy = mean(BFCL_avg, τ-bench_avg, unhappy_avg)` (each in pp/100)
- `speed = decode_tok_s / 50` (50 tok/s is the realtime floor)
- `cost = 2 / (peak_rss_gb)` (cheaper = more headroom on a 48 GB Mac)

This mirrors `scripts/score_formula.py:230` rather than redefining the
formula here — if you change the weights, change them there and let
this doc inherit.

**How to add a model** (one command + a manifest line):

```
# 1. Run all four suites against the model
scripts/eval_slm_full.sh <lm-studio-model-id> <tag>

# 2. Add a row to the manifest
cat docs/research/data/leaderboard_manifest.json
# {"rows": [
#   {"label": "gemma-3-12b-it", "params": "12B",
#    "unhappy_tag": "h2-gemma-12b",
#    "bfcl_json":   "docs/research/data/bfcl-gemma-12b.json",
#    "tau_json":    "docs/research/data/tau-gemma-12b.json",
#    "decode_json": "docs/research/data/decode-gemma-12b.json"}
# ]}

# 3. Rebuild this page
python3 scripts/build_slm_leaderboard.py \\
    --manifest docs/research/data/leaderboard_manifest.json
```

## Leaderboard

<!-- TABLE START -->

(no rows yet — run `eval_slm_full.sh` against a model and re-run
`build_slm_leaderboard.py`.)

<!-- TABLE END -->

## What each column measures

- **decode tok/s** — median over 20 streamed runs at gen=128 against
  the model's OpenAI-compatible endpoint. From
  `scripts/bench_decode.py`. The number that gates "is this realtime?"
- **TTFT p99 (ms)** — 99th-percentile time-to-first-token across the
  same 20 runs. Gates "does it feel responsive on the first reply?"
- **RSS p99 (MB)** — peak resident memory of the serving process,
  polled via `ps -o rss=` once per run. Gates "will it OOM on a 24 GB
  Mac?"
- **BFCL avg** — `tinygpt eval-bfcl`'s 10-category average. Tool-calling
  capability.
- **τ-bench avg** — `tinygpt eval-tau-bench`'s retail + airline mean.
  Multi-turn agent capability.
- **unhappy avg** — Pace planner n=130 ambig/oos/destructive mean.
  Robustness on the cases that mis-route the most.
- **composite** — see formula above. Sortable by this column to find
  the best all-rounder.

## Caveats v0 will ship with

- All four suites must run against the same model session for the
  numbers to be comparable. The wrapper enforces that; manual
  re-runs are caller-discipline.
- The unhappy-path suite is the one most sensitive to system-prompt
  choice; the leaderboard pins the standard v11 prompt (no
  v11-compact) so cross-model deltas reflect the model, not the L1
  tiering A/B (that's E9's job).
- BFCL category averages mask category-level wins. Drill into the
  per-suite JSON when a model with a tied composite has very
  different per-category scores.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                    help="path to the leaderboard manifest (JSON; see "
                         "scripts/build_slm_leaderboard.py docstring)")
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    table = build_table(rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": table}, indent=2,
                                         default=str) + "\n")

    if not table:
        # Empty manifest is a legitimate state (just-scaffolded). Render
        # the stub-with-empty-table; downstream readers know how to
        # interpret the "(no rows yet)" sentinel.
        write_doc(args.out_md, "(no rows yet — run `eval_slm_full.sh` "
                  "against a model and re-run `build_slm_leaderboard.py`.)",
                  0)
        print(f"wrote empty leaderboard → {args.out_md}", file=sys.stderr)
        return

    write_doc(args.out_md, render_markdown(table), len(table))
    print(f"wrote {len(table)}-row leaderboard → {args.out_md}\n"
          f"machine view → {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
