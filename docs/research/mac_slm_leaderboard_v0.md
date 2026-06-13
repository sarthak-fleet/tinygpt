---
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
python3 scripts/build_slm_leaderboard.py \
    --manifest docs/research/data/leaderboard_manifest.json
```

## Leaderboard

<!-- TABLE START -->

| rank | model | params | decode tok/s | TTFT p99 (ms) | RSS p99 (MB) | BFCL avg | τ-bench avg | unhappy avg | composite |
|---|---|---|---|---|---|---|---|---|---|
| 1 | google/gemma-3-12b | 12B | 36.3 | 187.9 | 9097 | — | — | 64.2 | 0.105 |
| 2 | google/gemma-3-12b (v11-compact) | 12B | 36.3 | 187.9 | 9097 | — | — | 57.5 | 0.094 |

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
