#!/usr/bin/env bash
# scripts/score-baselines.sh — pre-score HF baseline models against our eval suite.
#
# Usage:
#   ./scripts/score-baselines.sh [out.jsonl]
#
# Scores a fixed set of small open-weight baselines (SmolLM2-135M,
# SmolLM2-360M, Qwen3-0.6B, TinyLlama-1.1B, Phi-3-mini-4K) on the same
# tasks the post-N02 sweep uses. Output rows are E0-shaped, so the same
# `tinygpt eval-compare` consumes them.
#
# Why pre-score: when N02 lands and `score-run.sh` runs, the comparison
# table only has SmolLM2 as the cross-model anchor. Pre-scoring the
# others now means landing-day shows a 5-model leaderboard, not 2.
#
# GPU contention note: lm-eval-harness's HF backend uses MPS by default
# on Mac. If N02 is still training, set DEVICE=cpu to avoid contention
# (10× slower per model but safe). After N02, leave DEVICE unset (mps).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINYGPT="$REPO_ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt"

OUT="${1:-$REPO_ROOT/docs/artifacts/baselines-$(date +%Y%m%d).jsonl}"
TASKS="${TASKS:-arc_easy}"
LIMIT="${LIMIT:-30}"
DEVICE="${DEVICE:-mps}"
# After N02 completes, bump TASKS to "arc_easy,hellaswag,piqa" and
# LIMIT to 200 for full leaderboard volume.

mkdir -p "$(dirname "$OUT")"

# 5 baselines, ranked small → mid (skip Phi-3 if it OOMs on 48 GB).
BASELINES=(
    "HuggingFaceTB/SmolLM2-135M"
    "HuggingFaceTB/SmolLM2-360M"
    "Qwen/Qwen3-0.6B"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "microsoft/Phi-3-mini-4k-instruct"
)

if [[ ! -x "$TINYGPT" ]]; then
    echo "tinygpt binary missing — run \`cd native-mac && swift build -c release\`" >&2
    exit 1
fi

echo "=== score-baselines ==="
echo "  models: ${#BASELINES[@]}"
echo "  tasks:  $TASKS"
echo "  limit:  $LIMIT"
echo "  device: $DEVICE"
echo "  out:    $OUT"
echo ""

i=0
for repo in "${BASELINES[@]}"; do
    i=$((i+1))
    name="$(basename "$repo")"

    # Skip if already in the JSONL.
    if [[ -f "$OUT" ]] && grep -q "\"model_name\":\"$name\"" "$OUT"; then
        echo "[$i/${#BASELINES[@]}] $name — already scored, skipping"
        continue
    fi

    echo "[$i/${#BASELINES[@]}] $name"
    "$TINYGPT" run-lm-eval \
        --hf-model "$repo" \
        --tasks "$TASKS" \
        --limit "$LIMIT" \
        --device "$DEVICE" \
        --model-name "$name" \
        --baseline \
        --out "$OUT" || {
            echo "  failed (continuing)" >&2
        }
    echo ""
done

echo ""
echo "=== baselines summary ==="
"$TINYGPT" eval-compare "$OUT" --by model
