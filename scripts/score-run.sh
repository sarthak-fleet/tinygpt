#!/usr/bin/env bash
# scripts/score-run.sh — score every history checkpoint of a training run.
#
# Usage:
#   ./scripts/score-run.sh <canonical.tinygpt> [out.jsonl]
#   ./scripts/score-run.sh ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt
#
# Given a canonical training output, finds the sibling step-N history
# checkpoints (from `--save-history`), scores each one + the canonical
# + SmolLM2 baseline, and renders the cross-checkpoint emergence view.
#
# This is the fire-and-forget "what to do when N02 finishes" runbook.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINYGPT="$REPO_ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <canonical.tinygpt> [out.jsonl]" >&2
    exit 1
fi

CKPT="$1"
CKPT_BASE="$(basename "$CKPT" .tinygpt)"
CKPT_DIR="$(dirname "$CKPT")"
OUT="${2:-$REPO_ROOT/docs/artifacts/score-$CKPT_BASE-$(date +%Y%m%d).jsonl}"
mkdir -p "$(dirname "$OUT")"

if [[ ! -f "$CKPT" ]]; then
    echo "canonical checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -x "$TINYGPT" ]]; then
    echo "tinygpt binary missing — run \`cd native-mac && swift build -c release\`" >&2
    exit 1
fi

# Collect history checkpoints (numeric-sorted) + the canonical at the end.
mapfile -t HISTORY < <(ls "$CKPT_DIR/$CKPT_BASE".step-*.tinygpt 2>/dev/null | sort -V)
ALL_CKPTS=("${HISTORY[@]}" "$CKPT")

echo "=== score-run: $CKPT_BASE ==="
echo "  ${#HISTORY[@]} history checkpoint(s) + 1 canonical → ${#ALL_CKPTS[@]} total"
echo "  out: $OUT"
echo ""

i=0
for c in "${ALL_CKPTS[@]}"; do
    i=$((i+1))
    echo "[$i/${#ALL_CKPTS[@]}] $(basename "$c")"
    "$REPO_ROOT/scripts/score-checkpoint.sh" "$c" "$OUT" || {
        echo "  (continuing despite failure)" >&2
    }
    echo ""
done

echo ""
echo "=== summary ==="
echo ""
echo "--- by step (training emergence) ---"
"$TINYGPT" eval-compare "$OUT" --by step
echo ""
echo "--- by model (vs SmolLM2 baseline) ---"
"$TINYGPT" eval-compare "$OUT" --by model
echo ""
echo "--- by task ---"
"$TINYGPT" eval-compare "$OUT" --by task

echo ""
echo "raw rows preserved at: $OUT"
