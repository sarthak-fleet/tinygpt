#!/usr/bin/env bash
# scripts/score-checkpoint.sh — fire-and-forget post-training scoring.
#
# Usage:
#   ./scripts/score-checkpoint.sh <ckpt.tinygpt> [out.jsonl]
#   ./scripts/score-checkpoint.sh ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt
#
# Defaults out.jsonl to docs/artifacts/score-$(basename ckpt)-$(date).jsonl
#
# Runs `tinygpt run-lm-eval` against the checkpoint for a fixed sweep
# of tasks, appends rows to the JSONL, prints `eval-compare --by model`.
# Optionally scores SmolLM2-135M baseline on the same tasks if not
# already in the JSONL (idempotent via `--model-name` dedup at the
# eval-compare layer).
#
# Designed to be fired after N02 lands without intervention. Run it
# against the canonical checkpoint AND each step-N history checkpoint
# to render the full --by step emergence view.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINYGPT="$REPO_ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <ckpt.tinygpt> [out.jsonl] [--tasks csv] [--limit N]" >&2
    exit 1
fi

CKPT="$1"; shift

if [[ ! -f "$CKPT" ]]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -x "$TINYGPT" ]]; then
    echo "tinygpt binary missing — run \`cd native-mac && swift build -c release\`" >&2
    exit 1
fi

CKPT_BASE="$(basename "$CKPT" .tinygpt)"
DEFAULT_OUT="$REPO_ROOT/docs/artifacts/score-$CKPT_BASE-$(date +%Y%m%d).jsonl"
OUT="${1:-$DEFAULT_OUT}"
[[ $# -gt 0 ]] && shift || true
mkdir -p "$(dirname "$OUT")"

# Defaults — kept small to survive GPU contention if training is still
# running. The 2026-06-05 validation found `serve` drops responses
# under combined train+serve load past ~200 lm-eval requests. After N02
# completes, bump TASKS to "arc_easy,hellaswag,piqa,arc_challenge" and
# LIMIT to 200 for a richer signal.
TASKS="${TASKS:-arc_easy}"
LIMIT="${LIMIT:-30}"

# Tokenizer (SmolLM2-135M BPE — matches what training used).
TOKENIZER_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"
TOKENIZER_PATH="$(find "$TOKENIZER_DIR/snapshots" -name "tokenizer.json" 2>/dev/null \
                  | head -1 | xargs -I {} dirname {} || true)"

# Try to extract step from filename like "<stem>.step-N.tinygpt".
STEP_FROM_NAME=""
if [[ "$CKPT_BASE" =~ \.step-([0-9]+)$ ]]; then
    STEP_FROM_NAME="--model-step ${BASH_REMATCH[1]}"
fi

# Model-name strip step suffix so all checkpoints of one run share a name.
MODEL_NAME="$(echo "$CKPT_BASE" | sed 's/\.step-[0-9]*$//')"

echo "=== scoring $CKPT ==="
echo "  tasks:  $TASKS"
echo "  limit:  $LIMIT"
echo "  out:    $OUT"
echo ""

"$TINYGPT" run-lm-eval \
    --tinygpt-model "$CKPT" \
    --tokenizer "$TOKENIZER_PATH" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --model-name "$MODEL_NAME" \
    $STEP_FROM_NAME \
    --out "$OUT"

# Score SmolLM2 baseline once per JSONL (skip if already present).
if ! grep -q '"model_name":"SmolLM2-135M"' "$OUT" 2>/dev/null; then
    echo ""
    echo "=== scoring SmolLM2-135M baseline ==="
    "$TINYGPT" run-lm-eval \
        --hf-model HuggingFaceTB/SmolLM2-135M \
        --tasks "$TASKS" \
        --limit "$LIMIT" \
        --model-name "SmolLM2-135M" \
        --baseline \
        --out "$OUT" || echo "baseline scoring failed (non-fatal — TinyGPT row already written)"
fi

echo ""
echo "=== summary ==="
"$TINYGPT" eval-compare "$OUT" --by model
