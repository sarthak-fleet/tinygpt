#!/usr/bin/env bash
# scripts/sae-run.sh — train SAEs across every history checkpoint of a run.
#
# Usage:
#   ./scripts/sae-run.sh <canonical.tinygpt> [out-dir]
#   ./scripts/sae-run.sh ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt
#
# Sibling of score-run.sh. Runs `tinygpt sae --checkpoint-dir` against
# the history checkpoints + canonical, emitting a JSONL timeline that
# `browser/src/pages/sae-timeline.astro` (when the elf ships it) will
# render as a feature-emergence chart.
#
# All inputs are CPU-light other than the model forwards; runs serially
# so it doesn't fight a concurrent training run for the GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINYGPT="$REPO_ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <canonical.tinygpt> [out-dir]" >&2
    exit 1
fi

CKPT="$1"
CKPT_BASE="$(basename "$CKPT" .tinygpt)"
CKPT_DIR="$(dirname "$CKPT")"
OUT_DIR="${2:-$REPO_ROOT/docs/artifacts/sae-$CKPT_BASE-$(date +%Y%m%d)}"
TIMELINE_JSONL="$OUT_DIR/timeline.jsonl"

if [[ ! -f "$CKPT" ]]; then
    echo "canonical checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -x "$TINYGPT" ]]; then
    echo "tinygpt binary missing — run \`cd native-mac && swift build -c release\`" >&2
    exit 1
fi

# Pick a sensible default corpus: the same one the model trained on, if
# discoverable; otherwise FineWeb-Edu (the N02 default).
CORPUS="${CORPUS:-/tmp/fineweb-edu.txt}"
if [[ ! -f "$CORPUS" ]]; then
    echo "training corpus missing at $CORPUS — set CORPUS=<path> and retry" >&2
    exit 1
fi

# SAE training hyperparameters — kept small so the sweep stays under
# an hour for ~10 checkpoints. Override via env if you want broader runs.
LAYER="${LAYER:-6}"             # middle-ish layer for Huge (12L)
D_FEATURES="${D_FEATURES:-2048}"
SAE_STEPS="${SAE_STEPS:-1000}"
SAE_BATCH="${SAE_BATCH:-8}"
SAE_CTX="${SAE_CTX:-256}"

mkdir -p "$OUT_DIR"
echo "=== sae-run: $CKPT_BASE ==="
echo "  layer:      $LAYER"
echo "  d_features: $D_FEATURES"
echo "  steps:      $SAE_STEPS"
echo "  corpus:     $CORPUS"
echo "  out-dir:    $OUT_DIR"
echo "  timeline:   $TIMELINE_JSONL"
echo ""

# `tinygpt sae --checkpoint-dir <dir>` walks all .tinygpt files (history
# + canonical) and trains one SAE per checkpoint. JSONL row schema is
# whatever B13 (SAE.swift) emits — see docs/prds/sae-timeline-viewer.md.
# Note: `tinygpt sae --checkpoint-dir` walks every .tinygpt under that
# dir. Symlink the run's checkpoints into a clean staging dir so we
# don't accidentally pick up unrelated checkpoints (the default save
# location was /tmp/ before persistent-output PRD; staging dir stays local).
STAGE_DIR="$OUT_DIR/checkpoints-staged"
mkdir -p "$STAGE_DIR"
for f in "$CKPT" "$CKPT_DIR/$CKPT_BASE".step-*.tinygpt; do
    [[ -f "$f" ]] && ln -sf "$f" "$STAGE_DIR/"
done

"$TINYGPT" sae \
    --checkpoint-dir "$STAGE_DIR" \
    --corpus "$CORPUS" \
    --layer "$LAYER" \
    --features "$D_FEATURES" \
    --steps "$SAE_STEPS" \
    --batch "$SAE_BATCH" \
    --ctx "$SAE_CTX" \
    --out "$OUT_DIR/sae" \
    --timeline-out "$TIMELINE_JSONL"

echo ""
echo "=== sae-run complete ==="
echo "  rows written: $(wc -l < "$TIMELINE_JSONL") to $TIMELINE_JSONL"
echo ""
echo "  next: drop $TIMELINE_JSONL into /sae-timeline.astro (when shipped)"
echo "        or grep the rows: jq '. | {step, mse, l0_frac}' < $TIMELINE_JSONL"
