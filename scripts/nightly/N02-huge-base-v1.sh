#!/usr/bin/env bash
# N02-huge-base-v1.sh — pretrain a Huge base on FineWeb-Edu.
#
# ~8-10 hrs wall on M5 Pro. Produces the reusable base every subsequent
# specialist SFT will start from. All later runs amortise this cost.
#
# Outputs:
#   /tmp/huge-base-v1.tinygpt              canonical (atomic-saved every 1000 steps)
#   /tmp/huge-base-v1.step-*.tinygpt       history (B13 interp)
#   /tmp/huge-base-v1.jsonl                /training-dashboard log
#
# Recipe choices (all set with intent — change deliberately):
#   --preset huge                   12L · d=256 · 10M body params · fits 48 GB
#   --tokenizer SmolLM2-135M        49K-vocab BPE; right-sized for 10M body
#   --ctx 256                       Huge default; pretrain stays cheap
#   --batch 8 --accum 2             effective batch 16 — stable LM training
#   --steps 100000                  ~410M tokens (over-Chinchilla — fine)
#   --lr-schedule wsd               decay phase doubles as annealing
#   --warmup 1000 --decay-steps 10000   1% warmup, 10% decay (MiniCPM-ish)
#   --max-lr 3e-4 --min-lr 3e-5     standard transformer endpoints
#   --save-every 1000 --save-history   100 checkpoints kept (~12 GB)
#   --seed 42                       reproducible init for A/B
#   --grad-clip 1.0                 already default; explicit for the log
#   spike detector ON               will flag spike > 3× MA over 50 steps

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TINYGPT="$REPO_ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt"
CACHE="$HOME/.cache/tinygpt/datasets"
TOKENIZER_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"
# SmolLM2 ships with multiple snapshot/<commit>/ dirs; pick the latest
# snapshot dir that has tokenizer.json.
TOKENIZER_PATH="$(find "$TOKENIZER_DIR/snapshots" -name "tokenizer.json" 2>/dev/null \
                  | head -1 | xargs -I {} dirname {} || true)"

if [[ ! -x "$TINYGPT" ]]; then
    echo "tinygpt binary not found at $TINYGPT — run \`cd native-mac && swift build -c release\` first" >&2
    exit 1
fi

if [[ -z "$TOKENIZER_PATH" ]] || [[ ! -f "$TOKENIZER_PATH/tokenizer.json" ]]; then
    echo "SmolLM2 tokenizer not found under $TOKENIZER_DIR" >&2
    echo "Run N01-pull-datasets.sh first." >&2
    exit 1
fi

# FineWeb-Edu lands as JSONL; the train command wants a raw text file.
# Extract the "text" field once; cache the result so reruns are instant.
CORPUS_JSONL="$CACHE/fineweb-edu.sample.jsonl"
CORPUS_TXT="$CACHE/fineweb-edu.sample.txt"

if [[ ! -f "$CORPUS_JSONL" ]]; then
    echo "pretrain corpus not found at $CORPUS_JSONL — run N01 first" >&2
    exit 1
fi

if [[ ! -f "$CORPUS_TXT" ]] || [[ "$CORPUS_JSONL" -nt "$CORPUS_TXT" ]]; then
    echo "extracting raw text from $CORPUS_JSONL …"
    # Use jq if available (precise); fall back to python (always available
    # on macOS via /usr/bin/python3).
    if command -v jq >/dev/null 2>&1; then
        jq -r '.text // .content // empty' "$CORPUS_JSONL" > "$CORPUS_TXT"
    else
        /usr/bin/python3 -c "
import json, sys
with open('$CORPUS_JSONL') as f, open('$CORPUS_TXT', 'w') as out:
    for line in f:
        try:
            obj = json.loads(line)
            text = obj.get('text') or obj.get('content') or ''
            if text:
                out.write(text + '\n\n')
        except Exception:
            pass
"
    fi
    echo "  extracted $(wc -c < "$CORPUS_TXT") bytes"
fi

OUT="/tmp/huge-base-v1.tinygpt"
LOG="/tmp/huge-base-v1.jsonl"

echo ""
echo "=== huge-base-v1 pretrain ==="
echo "  corpus:    $CORPUS_TXT ($(wc -c < "$CORPUS_TXT") bytes)"
echo "  tokenizer: $TOKENIZER_PATH"
echo "  out:       $OUT"
echo "  log:       $LOG"
echo "  expected wall: ~8-10 hrs"
echo ""

"$TINYGPT" train \
    --preset huge \
    --tokenizer "$TOKENIZER_PATH" \
    --corpus "$CORPUS_TXT" \
    --dtype bfloat16 \
    --ctx 256 \
    --batch 8 \
    --accum 2 \
    --steps 100000 \
    --lr-schedule wsd \
    --warmup 1000 \
    --decay-steps 10000 \
    --max-lr 3e-4 \
    --min-lr 3e-5 \
    --grad-clip 1.0 \
    --save-every 1000 \
    --save-history \
    --val-split 0.005 \
    --val-every 500 \
    --seed 42 \
    --log-jsonl "$LOG" \
    --out "$OUT" \
    --sample-every 5000

echo ""
echo "huge-base-v1 pretrain complete."
echo "next: N03-sft-toolcaller-v1.sh (LoRA SFT on this base)"
