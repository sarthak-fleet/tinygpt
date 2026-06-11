#!/usr/bin/env bash
# clarify-adapter v1 pipeline: train, bake, serve, gate vs cloud baseline.
# 38 hand-curated contrastive rows (19 ambig→clarify / 19 matched→action).
# Win condition: ambig-h2 > 15% (Claude cloud baseline) without regressing
# the action twins or other h2 suites.
set -euo pipefail
BASE=~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/main
DATA=~/.cache/tinygpt/datasets/clarify-train-v1.jsonl
RUN_DIR=~/.cache/tinygpt/runs/clarify-v1
LORA="$RUN_DIR/clarify-v1.lora"
BAKED="$RUN_DIR/baked-hf"
TINYGPT=/Users/sarthak/Desktop/fleet/tinygpt/native-mac/.build/arm64-apple-macosx/release/tinygpt
TGT=/Users/sarthak/Desktop/fleet/tinygpt
SYSP="$TGT/grammars/pace-system-prompt-v11.txt"
GRAMMAR="$TGT/grammars/pace-fm-response-v11.schema.json"

mkdir -p "$RUN_DIR"
echo "[1/4] Training Qwen3-4B + clarify-v1 (38 rows, DoRA off, rank 32 alpha 64, 600 steps)..."
caffeinate -i "$TINYGPT" sft "$BASE" --data "$DATA" --out "$LORA" \
  --template chatml --rank 32 --alpha 64 --steps 600 --lr 1e-4 --batch 2 \
  --no-dora --metal-cache-gb 12 > "$RUN_DIR/train.log" 2>&1
echo "  trained: $LORA"

echo "[2/4] Baking adapter..."
"$TINYGPT" bake-lora "$BASE" "$LORA" --out "$BAKED" 2>&1 | tail -2

echo "[3/4] Serving baked model on :8765..."
pkill -f "tinygpt serve" 2>/dev/null || true; sleep 2
"$TINYGPT" serve "$BAKED" --grammar "$GRAMMAR" --port 8765 \
  > "$RUN_DIR/serve.log" 2>&1 &
for i in $(seq 1 60); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/v1/models 2>/dev/null | grep -q 200 && break
  sleep 0.5
done; echo "  ready"

echo "[4/4] Gate vs Claude on h2 suites..."
URL=http://127.0.0.1:8765/v1/chat/completions
for SUITE in ambig oos destructive; do
  echo "--- $SUITE-h2 ---"
  python3 "$TGT/scripts/eval_pace_unhappy.py" \
    --fixtures-dir "$TGT/evals/fm-fixtures-$SUITE-h2" \
    --serve-url "$URL" --sys-prompt "$SYSP" \
    --out "$RUN_DIR/$SUITE.json" 2>&1 | tail -2
done
pkill -f "tinygpt serve" 2>/dev/null || true
echo "=== done. results in $RUN_DIR ==="
