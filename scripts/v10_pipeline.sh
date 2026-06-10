#!/usr/bin/env bash
# v10 cascade: train + bake + serve + eval + score
#
# Assumes:
#   ~/.cache/tinygpt/datasets/pace-v10-multiplied.jsonl exists (output of teacher multiplier)
#   Qwen3-0.6B base at ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/.../
#   No serve running on port 8765
#   LM Studio Qwen3-14B can be unloaded (frees GPU for training)
#
# Time: ~90 min train + ~5 min bake + ~10 min eval = ~105 min wall total.

set -euo pipefail

BASE=/Users/sarthak/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
DATA=/Users/sarthak/.cache/tinygpt/datasets/pace-v10-multiplied.jsonl
RUN_DIR=/Users/sarthak/.cache/tinygpt/runs/pace-planner-v10
LORA="$RUN_DIR/pace-planner-v10.lora"
BAKED="$RUN_DIR/baked-hf"
LOG="$RUN_DIR/train.log"
TINYGPT=/Users/sarthak/Desktop/fleet/tinygpt/native-mac/.build/release/tinygpt

mkdir -p "$RUN_DIR"

if [ ! -f "$DATA" ]; then
  echo "ERROR: multiplied dataset missing at $DATA"
  exit 1
fi

DATA_ROWS=$(wc -l < "$DATA")
echo "=== v10 pipeline start $(date) ==="
echo "  base: $BASE"
echo "  data: $DATA ($DATA_ROWS rows)"
echo "  run:  $RUN_DIR"
echo ""

# Step 1: Unload LM Studio Qwen3-14B if loaded (free GPU for training)
echo "[1/5] Unloading LM Studio models..."
lms unload --all 2>/dev/null || lms ps 2>&1 | head || true
echo ""

# Step 2: Train v10 with DoRA (now persists magnitudes via #248)
echo "[2/5] Training v10 (DoRA, rank 32, alpha 64, 3000 steps)..."
caffeinate -i "$TINYGPT" sft \
  "$BASE" \
  --data "$DATA" \
  --out "$LORA" \
  --template chatml \
  --rank 32 \
  --alpha 64 \
  --steps 3000 \
  --lr 1e-4 \
  --batch 4 \
  --dora > "$LOG" 2>&1

if [ ! -f "$LORA" ]; then
  echo "ERROR: training failed, no lora file written"
  tail -20 "$LOG"
  exit 1
fi
LORA_SIZE=$(ls -la "$LORA" | awk '{print $5}')
echo "  trained: $LORA ($LORA_SIZE bytes)"
echo ""

# Step 3: Bake v10 LoRA into HF dir
echo "[3/5] Baking v10 LoRA into merged HF dir..."
"$TINYGPT" bake-lora "$BASE" "$LORA" --out "$BAKED" 2>&1 | tail -5
if [ ! -f "$BAKED/model.safetensors" ]; then
  echo "ERROR: bake failed"
  exit 1
fi
echo ""

# Step 4: Boot serve + warm up
echo "[4/5] Starting serve with v10 + v9-grammar..."
pkill -f "tinygpt serve" 2>/dev/null || true
sleep 2
mkdir -p /tmp/tinygpt-cache-v10
# Kill the serve process on ANY exit (eval failure under set -e included),
# not just the happy path.
SERVE_PID=""
trap '[ -n "${SERVE_PID:-}" ] && kill "$SERVE_PID" 2>/dev/null || true' EXIT
"$TINYGPT" serve \
  "$BASE" --lora "$LORA" \
  --grammar /Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-fm-label-response-v9.schema.json \
  --prompt-cache-dir /tmp/tinygpt-cache-v10 \
  --port 8765 > /tmp/serve-v10.log 2>&1 &
SERVE_PID=$!
echo $SERVE_PID > $RUN_DIR/serve.pid
echo "  serve pid=$SERVE_PID"

# Wait for readiness
SERVE_READY=0
for i in $(seq 1 90); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/v1/models 2>/dev/null | grep -q "200"; then
    echo "  ready after $((i * 2 / 10)).$((i * 2 % 10))s"
    SERVE_READY=1
    break
  fi
  sleep 0.2
done
if [ "$SERVE_READY" -ne 1 ]; then
  echo "ERROR: serve never became ready on port 8765; see /tmp/serve-v10.log"
  tail -20 /tmp/serve-v10.log
  exit 1
fi
echo ""

# Step 5: Eval suite
echo "[5/5] Running eval suite..."
EVAL_LOG="$RUN_DIR/eval.log"

# 5a: fm-fixtures-v2 with tightened v9-compose-v2 prompt
echo "--- v10 vs fm-fixtures-v2 (with v9-compose-v2 prompt) ---" | tee "$EVAL_LOG"
python3 /Users/sarthak/Desktop/fleet/tinygpt/scripts/eval_pace_v2.py \
  --fixtures-dir /Users/sarthak/Desktop/fleet/pace/evals/fm-fixtures-v2 \
  --serve-url http://127.0.0.1:8765/v1/chat/completions \
  --sys-prompt /Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v9-compose-v2.txt \
  2>&1 | tee -a "$EVAL_LOG" | tail -10

# 5b: fm-fixtures-holdout
echo "" | tee -a "$EVAL_LOG"
echo "--- v10 vs fm-fixtures-holdout ---" | tee -a "$EVAL_LOG"
python3 /Users/sarthak/Desktop/fleet/tinygpt/scripts/eval_pace_v2.py \
  --fixtures-dir /Users/sarthak/Desktop/fleet/pace/evals/fm-fixtures-holdout \
  --serve-url http://127.0.0.1:8765/v1/chat/completions \
  --sys-prompt /Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v9-compose-v2.txt \
  2>&1 | tee -a "$EVAL_LOG" | tail -10

# 5c: compose fixtures
echo "" | tee -a "$EVAL_LOG"
echo "--- v10 vs fm-fixtures-compose ---" | tee -a "$EVAL_LOG"
python3 /Users/sarthak/Desktop/fleet/tinygpt/scripts/eval_pace_v2.py \
  --fixtures-dir /Users/sarthak/Desktop/fleet/pace/evals/fm-fixtures-compose \
  --serve-url http://127.0.0.1:8765/v1/chat/completions \
  --sys-prompt /Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v9-compose-v2.txt \
  2>&1 | tee -a "$EVAL_LOG" | tail -10

# 5d: BFCL (smaller sample to keep wall-clock bounded)
echo "" | tee -a "$EVAL_LOG"
echo "--- v10 vs BFCL (15 per category) ---" | tee -a "$EVAL_LOG"
python3 /Users/sarthak/Desktop/fleet/tinygpt/scripts/eval_bfcl.py \
  --max-per-category 15 \
  --serve-url http://127.0.0.1:8765/v1/chat/completions \
  2>&1 | tee -a "$EVAL_LOG" | tail -15

# 5e: formula score
echo "" | tee -a "$EVAL_LOG"
echo "--- v10 formula score ---" | tee -a "$EVAL_LOG"
python3 /Users/sarthak/Desktop/fleet/tinygpt/scripts/score_formula.py \
  --label "v10-DoRA-fp16" \
  --model-dir "$BAKED" \
  --serve-pid "$SERVE_PID" \
  --sys-prompt /Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v9-compose-v2.txt \
  2>&1 | tee -a "$EVAL_LOG" | tail -15

# Cleanup
kill $SERVE_PID 2>/dev/null || true
echo ""
echo "=== v10 pipeline complete $(date) ==="
echo "  artifacts:"
echo "    lora:    $LORA"
echo "    baked:   $BAKED"
echo "    eval:    $EVAL_LOG"
