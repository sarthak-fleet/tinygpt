#!/usr/bin/env bash
# v11 cascade: [amplify] → merge → train → bake → serve → 6-dim ship-gate eval
#
# ONE COMMAND to produce a v11 verdict against docs/prds/pace-planner-v11-ship-gate.md.
#
# Usage:
#   bash scripts/v11_pipeline.sh                # train on whatever corpus exists
#   bash scripts/v11_pipeline.sh --amplify     # run the thinking-teacher amplifier first
#                                              # (needs LM Studio + Qwen3-14B loaded, ~90min extra)
#
# Preconditions:
#   - GPU free (no other training / no other agent using LM Studio)
#   - ~/.cache/tinygpt/datasets/pace-v10-multiplied.jsonl exists
#   - ~/.cache/tinygpt/datasets/pace-v11-seed.jsonl exists
#   - for --amplify: LM Studio running with qwen/qwen3-14b loaded
#
# Wall: ~90min train + ~5min bake + ~25min eval  (+~90min if --amplify)

set -euo pipefail

BASE=/Users/sarthak/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
DS=/Users/sarthak/.cache/tinygpt/datasets
RUN_DIR=/Users/sarthak/.cache/tinygpt/runs/pace-planner-v11
LORA="$RUN_DIR/pace-planner-v11.lora"
BAKED="$RUN_DIR/baked-hf"
LOG="$RUN_DIR/train.log"
TGT=/Users/sarthak/Desktop/fleet/tinygpt
TINYGPT="$TGT/native-mac/.build/release/tinygpt"
SYSP="$TGT/grammars/pace-system-prompt-v11.txt"
GRAMMAR="$TGT/grammars/pace-fm-response-v11.schema.json"
PACE_EVAL=/Users/sarthak/Desktop/fleet/pace/evals

mkdir -p "$RUN_DIR"

echo "=== v11 pipeline start $(date) ==="

# Step 0 (optional): amplify
if [[ "${1:-}" == "--amplify" ]]; then
  echo "[0/6] Running thinking-teacher amplifier (~90min)..."
  python3 -u "$TGT/scripts/v11-amplify.py" \
    --seeds "$DS/pace-v11-seed.jsonl" \
    --out   "$DS/pace-v11-amplified.jsonl" \
    --cooldown-seconds 5
  echo ""
fi

# Step 1: merge corpus
echo "[1/6] Merging v11 corpus..."
python3 "$TGT/scripts/build-v11-corpus.py"
DATA="$DS/pace-v11-train.jsonl"
DATA_ROWS=$(wc -l < "$DATA")
echo "  corpus: $DATA ($DATA_ROWS rows)"
echo ""

# Step 2: free the GPU
echo "[2/6] Unloading LM Studio models..."
lms unload --all 2>/dev/null || true
echo ""

# Step 3: train
# Plain LoRA, NOT DoRA: the DoRA inference path is broken end-to-end as of
# 2026-06-10 (serve-apply degenerates generations, bake-lora rejects
# magnitudes). v9 shipped on plain LoRA; stay there until the DoRA gate task
# is fixed and verified.
echo "[3/6] Training v11 (plain LoRA, rank 32, alpha 64, 3000 steps)..."
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
  --no-dora > "$LOG" 2>&1
  # --no-dora is REQUIRED, not optional: sft defaults useDora=true, so
  # merely omitting --dora still trains DoRA (discovered 2026-06-10).

if [ ! -f "$LORA" ]; then
  echo "ERROR: training failed"; tail -20 "$LOG"; exit 1
fi
echo "  trained: $LORA ($(ls -la "$LORA" | awk '{print $5}') bytes)"
echo ""

# Step 4: bake
echo "[4/6] Baking LoRA into merged HF dir..."
"$TINYGPT" bake-lora "$BASE" "$LORA" --out "$BAKED" 2>&1 | tail -3
echo ""

# Step 5: serve with v11 grammar + prompt
echo "[5/6] Starting serve..."
pkill -f "tinygpt serve" 2>/dev/null || true
sleep 2
mkdir -p /tmp/tinygpt-cache-v11
"$TINYGPT" serve \
  "$BASE" --lora "$LORA" \
  --grammar "$GRAMMAR" \
  --prompt-cache-dir /tmp/tinygpt-cache-v11 \
  --port 8765 > /tmp/serve-v11.log 2>&1 &
SERVE_PID=$!
echo $SERVE_PID > "$RUN_DIR/serve.pid"
for i in $(seq 1 90); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/v1/models 2>/dev/null | grep -q 200; then
    echo "  ready"; break
  fi
  sleep 0.2
done
echo ""

# Step 6: the full 6-dimension ship-gate eval
echo "[6/6] Ship-gate eval (6 dimensions + non-regression)..."
EVAL_LOG="$RUN_DIR/eval.log"
SERVE_URL=http://127.0.0.1:8765/v1/chat/completions

run_dim () {  # name, fixtures-dir
  echo "--- $1 ---" | tee -a "$EVAL_LOG"
  python3 "$TGT/scripts/eval_pace_v2.py" \
    --fixtures-dir "$2" --serve-url "$SERVE_URL" --sys-prompt "$SYSP" \
    2>&1 | tee -a "$EVAL_LOG" | tail -3
  echo "" | tee -a "$EVAL_LOG"
}

run_unhappy () {  # name, fixtures-dir
  echo "--- $1 ---" | tee -a "$EVAL_LOG"
  python3 "$TGT/scripts/eval_pace_unhappy.py" \
    --fixtures-dir "$2" --serve-url "$SERVE_URL" --sys-prompt "$SYSP" \
    --out "$RUN_DIR/eval-$1.json" \
    2>&1 | tee -a "$EVAL_LOG" | tail -3
  echo "" | tee -a "$EVAL_LOG"
}

: > "$EVAL_LOG"
# Dim 1: happy path
run_dim "dim1-fm-fixtures-v2" "$PACE_EVAL/fm-fixtures-v2"
# Dim 2: BFCL pace-12
echo "--- dim2-bfcl-pace12 ---" | tee -a "$EVAL_LOG"
python3 "$TGT/scripts/eval_bfcl.py" \
  --serve-url "$SERVE_URL" \
  --bfcl-dir "$HOME/.cache/tinygpt/datasets/bfcl" \
  --categories pace12 \
  --out "$RUN_DIR/eval-bfcl-pace12.json" \
  2>&1 | tee -a "$EVAL_LOG" | tail -6
echo "" | tee -a "$EVAL_LOG"
# Dims 3, 4, 6: unhappy paths
run_unhappy "dim3-oos"         "$PACE_EVAL/fm-fixtures-oos"
run_unhappy "dim4-ambig"       "$PACE_EVAL/fm-fixtures-ambig"
run_unhappy "dim6-destructive" "$PACE_EVAL/fm-fixtures-destructive"
# Non-regression: compose + holdout
run_dim "nonreg-compose" "$PACE_EVAL/fm-fixtures-compose"
run_dim "nonreg-holdout" "$PACE_EVAL/fm-fixtures-holdout"
# Formula score
echo "--- formula score ---" | tee -a "$EVAL_LOG"
python3 "$TGT/scripts/score_formula.py" \
  --label "v11-DoRA-fp16" \
  --model-dir "$BAKED" \
  --serve-pid "$SERVE_PID" \
  --sys-prompt "$SYSP" \
  2>&1 | tee -a "$EVAL_LOG" | tail -10

kill $SERVE_PID 2>/dev/null || true
echo ""
echo "=== v11 pipeline complete $(date) ==="
echo "  eval log: $EVAL_LOG"
echo ""
echo "  NOW: compare against docs/prds/pace-planner-v11-ship-gate.md thresholds."
echo "  Dim5 (schema validity) = check AST-valid rate inside the BFCL output JSON."
echo "  Ship iff ALL six >= threshold AND compose >= 65% AND TTFW <= 140ms."