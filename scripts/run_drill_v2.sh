#!/usr/bin/env bash
# Clean drill driver — no `lms unload --all` race, single-load per model.
# Output: per-model {ambig, oos, destructive}.json in
#   ~/.cache/tinygpt/runs/h2-combined-<tag>/
set -uo pipefail
TGT=/Users/sarthak/Desktop/fleet/tinygpt
SYSP="$TGT/grammars/pace-system-prompt-v11.txt"
LOG=/tmp/drill_v2.log
echo "drill_v2 starting at $(date)" > "$LOG"

# Wait until a freshly loaded model answers a smoke prompt.
wait_for_model() {
  local id=$1
  for i in $(seq 1 30); do
    local out
    out=$(curl -s -m 20 http://127.0.0.1:1234/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$id\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":3,\"temperature\":0}" 2>&1)
    if echo "$out" | grep -q '"choices"'; then return 0; fi
    sleep 4
  done
  return 1
}

run_eval() {
  local TAG=$1 URL=$2 MODEL=$3 RUN=$4
  echo "[$(date +%H:%M:%S)] EVAL $TAG via $URL model=$MODEL" | tee -a "$LOG"
  bash "$TGT/scripts/eval_combined.sh" "$TAG" "$URL" "$MODEL" "$SYSP" "$RUN" 2>&1 | tee -a "$LOG"
}

# Stage 1: two-stage shim wrapping qwen3-4b-instruct (already loaded).
echo "[$(date +%H:%M:%S)] STAGE 1: two-stage-v2 on 4B" | tee -a "$LOG"
pkill -f two_stage_shim.py 2>/dev/null || true
sleep 1
python3 "$TGT/scripts/two_stage_shim.py" \
  --upstream http://127.0.0.1:1234/v1/chat/completions \
  --model qwen3-4b-instruct-2507 --port 8769 \
  > /tmp/two_stage_shim.log 2>&1 &
SHIM_PID=$!
sleep 2
run_eval two-stage-v2 http://127.0.0.1:8769/v1/chat/completions qwen3-4b-instruct-2507 two-stage-v2
kill $SHIM_PID 2>/dev/null || true
sleep 2

# Stage 2: each candidate model — load, warm, eval. NO unload between
# (LM Studio handles JIT swap; manual unload was the wedge cause).
for MODEL in \
    "qwen3-4b-thinking-2507|qwen3-4b-thinking" \
    "deepseek-r1-distill-qwen-7b|deepseek-r1-7b" \
    "meta-llama-3.1-8b-instruct|llama-3.1-8b" \
    "google/gemma-3-12b|gemma-3-12b" ; do
  IFS='|' read -r MID TAG <<< "$MODEL"
  echo "[$(date +%H:%M:%S)] STAGE 2: $TAG (id=$MID) — loading" | tee -a "$LOG"
  # JIT-load via API: send a tiny request, LM Studio loads on first hit.
  if ! wait_for_model "$MID"; then
    echo "[$(date +%H:%M:%S)] $TAG: load failed, skipping" | tee -a "$LOG"
    continue
  fi
  run_eval "$TAG" http://127.0.0.1:1234/v1/chat/completions "$MID" "$TAG"
done

echo "[$(date +%H:%M:%S)] drill_v2 complete" | tee -a "$LOG"
