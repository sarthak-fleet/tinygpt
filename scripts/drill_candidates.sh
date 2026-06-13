#!/usr/bin/env bash
# Sequenced evaluation of: 4B + two-stage shim, then four new candidate
# bases. Each gets the full combined h2 + h2-ext suite (ambig 40, oos 60,
# destructive 30). Run after the current verification chain finishes.
set -euo pipefail
TGT=/Users/sarthak/Desktop/fleet/tinygpt
SYSP="$TGT/grammars/pace-system-prompt-v11.txt"

cleanup() {
  pkill -f "two_stage_shim" 2>/dev/null || true
  pkill -f "tinygpt serve" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until LM Studio is idle on whatever was loaded.
sleep 2
echo "=== 1. 4B + two-stage shim ==="
lms load qwen3-4b-instruct-2507 2>&1 | tail -1 || true
sleep 3
python3 "$TGT/scripts/two_stage_shim.py" --upstream http://127.0.0.1:1234/v1/chat/completions \
  --model qwen3-4b-instruct-2507 --port 8769 > /tmp/two_stage_shim.log 2>&1 &
sleep 3
bash "$TGT/scripts/eval_combined.sh" two-stage-4b http://127.0.0.1:8769/v1/chat/completions \
  "qwen3-4b-instruct-2507" "$SYSP" two-stage-4b
pkill -f two_stage_shim 2>/dev/null || true
sleep 2

# Each candidate: load, warm, eval, unload.
for MODEL in \
    "qwen3-4b-thinking-2507-4bit|mlx-community/Qwen3-4B-Thinking-2507-4bit|qwen3-4b-thinking" \
    "deepseek-r1-distill-qwen-7b-4bit|mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit|deepseek-r1-7b" \
    "meta-llama-3.1-8b-instruct-4bit|mlx-community/Meta-Llama-3.1-8B-Instruct-4bit|llama-3.1-8b" \
    "gemma-3-12b-it-qat-4bit|mlx-community/gemma-3-12b-it-qat-4bit|gemma-3-12b"; do
  IFS='|' read -r MID MPATH TAG <<< "$MODEL"
  echo "=== $TAG ($MID) ==="
  lms unload --all 2>&1 | tail -1 || true
  sleep 2
  lms load "$MPATH" 2>&1 | tail -1 || true
  sleep 3
  # warmup
  curl -s -m 30 http://127.0.0.1:1234/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" > /dev/null || true
  bash "$TGT/scripts/eval_combined.sh" "$TAG" http://127.0.0.1:1234/v1/chat/completions "$MID" "$SYSP" "$TAG"
done

echo "=== drill_candidates done ==="
