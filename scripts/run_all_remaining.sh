#!/usr/bin/env bash
# Run the remaining models against the combined h2 + h2-ext suite:
# Apple FM -> clarify-v1 -> v9 -> v11
# (4B and 14B already done in separate runs)
set -euo pipefail
TGT=/Users/sarthak/Desktop/fleet/tinygpt
TINYGPT="$TGT/native-mac/.build/arm64-apple-macosx/release/tinygpt"
BASE_0_6B=/Users/sarthak/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
GRAMMAR="$TGT/grammars/pace-fm-response-v11.schema.json"
SYSP="$TGT/grammars/pace-system-prompt-v11.txt"

# --- 1. Apple FM (guided bridge) ---
echo "=== Apple FM ==="
pkill -f fm_shim 2>/dev/null || true
pkill -f "tinygpt serve" 2>/dev/null || true
sleep 2
(python3 "$TGT/scripts/fm_shim.py" --port 8766 > /tmp/fm_shim.log 2>&1 &)
sleep 4
bash "$TGT/scripts/eval_combined.sh" apple-fm http://127.0.0.1:8766/v1/chat/completions apple-foundation-models "$SYSP" apple-fm
pkill -f fm_shim 2>/dev/null || true
sleep 2

# --- 2. clarify-v1 (4B fine-tune, baked) ---
echo "=== clarify-v1 (4B fine-tune) ==="
"$TINYGPT" serve "$HOME/.cache/tinygpt/runs/clarify-v1/baked-hf" --grammar "$GRAMMAR" --port 8770 > /tmp/serve-cv1.log 2>&1 &
for i in $(seq 1 90); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8770/v1/models 2>/dev/null | grep -q 200 && break; sleep 0.5; done
bash "$TGT/scripts/eval_combined.sh" clarify-v1 http://127.0.0.1:8770/v1/chat/completions "" "$SYSP" clarify-v1
pkill -f "tinygpt serve" 2>/dev/null || true
sleep 2

# --- 3. v9-LoRA (0.6B + Pace's shipped adapter) ---
echo "=== v9-LoRA ==="
"$TINYGPT" serve "$BASE_0_6B" --lora "$HOME/.cache/tinygpt/runs/pace-planner-v9-lora/pace-planner-v9-lora.lora" --grammar "$GRAMMAR" --port 8771 > /tmp/serve-v9.log 2>&1 &
for i in $(seq 1 90); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8771/v1/models 2>/dev/null | grep -q 200 && break; sleep 0.5; done
bash "$TGT/scripts/eval_combined.sh" v9-lora http://127.0.0.1:8771/v1/chat/completions "" "$SYSP" v9-lora
pkill -f "tinygpt serve" 2>/dev/null || true
sleep 2

# --- 4. v11-LoRA (0.6B + the failed specialist) ---
echo "=== v11-LoRA (the discarded specialist) ==="
"$TINYGPT" serve "$BASE_0_6B" --lora "$HOME/.cache/tinygpt/runs/pace-planner-v11/pace-planner-v11-plain.lora" --grammar "$GRAMMAR" --port 8772 > /tmp/serve-v11.log 2>&1 &
for i in $(seq 1 90); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8772/v1/models 2>/dev/null | grep -q 200 && break; sleep 0.5; done
bash "$TGT/scripts/eval_combined.sh" v11-lora http://127.0.0.1:8772/v1/chat/completions "" "$SYSP" v11-lora
pkill -f "tinygpt serve" 2>/dev/null || true

echo "=== ALL DONE ==="
