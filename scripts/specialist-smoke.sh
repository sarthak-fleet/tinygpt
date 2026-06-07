#!/usr/bin/env bash
# scripts/specialist-smoke.sh — end-to-end smoke for the specialist arc.
#
# Usage:
#   ./scripts/specialist-smoke.sh <hf-base-dir> <adapter.lora> [prompt]
#
# Loads the HF base + applies the LoRA adapter + samples a short
# completion. Confirms the full train→apply→inference loop works.
#
# Requires: tinygpt binary built (`cd native-mac && swift build -c release`).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINYGPT="$REPO_ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <hf-base-dir> <adapter.lora> [prompt]" >&2
    echo "" >&2
    echo "example (after Qwen3 FC SFT completes):" >&2
    echo "  $0 ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de... \\" >&2
    echo "     ~/.cache/tinygpt/runs/qwen3-fc-v2/qwen3-fc-v2.lora \\" >&2
    echo "     'What is 2 + 2?'" >&2
    exit 1
fi

BASE_DIR="$1"
LORA_PATH="$2"
PROMPT="${3:-What is 2 + 2?}"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "error: base dir not found: $BASE_DIR" >&2
    exit 1
fi
if [[ ! -f "$LORA_PATH" ]]; then
    echo "error: lora not found: $LORA_PATH" >&2
    exit 1
fi
if [[ ! -x "$TINYGPT" ]]; then
    echo "error: tinygpt binary missing — run 'cd native-mac && swift build -c release'" >&2
    exit 1
fi

echo "=== specialist smoke ==="
echo "  base:   $BASE_DIR"
echo "  lora:   $LORA_PATH (size $(du -h "$LORA_PATH" | cut -f1))"
echo "  prompt: $PROMPT"
echo ""

echo "--- baseline (no LoRA) ---"
"$TINYGPT" hf-load "$BASE_DIR" --sample --prompt "$PROMPT" --tokens 50 --temperature 0.0 2>&1 | tail -10
echo ""
echo "--- with LoRA ---"
"$TINYGPT" hf-load "$BASE_DIR" --lora "$LORA_PATH" --sample --prompt "$PROMPT" --tokens 50 --temperature 0.0 2>&1 | tail -10
echo ""
echo "=== smoke complete ==="
