#!/usr/bin/env bash
# N01-pull-datasets.sh — fetch every dataset + tokenizer A1 needs.
#
# Runs in ~30-60 min depending on network. Idempotent: anything already
# under ~/.cache/tinygpt/datasets or ~/.cache/huggingface gets skipped.
#
# Outputs:
#   ~/.cache/tinygpt/datasets/fineweb-edu.sample.jsonl   pretrain corpus
#   ~/.cache/tinygpt/datasets/xlam.jsonl                 SFT (tool-calling)
#   ~/.cache/tinygpt/datasets/hermes-fc.jsonl            SFT (tool-calling)
#   ~/.cache/tinygpt/datasets/ultrafeedback.jsonl        DPO (preference)
#   ~/.cache/tinygpt/datasets/bfcl.jsonl                 eval (BFCL)
#   ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M  tokenizer

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TINYGPT="$REPO_ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt"
CACHE="$HOME/.cache/tinygpt/datasets"
mkdir -p "$CACHE"

# Sanity: binary exists.
if [[ ! -x "$TINYGPT" ]]; then
    echo "tinygpt binary not found at $TINYGPT — run \`cd native-mac && swift build -c release\` first" >&2
    exit 1
fi

echo "=== [1/6] FineWeb-Edu sample (pretrain corpus, ~200-500 MB) ==="
if [[ ! -f "$CACHE/fineweb-edu.sample.jsonl" ]]; then
    "$TINYGPT" download-dataset HuggingFaceFW/fineweb-edu \
        --format plain --max-files 1 \
        --out "$CACHE/fineweb-edu.sample.jsonl"
else
    echo "  already cached, skipping"
fi

echo "=== [2/6] xlam-function-calling-60k (SFT) ==="
if [[ ! -f "$CACHE/xlam.jsonl" ]]; then
    "$TINYGPT" download-dataset Salesforce/xlam-function-calling-60k \
        --format sft --out "$CACHE/xlam.jsonl"
else
    echo "  already cached, skipping"
fi

echo "=== [3/6] hermes-function-calling-v1 (SFT) ==="
if [[ ! -f "$CACHE/hermes-fc.jsonl" ]]; then
    "$TINYGPT" download-dataset NousResearch/hermes-function-calling-v1 \
        --format sft --out "$CACHE/hermes-fc.jsonl"
else
    echo "  already cached, skipping"
fi

echo "=== [4/6] ultrafeedback-binarized (DPO) ==="
if [[ ! -f "$CACHE/ultrafeedback.jsonl" ]]; then
    "$TINYGPT" download-dataset HuggingFaceH4/ultrafeedback_binarized \
        --format dpo --out "$CACHE/ultrafeedback.jsonl"
else
    echo "  already cached, skipping"
fi

echo "=== [5/6] BFCL (eval) ==="
if [[ ! -f "$CACHE/bfcl.jsonl" ]]; then
    "$TINYGPT" download-dataset gorilla-llm/Berkeley-Function-Calling-Leaderboard \
        --format plain --out "$CACHE/bfcl.jsonl"
else
    echo "  already cached, skipping"
fi

echo "=== [6/6] SmolLM2-135M tokenizer ==="
# Use huggingface-cli if available; fall back to skipping (the user can
# pull it manually). SmolLM2's 49K-vocab BPE is the right size for our
# 10M-param Huge model (vs Qwen3's 150K which dominates the param budget).
TOKENIZER_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"
if [[ ! -d "$TOKENIZER_DIR" ]]; then
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download HuggingFaceTB/SmolLM2-135M \
            tokenizer.json tokenizer_config.json config.json special_tokens_map.json
    else
        echo "  WARN: huggingface-cli not found." >&2
        echo "        Run: pip install huggingface_hub && huggingface-cli login" >&2
        echo "        Then re-run N01. N02 needs this tokenizer." >&2
        exit 1
    fi
else
    echo "  already cached at $TOKENIZER_DIR"
fi

echo ""
echo "=== summary ==="
du -sh "$CACHE" 2>/dev/null || true
ls -la "$CACHE"
echo "tokenizer: $TOKENIZER_DIR"
echo ""
echo "N01 complete. Next job: N02-huge-base-v1.sh"
