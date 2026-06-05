#!/usr/bin/env bash
# N01-pull-datasets.sh — verify and prepare the corpora A1 needs.
#
# Honest current state (2026-06-05):
#   ✅ Gutenberg corpus (32 MB clean text) via fetch_corpora.sh        — pretrain
#   ✅ hermes-function-calling-v1 (~50 MB JSONL)                        — SFT
#   ✅ SmolLM2-135M tokenizer in ~/.cache/huggingface/hub               — BPE
#   ❌ FineWeb-Edu (parquet — tinygpt has no parquet decoder yet)        — pretrain v2
#   ❌ xlam-function-calling-60k (gated — needs HF_TOKEN)                — SFT v2
#   ❌ UltraFeedback (parquet)                                           — DPO
#   ❌ BFCL (cache present, flat-emit broken)                            — eval
#
# This script verifies the ✅ items are in place. The ❌ items are
# tracked in NIGHTLY.md as "blocked by parquet support / HF_TOKEN /
# flat-emit bug" — they unblock more ambitious base + DPO + eval runs.

set -euo pipefail

CACHE="$HOME/.cache/tinygpt/datasets"
GUTENBERG_DIR="/tmp/tinygpt-corpora"
EVERYTHING="$GUTENBERG_DIR/everything.txt"
TOKENIZER_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"

ok=0
warn=0

check() {
    if [[ "$1" == "ok" ]]; then
        echo "  ✓ $2"
        ok=$((ok+1))
    else
        echo "  ✗ $2"
        warn=$((warn+1))
    fi
}

echo "=== [1/3] Gutenberg pretrain corpus ==="
if [[ ! -d "$GUTENBERG_DIR" ]] || ! ls "$GUTENBERG_DIR"/*.txt >/dev/null 2>&1; then
    echo "  fetching via scripts/fetch_corpora.sh..."
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/fetch_corpora.sh"
fi
if ls "$GUTENBERG_DIR"/*.txt >/dev/null 2>&1; then
    if [[ ! -f "$EVERYTHING" ]] || [[ "$EVERYTHING" -ot "$GUTENBERG_DIR/shakespeare-complete.txt" ]]; then
        echo "  concatenating → $EVERYTHING"
        cat "$GUTENBERG_DIR"/*.txt > "$EVERYTHING"
    fi
    SZ=$(wc -c < "$EVERYTHING")
    check ok "Gutenberg combined: $EVERYTHING ($SZ bytes)"
else
    check fail "Gutenberg corpus missing"
fi

echo "=== [2/3] SmolLM2-135M tokenizer ==="
if [[ -d "$TOKENIZER_DIR" ]] && find "$TOKENIZER_DIR/snapshots" -name "tokenizer.json" 2>/dev/null | grep -q .; then
    check ok "SmolLM2 tokenizer cached: $TOKENIZER_DIR"
else
    echo "  needs huggingface-cli login + download. See NIGHTLY.md for the one-time setup."
    check fail "SmolLM2 tokenizer missing"
fi

echo "=== [3/3] SFT corpus (hermes-fc) ==="
if [[ -f "$CACHE/hermes-fc.jsonl" ]] && [[ -s "$CACHE/hermes-fc.jsonl" ]]; then
    SZ=$(wc -c < "$CACHE/hermes-fc.jsonl")
    LINES=$(wc -l < "$CACHE/hermes-fc.jsonl")
    check ok "hermes-fc.jsonl: $SZ bytes, $LINES lines"
else
    check fail "hermes-fc.jsonl missing or empty — run \`tinygpt download-dataset NousResearch/hermes-function-calling-v1 --format sft --out $CACHE/hermes-fc.jsonl\`"
fi

echo ""
echo "=== summary ==="
echo "  $ok ready, $warn missing"

if [[ "$warn" -gt 0 ]]; then
    echo ""
    echo "Missing items above block A1. Resolve them and re-run N01."
    exit 1
fi

cat <<'NOTE'

Known parquet/auth-blocked corpora (NIGHTLY.md tracks these as v2 work):
  - FineWeb-Edu  → 2.4 GB parquet on disk; needs a parquet→txt decoder
  - UltraFeedback → 5 parquet shards on disk; needs the same decoder
  - xlam-function-calling-60k → gated, needs HF_TOKEN (huggingface-cli login)
  - BFCL → flat-emit hit a bug; 0-byte output

N01 verifies the unblocked path. Proceed to N02 (huge-base pretrain).
NOTE
