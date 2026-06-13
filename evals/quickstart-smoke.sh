#!/usr/bin/env bash
# B33 smoke: `tinygpt quickstart --dry-run` resolves a sane (base, recipe)
# plan from fixture data and emits a valid project manifest — no GPU, no
# training. Verifies the exit-code + plan contract the CLI promises.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIX="$ROOT/evals/quickstart-fixtures"
NATIVE="$ROOT/native-mac"

BIN=""
for cand in "$NATIVE/.build/release/tinygpt" "$NATIVE/.build/debug/tinygpt"; do
  [ -x "$cand" ] && BIN="$cand" && break
done
if [ -z "$BIN" ]; then
  echo "no built binary found — building debug…"
  (cd "$NATIVE" && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build)
  BIN="$NATIVE/.build/debug/tinygpt"
fi
echo "binary: $BIN"

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

# 1) chat data → shape=chat, picks the chat-tagged base, emits adapter pin
echo "--- chat data ---"
out="$("$BIN" quickstart "$FIX/sample-chat.jsonl" --gallery "$FIX/gallery.json" --dry-run)" \
  || fail "chat dry-run exited non-zero"
echo "$out"
echo "$out" | grep -q "shape=chat"      || fail "expected shape=chat"
echo "$out" | grep -q "qwen3-0.6b-it"   || fail "expected chat base qwen3-0.6b-it (smaller, chat-tagged)"
echo "$out" | grep -q -- "--rank"       || fail "expected an sft recipe with --rank"
echo "$out" | grep -q "applies_to"      || fail "expected an adapter pin (applies_to) in the project preview"

# 2) tool-call data → shape=toolCall, longer max-seq, picks the tool-tagged base
echo "--- tool-call data ---"
out="$("$BIN" quickstart "$FIX/sample-toolcalls.jsonl" --gallery "$FIX/gallery.json" --dry-run)" \
  || fail "tool dry-run exited non-zero"
echo "$out" | grep -q "shape=toolCall"  || fail "expected shape=toolCall"
echo "$out" | grep -q "max-seq=2048"    || fail "expected max-seq=2048 for tool-call data"
echo "$out" | grep -q "qwen3-4b-tool"   || fail "expected tool base qwen3-4b-tool"

# 3) unreadable data → non-zero exit with guidance
echo "--- missing data file ---"
if "$BIN" quickstart "$FIX/does-not-exist.jsonl" --dry-run >/dev/null 2>&1; then
  fail "expected non-zero exit on a missing data file"
fi

echo "SMOKE PASS"
