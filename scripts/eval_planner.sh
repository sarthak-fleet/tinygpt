#!/usr/bin/env bash
# tinygpt eval-planner — one-command planner drilldown for a new model.
#
#   scripts/eval_planner.sh <lm-studio-model-id> [run-tag]
#
# The recurring question this answers: a new model dropped — does it beat
# Pace's current planner? Runs the h2+h2-ext unhappy-path suites
# (n=130: 40 ambig + 60 oos + 30 destructive) against an LM Studio model
# and prints the table vs the stored champion (evals/planner-champion.json).
#
# Takes ~30 min on an M5 Pro for a 12B model. Requires LM Studio server
# running with the model downloaded (JIT load is handled here).
set -uo pipefail
TGT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: eval_planner.sh <lm-studio-model-id> [run-tag]}"
TAG="${2:-$(echo "$MODEL" | tr '/' '-' | tr -cd 'a-zA-Z0-9._-')}"
SYSP="$TGT/grammars/pace-system-prompt-v11.txt"
URL="http://127.0.0.1:1234/v1/chat/completions"

if ! curl -s -m 3 http://127.0.0.1:1234/v1/models > /dev/null; then
  echo "LM Studio server not responding on :1234 — run 'lms server start'." >&2
  exit 1
fi

echo "[eval-planner] warming $MODEL (JIT load — large models take a minute)..."
loaded=0
for _ in $(seq 1 30); do
  out=$(curl -s -m 20 "$URL" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":3,\"temperature\":0}" 2>&1)
  if echo "$out" | grep -q '"choices"'; then loaded=1; break; fi
  sleep 4
done
if [ "$loaded" -ne 1 ]; then
  echo "[eval-planner] $MODEL never answered. Is it downloaded? ('lms ls' to check, 'lms get $MODEL --mlx' to fetch)" >&2
  exit 1
fi

echo "[eval-planner] running 3 suites (n=130) as run-tag '$TAG'..."
# Clear any previous run under this tag — a partial/aborted earlier run
# must not survive to be misread as fresh results by the report.
rm -rf "$HOME/.cache/tinygpt/runs/h2-combined-$TAG"
if ! bash "$TGT/scripts/eval_combined.sh" "$TAG" "$URL" "$MODEL" "$SYSP" "$TAG"; then
  echo "[eval-planner] eval aborted for $MODEL — no verdict (NOT a 0% score)." >&2
  exit 1
fi

python3 "$TGT/scripts/eval_planner_report.py" "$TAG"
