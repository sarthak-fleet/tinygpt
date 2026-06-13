#!/usr/bin/env bash
# tinygpt eval-planner — one-command planner drilldown for a new model.
#
#   scripts/eval_planner.sh <lm-studio-model-id> [run-tag] [--compact]
#
# The recurring question this answers: a new model dropped — does it beat
# Pace's current planner? Runs the h2+h2-ext unhappy-path suites
# (n=130: 40 ambig + 60 oos + 30 destructive) against an LM Studio model
# and prints the table vs the stored champion (evals/planner-champion.json).
#
# With --compact, runs the same model twice: once against the standard v11
# system prompt and once against pace-system-prompt-v11-compact (action
# registry collapsed to a one-line index per E9 in docs/PLAN.md). The
# report then shows an A/B delta — does pulling action schemas out of L1
# improve ambig/OOS without regressing action/destructive?
#
# Takes ~30 min on an M5 Pro for a 12B model. Requires LM Studio server
# running with the model downloaded (JIT load is handled here).
set -uo pipefail
TGT="$(cd "$(dirname "$0")/.." && pwd)"
COMPACT=0
POSARGS=()
for arg in "$@"; do
  case "$arg" in
    --compact) COMPACT=1 ;;
    *) POSARGS+=("$arg") ;;
  esac
done
set -- "${POSARGS[@]}"
MODEL="${1:?usage: eval_planner.sh <lm-studio-model-id> [run-tag] [--compact]}"
TAG="${2:-$(echo "$MODEL" | tr '/' '-' | tr -cd 'a-zA-Z0-9._-')}"
SYSP="$TGT/grammars/pace-system-prompt-v11.txt"
SYSP_COMPACT="$TGT/grammars/pace-system-prompt-v11-compact.txt"
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

if [ "$COMPACT" -eq 1 ]; then
  if [ ! -f "$SYSP_COMPACT" ]; then
    echo "[eval-planner] --compact requested but $SYSP_COMPACT is missing." >&2
    exit 1
  fi
  TAG_B="$TAG-compact"
  echo "[eval-planner] re-running n=130 against v11-compact prompt as '$TAG_B'..."
  rm -rf "$HOME/.cache/tinygpt/runs/h2-combined-$TAG_B"
  if ! bash "$TGT/scripts/eval_combined.sh" "$TAG_B" "$URL" "$MODEL" "$SYSP_COMPACT" "$TAG_B"; then
    echo "[eval-planner] compact eval aborted for $MODEL — A/B incomplete." >&2
    exit 1
  fi
  python3 "$TGT/scripts/eval_planner_report.py" "$TAG" --candidate-b "$TAG_B"
else
  python3 "$TGT/scripts/eval_planner_report.py" "$TAG"
fi
