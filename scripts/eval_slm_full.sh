#!/usr/bin/env bash
# eval_slm_full.sh — one model → all four leaderboard suites in one run.
#
#   scripts/eval_slm_full.sh <lm-studio-model-id> <tag>
#
# Runs (in order):
#   1. scripts/eval_planner.sh <model> <tag>        # unhappy-paths n=130
#   2. tinygpt eval-bfcl --base-url … --out …       # BFCL 10 categories
#   3. tinygpt eval-tau-bench --base-url … --out …  # retail + airline
#   4. scripts/bench_decode.py --url … --model …    # decode + RSS
#
# Produces per-suite artifacts under docs/research/data/<tag>/ and
# updates ~/.cache/tinygpt/runs/h2-combined-<tag>/. Compose them into
# the leaderboard table with:
#
#   scripts/build_slm_leaderboard.py \
#       --manifest docs/research/data/leaderboard_manifest.json
#
# Long-running (~45-90 min on M5 Pro for a 12B model). Skips any suite
# that already has an artifact unless --force is passed.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
URL_CHAT="http://127.0.0.1:1234/v1/chat/completions"
URL_BASE="http://127.0.0.1:1234/v1"
DATA_DIR="$REPO/docs/research/data"
TINYGPT="$REPO/native-mac/.build/arm64-apple-macosx/release/tinygpt"

FORCE=0
POSARGS=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) POSARGS+=("$arg") ;;
  esac
done
set -- "${POSARGS[@]}"

MODEL="${1:?usage: eval_slm_full.sh <lm-studio-model-id> <tag> [--force]}"
TAG="${2:?usage: eval_slm_full.sh <lm-studio-model-id> <tag> [--force]}"
OUT="$DATA_DIR/$TAG"
mkdir -p "$OUT"

UNHAPPY_JSON="$HOME/.cache/tinygpt/runs/h2-combined-$TAG/ambig.json"
BFCL_JSON="$OUT/bfcl.json"
TAU_JSON="$OUT/tau.json"
DECODE_JSON="$OUT/decode.json"

skip_or_run() {
  local label="$1"; local marker="$2"; shift 2
  if [ "$FORCE" -ne 1 ] && [ -e "$marker" ]; then
    echo "[$label] skip (artifact $marker exists; pass --force to re-run)"
    return 0
  fi
  echo "[$label] running"
  "$@"
}

# --- 1. unhappy-paths ---------------------------------------------------
skip_or_run "unhappy" "$UNHAPPY_JSON" bash "$REPO/scripts/eval_planner.sh" "$MODEL" "$TAG"

# --- 2. BFCL ------------------------------------------------------------
# tinygpt eval-bfcl writes to a temp dir; we copy its summary JSON to a
# stable path so the leaderboard manifest can point at it.
if [ ! -x "$TINYGPT" ]; then
  echo "[bfcl/tau] tinygpt binary missing — build it with:" >&2
  echo "    cd native-mac && swift build -c release" >&2
  echo "[bfcl/tau] skipping these two suites; leaderboard will show '—' for them." >&2
else
  skip_or_run "bfcl" "$BFCL_JSON" \
    "$TINYGPT" eval-bfcl \
      --base-url "$URL_BASE" --model "$MODEL" \
      --out "$BFCL_JSON" || \
      echo "[bfcl] failed — leaderboard column stays '—' for this model."

  skip_or_run "tau-bench" "$TAU_JSON" \
    "$TINYGPT" eval-tau-bench \
      --base-url "$URL_BASE" --model "$MODEL" \
      --out "$TAU_JSON" || \
      echo "[tau-bench] failed — leaderboard column stays '—' for this model."
fi

# --- 3. decode bench ----------------------------------------------------
# RSS-poll target: the LM Studio inference connector if present, else
# the lms ps output's PID column. The simple heuristic below catches
# every variant seen in the wild.
RSS_PID=$(pgrep -f 'lmlink-connector.*metal' 2>/dev/null | head -1 || true)
RSS_FLAG=()
[ -n "$RSS_PID" ] && RSS_FLAG=(--rss-pid "$RSS_PID")

skip_or_run "decode" "$DECODE_JSON" \
  bash -c "python3 '$REPO/scripts/bench_decode.py' \
    --url '$URL_CHAT' --model '$MODEL' \
    ${RSS_FLAG[*]} > '$DECODE_JSON.tmp' && mv '$DECODE_JSON.tmp' '$DECODE_JSON'"

echo
echo "[eval_slm_full] done; artifacts under $OUT/"
echo "  next: add a row to docs/research/data/leaderboard_manifest.json:"
cat <<EOF
    {"label": "$TAG", "params": "?B",
     "unhappy_tag": "$TAG",
     "bfcl_json":   "docs/research/data/$TAG/bfcl.json",
     "tau_json":    "docs/research/data/$TAG/tau.json",
     "decode_json": "docs/research/data/$TAG/decode.json"}
EOF
echo "  then: python3 scripts/build_slm_leaderboard.py --manifest docs/research/data/leaderboard_manifest.json"
