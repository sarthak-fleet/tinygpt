#!/bin/bash
# CPU-speedup-bundle benchmark runner.
#
# One binary, env-var-toggled feature flags:
#   TINYGPT_DISABLE_COMPILED_LR=1  → item #1 off
#   TINYGPT_DISABLE_FUSED_ACCUM=1  → item #2 off
#   TINYGPT_DISABLE_QOS=1          → item #3 off
#   --prefetch on/off              → item #4 toggle (default off)
#
# All items disabled + cosine+accum=4 + adamw  → matches main HEAD behaviour.
#
# Each config runs 3× and we report median step/s. The heavy flagship-huge-v5
# training is running on the GPU so absolute numbers are noisier than they'd
# be on an idle box; we measure differences, not absolute numbers.

set -euo pipefail

BIN=${BIN:-/tmp/tinygpt-smoke-cpubundle/Build/Products/Release/tinygpt}
STEPS=${STEPS:-60}
BATCH=${BATCH:-8}
PRESET=${PRESET:-tiny}
DATAFILE=$(mktemp /tmp/cpu_bundle_corpus.XXXXXX)
# 2 MB of pseudo-random ASCII so loss is non-degenerate.
head -c $((2*1024*1024)) /dev/urandom | base64 > "$DATAFILE" || true

CORPUS_FLAG="--corpus $DATAFILE"

run_once() {
    local env_str="$1"; shift
    local args="$@"
    local out
    out=$(env $env_str "$BIN" train --preset "$PRESET" --steps "$STEPS" --batch "$BATCH" $CORPUS_FLAG $args 2>&1 | tail -3 | grep "done" || echo "FAIL")
    local sps
    sps=$(echo "$out" | sed -nE 's/.*\(([0-9.]+) step\/s\).*/\1/p')
    if [ -z "$sps" ]; then sps="0.0"; fi
    echo "$sps"
}

median3() {
    printf '%s\n' "$@" | sort -g | awk 'NR==2 {print}'
}

bench_config() {
    local label="$1"; shift
    local env_str="$1"; shift
    local args="$@"
    local s1 s2 s3
    s1=$(run_once "$env_str" $args)
    s2=$(run_once "$env_str" $args)
    s3=$(run_once "$env_str" $args)
    local med
    med=$(median3 "$s1" "$s2" "$s3")
    printf '%-44s  %s / %s / %s  →  %s step/s\n' "$label" "$s1" "$s2" "$s3" "$med"
}

COSINE_ACCUM4="--lr-schedule cosine --warmup 5 --max-lr 3e-4 --min-lr 3e-5 --accum 4"

echo "preset=$PRESET, batch=$BATCH, steps=$STEPS, corpus=$DATAFILE"
echo "binary: $BIN"
echo "concurrent: $(ps -A | grep -i 'tinygpt train' | grep -v grep | wc -l | tr -d ' ') tinygpt train processes alive"
echo "-----------------------------------------------------------------------------"
echo "[A] baseline-on-bundle-binary: cosine+accum=4 with EVERY item off"
bench_config "  all items off (= HEAD baseline)" \
  "TINYGPT_DISABLE_COMPILED_LR=1 TINYGPT_DISABLE_FUSED_ACCUM=1 TINYGPT_DISABLE_QOS=1" \
  $COSINE_ACCUM4

echo
echo "[B] item-by-item incremental gains (each step adds one item to the above)"
bench_config "  +#3 QoS only" \
  "TINYGPT_DISABLE_COMPILED_LR=1 TINYGPT_DISABLE_FUSED_ACCUM=1" \
  $COSINE_ACCUM4
bench_config "  +#1 compile under cosine LR" \
  "TINYGPT_DISABLE_FUSED_ACCUM=1" \
  $COSINE_ACCUM4
bench_config "  +#2 fused accum (item #1+#2+#3)" \
  "" \
  $COSINE_ACCUM4
bench_config "  +#4 prefetch (all four)" \
  "" \
  "$COSINE_ACCUM4 --prefetch on"

echo
echo "[C] item-isolation tests (single-item vs zero-baseline)"
bench_config "  #1 alone (cosine, no accum)" \
  "TINYGPT_DISABLE_QOS=1" \
  "--lr-schedule cosine --warmup 5 --max-lr 3e-4 --min-lr 3e-5"
bench_config "  #2 alone (accum=4, no cosine)" \
  "TINYGPT_DISABLE_QOS=1" \
  "--accum 4"
bench_config "  #3 alone (const LR, no accum)" \
  "" \
  ""

echo
echo "[D] sanity: legacy compile path unchanged (const LR, no accum)"
bench_config "  default constant LR" \
  "" \
  ""
bench_config "  with #3 off" \
  "TINYGPT_DISABLE_QOS=1" \
  ""

rm -f "$DATAFILE"
