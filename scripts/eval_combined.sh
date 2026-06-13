#!/usr/bin/env bash
# Evaluate one model against the union of h2 + h2-ext on all unhappy dimensions.
# Args: TAG SERVE_URL MODEL_ID SYSP_PATH RUN_NAME
set -euo pipefail
TAG=$1; URL=$2; MODEL=$3; SYSP=$4; RUN_NAME=${5:-$TAG}
RUN_DIR=$HOME/.cache/tinygpt/runs/h2-combined-$RUN_NAME
mkdir -p "$RUN_DIR"
TGT=/Users/sarthak/Desktop/fleet/tinygpt

for SUITE in ambig oos destructive; do
  COMBINED=/tmp/h2-combined-$SUITE
  rm -rf $COMBINED && mkdir -p $COMBINED
  cp $TGT/evals/fm-fixtures-$SUITE-h2/*.txt $COMBINED/
  cp $TGT/evals/fm-fixtures-$SUITE-h2-ext/*.txt $COMBINED/
  python3 $TGT/scripts/eval_pace_unhappy.py \
    --fixtures-dir $COMBINED --serve-url "$URL" \
    ${MODEL:+--model "$MODEL"} --sys-prompt "$SYSP" \
    --out "$RUN_DIR/$SUITE.json" 2>&1 | tail -1
done
