#!/usr/bin/env bash
# distill-specialist.sh — compress a big model's capability on YOUR task into a
# small, local, deployable model. The validated cost-compression recipe
# (proven: a 1.7B matched a 4B teacher on tool-calling at ~2.3x smaller; a 0.6B
# matched function-selection at ~7x smaller). Runs entirely on the Mac via MLX.
#
# Usage:
#   distill-specialist.sh <data-dir> <student-model> <out-dir> [iters]
#
#   <data-dir>      dir with train.jsonl (+ optional valid.jsonl) in mlx_lm
#                   chat format: {"messages":[{role,content},...]}. Keep each
#                   example <= ~5500 chars so nothing truncates past max-seq
#                   (truncating away the response → NaN loss).
#   <student-model> HF id or local path of the small model to distill INTO
#                   (e.g. Qwen/Qwen3-1.7B for full fidelity, Qwen/Qwen3-0.6B for
#                   max compression).
#   <out-dir>       output dir for the fused, deployable specialist.
#   [iters]         training iters (default 400 ≈ ~1 epoch on ~800 rows).
#
# Output: a standalone model dir (LoRA fused in) — serve it via mlx_lm.server,
# oMLX, or LM Studio. No adapter wiring needed downstream.
set -euo pipefail

DATA="${1:?data-dir required}"
STUDENT="${2:?student-model required}"
OUT="${3:?out-dir required}"
ITERS="${4:-400}"
ADAPTER="${OUT}-adapter"

echo "== distilling into ${STUDENT}  (iters=${ITERS}) =="
python3 -m mlx_lm.lora \
  --model "$STUDENT" --train --data "$DATA" \
  --fine-tune-type lora --mask-prompt \
  --num-layers 16 --batch-size 2 --iters "$ITERS" \
  --learning-rate 1e-4 --max-seq-length 2048 --grad-checkpoint \
  --adapter-path "$ADAPTER" \
  --steps-per-report 25 --save-every 200 --val-batches 10 --seed 42

echo "== fusing LoRA into a standalone deployable model =="
python3 -m mlx_lm.fuse --model "$STUDENT" --adapter-path "$ADAPTER" --save-path "$OUT"

echo "✓ deployable specialist: ${OUT}"
echo "  serve:  python3 -m mlx_lm.server --model ${OUT}   (or point oMLX / LM Studio at it)"
echo "  next:   eval it vs the teacher on a held-out slice before trusting it (task-specific scorer)."
