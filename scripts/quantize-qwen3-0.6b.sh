#!/usr/bin/env bash
set -euo pipefail

# Convert a local Qwen3-0.6B HF checkout to Q4_K_M GGUF using llama.cpp.
# This script does not download models by itself; pass the HF model directory
# or set QWEN3_06B_DIR.

MODEL_DIR="${1:-${QWEN3_06B_DIR:-}}"
OUT="${2:-${HOME}/.cache/tinygpt/runs/qwen3-0.6b-q4_km.gguf}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "usage: scripts/quantize-qwen3-0.6b.sh <qwen3-0.6b-hf-dir> [out.gguf]" >&2
  exit 2
fi

if [[ ! -d "${MODEL_DIR}" || ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "not an HF model directory with config.json: ${MODEL_DIR}" >&2
  exit 2
fi

CONVERT="${LLAMA_CPP_CONVERT:-}"
QUANTIZE="${LLAMA_QUANTIZE:-}"

if [[ -z "${CONVERT}" ]]; then
  for candidate in \
    "${LLAMA_CPP_DIR:-}/convert_hf_to_gguf.py" \
    "${LLAMA_CPP_DIR:-}/convert.py" \
    "$(command -v convert_hf_to_gguf.py 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -f "${candidate}" ]] && CONVERT="${candidate}" && break
  done
fi

if [[ -z "${QUANTIZE}" ]]; then
  for candidate in \
    "${LLAMA_CPP_DIR:-}/build/bin/llama-quantize" \
    "${LLAMA_CPP_DIR:-}/llama-quantize" \
    "$(command -v llama-quantize 2>/dev/null || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] && QUANTIZE="${candidate}" && break
  done
fi

if [[ -z "${CONVERT}" || ! -f "${CONVERT}" ]]; then
  echo "could not find llama.cpp convert_hf_to_gguf.py; set LLAMA_CPP_DIR or LLAMA_CPP_CONVERT" >&2
  exit 1
fi
if [[ -z "${QUANTIZE}" || ! -x "${QUANTIZE}" ]]; then
  echo "could not find llama-quantize; set LLAMA_CPP_DIR or LLAMA_QUANTIZE" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"
TMP_F16="${OUT%.gguf}.f16.gguf"

python3 "${CONVERT}" "${MODEL_DIR}" --outfile "${TMP_F16}" --outtype f16
"${QUANTIZE}" "${TMP_F16}" "${OUT}" Q4_K_M

echo "wrote ${OUT}"
