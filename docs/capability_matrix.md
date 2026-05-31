---
title: Capability matrix — modalities × actions
description: Exhaustive map of input modalities tinygpt supports and what operations (train, distill, tune, quantize, etc.) are available for each. Honest status per cell — what's shipped, what's partial, what's only roadmapped.
---

# Capability matrix — modalities × actions

**Date**: 2026-05-31  ·  Commit: `218af10`
**Question**: For every (modality, action) pair, can tinygpt do it today?

## 1. Modalities

| Modality | Status | Notes |
|---|---|---|
| **Text — byte-level (vocab=256)** | ✅ | First-class. Default for from-scratch models. |
| **Text — BPE / HF tokenizer** | ✅ | Pinned via `--tokenizer` at train time; HF tokenizers loaded from model dir. SmolLM2 / Qwen3 vocabs verified. |
| **Code** | ✅ | Just text under the BPE tokenizer. Curated code datasets in registry: `the-stack-smol`, `python_code_instructions_18k_alpaca`, `codeforces-cots`, `commitpack`, `SWE-bench`. |
| **Structured (JSON, tool calls)** | ✅ | FSM-constrained generation (`ConstrainedGen`) enforces a target JSON schema during decode. Agent runtime + JSON-mode shipped. |
| **Multilingual text (Indic)** | ⚠️ | Tokenizer side OK (smollm2 / Qwen3 vocabs). MILU + IndicGenBench eval CLIs wired. No real Indic model trained yet. Sarvam-Edge / Airavata are the planned bases. |
| **Screen — accessibility tree (text)** | ✅ | `tinygpt screen tree` returns the focused window's AX tree as JSON. Text-only — LM-friendly out of the box. |
| **Screen — raw pixels (image)** | ⚠️ | `tinygpt screen capture` works in code; bare CLI hits the CGS-init quirk. Needs signed bundle / GUI-terminal context. NO vision encoder yet — just PNG bytes. |
| **Vision (image → embeddings)** | ⬜ | Vision encoder (ViT → tinygpt decoder) is **deferred research-grade work**. Not shipped. |
| **Audio (speech → text)** | ⬜ | Not in scope today. Roadmap notes Apple `Speech.framework` + `AVSpeechSynthesizer` as the eventual local choices. |
| **Audio (text → speech)** | ⬜ | Same — `AVSpeechSynthesizer` placeholder, not wired. |
| **Multi-modal — text + image** | ⬜ | Architecturally **not shipped**. Cider's mlx_vlm compatibility patches are referenced in the research doc but not consumed; no vision-language model is part of tinygpt's training or inference today. |
| **Multi-modal — text + audio** | ⬜ | Not built. |
| **Multi-modal — text + structured (tools)** | ✅ | Tool-calling specialists ARE multi-modal in the sense "input is user text + tool catalog, output is JSON call." Fully supported via agent runtime + mini-router. |

**Summary**: tinygpt is **a text-LM + structured-output + screen-text
hybrid**, with code as a first-class text variant. **No vision /
audio encoders are shipped.** Multi-modal in the vision/audio sense
is roadmapped (Wave 2.6 ViT, Wave 4 audio) but not implemented.

## 2. Actions

Every action below is a working CLI subcommand on M5 Pro
(audit confirmed in `docs/feature_audit_2026_05_31.md`).

### Training

| Action | CLI | Modality | Notes |
|---|---|---|---|
| Pretrain from scratch | `tinygpt train` | text (byte or BPE) | Standard transformer LM training. |
| Pretrain (training-throughput bench) | `tinygpt bench-train` | text | 42 ms/step Huge on M5 Pro |
| Continued pretrain (text corpus) | `tinygpt finetune` | text | Byte-level OK; updates base weights |
| Supervised fine-tune | `tinygpt sft` | text (BPE required) | DoRA default. JSONL `{instruction, response}`. |
| DPO / SimPO | `tinygpt dpo` | text (BPE required) | JSONL `{prompt, chosen, rejected}`. SimPO default. |
| Speculative-decoding heads | `tinygpt train-heads` | text | Medusa + EAGLE-2 variants |
| Tool-call extractor (mini-router) | `tinygpt train-extractor` | text → tool class | Tiny encoder + softmax over tool catalog |
| Knowledge distillation | `tinygpt distill` | text | Teacher → student via KL |
| Evolution strategies | `tinygpt es` | text | Gradient-free; experimental |
| Tuned-lens probes | `tinygpt tuned-lens` | text | Per-layer logit probes (research) |

### Adapter / PEFT (all under `tinygpt sft`)

| Variant | Flag | Status |
|---|---|---|
| LoRA | (default) | ✅ |
| DoRA | `--dora` | ✅ (in-session; disk format WIP) |
| VeRA | `--vera` | ✅ |
| RsLoRA | `--rs-lora` | ✅ |
| LoRA-FA | `--lora-fa` | ✅ |
| PISSA-init | `--pissa-init` | ✅ |
| LoftQ | `--loftq` | ✅ |
| AdaLoRA | `--adalora-target-rank N` | ✅ |
| LayerDrop | `--layer-drop F` | ✅ |
| LoRA+ | `--lora-plus` | ✅ |

### Quantization + compression

| Action | CLI | Status | Notes |
|---|---|---|---|
| GPTQ (Hessian-calibrated int N) | `tinygpt gptq` | ✅ | 2/3/4/8-bit · 0.10 rel error on smoke |
| HQQ (no calibration) | `tinygpt hqq` | ✅ | 2/3/4/8-bit · 0.09 rel error on smoke |
| QAT (in-training) | `tinygpt train --qat` | ✅ | Quantization-aware fine-tune |
| SmoothQuant (activation scale) | (internal, shipped per progress doc) | ✅ | Documented in `quantization_expansion.md` |
| Pruning — unstructured | `tinygpt prune-unstructured` | ✅ | 50% sparsity → -38.9% gz |
| Pruning — structured (heads/layers) | `tinygpt prune-structured` | ✅ | Frobenius head ranking; physical layer drop |
| LASER rank reduction (SVD) | `tinygpt laser` | ✅ | Per-tensor low-rank approximation |

### Inference + serving

| Action | CLI | Status |
|---|---|---|
| Sample / generate | `tinygpt sample` | ✅ — KV cached, speculative-decode flags |
| HTTP server (OpenAI surface) | `tinygpt serve` | ✅ — `/v1/{models,chat/completions,completions}` |
| HTTP server (Ollama surface) | `tinygpt serve` (same port) | ✅ — `/api/{tags,version,show,chat,generate}` |
| Agent runtime (multi-turn + tools) | `tinygpt agent` | ✅ — JSON-mode tool dispatch + persistent KV |
| Cloud escalation (direct call) | `tinygpt escalate` | ✅ — Anthropic + OpenAI |
| Cloud escalation (agent-driven) | `tinygpt agent --cloud-escalate` | ✅ — synthetic `escalate` tool added to schema |
| Mini-router inference | `tinygpt extract` | ✅ — top-K tool prediction with softmax confidence |

### Evaluation

| Action | CLI | Status |
|---|---|---|
| Loss on a corpus | `tinygpt eval` | ✅ |
| Standardized benchmarks | `tinygpt score-bench` | ✅ — HellaSwag / MMLU-Pro / GSM8K via lm-eval adapter |
| Indic benchmarks (MILU, IndicGenBench) | `tinygpt eval-indic` | ⚠️ — CLI wired; real-data baseline pending dataset fetch |
| Inference benchmark (TTFT/ITL/throughput) | `tinygpt bench` | ✅ |
| Training-throughput benchmark | `tinygpt bench-train` | ✅ |
| Base ↔ LoRA comparison | `tinygpt compare` | ✅ — side-by-side eval |

### Data acquisition + manipulation

| Action | CLI | Status |
|---|---|---|
| HF Datasets registry (curated) | `tinygpt list-datasets` | ✅ — 22 entries |
| Download HF dataset | `tinygpt download-dataset` | ✅ |
| Inspect HF model dir | `tinygpt hf-load` / `hf-inspect` | ✅ |
| GitHub issue→PR / commits / reviews | `tinygpt fetch-github` | ✅ |
| Synthetic instructions via Magpie | `tinygpt magpie` | ✅ — needs chat-tuned base |
| Tool-call training data (BFCL/τ-bench) | `tinygpt extractor-data` | ✅ — JSONL `{query, tool}` pairs |

### Inspection + debugging

| Action | CLI | Status |
|---|---|---|
| Inspect .tinygpt file | `tinygpt inspect` | ✅ |
| Round-trip validate | `tinygpt validate` | ✅ |
| Debug helpers | `tinygpt debug-{dtypes,load,logits,loss,names}` | ✅ |

### Cloud / storage

| Action | CLI | Status |
|---|---|---|
| Push checkpoint to R2 | `tinygpt push` | ✅ — env-gated, dry-run supported |
| Pull checkpoint from R2 | `tinygpt pull` | ✅ |
| R2 bucket management | `tinygpt cloud {list,delete,setup}` | ✅ |

### Screen reading

| Action | CLI | Status |
|---|---|---|
| Capture active window (image) | `tinygpt screen capture` | ⚠️ — CGS-init quirk from bare CLI |
| Read AX tree (text) | `tinygpt screen tree` | ✅ |
| Capture both | `tinygpt screen both` | ⚠️ — image half subject to same quirk |

## 3. Modality × action coverage matrix

Reading: rows are modalities, columns are actions. ✅ = supported,
⚠️ = partial, ⬜ = not built, — = N/A.

| Modality / Action | Pretrain | Finetune | SFT | DPO | Distill | Quantize | Prune | Sample | Serve | Agent | Eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Text (byte-level) | ✅ | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Text (BPE) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Code (BPE) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Structured (JSON, tools) | ✅* | ✅* | ✅* | ✅* | ✅* | — | — | ✅ | ✅ | ✅ | ⚠️† |
| Multilingual / Indic | ✅* | ✅* | ✅* | ✅* | ✅* | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️† |
| Screen AX tree | — | ✅* | ✅* | ✅* | — | — | — | ✅ | ✅ | ✅* | ⬜ |
| Screen raw pixels | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — | — | ⬜ | ⬜ | ⚠️† | ⬜ |
| Vision (image → embed) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Audio (speech in/out) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Multi-modal text+image | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — | — | ⬜ | ⬜ | ⬜ | ⬜ |
| Multi-modal text+audio | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — | — | ⬜ | ⬜ | ⬜ | ⬜ |
| Multi-modal text+tools | ✅* | ✅* | ✅* | ✅* | ✅* | — | — | ✅ | ✅ | ✅ | ⚠️† |

`*` = the *base* model still gets trained as text; we don't have a
separate "structured" or "Indic" pretrainer — they're text under a
tokenizer + appropriate training data.

`†` = the eval CLI exists but no real-data baseline has been run yet.

## 4. What's exposed vs what would need building

### Things tinygpt **can do today** that you might not realize

- **Train a Hindi specialist with MILU + IndicGenBench evals**:
  pick a base with the right tokenizer (Sarvam / Airavata path),
  run `sft` on Indic instruction data, eval via `eval-indic`.
- **Train a tool-using specialist with constrained JSON output**:
  `sft` on BFCL/τ-bench pairs, agent runtime enforces JSON via FSM.
- **Specialist + cloud-escalation hybrid**: out of the box, via
  the `--cloud-escalate` flag.
- **Mini-router that pre-selects tools**: `train-extractor` →
  `tinygpt agent --router router.tinygpt`.
- **Quantize + LoRA-tune any HF model**: GPTQ to int4, then SFT
  with `--loftq` (which compensates for the int4 error).

### Things tinygpt **cannot do today** (and the missing piece)

- **Vision-language model** (image → text): need a ViT (or other
  image encoder) + a projector to the LM's embedding space. ~2
  weeks of research-grade work per the roadmap.
- **Speech-to-text or text-to-speech**: would wire Apple
  `Speech.framework` + `AVSpeechSynthesizer`. Not in scope today.
- **Audio-to-text training data**: no whisper-style training is
  wired. Datasets in the registry are text-only.
- **Diffusion / image generation**: out of scope by design;
  cider's mlx_vlm patches mention it but tinygpt isn't a
  diffusion project.

## 5. The honest one-line summary

**tinygpt is a text + code + structured-output transformer toy
with quantization / PEFT / spec-decoding / agent-runtime / screen-
text-via-AX, train-able from scratch on Apple Silicon and browser
WebGPU. It is NOT yet multi-modal in the vision/audio sense; that's
roadmapped, not built.**

For the avoidance of doubt:
- ✅ Anything text-shaped (chat, code, Indic, structured, tool-calling)
- ⚠️ Screen-derived text (AX tree works, raw image capture has a CGS quirk)
- ⬜ Anything vision/audio/diffusion (deferred research-grade work)
