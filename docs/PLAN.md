---
title: TinyGPT — master plan (shipped / skipped / TODO)
description: Single source of truth for what's shipped, skipped, and still to build. Consolidated from docs/roadmap/*, docs/progress.md, docs/backlog.md, docs/feature_audit_2026_05_31.md, and docs/roadmap/recent_research.md (paper catalogue). Replaces them as the canonical reference; the older docs are now pointer stubs or archived under docs/archive/.
---

# TinyGPT — master plan

**Last verified against codebase**: 2026-06-02 (third pass — multiple re-audits this session caught additional stale ⬜ markers)
**Sources merged**: `docs/roadmap/*` · `docs/progress.md` · `docs/backlog.md` · `docs/feature_audit_2026_05_31.md` · `docs/roadmap/recent_research.md` (paper catalogue → §4)

Three sections — **shipped**, **skipped**, **TODO**. Every claim verified
against the code. The first audit caught Lion/Sophia/Muon/PEFT-bundle/
gradient-clipping; the second caught YOCO + GPTQ-reader + token-elim
(dropped under value-add filter); the third caught embedding RMSNorm,
cosine warmup, layer-wise LR decay, DeepNorm, BPE-dropout, Real CI —
all shipped, all previously marked ⬜.

### Status legend

| Mark | Meaning |
|---|---|
| ✅ | shipped — verified against code today |
| 🟡 | partial / in-session-only / verified-with-caveat |
| ⬜ | TODO — in active backlog |
| ⏸ | deferred — would build but waiting on external trigger |
| ❌ | skipped — intentionally not built (better alternative exists) |
| 🚧 | blocked — would build but cannot right now (hardware / upstream / budget) |

---

# 1. SHIPPED

## Mac runtime + CLI

**Audit baseline**: every CLI smoke-tested on M5 Pro 2026-05-31. See
`feature_audit_2026_05_31.md` for the full smoke trace. 30+ subcommands all green.

- ✅ Cold-start bundle (mmap + lazy embed + async load + compile cache) — 24 ms in-process TTFT on 1B
- ✅ KV cache (GQA + in-place + persistent across sessions)
- ✅ Pausable training (cooperative SIGINT + atomic save + `--resume`)
- ✅ Cross-process GPU lock (`~/.cache/tinygpt/gpu.lock`)
- ✅ CF R2 cloud save/load pipeline (push / pull / list / setup; zero egress)
- ✅ `tinygpt serve` — OpenAI + Ollama surfaces on the same socket
- ✅ `tinygpt agent` — multi-turn + tool dispatch + persistent KV + `--cloud-escalate`
- ✅ JSON-mode constrained generation (FSM token masking)
- ✅ Cloud API client (Anthropic + OpenAI via curl) + SSE streaming + cancellation
- ✅ Continue.dev / Ollama-compat provider (`/api/tags`, `/api/version`, `/api/show`, `/api/chat`, `/api/generate`)
- ✅ `tinygpt escalate` (direct cloud-API call)

## Mac training + post-training

- ✅ Pretrain (`tinygpt train`) — 42 ms/step Huge on M5 Pro, 17.2× browser
- ✅ Finetune (`tinygpt finetune`)
- ✅ SFT (`tinygpt sft`) — DoRA default + every PEFT variant
- ✅ DPO / SimPO / KTO / ORPO (all in `tinygpt dpo` via flags)
- ✅ Knowledge distillation (`tinygpt distill`) — KL teacher → student
- ✅ Speculative-decoding head training (`tinygpt train-heads --type medusa|eagle`)
- ✅ Evolution Strategies trainer (`tinygpt es`)
- ✅ Tuned-lens trainer (`tinygpt tuned-lens`)
- ✅ Mini-router trainer (`tinygpt train-extractor`)
- ✅ Magpie synthetic-instruction generator (`tinygpt magpie`)
- ✅ Sequence packing for SFT
- ✅ NEFTune (noisy embeddings) — `--neftune-alpha` in `sft` + `dpo` (matches the paper's "Noisy Embeddings Improve **Instruction Finetuning**" scope; not in the pretrain path)
- ✅ Gradient clipping (`--grad-clip F`, default 1.0, on train + sft + dpo)
- ✅ z-loss auxiliary (`--z-loss-weight F`)
- ✅ Embedding tying (`tieEmbeddings` config flag)
- ✅ Document-level shuffling (implicit via batch sampler)
- ✅ Gradient checkpointing (CustomFunction VJP workaround for missing `mlx_checkpoint`)
- ✅ QAT (in-training, `--qat`)
- ✅ Persistent tokenized cache (TokenCache.swift; wired into Train + Eval + Distill + Finetune)

## Training stability (verified 2026-06-02 — these were all marked ⬜ in older docs)

- ✅ Embedding RMSNorm (`--embedding-rmsnorm` / `cfg.useEmbeddingRMSNorm`)
- ✅ DeepNorm residual scaling (`--deep-norm` / `cfg.useDeepNorm` + `deepNormAlpha/Beta`)
- ✅ Layer-wise LR decay (`cfg.lrLayerDecay`)
- ✅ Cosine warmup (`--lr-schedule cosine --warmup 500` — the curated default)
- ✅ BPE-dropout (`BPEDropout.swift` — per-merge skip during encoding for regularization)

## PEFT bundle

All in `native-mac/Sources/TinyGPTModel/PeftVariants.swift`, all gated through `tinygpt sft`:

- ✅ LoRA · Multi-LoRA composition · LoRA+ (different LR for A/B)
- ✅ DoRA (in-session; on-disk format pending — see Tier C)
- ✅ VeRA · LoftQ · AdaLoRA · RsLoRA · PISSA · LoRA-FA · LayerDrop

## Inference + sampling

- ✅ KV cache + flash-attention forward (`MLXFast.scaledDotProductAttention`) + backward
- ✅ Quantized inference (int4 / int8 via `MLXNN.quantize`)
- ✅ Speculative decoding (vanilla + Medusa + EAGLE-2 heads)
- ✅ Prefix / prompt caching
- ✅ Streaming-LLM attention sink
- ✅ KV cache quantization (KIVI)
- ✅ Multi-Token Prediction (MTP) inference path
- ✅ Multi-Query Attention (free via `nKvHeads: 1`)
- ✅ Sliding window attention (`--sliding-window N`)
- ✅ ALiBi position bias (`--alibi`)

## Quantization + compression

- ✅ HQQ (`tinygpt hqq` — int4 q-then-dq, 0.087 rel error)
- ✅ GPTQ (`tinygpt gptq` — from-scratch int4 quant of own model, 0.102 rel error)
- ✅ AWQ safetensors reader (loads HF AWQ-quantized models)
- ✅ GPTQ safetensors reader (`GPTQReader.swift` — loads HF GPTQ-format models; tested 72 tensors quantised in 31s)
- ✅ GGUF reader (`GGUFReader.swift` + `tinygpt gguf-inspect`) — parses v2/v3 header + metadata + tensor inventory; dequantises F32 / F16 / Q4_0 / Q8_0 tensors to fp32. K-quants (Q4_K / Q6_K / etc.) slot into the same switch when needed.
- ✅ SmoothQuant (in-training)
- ✅ Pruning — unstructured (`tinygpt prune-unstructured`) + structured (`tinygpt prune-structured`)
- ✅ LASER selective rank reduction (`tinygpt laser`)

## Optimizers

- ✅ AdamW · Lion · Sophia · Muon · Adafactor (all in `Optimizers.swift`)
- ✅ GaLore (gradient low-rank projection)

## Architecture variants

- ✅ Standard transformer (RoPE + RMSNorm + SwiGLU + GQA)
- ✅ Sliding window · ALiBi · Multi-Token Prediction · MQA · GQA
- ✅ MoE (dense routing — sparse hard routing blocked, see §2)
- ✅ Mixture of Depths (soft sigmoid gate — hard routing blocked, see §2)
- ✅ Differential attention (`--diff-attn`)
- ✅ YOCO cross-layer KV sharing (`--yoco`) — CrossAttention.swift module, second-half blocks reuse first-half K/V. See `docs/yoco_results.md`. *(Was marked "designed only" in older audit — actually shipped.)*

## Tokenization

- ✅ Byte-level (vocab=256) — from-scratch path
- ✅ HF BPE / SentencePiece via swift-transformers

## Interpretability tools (browser playground)

- ✅ Logit lens (button + worker route)
- ✅ Tuned lens (Mac CLI trainer + `.lenses` sidecar + browser upload)
- ✅ Attention heatmap ("Watch the model think" panel)
- ✅ Per-layer ablation ("Ablate & sample" button)
- ✅ Activation patching — both variants (zero + donor-swap, shipped 2026-06-02 in `17021bc`)
- ✅ Linear probes (`tinygpt linear-probe`) — train Linear(d_model → C) on per-layer hidden states + label data; `.lp` sidecar format. Detects whether a layer represents an arbitrary external property (Alain & Bengio 2016).
- ✅ ROME — surgical fact editing (`tinygpt rome`). Rank-1 update to one MLP's W_out, identity-Hessian first cut. Verified on shakespeare.tinygpt: `--target X --layer 11 --scale 10` flipped sampled next-token to X. Covariance-based ROME is the follow-up.
- ✅ MEMIT — batched fact editing (`tinygpt memit`). Rank-K least-squares ΔW = R(KᵀK + λI)⁻¹Kᵀ via hand-rolled Gauss-Jordan inverse on the small N×N system. Verified math: per-fact residual ~1e-4 at scale=1 (machine noise — least-squares is exact). Single-layer visibility-in-sampling tradeoff documented; multi-layer MEMIT (distribute update across 5-7 mid-network layers) is the next-cut.

## Browser / Web track

- ✅ Landing page + `/playground` route
- ✅ WebGPU training pipeline (Huge / Mega presets via capability detection)
- ✅ Browser BPE scorer + gallery model loader
- ✅ Browser-side benchmark runner ("Run benchmark on your loaded model")
- ✅ Doc consolidation — every doc visible at `/docs/[slug]`
- ✅ WASM SIMD (`-msimd128`) — measured 1.6×
- ✅ Multi-threaded WASM (pthreads + SAB) — measured ~2×
- ✅ Memory64 module (`tinygpt64.{js,wasm}`) — partial: Node ok, browser blocked at d_model ≥ 256 (ABI bug, task #66)
- ✅ Speedup curve vs WASM SIMD: Small 2.6× / Medium 6.8× / Large 9.3× / XL 12.1×
- ✅ **WebNN active probe** (`webnn_probe.ts`, builds a tiny MLGraph and verifies it computes, drives the `+WebNN (gpu/npu)` pill state — 2026-06-02 in `86433c3`). Full transformer-as-MLGraph follow-up unblocked.

## WebGPU kernels (in `webgpu/train*.wgsl`)

- ✅ Naive scalar matmul
- ✅ Blocked 4×4 matmul (`matmul_blocked_vec4`)
- ✅ Layer-norm subgroup variant (gated on `gpuFeatures.subgroups`)
- ✅ Cross-entropy subgroup variant
- ✅ Bias-grad subgroup variant
- ✅ FA2 forward in WGSL (flash attention in browser)
- ✅ f16-storage matmul (gated by `verifyF16Storage`)
- ✅ **f16-compute matmul forward + backward** (`train_f16_compute.wgsl`, gated by `verifyShaderF16Compute` — 2026-06-02 in `1ddf6ba` / `2cdedac`)
- ✅ **Coop-matrix matmul** (`train_coopmat.wgsl`, gated by `verifyCoopMatrix` — 2026-06-02 in `86433c3`)
- ✅ OPFS persistence
- ✅ Patch kernels (`patch_zero` + `patch_replace` — 2026-06-02)
- ✅ Subgroup matmul kernel (`matmul_sg` / `matmul_abt_sg` — gate currently fails on M5 Pro, falls back to vec4)

**Numerics-gate framework** — every fast path (f16-storage, f16-compute,
coop-matrix, subgroup) carries its own gate that compares against a f32
reference with a magnitude-aware tolerance. Gate-fail → silent fallback,
zero regression risk. See `docs/precision.md`.

## Datasets + data pipelines

- ✅ `tinygpt list-datasets` — 22 curated entries (tool-calling / debugger / code / math / reasoning)
- ✅ `tinygpt download-dataset` (canonical `hf://datasets/owner/name` form)
- ✅ HF Datasets / Hub integration (`hf-load`, `hf-inspect`)
- ✅ GitHub data fetcher (`fetch-github` — issue→PR pairs)
- ✅ Magpie synthetic instruction generator
- ✅ Extractor-data pipeline (`extractor-data` — BFCL/τ-bench → `{query, tool}` pairs)
- ✅ Indic eval pipeline (`eval-indic` — MILU MCQ + IndicGenBench-XQuAD, smoke-validated)

## Tooling + infra

- ✅ XCTest harness + swiftformat + lint CI (Mac)
- ✅ `tinygpt inspect` / `validate` (round-trip byte-compare verified on 110 MB model)
- ✅ `tinygpt bench` (TTFT/ITL/decode tok/s/peak RSS) + `tinygpt bench-train`
- ✅ `tinygpt eval` / `score-bench` (loss + benchmark scorers)
- ✅ `tinygpt compare` (side-by-side base vs LoRA-adapted)
- ✅ `tinygpt debug-*` (dtypes / load / logits / loss / names helpers)
- ✅ `tinygpt screen tree` (AX tree readout — focused-window JSON)
- ✅ lm-evaluation-harness MLX adapter

## Headline metrics (Mac, M5 Pro / 48 GB)

| | Value | Target | Headroom |
|---|---|---|---|
| TTFT (warm) | 5.8 ms p99 | < 50 ms | ✅ 10× under |
| ITL p99 | 4.9 ms | < 30 ms | ✅ 6× under |
| Decode tok/s | 293 (mega-pilot 960M) → 696 (huge 221M) | > 50 tok/s | ✅ 6× over |
| Cold start TTFT | 24 ms (1B) | < 50 ms | ✅ 2× under |
| Training Huge | 42 ms/step | (baseline) | — |
| Speedup vs browser | 17.2× | (baseline) | — |
| Largest model | 960 M params (1.1 GB) | — | — |

## Recent product surfaces (Wave 2.6, shipped 2026-05-31)

- ✅ Cloud-escalate wired into AgentLoop
- ✅ Continue.dev / Ollama-compat provider
- ✅ Tool-call extractor (mini-router) scaffold — ToolRouterModel + CLI pipeline
- ✅ ScreenCaptureKit + macOS Accessibility scaffold — AX tree works end-to-end from CLI

## Learning artifacts (docs)

- ✅ `docs/decision_log.md` — every architectural decision logged
- ✅ Research bundles in `docs/research/` (inference + quality benchmarks, kernel audit, mac decode baseline, wave-4 landscape, Indic evals)
- ✅ Session retrospectives (e.g., `session_2026_05_31.md`)
- ✅ Per-technique deep-dives (`distillation.md`, `evolution_strategies.md`, `moe.md`, `mtp.md`, `lora_guide.md`, `interpretability.md`, etc.)

---

# 2. SKIPPED

## ❌ Superseded by better alternatives

- **fp16 mixed-precision training** — bf16 strictly better, shipped
- **ZeRO / FSDP / pipeline parallelism** — multi-device only
- **State space models (Mamba / RWKV)** — different architecture; ~2-3 week port; better as side-project
- **PagedAttention / continuous batching** — multi-user inference only
- **Tree attention / lookahead decoding** — marginal over speculative
- **Adapter modules (Houlsby / Pfeiffer)** — LoRA's older cousin, superseded
- **BitFit** — train biases only; quality is poor
- **IA³** — element-wise scaling; superseded by LoRA family
- **Hyena / long-conv** — different architecture
- **fp8 training** — needs H100 / Blackwell hardware

## ❌ Dropped after audit (real cost, no payoff at our scale)

- **Flash Attention Metal kernel** — MLXFast SDPA already fused (`docs/research/wave_2_5_kernel_audit.md` §1)
- **Int4 packed matmul Metal kernel** — MLX `quantized_matmul` already hand-tuned (§3)
- **General SWE-bench leaderboard chase** — Sonnet 4.6 dominates regardless of wrapper; play local-first / on-device game instead
- **Tinker cloud fine-tune as differentiator** — use if needed; not a project differentiator (budget-ruled-out for solo)
- **Hooking into Apple App Intents** — no public API for third-party LLMs to replace Apple's FM

## ⏸ Deferred (waiting on external trigger)

| Item | Trigger | Why deferred |
|---|---|---|
| cider W8A8 adoption | a 3B+ specialist ships | At ≤ 1B, Mac already 10× under realtime; cider's prefill win is immaterial |
| ANE + GPU heterogeneous routing | Apple ships Stateful Models API (rumored late 2026) | Research-grade; current path uses private ANEMLL APIs |
| WebGPU subgroup matmul redesign | browser focus returns | Current gate fails (1415% mean_rel); fallback works |
| Vision encoder (ViT → tinygpt decoder) | vision-specialist demand becomes concrete | 2-week research-grade work; not critical-path |
| Audio I/O (Speech.framework + AVSpeechSynthesizer) | voice-mode demo becomes priority | Not in scope for Wave 3 |
| Async tool-call dispatch | parallel-tool specialist ships | LM dominates 5-100× over subprocess at current scales |
| ScreenCaptureKit raw image (CGS-init fix) | vision specialist needs raw bytes | AX tree sufficient for tool-calling specialists |
| Public launch (HF + writeup + HN) | ≥ 1 specialist beats a fair baseline | Nothing to launch yet |
| Phase 7 browser perf (subgroups / coop-matrix / WebNN) | post-HN v2 push | Current 12.1× lift is the launch story |

## 🚧 Blocked by hardware

- **Distributed training (ZeRO, FSDP, pipeline-parallel)** — single device only
- **Native FP4 training** — Apple M-series lacks FP4 tensor ops
- **Native FP8 training** — same
- **Hardware-accelerated MoE routing** — Apple silicon has no sparse-routing ops
- **ANE training acceleration** — ANE is inference-only

## 🚧 Blocked by upstream library state

| Item | Blocker | Workaround |
|---|---|---|
| QLoRA real-quantized base + LoRA | MLX-Swift quantized arrays don't autograd through | Manual fake-quant in fwd (pedagogical, no memory win) |
| Sparse MoE hard routing | MLX-Swift no `scatter_add` | Soft (dense) routing shipped |
| Mixture-of-Depths hard top-K | same | Soft sigmoid gate shipped |
| Fast BPE encoding | swift-transformers single-threaded; 2 GB corpus = ~30 min | Rust-backed encoder via FFI (future) |
| Native int4 / int8 WebGPU matmul | spec doesn't yet have quantized matmul extensions | Wait for subgroup / coop-matrix extensions |
| GGUF safetensors reader | not yet written | Could write (~2 days); AWQ + GPTQ readers already ship |

## 🚧 Blocked by budget

- **Synthetic SFT via frontier API ($1-10K)** — use open-weights teacher via Magpie instead
- **Multi-TB dataset downloads** — stream subsets (the HF importer does this)
- **Strong local judge for Constitutional AI / RLAIF** — no 70B+ runs usable on a Mac
- **Public RLHF / PPO pipeline** — 5× the code of DPO + 10× iteration; DPO covers 80-90% of the lift

---

# 3. TODO

ROI-ordered. Sourced from `backlog.md` (the living list, last sort 2026-05-31).

## Tier A — DO NEXT (north-star aligned; specialists)

Until A1 lands, every optimization is theoretical. Until **Tier E** (eval
pipelines) lands, every specialist is unmeasurable — A1's "ship" criterion
implicitly requires E1 + E3 wired before any score can be published.
Sequencing: Tier D (data) + Tier E (evals) → A1 specialist → Tier B
follow-ups.

- ⬜ **A1. Train first specialist end-to-end (tool-caller)** — 3-5 days execution + GPU hours. Validates north-star thesis.
- ⬜ **A2. Pull foundational datasets** (xlam-function-calling-60k, hermes-function-calling-v1, function-calling-chatml, SWE-bench_Verified, alpaca-cleaned, orca_dpo_pairs, MetaMathQA, ultrafeedback-binarized-preferences-cleaned, the-stack-smol, python_code_instructions_18k_alpaca) — ~1 hr wall
- ⬜ **A3. Fetch GitHub issue→PR corpus for debugger** — ~1 day with `GITHUB_TOKEN`
- ⬜ **A4. Pull BFCL + τ-bench via extractor-data** — ~30 min (DONE — sources at `~/.cache/tinygpt/datasets/_external/{gorilla-bfcl,tau-bench}/`; **wiring is Tier E**, not Tier A)
- ⬜ **A5. Pull Indic eval datasets (MILU + IndicGenBench-XQuAD)** — ~30 min (DONE — MILU is lm-eval-harness, source at `_external/MILU/`; wiring → E3)
- ⬜ **A6. Dataset inventory doc** — ~30 min after A2-A5
- ⬜ **A7. Real-data MILU baseline on flagship-huge-v5** — ~2 hr; depends on A5 + E3

## Tier D — DATA (gaps blocking specialists)

Pulled today: hermes-fc.jsonl, ultrafeedback.jsonl, MetaMathQA, alpaca-cleaned,
orca_dpo_pairs, FineWeb-Edu (50K-row sample via parquet decoder). Blocked /
missing for the planned specialists:

- ⬜ **D1. xlam-function-calling-60k** — GATED. Needs `export HF_TOKEN=hf_…` then re-run `tinygpt download-dataset Salesforce/xlam-function-calling-60k`. ~5 min user-side.
- ⬜ **D2. function-calling-chatml + SWE-bench_Verified** — public; rerun `tinygpt download-dataset` once xlam unblocks. ~15 min.
- ⬜ **D3. MS-MARCO + Natural Questions** — needed for B25 ScaleDown specialist. Pull via `tinygpt download-dataset` + decode parquet via `scripts/parquet_to_txt.py --jsonl`. ~30 min.
- ⬜ **D4. the-stack-smol + python_code_instructions_18k_alpaca** — needed for code specialist. ~30 min.
- ⬜ **D5. GSM8K + MATH + HumanEval + MBPP eval splits** — small JSONLs (test splits only), needed for math/code specialist eval. ~15 min. Pairs with E4 + E5.

## Tier E — EVAL PIPELINES (wire harnesses → automate scores)

Source code for BFCL / τ-bench / lm-eval-harness is already on disk under
`~/.cache/tinygpt/datasets/_external/`. **Pulling source ≠ usable evaluator.**
Each item below is the wiring work — a `tinygpt eval-<name>` subcommand that
takes a model path, runs the harness via subprocess, parses the score JSON,
returns a clean number. Until these land, "did the specialist learn anything?"
has no automated answer.

- ⬜ **E1. `tinygpt eval-bfcl <model>`** — subprocess to the Python BFCL harness at `_external/gorilla-bfcl/berkeley-function-call-leaderboard/`. Convert TinyGPT model → safetensors via existing `to-safetensors`, register as a model with the BFCL harness, run, parse the result JSON. ~1 day. **Blocks A1's "ship" criterion.**
- ⬜ **E2. `tinygpt eval-tau-bench <model>`** — same pattern, harness at `_external/tau-bench/run.py`. Multi-turn agent score. ~1 day. **Pairs with E1 for tool-caller credibility.**
- ⬜ **E3. `tinygpt run-lm-eval <model> --tasks mmlu,arc,hellaswag,gsm8k,…`** — wrap `_external/MILU/` (which IS lm-eval-harness). One wrapper unlocks **all** standard LM evals. **Highest leverage per day**; do first. ~1 day.
- ⬜ **E4. `tinygpt eval-gsm8k <model>`** — standalone scorer. Parse model's final numeric answer, compare to gold. Tiny — covered by E3 if lm-eval-harness lands, but a standalone fallback gets you a number in ~half-day if E3 slips.
- ⬜ **E5. `tinygpt eval-humaneval <model>` + sandbox** — sandboxed Python exec is the hard part (needs container or restricted-process). HumanEval + MBPP. ~1-2 days; the sandbox is what's risky.
- ⬜ **E6. `tinygpt eval-scaledown <model>`** — clone ScaleBench, wire to TinyGPT-loaded model, run. Prereq for B25 submission. ~half-day after E1's subprocess pattern is the template.
- ⬜ **E7. `tinygpt judge <out.jsonl> --judge <local-model>`** — LLM-as-judge shim. Pair preferences (or rate single outputs 1-10) via a local Qwen/SmolLM. Unlocks AlpacaEval / MT-Bench / RewardBench-style preference evals without an OpenAI API key. ~1 day.

**Total Tier E**: ~5-7 focused days. Do E3 + E1 first (highest leverage + A1 unblocker), then the rest as nightly arcs.

## Tier B — NEXT QUARTER (multi-specialist + product)

- ⬜ **B1. Second specialist (shell or SQL)** — 3-5 days; depends on A1
- ⬜ **B2. Mini-router on real BFCL data** — ~half day after A4
- ⬜ **B2b. Bake-off — classifier-head router vs pure-GPT-with-FSM** — settles whether architectural deviation is justified
- ⬜ **B3. FSM constraint-injection from router prediction** — ~3 days; depends on B2
- ⬜ **B4. Tool-call eval harness (subprocess refactor for BFCL/τ-bench)** — ~half day
- ⬜ **B5. Cloud-escalation training signal (`{"defer_to_cloud": true}`)** — ~1 week
- ⬜ **B6. Mac app demo** — ~1 week; depends on A1
- ⬜ **B7. Specialist routing model** — 1-2 weeks; depends on B1
- ⬜ **B8. Multilingual specialist (Sarvam-Edge / Airavata base)** — 1-2 weeks; depends on A7
- ⬜ **B9. Energy J/token measurement (needs sudo for `powermetrics`)** — ~1 day

**Pretrain + runtime quality (added 2026-06-04 — "good product" lens, not launch optics):**

- ⬜ **B10. Quality classifier on pretrain data (FineWeb-Edu-style)** — tiny fastText classifier on educational-quality labels, score corpus, keep top X%. Highest direct quality lift per dev-day. ~2 days. See §4.3.
- ⬜ **B11. WSD schedule (warmup-stable-decay)** — replaces cosine; the decay phase IS the annealing knob, unifying two adds. SmolLM/MiniCPM-validated. ~half-day. See §4.3.
- ⬜ **B12. Loss-spike recovery + replay** — grad-norm tracker triggers auto-rollback N steps + LR drop on spike. Saves hours of wasted compute on long runs. ~1 day.
- ⬜ **B13. Interp-on-checkpoints** — replay SAE / MEMIT / `tinygpt patch` across the multi-checkpoint timeline. No competitor ships this combination at small scale. Needs save-every-N flag + checkpoint-batch loader. ~1-2 days infra + ongoing analysis. See §4.3.
- ⬜ **B14. Speculative decoding (Mini-Llama draft for Mega)** — 2× runner throughput; natural fit with our model zoo. Needs numerics gate per the no-quality-regression rule. ~2-3 days. See §4.3.
- ⬜ **B15. Layer-wise LR decay for SFT** — lower layers stay stable; ULMFiT-era trick, still relevant. Flag-level add on existing optimizer. ~half-day. See §4.3.

**Competitor-aware additions (added 2026-06-04 — surfaced by web sweep, not Jan-2026 cutoff knowledge):**

- ⬜ **B16. M5 Neural Accelerator prefill benchmark + bump** — verify the claimed 3.5×–4× M5-vs-M4 prefill speedup is materializing on TinyGPT's MLX path. Current pin: `mlx-swift 0.31.3` on macOS 26.5 / M5 Pro (well past the 26.2 floor). Bump to latest (0.31.4) and benchmark. ~half-day. Free win if it's already on; bump is reversible. See §4.3.
- ⬜ **B17. SAE Lens interop / Neuronpedia format export** — compare our SAE to [SAELens](https://github.com/decoderesearch/SAELens). Decide: (a) keep ours, (b) port to theirs, (c) export our SAEs in their format so Neuronpedia consumes them. Option (c) is the cheapest interop win. ~2 days for (c). See §4.3.
- ⬜ **B18. nanochat-style `--depth` single-knob HP derivation** — one knob auto-derives width, heads, LR, batch, steps from depth via compute-optimal scaling laws. UX win. Pairs with µ-Transfer (deferred). ~1 day. See §4.3.
- ⬜ **B19. Group-SAE (layer-group SAE training)** — train one SAE per layer-group instead of per-layer; cuts SAE training cost meaningfully. Layered onto existing SAE infra. ~2-3 days. See §4.3.
- ⬜ **B20. Investigate learnable cross-stream attention** — modded-nanogpt speedrun trick; not yet a paper but on the GPT-2-quality speedrun playbook. Read-and-evaluate before adoption. ~half-day investigation, build cost TBD. See §4.3.
- ⬜ **B21. Micro-AutoMixer for specialist data mixes** — Poolside-style data mixture optimization, scaled down: train 6-12 proxy runs across code/math/tool/web ratios, score on fixed capability evals, fit a simple surrogate, then propose the next mix. Do this before expensive specialist training so data ratios stop being hand-wavy. ~2-3 days plus small proxy runs. See §4.3.
- ⬜ **B22. Token-preserving agent trajectory recorder** — store `input_ids`, sampled `output_ids`, tool calls, tool results, rewards, and checkpoint hash for every agent rollout. Poolside calls out token-in/token-out trajectories to avoid retokenization mismatch in off-policy RL; the same invariant helps TinyGPT SFT/DPO/RLVR even before full RL. ~2 days.
- ⬜ **B23. Agent eval protocol hardening** — Poolside reports pass@1 averaged over repeated runs with fixed step limits, sandbox resources, and sampling params. Mirror that discipline for BFCL/τ-bench/SWE-mini/Terminal-mini: fixed max steps, fixed sandbox budget, repeated seeds, and infra-patch notes. ~1 day.
- ⬜ **B24. Muon re-benchmark at 1B+ or skip** — Poolside reports Muon giving a large-step efficiency win at scale with distributed overhead below 1%; TinyGPT's current Muon smoke loses badly at small scale. Do not promote it until a ≥1B-ish run or a proxy matmul-dominated benchmark shows the overhead is amortized. ~half-day once a large run exists.

**External-leaderboard arc (added 2026-06-05 — first public competitive submission target):**

- ⬜ **B25. ScaleDown Challenge specialist — extractive context compression** — train a task-specific SLM that takes `(query, long_context)` and returns the subset of sentences relevant to the query. Token-level relevance classifier head on the residual stream → sentence-level aggregation → threshold-keep. Training data: MS-MARCO + Natural Questions + similar (query, doc, answer) triplets with teacher-labeled per-sentence relevance scores; teacher can be a local Qwen/SmolLM. Eval via [ScaleBench](https://tinyml.substack.com/p/benchmarking-scaledowns-summarization) (their open-source harness, downstream F1/EM after compression). Submit to the [ScaleDown Challenge leaderboard](https://main.d3hbeukddvrxcc.amplifyapp.com/leaderboard). **~3-5 days end-to-end**: dataset pulls (~1 hr, reuse `tinygpt download-dataset`), teacher-labeling pipeline (~half-day), classification-head module in `Sources/TinyGPTModel/` (~half-day), new `tinygpt compress` subcommand with token-level BCE loss (~1 day), ScaleBench integration + submission (~half-day). Pairs naturally with A1 (different domain, same A-track shape) and gives TinyGPT a public proof-point — "competitive task SLM trained from scratch on a Mac" — with an external scoreboard. See §4.3.

## Tier C — POLISH (mostly shipped this session)

- ✅ **C1. CLI cosmetic fixes** — 27 subcommands now `exit(0)` on `--help`; `bench-train --help` shows correct name. Shipped 2026-06-02 in `49dead5`.
- ✅ **C2. Roll up pre-switch CLI shims into main switch** — 17 shims absorbed; TinyGPT.swift -170 LoC. Shipped in `49dead5`.
- ⬜ **C3. DoRA on-disk adapter format** — ~1 day (today DoRA is in-session only)
- ⬜ **C4. Tool-call extractor: BPE tokenizer support** — ~2 days
- ⬜ **C5. Decode jitter under thermal load** — ~1 day (needs sustained workload measurement)
- ✅ **C6. ChatML template inline-system split** — `splitChatmlSystem` helper + 6 unit tests. Shipped in `49dead5`.
- ✅ **C7. Save+reload XCTest for LoRA adapters** — roundtrip + arch-mismatch coverage. Shipped in `49dead5`.
- ✅ **C8. Install-path discipline** — `~/.cache/tinygpt/` for adapters + corpus discovery; off `/tmp`. Shipped in `49dead5`.
- ⬜ **C9. Determinism harness** — bit-exact replay of step N for debugging spikes, drift, and grad-flow oddities. Pairs with B12. ~2 days.
- ⬜ **C10. Training-run dashboard** — W&B / TensorBoard or in-house; live visibility instead of guessing during long runs. ~1 day.

## Tier 5 — RESEARCH FRONTIER (2026 stretch goals)

Pauses the "training at 2024 fundamentals" cadence; deliverable is a paper-shaped artifact + reproducible code + a scaling-curve point, NOT a polished UX feature.

- ⬜ **5.1 Reasoning training on a 22M model** — 5-7 days; expected outcome is the *negative result* (CoT below emergence). Publishable.
- ⬜ **5.2 Test-time compute scaling** — 3-5 days; quality-vs-FLOPs plot at 22M-scale matching Snell et al. methodology. **Most cleanly publishable.**
- ⬜ **5.3 Vision-language toy** — ~2 weeks; ViT + projector + LLaVA-style. Smallest from-scratch VL model on consumer hardware.
- ⬜ **5.4 Diffusion LM micro-implementation** — 1-2 weeks; new paradigm via masked denoising loss.
- ⬜ **5.5 Real sparse MoE kernels** — 2-3 weeks; custom Metal kernel + measure FLOP reduction.
- ⬜ **5.6 TTS toy (text-to-speech via audio-token GPT)** — ~2-4 weeks; integrate EnCodec, train an autoregressive decoder over discrete audio tokens (VALL-E / MusicGen shape). The transformer side already exists in TinyGPT; the new pieces are codec integration, text→audio conditioning, vocoder decode, and an audio data pipeline. **Scoping note (2026-06-03): comes AFTER the Wave 3 specialist track (A1-B8) AND after 5.3 vision-language toy** — both higher-priority research arcs ahead of it.
- ⬜ **5.7 Specialized explainer-video model** — ~3-6 weeks for a Lamina-like toy: document/prompt → script → storyboard DSL → deterministic whiteboard/diagram render. This is NOT a Sora/Runway competitor; the first useful version is a specialized visual-planning model plus renderer. **Scoping note (2026-06-04): comes after A1-B8 and after 5.3 VL, because it needs both specialist training discipline and the text↔visual bridge.**

### 5.6 TTS toy — detailed scoping

What carries over from current TinyGPT:

| Piece | Reuse |
|---|---|
| Transformer decoder, KV cache, sampling, MTP heads (for K-codebook prediction) | direct |
| Training loop (`tinygpt train`) + PEFT bundle for downstream fine-tunes | direct |
| `CrossAttention.swift` (currently used for YOCO) | adapt to text-encoder K/V source for conditioning |

New code surface (~2 weeks of focused engineering + 3-7 days training):

| Piece | Effort |
|---|---|
| EnCodec encode/decode integration (Swift port of the HF EnCodec weights) | ~3-5 days |
| Text → conditioning surface (text encoder + cross-attention into decoder, OR text-as-prefix-tokens) | ~2-3 days |
| Audio data pipeline (LJSpeech / LibriTTS pre-tokenization to codec ids) | ~2-3 days |
| Eval (WER via Whisper transcription, MOS estimator) | ~2 days |
| First training run on LJSpeech single-speaker → intelligible speech | 2-4 days wall-clock |

**Realistic outcome at this scale:** smallest-published audio-token GPT (MusicGen-small) is ~300M; from-scratch on LJSpeech you get recognizable but not natural-sounding speech. The publishable artifact is the same shape as 5.3 — "smallest from-scratch ___ on consumer hardware."

**Why ordered after specialist + VL:**
- Specialist track validates the north-star thesis (Wave 3 work the project is actually about). Until at least one specialist beats a baseline, modality experiments are noise on top of unproven foundations.
- 5.3 vision-language toy is ahead because (a) it's the older Tier-5 item and (b) it stress-tests the same "external pretrained encoder + cross-attention into our decoder" pattern that TTS would reuse. Shipping VL first means TTS inherits a validated pattern instead of a speculative one.

### 5.7 Specialized explainer-video model — Lamina-like track

Reference product: Lamina Labs' Simi positions itself as an AI explainer
studio: prompt or document in, whiteboard-style educational video out
for students, course creators, customer training, and teams. The public
lesson is not "train a giant cinematic video model"; it is "make a
narrow video system that explains accurately, quickly, and consistently."

The TinyGPT version should start as a **structured explainer compiler**:

```
source document / prompt
  -> lesson script
  -> storyboard scenes
  -> visual DSL (objects, labels, arrows, equations, timeline)
  -> deterministic renderer (SVG/canvas/Remotion/Manim-style)
  -> captions + voiceover + MP4
```

What we would need:

| Piece | Build | Why |
|---|---|---|
| Scene/storyboard schema | JSON DSL for concepts, equations, diagrams, timings, camera/stroke actions | Gives the model a constrained target instead of free-form pixels |
| Renderer | Start with SVG/canvas frames; later Remotion/Manim export | Deterministic, debuggable, cheap to render |
| Visual-planner specialist | SFT/LoRA model: prompt/doc → storyboard DSL | This is the first "specialized video model" worth training |
| Asset/diagram library | Shapes, arrows, axes, code blocks, graph layouts, simple physics/math primitives | Explainers need reusable semantic primitives more than photorealism |
| Data pipeline | Pair open lessons/transcripts/docs with generated or human-edited storyboards | The scarce asset is supervised storyboard data |
| Eval set | Held-out concepts with rubric: factual correctness, visual grounding, pacing, label consistency, equation validity | Prevents "pretty but wrong" videos |
| Editing loop | User can regenerate one scene, lock script, lock diagrams, export MP4 | Real workflows need partial repair, not one-shot magic |

Model ladder:

1. **No learned video model**: use a strong text model or cloud model to
   produce the DSL; render deterministically. This validates product and
   schema fast.
2. **Tiny visual-planner specialist**: fine-tune tinygpt/HF-loaded base
   on prompt/doc → storyboard DSL. This is the first trainable model.
3. **Visual critic/evaluator**: model scores whether scene frames match
   the script and flags bad labels, missing objects, impossible diagrams.
4. **Optional diffusion/image/video model**: only for decorative assets
   or scene backgrounds after the deterministic explainer path works.

Good first eval tasks:

| Eval | Metric |
|---|---|
| Concept-to-storyboard | JSON validity + human/LLM rubric on lesson coverage |
| Equation/diagram correctness | Symbol/label exactness, graph/axis consistency |
| Script-to-scene grounding | Every narrated claim maps to an on-screen object/action |
| Pacing | Scene duration fits narration without overcrowding |
| Editability | Regenerate one scene without changing locked scenes |

Why this is plausible for TinyGPT:

- The project already has specialist SFT/LoRA, structured output,
  constrained generation, eval harnesses, and renderer-friendly web/native
  surfaces.
- A storyboard DSL is text. TinyGPT can train on that before any pixel
  generation exists.
- Deterministic rendering avoids the hardest part of video generation:
  long-horizon visual consistency.

Why it stays behind the current specialist track:

- It is a new modality product, not a training-foundation prerequisite.
- Data is the bottleneck. We need hundreds to thousands of good
  storyboard pairs before model training is meaningful.
- The first marketable version is mostly pipeline + UX, not raw model
  research. Build it only after the first text/tool specialist proves the
  project can beat a baseline.

## Unshipped techniques — after applying the value-add filter (and re-auditing)

Most items in the original roadmap-categories list either ship today
(third-audit corrections below) or were dropped under the user's
"don't list a technique unless it adds genuinely new capability" filter.
What's left:

**Genuinely new value-adds, not yet built**

After this session's batch closes, the non-training surface IS
exhausted — every capability item under your value-add filter has
shipped. Only niche residue remains:

- ⬜ **Sample packing (cross-source)** — niche, doesn't change capability at our scale
- ⬜ **Vocab trimming** — niche, only matters for embedded-deployment

After these: training-dependent (specialist Wave 3, Mini-Llama+ANE, Tier 5 modality arcs) or upstream-blocked (sparse MoE hard routing on `scatter_add`, real QLoRA on quantized-gradient flow).

**Shipped this session (third → fourth audit pass corrections):**

- ✅ Linear probes (`tinygpt linear-probe`)
- ✅ Deduplication (`tinygpt dedupe`, line + doc modes)
- ✅ ROME (`tinygpt rome`, identity-Hessian first cut)
- ✅ MEMIT (`tinygpt memit`, single-layer least-squares, exact per-fact residual at scale=1)
- ✅ Multi-layer MEMIT (`--layers SPEC`, residual partitioned across N layers; 8-14% per-layer rel vs 41-72% single-layer)
- ✅ MEMIT `--layer-weighting key-norm` (data-driven proxy for Meng 2023's causal-trace influence)
- ✅ GGUF reader (`GGUFReader.swift` + `tinygpt gguf-inspect` — F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K)
- ✅ GGUF model loader validator (`tinygpt gguf-load` — metadata parse, tensor-name mapping, shape validation against TinyGPT-HF op tree)
- ✅ Best-of-N + Snell-style scaling curve (`tinygpt bon --scan`)
- ✅ `bon --verifier corpus-ppl` (corpus-anchored PPL as scoring signal — distinct from self-likelihood)
- ✅ Sparse autoencoders (`tinygpt sae` — Bricken et al. 2023; encoder + decoder + L1, .sae sidecar)
- ✅ SAE feature explorer (`tinygpt sae-explore` — load .sae, scan corpus, surface top-K activating windows per feature)
- ✅ Activation patching CLI (`tinygpt patch` — Mac CLI for zero + donor-swap; reuses shipped `forwardWithPatch`)
- ✅ Causal trace CLI (`tinygpt causal-trace` — Meng et al. 2022 per-layer fact localization)
- ✅ MinHash near-duplicate dedup (`tinygpt dedupe --near-dup` — catches paraphrased boilerplate that exact-SHA misses)
- ✅ GGUF tokenizer + config extractor (`tinygpt gguf-extract` — writes tokenizer.json + config.json + manifest, the missing piece between gguf-load and runnable model)
- ✅ to-coreml conversion bridge (`tinygpt to-coreml` — generates a tailored Python conversion script for the user's coremltools install; now end-to-end runnable via safetensors hop)
- ✅ Safetensors writer (`TinyGPTModel/SafetensorsWriter.swift` — HF-compatible binary format; shared foundation)
- ✅ `tinygpt to-safetensors` — converts `.tinygpt` → `model.safetensors` with HF Llama tensor names (or `--keep-names` for native). Verified 196 tensors / 38.4 MB / valid HF format on the shakespeare gallery model.
- ✅ `gguf-extract` materializes weights to safetensors — output directory is now a complete HuggingFace model bundle loadable via `transformers.AutoModelForCausalLM.from_pretrained()`. Verified on a 21-tensor llama-shape GGUF: tokenizer.json + tokenizer_config.json + config.json + model.safetensors all populated.
- ✅ to-coreml safetensors bridge — Python script no longer stubbed; loads weights via `safetensors.torch.load_file()` with full HF Llama → TinyGPT name-map. `py_compile` clean.

**Stale ⬜ markers caught + corrected this session — now ✅:**

| Item | Where it ships |
|---|---|
| Embedding RMSNorm | `--embedding-rmsnorm` flag, `RMSNorm` module on token-embed |
| DeepNorm | `--deep-norm` flag, `cfg.useDeepNorm`/`deepNormAlpha`/`deepNormBeta` |
| Layer-wise LR decay | `cfg.lrLayerDecay` |
| Cosine warmup | `--lr-schedule cosine --warmup 500` (the curated default) |
| BPE-dropout | `BPEDropout.swift` |
| Real CI | `.github/workflows/ci.yml` + `deploy.yml` |
| Persistent tokenized cache | `TokenCache.swift` wired into Train+Eval+Distill+Finetune |
| Linear probes | `tinygpt linear-probe` (this session, `6dbe15c`) |
| YOCO cross-layer KV | `--yoco` flag, `CrossAttention.swift`, `docs/yoco_results.md` |
| GPTQ safetensors reader | `GPTQReader.swift` (72 tensors quantised in 31s) |

**Dropped under value-add filter (duplicate / inferior / niche):**

| Dropped | Why |
|---|---|
| ReLoRA | GaLore already gives "full fine-tune at LoRA memory cost" |
| Prefix tuning / soft prompts | LoRA covers the practical case |
| IPO | DPO with high β covers tiny-pair regularization |
| Token elimination | StreamingLLM + KIVI cover positional + per-entry-bits axes |
| Tree decoding | Speculative decode (vanilla + Medusa + EAGLE-2) covers the niche |
| Curriculum learning | Modest gains, scale-dependent; needs a difficulty metric we don't have |
| Self-instruct / Evol-instruct | Magpie subsumes (uses model's own distribution, no seed needed) |
| Hard example mining / Importance sampling | Marginal at our scale |
| Data quality filtering | PPL-filtering needs a ref model; basic dedup covers most of the value |
| BigBird / Longformer sparse attention | Only matters past ctx=8192 (we don't train at that length) |
| Linear attention (Performer / Linformer / Reformer) | Quality usually worse than flash attention |
| Hybrid attention/SSM (Jamba, Samba) | Different family; side-project |
| Pre-norm vs post-norm toggle | Config knob, not a feature |
| Tiktoken adoption | swift-transformers handles BPE-family tokenizers already |
| Subword regularization | Marginal vs BPE-dropout |
| Train own BPE on corpus | Modest gain (~5% PPL); blocked on Rust-FFI for speed |
| TinyGPT-as-library API | User explicitly deferred until specialists beat a baseline |

---

# Queued findings — ANE routing + Mac-vs-browser sampling

Triggered by the question "how do we get to 170× instead of 17×?" The
17.2× number is at Huge *training* — small bandwidth-bound model where
kernel-launch overhead dominates. Several legitimate paths to a much
larger ratio; each is queued with its honest cost.

## 1. Browser sampling tok/s harness — CHEAP, ~30 min

Closes a real missing measurement. We have Mac sampling tok/s
(293-696 by model size) but no analogous browser-side number. The
playground worker generates via `GpuModel.generate` already; we just
don't time it.

**What:** in `browser/src/worker.ts`, log per-token wall-clock in the
generate loop, post a `sampling_perf` message, display tok/s next to
the playground output.

**Expected ratio**: Mac-vs-browser sampling probably **30-80×** at
Huge based on shape priors (Mac is much less kernel-launch-overhead
sensitive during decode than during training). That alone changes the
headline from "17× training" to "30-80× sampling, 17× training."

**Why queued**: tiny work, just hasn't been done. No blockers.

## 2. ANE-routed inference via Mini-Llama TinyGPT — MEDIUM, 1-2 weeks

Apple Neural Engine routes only when the graph hits its preferred
shapes. The published numbers (ANEMLL, perf-quest memory) are 2-3×
sampling over the same model on Apple GPU when ANE engages cleanly,
not 100×+ end-to-end. The big win is the *combined* ratio: bigger
ANE-friendly model × ANE-routing × already-unfit-for-browser size.

### Why TinyGPT doesn't route today

ANE prefers `head_dim ∈ {64, 128}`, tensor dims multiples of 64,
fp16, RoPE-style attention, bias-free linears, RMSNorm. Our Huge
default is the opposite of all of these:

| Dimension | TinyGPT Huge | Llama 3.1 8B | ANE impact |
|---|---|---|---|
| `head_dim` | 32 | 128 | falls off ANE matrix engine |
| `d_model` | 256 | 4,096 | tiny matmuls under-utilize ANE tiles |
| `vocab` | 256 (byte) | 128,256 (BPE) | LM-head matmul too small to matter |
| Norm | LayerNorm | RMSNorm | RMSNorm has better ANE op coverage |
| Positional | learned absolute | RoPE | ANE's fused-attention paths assume RoPE |
| MLP activation | GELU | SwiGLU | SwiGLU is the ANE-tuned default |
| Linear bias | yes | no | bias-free fuses cleaner into matmul-add |

### What to build

A new ModelConfig preset — `mini-llama` — using only existing config
flags (every one of the above is already a knob):

```swift
ModelConfig(
    vocabSize: 32768,        // small BPE, multiple of 64
    contextLength: 2048,
    nLayers: 24,
    nHeads: 16,              // head_dim = 128
    nKvHeads: 4,             // GQA
    dModel: 2048,
    dMlp: 8192,
    useRoPE: true,
    useRMSNorm: true,
    useSwiGLU: true,
    tieEmbeddings: false,
)
// ~600M params; scale down to (1280, 16) for ~200M first cut
```

Plus `tinygpt to-coreml` exporter (~1-2 days): maps our transformer
ops to CoreML's op set, produces a `.mlpackage` that Instruments can
profile to see whether ANE actually engages.

### Realistic speedup expectations

| Path | Realistic tok/s |
|---|---|
| Current Huge on Mac GPU | 293-696 |
| Mini-Llama (~600M) on Mac GPU | ~150-400 |
| Mini-Llama on Mac ANE if it routes | ~400-1200 (~2-3× over its own GPU) |
| Mini-Llama in browser | ~5-20 (probably can't load; 600M near browser ceiling) |
| **Mac-ANE vs browser ratio** | **30-200×** depending on routing cleanliness |

### Probability analysis

Test 1 (ANEMLL works on Llama 3.1 on your machine) → confirms the
environment but NOT that our model routes. Independent reasons it
could still fail:

```
ANEMLL on Llama works?
├─ No  → done, environment broken
└─ Yes → environment confirmed
         └─ Build tinygpt to-coreml exporter
            └─ Convert + profile Mini-Llama
               ├─ All ops on ANE     → 🎉 ~30-50% chance, you win
               ├─ Partial split      → 🟡 ~40% chance, measure if net speedup
               └─ Nothing on ANE     → 😐 ~10-20% chance, GPU is the ceiling
```

### Cost-benefit (honest)

| Item | Cost | Outcome regardless of ANE result |
|---|---|---|
| Train Mini-Llama (200-600M) | 3-7 days mostly-background | Real Llama-architecture gallery model. Useful independently. |
| `tinygpt to-coreml` exporter | 1-2 days focused | Reusable for any future model. Useful independently. |
| Profile + iterate | 1-3 days unpredictable | Empirical learning either way. |

**Total**: 1-2 weeks calendar; dominated by training wall-clock.

### Why queued

- Requires lifting the current "no training" goal constraint
- The trained Mini-Llama IS a valuable artifact independent of ANE,
  so the conditional EV is positive — but only if you're willing to
  train.
- Doesn't deliver 10× on Mac-alone (realistic 2-3× ANE-over-GPU);
  delivers 30-200× only via the Mac-ANE-vs-browser combined ratio.
- The cheaper browser-sampling-benchmark (item 1 above) is a
  prerequisite to even know the current sampling ratio — should do
  that first.

### Apple's actual ANE landscape (for posterity)

- **CoreML** (public) — convert to `.mlpackage`, Apple's runtime
  decides per-op CPU/GPU/ANE dispatch. Heuristics are opaque. No way
  to force ANE.
- **ANEMLL** (community, github.com/Anemll/Anemll) — uses private
  CoreML internals to coerce more ops to ANE. Works on macOS
  Sequoia. **Historically breaks on every macOS update.** Hand-tuned
  for Llama-family.
- **"Stateful Models API"** (rumored late 2026) — would make ANE
  routing first-class. Not shipped.

There is no Apple-sanctioned "private beta" for ANE inference; that
phrasing was loose. The real options are the three above.

---

# 4. Research absorbed — paper × verdict

External-paper catalogue (was `docs/roadmap/recent_research.md`, now
archived at `docs/archive/recent_research.md`). Each row: technique →
one-line source → verdict pointing at where it lives in this codebase,
or why it doesn't.

## 4.1 Implemented (techniques we ship)

### Alignment / preference

| Technique | Source | Where it lives |
|---|---|---|
| DPO | [Rafailov et al., NeurIPS 2023](https://arxiv.org/abs/2305.18290) | `tinygpt dpo` |
| KTO | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) | `tinygpt dpo --variant kto` |
| ORPO | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) | `tinygpt dpo --variant orpo` |
| SimPO | [Meng et al., 2024](https://arxiv.org/abs/2405.14734) | `tinygpt dpo --variant simpo` |
| NEFTune | [Jain et al., NeurIPS 2023](https://arxiv.org/abs/2310.05914) | `--neftune` |

### PEFT

All in `native-mac/Sources/TinyGPTModel/PeftVariants.swift`, surfaced via `tinygpt sft`.

| Technique | Source | Where it lives |
|---|---|---|
| DoRA | [Liu et al., 2024](https://arxiv.org/abs/2402.09353) | default in `sft` |
| GaLore | [Zhao et al., 2024](https://arxiv.org/abs/2403.03507) | `Optimizers.swift` |
| LoftQ | [Li et al., ICLR 2024](https://arxiv.org/abs/2310.08659) | `PeftVariants.swift` |
| VeRA | [Kopiczko et al., ICLR 2024](https://arxiv.org/abs/2310.11454) | `PeftVariants.swift` |
| PISSA | [Meng et al., 2024](https://arxiv.org/abs/2404.02948) | `PeftVariants.swift` |
| LoRA+ | [Hayou et al., ICML 2024](https://arxiv.org/abs/2402.12354) | `PeftVariants.swift` |
| rsLoRA | [Kalajdzievski, 2023](https://arxiv.org/abs/2312.03732) | `PeftVariants.swift` |

### Quantization

| Technique | Source | Where it lives |
|---|---|---|
| GPTQ | [Frantar et al., ICLR 2023](https://arxiv.org/abs/2210.17323) | `tinygpt gptq` + `GPTQReader.swift` |
| AWQ | [Lin et al., MLSys 2024](https://arxiv.org/abs/2306.00978) | AWQ safetensors reader |
| HQQ | [Badri & Shaji, 2024](https://mobiusml.github.io/hqq_blog/) | `tinygpt hqq` |
| KIVI | [Liu et al., 2024](https://arxiv.org/abs/2402.02750) | KV cache quantization path |

### Inference / efficiency

| Technique | Source | Where it lives |
|---|---|---|
| Speculative decoding | [Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192) | `tinygpt train-heads --type medusa\|eagle` + decode loop |
| Medusa | [Cai et al., 2024](https://arxiv.org/abs/2401.10774) | same path, head type |
| EAGLE-2 | [Li et al., 2024](https://arxiv.org/abs/2406.16858) | same path, head type |
| StreamingLLM | [Xiao et al., ICLR 2024](https://arxiv.org/abs/2309.17453) | attention-sink path |

### Architecture variants

| Technique | Source | Where it lives |
|---|---|---|
| MTP | [Gloeckle et al., ICML 2024](https://arxiv.org/abs/2404.19737) | `Train.swift`, `docs/mtp.md` |
| Differential Transformer | [Microsoft 2024](https://arxiv.org/abs/2410.05258) | `DifferentialAttention.swift`, `--diff-attn` |
| Mixture of Depths | [Raposo et al., 2024](https://arxiv.org/abs/2404.02258) | soft sigmoid gate (hard top-K upstream-blocked) |
| LASER | [Sharma et al., ICLR 2024](https://arxiv.org/abs/2312.13558) | `tinygpt laser` |

### Optimizers

| Technique | Source | Where it lives |
|---|---|---|
| Sophia | [Liu et al., 2023](https://arxiv.org/abs/2305.14342) | `Optimizers.swift` |
| Lion | [Chen et al., NeurIPS 2023](https://arxiv.org/abs/2302.06675) | `Optimizers.swift` |
| Muon | [Jordan, 2024](https://kellerjordan.github.io/posts/muon/) | `Optimizers.swift` |
| GaLore | (see PEFT) | `Optimizers.swift` |

### Distillation

| Technique | Source | Where it lives |
|---|---|---|
| Soft-targets distillation | Hinton et al., 2015 | `tinygpt distill` |

### Synthetic data

| Technique | Source | Where it lives |
|---|---|---|
| Magpie | [Xu et al., ICLR 2025](https://arxiv.org/abs/2406.08464) | `tinygpt magpie` |
| TinyStories | [Eldan & Li, 2023](https://arxiv.org/abs/2305.07759) | dataset source |

### Test-time compute

| Technique | Source | Where it lives |
|---|---|---|
| Best-of-N | [Snell et al., 2024](https://arxiv.org/abs/2408.03314) | `tinygpt bon --scan` |

### Evolution Strategies

| Technique | Source | Where it lives |
|---|---|---|
| ES at scale | [Qiu et al., Sept 2025](https://arxiv.org/abs/2509.24372) | `tinygpt es`, `docs/evolution_strategies.md` |

## 4.2 Cannot — blocked, parked, or skipped

### 🚧 Blocked by hardware

| Technique | Source | Why parked |
|---|---|---|
| BitNet b1.58 | [Ma et al., 2024](https://arxiv.org/abs/2402.17764) | Ternary from-scratch needs 100B+ tokens to validate; not differentiating at <1B params on our hardware. Park; revisit if a clear gallery-model use case appears. |
| FP4 training (NVFP4 / Quartet) | [Wang Jan 2025](https://arxiv.org/abs/2501.17116) · [Quartet II Jan 2026](https://arxiv.org/abs/2601.22813) | Apple M-series has no native FP4 ops |
| FP8 training | — | Needs H100 / Blackwell |

### 🚧 Blocked upstream

| Technique | Source | Why parked |
|---|---|---|
| Hard sparse MoE routing | DeepSeek-V3 family | MLX-Swift no `scatter_add`; soft (dense) routing ships |
| Real QLoRA | [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) | MLX-Swift quantized arrays don't autograd through; manual fake-quant shipped (pedagogical, no memory win) |

### ❌ Skipped — different family / not worth the seat

| Technique | Source | Why skipped |
|---|---|---|
| Mamba / Mamba-2 | [Gu & Dao, 2023/2024](https://arxiv.org/abs/2312.00752) | Linear-time SSM, different family; better as side-project |

### ❌ Dropped — value-add filter (subsumed by what ships)

| Technique | Source | Subsumed by |
|---|---|---|
| IPO | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) | DPO with high β regularizes equivalently |
| CPO | [Xu et al., 2024](https://arxiv.org/abs/2401.08417) | DPO + BC term marginal over SimPO at our scale |
| Self-Instruct | [Wang et al., 2023](https://arxiv.org/abs/2212.10560) | Magpie (model's own distribution; no seed needed) |
| Evol-Instruct | [Xu et al., 2024 (WizardLM)](https://arxiv.org/abs/2304.12244) | Magpie subsumes |
| MiniPLM | [Gu et al., NeurIPS 2024](https://openreview.net/forum?id=tJHDw8XfeC) | Distill-for-pretraining — needs a teacher-student pair we don't have |
| Distillation with Training Wheels | [Feb 2025](https://arxiv.org/abs/2502.17717) | `cloud-escalate` already provides the analogous "student asks teacher" deployment shape |
| DEITA | [Liu et al., 2024](https://arxiv.org/abs/2312.15685) | Instruction-data quality framework — only matters once SFT corpus > 1M samples |

## 4.3 Planned — queued for a future training run

| Item | Source | Where in §3 |
|---|---|---|
| GRPO / DAPO (RLVR pipeline) | [DeepSeek-R1, Jan 2025](https://arxiv.org/abs/2501.12948) · [DAPO, March 2025](https://arxiv.org/abs/2503.14476) | **Tier 5 §5.1** — Reasoning training on a 22M model. GRPO = mental model; DAPO = implementation. |
| Reasoning-trace distillation | DeepSeek-R1-Distill series, OpenThoughts | Tier 5 §5.1 — SFT-on-traces is the first half of §5.1 before RLVR |
| Snell test-time-compute scaling experiment | [Snell et al., 2024](https://arxiv.org/abs/2408.03314) | Tier 5 §5.2 — `bon` shipped; the scaling-curve experiment at 22M matches Snell methodology |
| Vision-language toy | LLaVA family | Tier 5 §5.3 |
| Diffusion LM micro | (multiple) | Tier 5 §5.4 |
| Real sparse MoE kernels | DeepSeek-V3 style | Tier 5 §5.5 (also upstream-blocked on `scatter_add`) |
| TTS toy | VALL-E / MusicGen family | Tier 5 §5.6 |

**Small additions, no current owner — append when a slot opens:**

| Item | Source | Effort |
|---|---|---|
| LISA optimizer | [Pan et al., 2024](https://arxiv.org/abs/2403.17919) | ~1 day; layerwise importance sampling, drop-in alongside Sophia/Muon |
| MiniLLM KL variants | [Gu et al., ICLR 2024](https://arxiv.org/abs/2306.08543) | ~1-2 days; reverse-KL / skew-KL switches on top of existing `tinygpt distill` |
| Distilling Step-by-Step | [Hsieh et al., ACL 2023](https://arxiv.org/abs/2305.02301) | ~1-2 days; rationale-distillation recipe on top of `tinygpt distill` |
| DoReMi data-mixture optimization | [Xie et al., NeurIPS 2023](https://arxiv.org/abs/2305.10429) | Park until ≥3 distinct domains are mixed at non-trivial scale |
| Quality classifier (FineWeb-Edu-style) | [Penedo et al., 2024 — FineWeb / FineWeb-Edu](https://arxiv.org/abs/2406.17557) | §3 B10 — ~2 days; tiny fastText scorer + top-X% filter |
| WSD schedule (warmup-stable-decay) | [MiniCPM, Hu et al., 2024](https://arxiv.org/abs/2404.06395) · [SmolLM blog](https://huggingface.co/blog/smollm) | §3 B11 — ~half-day; decay phase doubles as annealing |
| Interp-on-checkpoints methodology | [Pythia, Biderman et al., 2023](https://arxiv.org/abs/2304.01373) · [OLMo, Groeneveld et al., 2024](https://arxiv.org/abs/2402.00838) | §3 B13 — 1-2 days infra + ongoing analysis; replay SAE / MEMIT across the checkpoint timeline |
| Speculative decoding | [Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192) · [Chen et al., 2023](https://arxiv.org/abs/2302.01318) | §3 B14 — 2-3 days; Mini-Llama draft for Mega; numerics gate required |
| Layer-wise LR decay (SFT) | [ULMFiT, Howard & Ruder, 2018](https://arxiv.org/abs/1801.06146) | §3 B15 — ~half-day flag add on existing optimizer |
| M5 GPU Neural Accelerator prefill benchmark | [Apple ML Research, 2026](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) | §3 B16 — ~half-day; verify the claimed 3.5× M5-vs-M4 prefill speedup is materializing on our path |
| SAE Lens interop / Neuronpedia format export | [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens) | §3 B17 — ~2 days for format-export option; compare-and-decide before building |
| nanochat-style `--depth` single-knob HP derivation | [karpathy/nanochat](https://github.com/karpathy/nanochat) | §3 B18 — ~1 day; one knob auto-derives width / heads / LR / batch / steps; UX win |
| Group-SAE (layer-group SAE training) | [Wang et al., 2024](https://arxiv.org/abs/2410.21508) | §3 B19 — 2-3 days; trains SAEs once per layer-group instead of per-layer; cuts SAE training cost |
| Learnable cross-stream attention (modded-nanogpt speedrun trick) | [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | §3 B20 — read-and-evaluate; speedrun-specific, not yet a paper |
| ScaleDown extractive context compression SLM | [ScaleDown blog](https://tinyml.substack.com/p/how-we-train-small-language-models) · [Challenge leaderboard](https://main.d3hbeukddvrxcc.amplifyapp.com/leaderboard) · [scaledown.ai](https://scaledown.ai/) | §3 B25 — 3-5 days; token-level relevance head + sentence aggregation; submit to public leaderboard as a "specialist trained on a Mac" proof-point |
| Micro-AutoMixer for specialist data mixes | [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive) · RegMix/DoReMi-style mixture search | §3 B21 — small proxy-run version of Poolside's automixing; optimize specialist ratios before full training |
| Token-preserving agent trajectory recorder | [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive) | §3 B22 — preserve token IDs through rollout → training so agent traces cannot drift through retokenization |
| Agent eval protocol hardening | [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive) | §3 B23 — repeated pass@1, fixed step/resource/sampling budgets, and explicit infra-patch notes |
| Muon large-scale re-benchmark | [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive) · [Jordan, 2024](https://kellerjordan.github.io/posts/muon/) | §3 B24 — only revisit if large/proxy matmul-dominated runs amortize Newton-Schulz overhead |

## 4.4 Reference reads (no verdict — context only)

For mental-model framing, not techniques to implement:

- **State of GPT** (Karpathy, 2023) — pretrain → SFT → RM → PPO; we skip RM/PPO for DPO
- **Tulu 3** ([Lambert et al., 2024](https://arxiv.org/abs/2411.15124)) — open RLVR recipe; informs §5.1
- **SmolLM blog** ([HF, 2024](https://huggingface.co/blog/smollm)) — 135M/360M/1.7B small-model recipe
- **HuggingFace Alignment Handbook** ([repo](https://github.com/huggingface/alignment-handbook)) — reference SFT/DPO recipes at 7B
- **Survey of LLMs** ([Zhao et al., arXiv 2303.18223](https://arxiv.org/abs/2303.18223)) — broad survey, continuously updated
- **On-Policy Distillation Survey** ([April 2026](https://arxiv.org/abs/2604.00626)) — confirms distillation dominates for shipping small models

**2026 small-model peers** (for positioning, not adoption): SmolLM3-3B · Qwen3.5-0.8B · Phi-4-mini-instruct · Gemma-3n-E2B-IT · [Gemma-4-12B Unified](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) (encoder-free multimodal, 256K ctx, MLX variants exist). Implication: the niche is "browser-trainable + every byte of training code is here," not "perf-competitive with Phi-4."

**Direct from-scratch peers (full pipeline, not just pretrain):**
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — tokenizer → pretrain → SFT → RL → CLI/web chat in one repo. $48/2h on 8×H100. Apple Silicon mode exists via `runs/runcpu.sh` (degraded scale). **No interpretability story.** Single `--depth` knob auto-derives all HPs. Closest head-on competitor; differentiation = Mac-first + interp lab.
- [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — speedrun fork; April 2026 record 1.35 min to GPT-2 quality on 8×H100. Playbook: Muon (we have) · FA3 · FP8 head (HW-blocked) · learnable cross-stream attention · MTP (queued).
- [Poolside Laguna XS.2 / M.1 deep dive](https://poolside.ai/blog/laguna-a-deeper-dive) — agentic coding models with open XS.2 weights, strong SWE/Terminal benchmark protocol, quality+diversity data curation, synthetic data throughout pretraining, automixed data ratios, Muon at scale, and async agent RL. **Steal the workflow discipline, not the scale:** data-mix proxy sweeps, token-preserved agent traces, repeated eval protocol, and Muon only after large-scale re-benchmark.

**Tools worth knowing**:
- [Unsloth](https://github.com/unslothai/unsloth) — Triton-kernel fine-tune framework; not Mac/MLX but study for technique transfer. **Feb 2026**: 12× faster MoE training + embedding model support + ultra-long-context RL.
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) — config-driven multi-GPU production fine-tuner; multimodal support landed 2026
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — web-UI fine-tuner (LlamaBoard); zero-config entry point
- [TorchTune](https://github.com/pytorch/torchtune) — Meta's PyTorch-native fine-tuner; ~20-24% speedup via PyTorch 2.5 compile
- [Argilla Distilabel](https://github.com/argilla-io/distilabel) — Python pipeline for synthetic SFT/DPO (wraps Magpie/DEITA)

**Apple Silicon ecosystem (direct peers on our platform):**
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple's official MLX inference + LoRA / DoRA / QLoRA / full fine-tune + OpenAI-compatible server. Direct overlap with our SFT/DPO LoRA path; differentiation = pretrain + interp + GGUF/CoreML export.
- [Ollama + MLX backend (v0.19, March 2026)](https://markaicode.com/run-fine-tune-llms-mac-mlx-lm/) — prefill 1154→1810 tok/s, decode 58→112 tok/s on Apple Silicon. Direct competition for our GGUF runner.
- [exo-explore/exo](https://github.com/exo-explore/exo) — multi-Mac P2P distributed inference. JACCL collectives over RDMA-on-Thunderbolt-5 on macOS 26.2 → 1.8×/3.2× speedup on 2/4 devices. Out of single-machine scope, but the infra is new.

**Interpretability ecosystem (overlap with our interp lab):**
- [SAELens](https://github.com/decoderesearch/SAELens) — established SAE training/analysis library; integrates with TransformerLens + HF + nnsight + [Neuronpedia](https://www.neuronpedia.org/). Our SAE may be reinventing; B18 task = compare + decide on interop format.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) · [nnsight (NDIF)](https://nnsight.net/) — PyTorch interp infra; complementary to SAELens. We have native Swift/MLX equivalents.

**Proprietary / out of scope**: OpenAI o1 / o3 (closed-weights; reframed the field around test-time compute, no adoptable artifact). DeepSeek-V3 (671B-MoE, scale-blocked; informs MTP + MoE design). Qwen3 (model family, not a technique).

## 4.5 Coverage cutoff

The catalogue was hand-curated up to assistant knowledge cutoff
**January 2026** plus best-effort web-search additions for Feb-May
2026 (coverage spottier there). Today is **2026-06-04**.

**2026-06-04 web sweep folded in** — five surfaces were checked
(Apple Silicon training, nanoGPT successors, Mac inference runtimes,
interpretability libraries, fine-tune frameworks). Results: nanochat
+ modded-nanogpt added as direct from-scratch peers; mlx-lm +
Ollama-MLX + EXO added as Apple Silicon ecosystem peers; SAELens
added as interp peer; B16-B20 queued in §3 from surfaced gaps;
Unsloth Feb-2026 release notes folded into tools row. Coverage of
Feb-Jun 2026 papers is now meaningfully better but still not
exhaustive.

Future papers append row-by-row into §4.1 / §4.2 / §4.3.

---

# Appendix — index of source docs absorbed by this file

This doc replaces the multi-file roadmap split. The source docs are kept for context but should be treated as historical; **edit this file**, not them.

| Old doc | What it covered | Status |
|---|---|---|
| `docs/roadmap/index.md` | TOC for the multi-file split | Superseded — point at this file |
| `docs/roadmap/tier1.md` / `tier2.md` / `tier3.md` | ROI-tiered technique inventory | Absorbed; markers refreshed |
| `docs/roadmap/tier4_skip.md` | Intentionally-not-built items | Absorbed into §2 |
| `docs/roadmap/tier5_frontier_2026.md` | 2026 research frontier | Absorbed into §3 Tier 5 |
| `docs/roadmap/categories.md` | Orthogonal technique taxonomy (had stale markers) | Absorbed; refreshed against code |
| `docs/roadmap/blockers.md` | What we can't build + Phase 9/10 status appendix | Absorbed into §2 + §1 |
| `docs/roadmap/phased_plan.md` | 7-week sequential plan | Mostly shipped; remainder in §3 |
| `docs/roadmap/recommended_order.md` | Top-10 next | Superseded by Tier A/B ordering in §3 |
| `docs/roadmap/honest_summary.md` | "CAN / CAN'T / SHOULDN'T" framing | Absorbed |
| `docs/progress.md` | Mac+Web shipped dashboard | Absorbed into §1 |
| `docs/backlog.md` | ROI-ordered "what's left" (Tier A/B/C/D) | Absorbed into §3 |
| `docs/feature_audit_2026_05_31.md` | CLI smoke audit | Cross-referenced; was the verification baseline |
| `docs/roadmap/recent_research.md` | Paper catalogue (2024-2026) | Absorbed into §4; archived at `docs/archive/recent_research.md` |

**Still canonical (deep dives, not absorbed)**: `docs/roadmap/datasets.md`,
`docs/roadmap/north_star_refined.md`, and the per-technique docs
(`distillation.md`, `interpretability.md`,
`moe.md`, `mtp.md`, `lora_guide.md`, `precision.md`, `memory_tradeoffs.md`,
`perf_quest.md`, `decision_log.md`). Those don't duplicate planning — they
explain *how* shipped pieces work.
