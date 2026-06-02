---
title: TinyGPT — master plan (shipped / skipped / TODO)
description: Single source of truth for what's shipped, skipped, and still to build. Consolidated from docs/roadmap/*, docs/progress.md, docs/backlog.md, docs/feature_audit_2026_05_31.md. Replaces them as the canonical reference; the older docs are now pointer stubs to this file.
---

# TinyGPT — master plan

**Last verified against codebase**: 2026-06-02
**Sources merged**: `docs/roadmap/*` · `docs/progress.md` · `docs/backlog.md` · `docs/feature_audit_2026_05_31.md`

Three sections — **shipped**, **skipped**, **TODO**. Every claim verified
against the code on 2026-06-02 (categories.md had stale ⬜ markers for
items that shipped weeks ago — fixed here).

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
- ✅ NEFTune (noisy embeddings)
- ✅ Gradient clipping (`--grad-clip F`, default 1.0, on train + sft + dpo)
- ✅ z-loss auxiliary (`--z-loss-weight F`)
- ✅ Embedding tying (`tieEmbeddings` config flag)
- ✅ Document-level shuffling (implicit via batch sampler)
- ✅ Gradient checkpointing (CustomFunction VJP workaround for missing `mlx_checkpoint`)
- ✅ QAT (in-training, `--qat`)

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
- ✅ GPTQ (`tinygpt gptq` — int4, 0.102 rel error)
- ✅ AWQ safetensors reader (loads HF AWQ-quantized weights)
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

## Tokenization

- ✅ Byte-level (vocab=256) — from-scratch path
- ✅ HF BPE / SentencePiece via swift-transformers

## Interpretability tools (browser playground)

- ✅ Logit lens (button + worker route)
- ✅ Tuned lens (Mac CLI trainer + `.lenses` sidecar + browser upload)
- ✅ Attention heatmap ("Watch the model think" panel)
- ✅ Per-layer ablation ("Ablate & sample" button)
- ✅ Activation patching — both variants (zero + donor-swap, shipped 2026-06-02 in `17021bc`)

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

## WebGPU kernels (in `webgpu/train.wgsl` + `train_sg.wgsl`)

- ✅ Naive scalar matmul
- ✅ Blocked 4×4 matmul (`matmul_blocked_vec4`)
- ✅ Layer-norm subgroup variant (gated on `gpuFeatures.subgroups`)
- ✅ Cross-entropy subgroup variant
- ✅ Bias-grad subgroup variant
- ✅ FA2 forward in WGSL (flash attention in browser)
- ✅ f16 storage (gated)
- ✅ OPFS persistence
- ✅ Patch kernels (`patch_zero` + `patch_replace` — 2026-06-02)

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
| GPTQ / GGUF safetensors readers | not yet written | Could write — just hasn't been done; AWQ is shipped |
| YOCO cross-layer KV sharing | API plumbing, not blocked technically | ~150 lines, designed |

## 🚧 Blocked by budget

- **Synthetic SFT via frontier API ($1-10K)** — use open-weights teacher via Magpie instead
- **Multi-TB dataset downloads** — stream subsets (the HF importer does this)
- **Strong local judge for Constitutional AI / RLAIF** — no 70B+ runs usable on a Mac
- **Public RLHF / PPO pipeline** — 5× the code of DPO + 10× iteration; DPO covers 80-90% of the lift

---

# 3. TODO

ROI-ordered. Sourced from `backlog.md` (the living list, last sort 2026-05-31).

## Tier A — DO NEXT (north-star aligned; specialists)

Until A1 lands, every optimization is theoretical.

- ⬜ **A1. Train first specialist end-to-end (tool-caller)** — 3-5 days execution + GPU hours. Validates north-star thesis.
- ⬜ **A2. Pull foundational datasets** (xlam-function-calling-60k, hermes-function-calling-v1, function-calling-chatml, SWE-bench_Verified, alpaca-cleaned, orca_dpo_pairs, MetaMathQA, ultrafeedback-binarized-preferences-cleaned, the-stack-smol, python_code_instructions_18k_alpaca) — ~1 hr wall
- ⬜ **A3. Fetch GitHub issue→PR corpus for debugger** — ~1 day with `GITHUB_TOKEN`
- ⬜ **A4. Pull BFCL + τ-bench via extractor-data** — ~30 min
- ⬜ **A5. Pull Indic eval datasets (MILU + IndicGenBench-XQuAD)** — ~30 min
- ⬜ **A6. Dataset inventory doc** — ~30 min after A2-A5
- ⬜ **A7. Real-data MILU baseline on flagship-huge-v5** — ~2 hr; depends on A5

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

## Tier C — POLISH (pick up when blocked)

- ⬜ **C1. CLI cosmetic fixes** — `tinygpt validate --help`, `hf-inspect --help`, `score-bench`, `prune-structured` arg-parsing quirks (~1 hr total)
- ⬜ **C2. Roll up pre-switch CLI shims into main switch** (`score-bench`, `pruning`, `agent`, `cloud-merge`, `hf-datasets-merge`, `github-data-merge`) — ~half day
- ⬜ **C3. DoRA on-disk adapter format** — ~1 day (today DoRA is in-session only)
- ⬜ **C4. Tool-call extractor: BPE tokenizer support** — ~2 days
- ⬜ **C5. Decode jitter under thermal load** — ~1 day
- ⬜ **C6. ChatML template: detect inline `system:` prefix and split** — ~half day (footgun in hermes-function-calling-v1)
- ⬜ **C7. Save+reload XCTest for LoRA adapters** — ~2 hr (regression coverage for the A0 bug)
- ⬜ **C8. Install-path discipline (no more `/tmp` for caches)** — ~1 hr (default to `~/.cache/tinygpt/`)

## Tier 5 — RESEARCH FRONTIER (2026 stretch goals)

Pauses the "training at 2024 fundamentals" cadence; deliverable is a paper-shaped artifact + reproducible code + a scaling-curve point, NOT a polished UX feature.

- ⬜ **5.1 Reasoning training on a 22M model** — 5-7 days; expected outcome is the *negative result* (CoT below emergence). Publishable.
- ⬜ **5.2 Test-time compute scaling** — 3-5 days; quality-vs-FLOPs plot at 22M-scale matching Snell et al. methodology. **Most cleanly publishable.**
- ⬜ **5.3 Vision-language toy** — ~2 weeks; ViT + projector + LLaVA-style. Smallest from-scratch VL model on consumer hardware.
- ⬜ **5.4 Diffusion LM micro-implementation** — 1-2 weeks; new paradigm via masked denoising loss.
- ⬜ **5.5 Real sparse MoE kernels** — 2-3 weeks; custom Metal kernel + measure FLOP reduction.

## Unshipped techniques (technique-inventory residue)

These were in the original roadmap categories but never built; none currently blocked, all low-priority unless triggered by a specific use case.

**Post-training**
- ⬜ IPO (DPO variant for small ~1K-pair datasets)
- ⬜ Prefix tuning / soft prompts (~1 day)
- ⬜ ReLoRA (periodic merge + restart, ~2 days)

**Data techniques**
- ⬜ Curriculum learning · DoReMi (data-domain mixing ratios) · Data quality filtering (PPL-based)
- ⬜ Deduplication (matters mainly for raw web scrapes) · Hard example mining
- ⬜ Importance sampling · Self-instruct · Evol-instruct
- ⬜ Sample packing (cross-source, different from sequence packing)

**Tokenization**
- ⬜ BPE-dropout · Train own BPE on our corpus · Vocabulary trimming · tiktoken adoption · Subword regularization

**Training stability**
- ⬜ Embedding RMSNorm · Layer-wise LR decay · Cosine / exp warmup · DeepNorm (matters past ~50 layers)

**Inference**
- ⬜ Token elimination · Tree decoding

**Interpretability**
- ⬜ Linear probes · Sparse autoencoders (substantial build) · Knowledge editing (ROME / MEMIT)

**Architecture**
- ⬜ BigBird / Longformer sparse attention (Tier 4 unless we go past ctx=8192)
- ⬜ Linear attention (Performer / Linformer / Reformer)
- ⬜ Hybrid attention/SSM (Jamba / Samba)
- ⬜ Pre-norm vs post-norm toggle

**Infra**
- ⬜ TinyGPT-as-library API (surface `forward_backward`, `optim_step`, `sample`, `save_state`)
- ⬜ Real CI (GitHub Actions: build + test on every PR)
- ⬜ Persistent tokenized cache (saves the 30-min BPE-encode cost on every run)

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

**Still canonical (deep dives, not absorbed)**: `docs/roadmap/datasets.md`,
`docs/roadmap/recent_research.md`, `docs/roadmap/north_star_refined.md`,
and the per-technique docs (`distillation.md`, `interpretability.md`,
`moe.md`, `mtp.md`, `lora_guide.md`, `precision.md`, `memory_tradeoffs.md`,
`perf_quest.md`, `decision_log.md`). Those don't duplicate planning — they
explain *how* shipped pieces work.
