---
name: ANE research dossier — synthesize Apple's published patterns + beyond
status: shipped-v1-2026-06-07
owner: unassigned (parallel-agent task — research-only, no code)
created: 2026-06-08
priority: P1 — equips the ANE elf for M6/M7/M8; not itself implementation
parallel-safe: yes (research output only; no file conflicts with ANE elf)
supports: factory-ane-inference-pace.md (M6, M7, M8)
---

# PRD — ANE research dossier

## Ship note — 2026-06-07

V1 landed at `docs/learn/ane-research/dossier.md`. It covers ANE
layout patterns, Core ML stateful-model constraints, quantization
order, Qwen3-specific risk, blocker hypotheses, and a prioritized
M6/M7/M8 experiment sequence. This is research synthesis only; it does
not claim ANE runtime success.

## Why this PRD exists

The ANE elf is about to embark on M6 (diagnostic), M7 (`ml-ane-
transformers` layout port), and M8 (beyond-reference research). They
need a synthesized reference doc — pulled from Apple's papers,
published code, and the coremltools release notes — to draw from.
Building that synthesis is research work, separate from the elf's
diagnostic + porting work.

This is a **research task with documentation output, not code**.

## Goal

Produce `docs/learn/ane-research/dossier.md` (300-500 lines) that
synthesizes everything the ANE elf needs to know about getting
arbitrary transformers running on the Apple Neural Engine in 2026,
on macOS 26 + coremltools 9.

The dossier becomes the reference document for the ANE elf and for
future ANE work generally.

## Scope — sections the dossier should cover

### 1. ANE compute model (background)

- What the Apple Neural Engine actually is (16-core / 32-core matrix
  engines, NE1 / NE2 generations, M-series differences)
- INT8 / FP16 native support; what doesn't run natively
- Memory bandwidth, cache hierarchy, dispatch latency characteristics
- M5 Pro / M5 Max specifics (relevant for our hardware)

### 2. Apple's published patterns

- **ml-ane-transformers paper** (2022): "Deploying Transformers on the
  Apple Neural Engine". Key takeaways:
  - The `(B, C, 1, S)` layout convention and why
  - Conv2d-as-Linear pattern
  - LayerNormANE (custom layer norm)
  - ane_gelu / ane_silu (approximated activations that ANE compiles well)
- **ml-ane-transformers repo** (github.com/apple/ml-ane-transformers):
  - File-by-file walkthrough — what each module does
  - LLaMA reference implementation; what to copy for Qwen3
- **Apple's WWDC sessions** relevant to on-device LLM (2023-2026)
- **MobileCLIP / FastVLM** papers (Apple ML research, recent) —
  novel ANE patterns Apple has been publishing

### 3. coremltools 9 capabilities (and limits)

- Stateful models (MLState, MLStateModel) — what works, what doesn't
- Activation quantization (`ct.optimize.coreml.linear_quantize_activations`)
- Weight quantization options
- Known ANECCompile error codes and what they mean (especially -14)
- ANE supported / unsupported ops list (curated from coremltools
  source if not published)

### 4. Qwen3-specific concerns

- Architecture summary: GQA 16Q/8KV, QK-Norm, head_dim=128, SwiGLU,
  RoPE@1e6, tied embeddings, 28 layers
- For each architectural feature: how Apple's reference implementations
  (LLaMA, Mistral) handle the equivalent or similar feature
- Specific challenges Qwen3 brings beyond Apple's references

### 5. The blocker analysis

What we currently know about why ANECCompile fails on the full
Qwen3-0.6B graph:
- ANECCompile error -14 → "too many state slots" (resolved via
  consolidation per M3 ship note) AND/OR graph size limit
- Hypothesis tree: layer count, individual ops, layout mismatch
- What's been tried; what hasn't

### 6. Beyond-reference research arcs

For each M8 arc the PRD lists, what's known publicly:
- **Hybrid ANE/GPU dispatch**: what's documented in CoreML compute
  unit configs; what's not but might be possible via MPSGraph
- **Layer-chunked conversion**: how Apple does this in their own
  multi-billion-parameter examples (Stable Diffusion sub-model
  partitioning is a precedent)
- **ANE-native INT4/INT8**: coremltools 9 quantization APIs +
  recent papers (Apple's ML research has hinted at INT4 paths)
- **MPSGraph bridge**: when to use raw MPSGraph instead of CoreML
- **Sparse attention on ANE**: FastVLM paper specifics
- **Halved-depth distillation**: known precedents for 14-28 layer
  distillation in Qwen3 family or similar

### 7. Concrete next-steps recommendations

End the dossier with a prioritized recommendation list for the ANE
elf:
- M6 diagnostic experiments (which bisects to run first)
- M7 port specifics (which Apple reference is closest to Qwen3)
- M8 research arcs ranked by tractability vs payoff

## Sources to consult

**Required**:
- github.com/apple/ml-ane-transformers (full read)
- "Deploying Transformers on the Apple Neural Engine" (Apple ML
  research blog post / paper, 2022)
- coremltools 9.0 release notes + API reference
- developer.apple.com/documentation/coreml (stateful models section)

**Strongly recommended**:
- WWDC 2024 session "Bring your machine learning models to Apple
  silicon" (and equivalents from 2025/2026 if released)
- FastVLM paper (Apple ML research, recent)
- MobileCLIP paper (Apple ML research)
- ml-stable-diffusion repo (sub-model partitioning pattern)
- ml-mistral-coreml (if exists; community port)

**Worth scanning**:
- Apple Developer forums threads on CoreML + LLMs (2024-2026)
- coremltools GitHub issues tagged 'NeuralEngine'
- Recent papers citing ml-ane-transformers

## Scope — out

- Implementation work (that's M6/M7 in the ANE PRD)
- Benchmarking (that's M4/M5 in the ANE PRD)
- New paper writing (purely synthesis)
- Confidential / Apple-internal info (we have none)

## Acceptance

1. `docs/learn/ane-research/dossier.md` exists at 300-500 lines
2. All 7 sections covered with citations/links to sources
3. Section 7 (recommendations) is actionable — the ANE elf can pick
   up M6 with a concrete starting experiment
4. Future ANE work (M8 arcs) has a documented starting point per arc
5. Dossier readable by a maintainer-level reviewer in 30 minutes

## Estimated effort

**1-2 days** of research + writing. Reading is the bulk; synthesis
is mechanical once the sources are read.

## Why this is leverage (not redundant with the ANE elf)

The ANE elf is doing implementation. This task is doing synthesis.
Done in parallel, the elf has reference material when they need it
instead of having to interrupt to read sources. Pure leverage.

## Won't conflict with other elves

- No code touched, only docs
- New file in `docs/learn/ane-research/` (new directory)
- ANE elf doesn't write docs; we don't write code
