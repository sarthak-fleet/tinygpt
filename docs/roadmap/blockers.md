# Roadmap — what we can't add right now

Categorized blockers. These are NOT the "skip" items
([`tier4_skip.md`](tier4_skip.md)) — those are things we deliberately
won't build because better alternatives exist. These are things we'd
build but **can't** for external reasons.

## Blocked by hardware

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Distributed training (ZeRO, FSDP, pipeline parallelism, tensor parallelism)** | Single device only; nothing to parallelize across | Buy/rent a multi-GPU cluster — not the project's scope |
| **Native FP4 training** | Mac M-series GPU lacks FP4 tensor ops | Apple ships FP4 support (rumored on future M-series; not current) |
| **Native FP8 training** | Same — no FP8 ops on Apple silicon | Same |
| **Hardware-accelerated MoE routing** | Apple silicon doesn't have specialized sparse-routing ops | Same |
| **ANE (Apple Neural Engine) acceleration of training** | ANE is inference-only; not exposed for training | Apple opens ANE training APIs (no public roadmap) |

## Blocked by external library state

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Gradient checkpointing as first-class** | MLX-Swift doesn't expose it yet (would write custom forward — possible but invasive) | MLX-Swift adds API (tracked upstream); or we ship a custom impl as Phase 6 |
| **Fast BPE encoding** | swift-transformers BPE is single-threaded; 2 GB corpus takes ~30 min | Wait for swift-transformers improvements OR write a Rust-backed encoder via FFI |
| **Native int4 / int8 matmul on browser WebGPU** | WebGPU doesn't yet have quantized matmul extensions | Wait for WebGPU spec (subgroup / coop-matrix extensions in Phase 7 help) |
| **AWQ / GPTQ / GGUF model loading** | Some Swift readers don't exist yet (AWQ shipped; GPTQ + GGUF pending) | We could write them — just hasn't been done |
| **`scatter_add` for sparse MoE / MoD compute savings** | MLX-Swift doesn't expose it; blocks the hard-top-K + scatter variants. Soft routing ships in both. | Upstream PR. |

## Blocked by budget / cost

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Tinker / managed cloud training APIs** | Usage-based pricing; not affordable for solo project | Project becomes funded |
| **Large-scale synthetic data generation via GPT-4 / Claude API** | $1K-$10K to generate Magpie-scale (~1M pairs) of frontier-quality SFT data | Use open-weights teachers instead (Magpie pipeline does this) |
| **Multi-TB dataset downloads** | Bandwidth + disk for full Common Crawl / Pile | Stream subsets (the HF importer does this); full corpora not needed at our scale |
| **Strong local judge model for Constitutional AI / RLAIF** | No 70B+ model fits + runs at usable speed on a single Mac | Hardware grows OR use a smaller (worse) judge with explicit caveat |

## Blocked by knowledge cutoff

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Anything published after January 2026** | Assistant training cutoff | User pastes URLs / paper names; folded in |
| **Late-2025 / early-2026 alignment recipes** | Patchy coverage of Nov 2025 onward | Same |
| **Cutting-edge benchmark / dataset releases** | Same | Same — see how DeepSeek-R1, DAPO, Magpie all needed web search to verify |

## Blocked by integration scope

| Item | Why blocked | Unblock condition |
|---|---|---|
| **Full RLHF / PPO pipeline with reward model training** | Real cost is 5× the code of DPO + 10× the iteration time; usually skipped at our scale | DPO already covers 80-90% of the value |
| **Mass-scale Constitutional AI / RLAIF** | Requires generating + judging millions of model outputs | Smaller-scale exploration possible if needed |
| **State space models (Mamba/Mamba-2)** | Whole different architecture; ~2-3 week port; reuses almost nothing | Become a separate side-project (Tier 4) |
| **Diffusion language models** | Different paradigm; whole new codebase | Side-project |

## Cross-cutting blockers (root causes)

1. **MLX-Swift doesn't expose `mlx_checkpoint`** — blocks gradient
   checkpointing (Phase 6). The C primitive exists; the Swift
   wrapper doesn't. Workarounds in
   [`docs/memory_tradeoffs.md`](../memory_tradeoffs.md).
2. **MLX-Swift doesn't expose `scatter_add`** — blocks sparse MoE
   compute and MoD compute savings (Phase 5, Phase 10). Workarounds
   in [`docs/moe.md`](../moe.md) and below.
3. **Cmlx is internal to MLX-Swift** — neither of the above
   primitives can be bridged from outside the package without
   forking MLX-Swift. The right resolution is upstream PRs.

These are real engineering tasks, not session-sized work. Each
unblocks several roadmap items simultaneously — landing them is
the highest-leverage move for the next phase of work.

---

# Appendix — Phase 9 / 10 status detail

This appendix closes out the remaining Phase 9 (quantization) and Phase 10
(architecture menu) items. For each: what's shipped today, and for the
items not yet shipped, what's needed to land them.

## Phase 9 — quantization

| Item | Status | Notes |
|---|---|---|
| DoRA | ✅ shipped | `--dora` flag on sft + dpo. Adapter file format extension is queued. |
| LASER selective rank reduction | ✅ shipped | `tinygpt laser` command. File-level SVD truncation. |
| HQQ (half-quadratic quantization) | ✅ shipped — storage-only | `tinygpt hqq` command. IRLS solver with sub-quadratic loss runs in Swift; writes a model whose weights have been quantize-then-dequantised. Inference-time memory win still needs a packed-int4 matmul kernel. |
| AWQ safetensors reader | ✅ shipped | `AWQReader.swift`. Detects qweight/scales/qzeros triples in HF safetensors, unpacks the GEMM-pack int4 layout into dense fp32 weights the existing HFModelLoader consumes. |
| QLoRA (int4 base + LoRA) | 📋 designed | Blocker: MLX-Swift's quantized arrays don't yet fwd-prop gradients through to the underlying float matrices — see "QLoRA" below. |

### QLoRA — what's needed

Concept: load the BASE model in int4 (e.g. via existing `--quantize int4`
or AWQ), then attach a normal LoRA on top. Training only updates the
LoRA — gradient flows through the int4 base as a constant.

Two pieces are missing:

1. **Gradient passes through quantized weights.** Today,
   `MLXNN.quantize(model:...)` swaps Linear for QuantizedLinear, which
   is purely an inference module — its weight isn't a regular
   `@ParameterInfo` MLXArray that autograd accepts. Until MLX-Swift
   either makes quantized weights gradient-transparent (treating them
   as no-grad constants in the trace) OR exposes a "frozen quantized
   constant" type that gradient can flow PAST, we can't run backward
   through a quantized base.

   Workaround idea: do the quantization MANUALLY in user code — keep
   the base as a regular fp32/bf16 `Linear` whose `weight` is held
   constant via `freeze()`, but apply a fake-quant function in the
   forward (cast → round → cast back). Loses the memory win but
   preserves the gradient flow. Useful pedagogically; not the real
   QLoRA story.

2. **Persistent quantized base loading.** If we want QLoRA on an
   AWQ-quantized HF model, the AWQ reader below is the prerequisite.

### AWQ reader

AWQ (Lin et al., 2023) safetensors files store weights as
`qweight` (int32-packed 4-bit), `qzeros`, and `scales` per output
channel. Reading is mechanical:

```swift
// inside HFModelLoader.makeMLXArray when dtype == "I32" and name
// ends in ".qweight", and a sibling "scales" + "qzeros" exist:
let unpacked = unpackAwqInt4(qweight, scales, qzeros)
return MLXArray(unpacked, originalShape)
```

The conversion produces a dense fp16/fp32 representation that the
existing forward path can use unchanged. The pure-AWQ runtime
(matmul against packed int4 directly) would need a kernel.

### HQQ

HQQ (Badri & Shaji, 2023) uses convex optimization to find better
quantization scales than the naive min-max approach. The algorithm:

1. Group weights into blocks of size G (e.g. 64).
2. For each block, solve a small convex problem:
   minimise `‖W - dequant(quantize(W; scale, zero))‖₂` over (scale, zero).
3. Store (quantized weights, scale, zero) per block.

The optimisation is fast (closed-form per block). The inference-time
win requires a Metal kernel that does grouped int4 matmul against
the block layout — same kernel-engineering bar as the sparse MoE
dispatch. The quantization step itself is Swift-side and feasible.

## Phase 10 — architecture menu

| Item | Status | Notes |
|---|---|---|
| Sliding window attention | ✅ shipped | `--sliding-window N` flag, persisted in header. |
| ALiBi position bias | ✅ shipped | `--alibi` flag, per-head geometric slopes. |
| Differential attention | ✅ shipped | `--diff-attn` flag. `DifferentialAttention.swift` with 2× Q/K projections, learnable λ. Wired via Optional sibling on TransformerBlock (same pattern as MoE). |
| Mixture of Depths | ✅ shipped — soft routing | `--mod` flag. Per-token sigmoid gate on each block's residual contribution. Soft routing (no STE) means it's trainable end-to-end. Hard top-K + scatter still blocked on `scatter_add`. |
| YOCO cross-layer KV sharing | 📋 designed | Needs CausalSelfAttention to accept externally-cached K/V — bigger API change than other items. Mechanism in detail below. |

### Differential attention (Ye et al., 2024)  *(shipped)*

`DifferentialAttention.swift` + `--diff-attn` flag on `tinygpt train`.
Each attention head computes TWO independent softmax attention maps
and subtracts them, weighted by a learnable scalar λ:

```
A = softmax(Q1 K1ᵀ / √d) − λ · softmax(Q2 K2ᵀ / √d)
out = A · V
```

Wired via an Optional sibling on TransformerBlock — when
`cfg.useDifferentialAttention` is set, `diffAttn` is constructed
alongside the standard `attn` and the forward routes through it.
The standard `attn` stays constructed (small constant overhead) in
exchange for keeping every existing LoRA / KVCache / Debug call site
that touches `block.attn.qProj` etc. unchanged.

Simplifications from the paper:
- λ is a SINGLE learnable scalar, not the per-head re-parameterised
  `λ_init − exp(λ_q · λ_k)`.
- λ_init defaults to 0.5 (paper uses depth-dependent init).
Both are precision improvements — bounded follow-up.

### YOCO — "You Only Cache Once"  *(still designed)*

Lin et al., 2024. The model is split in two halves. The first half
computes K, V normally. The second half does CROSS-ATTENTION onto the
last K, V produced by the first half — no new K, V are computed for
those layers. KV cache memory drops by ~2× at long context.

**Why it didn't ship in this round**: CausalSelfAttention's forward
treats Q, K, V as locally-computed. Adding cross-attention requires
either:

1. A second "CrossAttention" module with the same call surface but
   K, V come from a caller-supplied source. Then half the blocks
   construct CausalSelfAttention, half construct CrossAttention.
   The model's forward captures the last K, V of the first half and
   plumbs them through. ~150 lines.
2. Refactoring CausalSelfAttention itself to optionally take
   external K, V tensors. Less new code but more invasive (every
   existing call site has to ignore the new optional). ~100 lines.

Either works; both need a careful pass on the KV-cached sampling
path (`KVCache.swift`, `KVCacheHF.swift`) where the cross-attention
layers DON'T grow their own cache.

### Mixture of Depths (Raposo et al., 2024)  *(shipped — soft routing)*

`--mod` flag on `tinygpt train`. Each TransformerBlock gains a
per-token sigmoid gate:

```
out = x + sigmoid(router(x)) · (block(x) − x)
```

Tokens the router scores low pass through unchanged; tokens it scores
high get the full block treatment. Init bias zero → gate ≈ 0.5 → block
fires half-strength at init; training pushes the gate towards 0 or 1
per token.

**Shipped variant**: soft routing only. The hard-top-K + scatter
variant (the version that ACTUALLY saves compute) is blocked on the
same `scatter_add` upstream gap as sparse MoE — see
[`docs/moe.md`](../moe.md). Soft routing gives the architectural change
+ training signal without the compute saving. When `scatter_add` lands,
swap the sigmoid gate for argTopK + STE and the compute saving lands too.

## Phase 8 — interpretability remainder

| Item | Status | Notes |
|---|---|---|
| Logit lens | ✅ shipped | Button in browser playground. |
| Attention heatmap | ✅ shipped | Existing "Watch the model think" panel. |
| Per-layer ablation | ✅ shipped | "Ablate & sample" button. |
| Activation patching | ✅ shipped — both variants | Worker `patch` message + `GpuModel.generatePatched`. Zero-patch zeroes the residual stream at (layer, position); donor → recipient SWAP captures another prompt's hidden state at (donorLayer, donorPosition) and substitutes it at the recipient. Both wired to the playground's Inspect panel ("Zero & sample" and "Swap & sample" buttons). |
| Tuned lens | ✅ shipped | `tinygpt tuned-lens` Mac CLI command trains per-layer probes on a frozen base. Sidecar `.lenses` file format. `TinyGPTModel.forwardTunedLens` for inference once loaded. Browser playground accepts the sidecar via the "Upload .lenses" affordance and routes the lens button through the tuned probes when loaded. |

### Activation patching (Meng et al., 2022)  *(shipped — zero + donor-swap)*

`webgpu/train.wgsl` exposes two patching kernels: `patch_zero` replaces
one residual-stream row with zeros, `patch_replace` replaces it with
the contents of a separately captured donor row. The worker's `patch`
message accepts an optional `donor: { prompt, layer, position }`
field per patch; when present, `GpuModel.captureHidden` runs a forward
over the donor prompt, downloads the residual-stream slice at the
donor coords, and `GpuModel.generatePatched` uploads it and routes the
recipient's forward through `patch_replace`.

UI: the playground's Inspect panel exposes both as separate buttons —
"Zero & sample" for the causal intervention (which tokens depended on
that representation being intact?) and "Swap & sample" for the full
Meng et al. variant (which downstream outputs causally depend on the
donor's representation at that coord?). The donor prompt, layer, and
position are entered alongside the recipient coords.
