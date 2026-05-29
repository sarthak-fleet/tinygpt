# Phase 9 + 10 — status and follow-up design

This doc closes out the remaining Phase 9 (quantization) and Phase 10
(architecture menu) items. For each: what's shipped today, and for the
items not yet shipped, what's needed to land them.

---

## Phase 9 — quantization

| Item | Status | Notes |
|---|---|---|
| DoRA | ✅ shipped | `--dora` flag on sft + dpo. Adapter file format extension is queued. |
| LASER selective rank reduction | ✅ shipped | `tinygpt laser` command. File-level SVD truncation. |
| HQQ (half-quadratic quantization) | ✅ shipped — storage-only | `tinygpt hqq` command. IRLS solver with sub-quadratic loss runs in Swift; writes a model whose weights have been quantize-then-dequantised. Inference-time memory win still needs a packed-int4 matmul kernel. |
| AWQ safetensors reader | ✅ shipped | `AWQReader.swift`. Detects qweight/scales/qzeros triples in HF safetensors, unpacks the GEMM-pack int4 layout into dense fp32 weights the existing HFModelLoader consumes. |
| QLoRA (int4 base + LoRA) | 📋 designed | Blocker: MLX-Swift's quantized arrays don't yet fwd-prop gradients through to the underlying float matrices — see "QLoRA" section below. |

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

---

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
layers DON'T grow their own cache. The other Phase 10 items shipped
without touching CausalSelfAttention's call surface; YOCO is the
exception. Sized as "next focused session" rather than "drop-in to
this batch".

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
same `scatter_add` upstream gap as sparse MoE — see `docs/moe.md`.
Soft routing gives the architectural change + training signal
without the compute saving. When `scatter_add` lands, swap the
sigmoid gate for argTopK + STE and the compute saving lands too.

---

## Phase 8 — interpretability remainder

| Item | Status | Notes |
|---|---|---|
| Logit lens | ✅ shipped | Button in browser playground. |
| Attention heatmap | ✅ shipped | Existing "Watch the model think" panel. |
| Per-layer ablation | ✅ shipped | New "Ablate & sample" button. |
| Activation patching | ✅ shipped — position-zeroing variant | Worker `patch` message + `GpuModel.generatePatched`. Zeroes the residual stream at (layer, position); donor → recipient SWAP is the next iteration. |
| Tuned lens | ✅ shipped | `tinygpt tuned-lens` Mac CLI command trains per-layer probes on a frozen base. Sidecar `.lenses` file format. `TinyGPTModel.forwardTunedLens` for inference once loaded. |

### Activation patching (Meng et al., 2022)  *(shipped — zero-patch variant)*

`webgpu/train.wgsl` gains a `patch_zero` kernel; worker exposes a
`patch` message. The simplest causal intervention: at the specified
(layer, position), ZERO OUT the residual stream value. The output
reveals whether that token's representation at that depth was
load-bearing.

The full donor → recipient SWAP (Meng et al., 2022's original
variant) requires:
- A second forward over the donor prompt with hidden-state capture
  at (layer, position) coords (download to CPU is fine for the
  small models we run).
- An "upload + scatter into a row" GPU op (slot the donor's value
  into the recipient's residual stream at that position).
- A two-prompt UI to pick donor and recipient.

The shipped zero-patch is mechanically the same gate (replace one
row of x); the donor-swap path differs only in WHAT we put in that
row. Bounded follow-up.

### Tuned lens (Belrose et al., 2023)  *(shipped)*

`tinygpt tuned-lens <model> --corpus <text>` trains one
`Linear(d_model → vocab)` per layer with the base model frozen.
Cross-entropy on each layer's projection, mean across layers, AdamW.
Output: a small `.lenses` sidecar (~`L × (vocab+1) × d_model` floats)
in a custom "TGTL v1" format.

Inference side: `TinyGPTModel.forwardTunedLens(idx)` runs the base
forward with `forwardLayerwise`, then applies the per-layer probes —
cleaner than the raw logit lens for "what does layer 3 think the
next token is?" questions. The browser playground's lens button
still uses the raw final-LN+LM-head projection; wiring the tuned
sidecar into the browser is the next iteration.

---

## Cross-cutting blockers

These items appear across multiple phases and share a root cause:

1. **MLX-Swift doesn't expose `mlx_checkpoint`** — blocks gradient
   checkpointing (Phase 6). The C primitive exists; the Swift
   wrapper doesn't. Workarounds in `docs/memory_tradeoffs.md`.
2. **MLX-Swift doesn't expose `scatter_add`** — blocks sparse MoE
   compute and MoD compute savings (Phase 5, Phase 10). Workarounds
   in `docs/moe.md` and above.
3. **Cmlx is internal to MLX-Swift** — neither of the above
   primitives can be bridged from outside the package without
   forking MLX-Swift. The right resolution is upstream PRs.

These are real engineering tasks, not session-sized work. Each
unblocks several roadmap items simultaneously — landing them is
the highest-leverage move for the next phase of work.
