# Wave 2.5 kernel audit — BUILD / DEFER / DROP

**Date**: 2026-05-31
**Question**: For each of 5 proposed low-level kernel items, is it worth
the engineering vs MLX-Swift / CoreML / Apple defaults?
**Outcome**: 4 of 5 collapse — most "Wave 2.5" Metal kernel work goes
away. One real win, one viable adoption on M5+, three drops, one defer.

## Verdicts at a glance

| Item | Verdict | Effort | Reason (1 line) |
|---|---|---|---|
| Flash Attention Metal kernel | **DROP** | — | MLX-Swift `MLXFast.scaledDotProductAttention` already fused FA-style; all 9 of our call sites hit the fast path |
| Int8 matmul Metal kernel (W8A8) | **DEFER / adopt cider on M5+** | 1-2 days adopt | `Mininglamp-AI/cider` ships this; needs M5 `cooperative_tensor`. M3/M4 has no hardware path |
| Int4 packed matmul Metal kernel | **DROP** | — | MLX `quantized_matmul` is hand-tuned int4. Custom = 0-5% delta |
| ANE+GPU heterogeneous routing | **DEFER** | 3-6 weeks | Research-grade. Gated on Apple Stateful Models API (rumored late 2026). Real win when screen-watching specialist ships |
| WebGPU subgroup matmul | **BUILD** | 3-5 days | 2-5× browser train. Scaffolding exists (`webgpu/train_sg.wgsl`) |

## 1. Flash Attention Metal kernel — DROP

MLX-Swift's `MLXFast.scaledDotProductAttention` is already a fused
FlashAttention-style Metal kernel: `do_causal` mode is built in, no
host-side mask allocation, tiled, matches Apple MPSGraph.

**Evidence**:
- In-repo audit `docs/perf_audit_mlxfast_tied.md` confirms all 9 of
  our call sites hit the fast path on head dims ∈ {64, 80, 96, 128, 256}.
- The credible competitor [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
  v2 is ~20% faster than MLX SDPA at 4096 ctx per [Liu Liu's benchmarks](https://x.com/liuliu/status/1877040179942146237) —
  but only at long context (we cap at 1024) and only for diffusion-style
  workloads.
- MLX's only honest weakness is that it uses the unfused path during
  *training* (fused VJP not optimized — see
  [scaled_dot_product_attention.cpp:598-602](https://github.com/ml-explore/mlx))
  and lacks PagedAttention ([ml-explore/mlx#2955](https://github.com/ml-explore/mlx/issues/2955)).

**Verdict**: Building our own FA Metal kernel is research-grade work
(4-8 weeks of sustained kernel-tuning) for a 10-20% delta on inputs we
don't run. MLX/Apple have effectively solved this for our workloads.
The repo's existing custom FA2-forward in WGSL (browser side) is where
our FA effort belongs.

## 2. Int8 matmul Metal kernel (W8A8) — DEFER / adopt cider on M5+

MLX's `quantize` is **weight-only**; true W8A8 inference needs INT8
activations + INT8×INT8→INT32 GEMM. The open-source
[Mininglamp-AI/cider](https://github.com/Mininglamp-AI/cider) project
ships exactly this as MLX custom primitives and reports
**1.2-1.9× faster prefill, ~8% decode** on M5 Pro (Qwen3-VL-2B).

**Hardware constraint**: cider explicitly disables itself on M4 and
earlier because it relies on `mpp::tensor_ops::matmul2d` from Metal 4's
`cooperative_tensor` API — INT8 GEMM hardware acceleration is M5-only.
On M3/M4 there is no hardware path; fallback to packed-int dot products
is slower than bf16.

**Decode benefit is small** because LLM decode is memory-bound; the
int8 win at long context is dominated by KV cache (still fp16) per
[ml-explore/mlx#3209](https://github.com/ml-explore/mlx/discussions/3209).

**Verdict**: If the target Mac is M5/M5 Pro, **adopt cider** (1-2 days
integration). Building from scratch = 2-3 weeks to recreate what cider
already published. tinygpt's primary dev machine is M5 Pro (confirmed
2026-05-31), so this is viable and is the **largest single Mac-speed
lever currently available**.

## 3. Int4 packed matmul Metal kernel — DROP

MLX's `quantized_matmul` is int4 by default and is already a hand-tuned
Metal kernel — what `mlx-lm`, `llmcheck.net`, and the
[ml-explore/mlx#3209](https://github.com/ml-explore/mlx/discussions/3209)
benchmarks all measure.

**Evidence**:
- On M3 Ultra: q4 hits ~31 tok/s on Qwen-32B at 1K ctx vs 10.4 tok/s
  fp16 — 3× memory-bandwidth-bound throughput, the expected payoff.
- [SwiftLM](https://github.com/SharpAI/SwiftLM) (the most serious
  MLX-Swift LLM server) doesn't roll its own — it uses stock
  `quantized_matmul`. Where it wins is KV-cache compression and SSD
  streaming, not matmul.
- tinygpt's quantization stack (HQQ/AWQ/GPTQ readers + QAT — see
  `docs/quantization_expansion.md`) is already aimed at this kernel.
  The algorithmic pieces are shipped; the matmul itself is MLX's.

**Verdict**: Building a competing int4 kernel is pure-implementation
grunt-work for a 0-5% delta. MLX already solved this.

## 4. ANE + GPU heterogeneous routing — DEFER

The only item with a real product-shaped win — but research-grade.

**Evidence**:
- [SqueezeBits — Disaggregated inference NPU prefill + GPU decode](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
  + [AtomGradient/hybrid-ane-mlx-bench](https://github.com/AtomGradient/hybrid-ane-mlx-bench):
  M2 Ultra, 268 tok/s ANE batched prefill, **11.3× over sequential**,
  comparable to GPU prefill but at **0.22 W vs 62 W — 282× lower power**.
- The [Orion paper (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728)
  reports 170 tok/s GPT-2 124M decode on M4 Max — but their own CPU
  baseline at 283 tok/s beats it. Decode on ANE is dominated by
  IOSurface dispatch overhead.
- ANEMLL benchmarks on M4: Llama-3.2-1B at 47-62 tok/s on ANE vs
  204 tok/s on MLX-GPU — ANE loses on raw decode throughput by 3-4×
  but wins on power 10×, and used ~500 MB vs MLX's 8 GB.

**The honest pattern**: ANE wins on prefill and always-on background
work; GPU wins on low-latency decode. Maps perfectly onto the agent
runtime + screen-watching specialist in Wave 2.6.

**Engineering reality**:
- CoreML conversion times: 2-97 min per fixed shape
- Multi-function package feature needed to switch prefill/decode without
  weight duplication
- Production path uses reverse-engineered private APIs (ANEMLL is
  "beta, one macOS update could break things")
- Apple's Stateful Models API (rumored late 2026 per `docs/perf_research.md`)
  is the gating prerequisite for int4 stateful ANE

**Verdict**: tinygpt already has a CoreML fp32 path (365 pass/s ANE
forward measured). Extending to int4 stateful models is gated on Apple
shipping the Stateful Models API. **Defer until at least one full
specialist ships; revisit when Stateful Models lands OR when
battery-life-driven always-on becomes the demo.** Pairs naturally with
the screen-watching specialist work in Wave 2.6.

## 5. WebGPU subgroup matmul — BUILD

The one clear win.

**Evidence**:
- tinygpt already ships subgroup variants for LN, cross-entropy, and
  bias_grad (`webgpu/train_sg.wgsl`) — explicitly **not** matmul yet.
- The [nuss-and-bolts WebGPU matmul writeup](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)
  hit 1 TFLOP/s on M2 Pro = ~17% of peak (6 TFLOP/s) and flagged
  subgroups as the obvious next step.
- Chrome's own data: 2.3-2.9× on Google Meet's matvec, up to 13× on
  some GPUs (combined with swizzle, 26×) per
  [Chrome 125](https://developer.chrome.com/blog/new-in-webgpu-125) /
  [129](https://developer.chrome.com/blog/new-in-webgpu-129) release blogs.
- Subgroups went stable around Chrome 128-129 (out of origin trial mid-2024);
  Safari 26 / iOS 26 ships WebGPU on Apple Silicon.

**Implementation shape**:
- Write `matmul_sg` (and ideally `matmul_backward_sg`) using `subgroupAdd`
  for the dot-product reduction
- Gate behind `gpuFeatures.subgroups` exactly like the existing `train_sg.wgsl`
  kernels
- Run the existing 500-step Shakespeare numerics gate (Decision 19 —
  auto-disable kernel on numerics mismatch)
- **Watch-out**: Apple Metal driver subgroup behavior has historically
  been the buggiest WebGPU backend ([ml-explore/mlx#2205](https://github.com/ml-explore/mlx/issues/2205)
  is the canonical example) — the auto-disable-on-numerics-fail gate
  from Decision 19 is mandatory.

**Verdict**: 3-5 days implementation-grade work. Realistic delta:
2-5× on flagship browser training where matmul dominates. Cleanest ROI
of the five.

## Recommendation summary

**If 2 weeks solo**:
1. Ship the WebGPU subgroup matmul (item 5, 3-5 days, 2-5× browser train)
2. Spend the rest evaluating cider's W8A8 on actual M5+ hardware
   (item 2, adopt if numbers hold)
3. Drop items 1 and 3 outright
4. Write a one-page "ANE routing — revisit when Stateful Models API
   ships" decision-log for item 4

**As executed (2026-05-31)**:
- User prioritizes Mac speed first → cider W8A8 adoption is the Mac-side
  primary lever (M5 Pro confirmed)
- WebGPU subgroup matmul deferred to background agent (browser-side
  work, not user's current focus)
- Items 1, 3 marked done (DROP) in task tracker
- Item 4 deferred per this doc

## Sources

- [MLXFast SDPA audit (in-repo)](../perf_audit_mlxfast_tied.md)
- [tinygpt perf_research notes (in-repo)](../perf_research.md)
- [tinygpt quantization_expansion notes (in-repo)](../quantization_expansion.md)
- [tinygpt train_sg.wgsl (in-repo)](../../webgpu/train_sg.wgsl)
- [MLX PagedAttention proposal — ml-explore/mlx#2955](https://github.com/ml-explore/mlx/issues/2955)
- [MLX inference benchmarks discussion — ml-explore/mlx#3209](https://github.com/ml-explore/mlx/discussions/3209)
- [Metal FlashAttention 2.0 writeup — Liu Liu (Draw Things)](https://medium.com/engineering-draw-things/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
- [metal-flash-attention — philipturner](https://github.com/philipturner/metal-flash-attention)
- [Mininglamp-AI/cider — W8A8/W4A8 for MLX on M5+](https://github.com/Mininglamp-AI/cider)
- [SwiftLM — MLX-Swift LLM server](https://github.com/SharpAI/SwiftLM)
- [Apple ML — M5 GPU Neural Accelerators in MLX](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [SqueezeBits — Disaggregated inference NPU prefill + GPU decode](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [AtomGradient/hybrid-ane-mlx-bench](https://github.com/AtomGradient/hybrid-ane-mlx-bench)
- [Orion paper — programming the ANE for LLM train+inf](https://arxiv.org/abs/2603.06728)
- [InsiderLLM — ANE LLM inference, ANEMLL benchmarks on M4](https://insiderllm.com/guides/apple-neural-engine-llm-inference/)
- [Apple ML — Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Chrome 125 — WebGPU subgroups intro](https://developer.chrome.com/blog/new-in-webgpu-125)
- [Chrome 129 — WebGPU subgroup builtins shipped](https://developer.chrome.com/blog/new-in-webgpu-129)
- [Optimizing a WebGPU Matmul Kernel for 1 TFLOP+ on M2 Pro](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)
