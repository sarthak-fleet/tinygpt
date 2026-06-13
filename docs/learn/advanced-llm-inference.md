---
title: Advanced LLM inference & serving — interview-grade map
description: Senior/staff inference-optimization interview topics — KV cache & paging, batching, speculative decoding, quantization, attention kernels/variants, long context, serving architecture — mapped to the best external source and to where this repo touches it.
---

# Advanced LLM inference & serving — interview-grade map

Inference is the half a speech/latency background is strongest in. Format:
**what's probed**, the **single best source**, **in the repo** where real.

## Fundamentals

**1. Roofline: memory-bound decode vs compute-bound prefill.** Decode
reuses huge weight matrices once per token → bandwidth-bound; prefill is
parallel → compute-bound. Derive arithmetic intensity. "Roofline for
prefill vs decode on an H100 — where does each bottleneck?"
*Learn:* [LLM Inference Unveiled (roofline)](https://arxiv.org/abs/2402.16363) · *senior*
*In repo:* `docs/research/mac_decode_baseline_m5pro.md` — native models hit
293–767 tok/s; today's bf16 4B via the HF path managed **7 tok/s**, a
textbook bandwidth-bound-plus-unoptimized-kernel case.

**2. KV-cache memory math.** Size it by hand:
`2·n_layers·n_kv_heads·head_dim·seq·batch·bytes`; why KV (not weights) caps
batch/context. "KV for Llama-3-70B @ 8k, batch 32 — what limits concurrency?"
*Learn:* [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) · *senior*
*In repo:* `docs/kv_cache_optimization.md`; `Sample.swift` exposes
`--kv-quantize` / `--kv-preallocate`.

**3. TTFT vs ITL vs goodput.** TTFT (prefill-bound), ITL/TPOT
(decode-bound), goodput under SLOs; batch size trades TTFT against ITL.
"p95 TTFT<300ms, ITL<40ms — tune the scheduler." *Learn:* [Inference Trilemma](https://www.digitalocean.com/blog/llm-inference-tradeoffs) · *mid*
*In repo:* Pace's TTFW hunt (330→119ms) is a TTFT story;
[`speech-and-systems-topics.md` §1](speech-and-systems-topics.md).

## Memory & scheduling

**4. PagedAttention.** Naive contiguous KV wastes 60–80% (fragmentation);
OS-style paging + block tables fix it and enable copy-on-write prefix
sharing. *Learn:* [PagedAttention](https://arxiv.org/abs/2309.06180) · *senior*

**5. Prefix / prompt caching (RadixAttention).** Reuse shared system
prompts / few-shot / chat history via a radix tree of KV blocks. "2k-token
shared system prompt — avoid recomputing it per request?"
*Learn:* [SGLang/RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) · *senior*
*In repo:* Pace sends `cache_prompt: true` on every request — this exact win.

**6. Continuous / in-flight batching.** Iteration-level scheduling injects
/ evicts requests every decode step so the GPU never idles on stragglers.
"Why does static batching tank throughput with heterogeneous lengths?"
*Learn:* [Orca (OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu) · *senior*
*In repo:* tinygpt serve is single-stream — know this as the throughput
lever you'd add for multi-tenant serving.

## Decoding & quantization

**7. Speculative decoding.** Draft proposes k tokens, target verifies in
one parallel pass; rejection sampling keeps the distribution **exact**.
"Why doesn't it change outputs, and when does it lose?" *Learn:* [Speculative Decoding](https://arxiv.org/abs/2211.17192) · *senior*
*In repo:* `SpeculativeDecode.swift` (B14, T=0 byte-equality gate).

**8. Self-speculation (Medusa, EAGLE).** Bolt-on heads / feature-level
autoregression beat a separate draft model on acceptance and avoid serving
two models. *Learn:* [EAGLE](https://arxiv.org/abs/2401.15077) · *staff*

**9. Weight quantization (GPTQ vs AWQ).** GPTQ's Hessian/second-order error
compensation vs AWQ's activation-aware salient-channel scaling; weight-only
int4 helps memory-bound decode. "Pick a scheme for a latency-sensitive 70B."
*Learn:* [AWQ](https://arxiv.org/abs/2306.00978) · *senior*
*In repo:* `tinygpt gptq` / `hqq` (`GPTQ.swift`, `HQQ.swift`); quantized-HF
checkpoint loading (commit ccf8937) is what let today's A1 load a 4-bit base.

**10. fp8 / activation quant / formats.** fp8 on Hopper/Blackwell, int8
SmoothQuant for activation outliers, GGUF for CPU/edge. "weight-only int4
vs fp8 weight+act — which for throughput, which for accuracy?"
*Learn:* [TensorRT-LLM quantization](https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html) · *senior*
*In repo:* `tinygpt gguf-load` / `to-coreml` (the edge/ANE path).

**11. KV-cache quantization.** The lever for long context + large batch;
harder than weight quant (outlier keys, accuracy cliffs). *Learn:* [roofline survey](https://arxiv.org/abs/2402.16363) · *staff*
*In repo:* `Sample.swift --kv-quantize`.

## Kernels, attention & long context

**12. Attention variants MHA/MQA/GQA/MLA.** KV-head sharing shrinks the
cache and raises arithmetic intensity; GQA is the dense standard, MLA
(DeepSeek) compresses KV to a low-rank latent (~90%+ reduction). "Why
MHA→GQA, and what does MLA add?" *Learn:* [GQA](https://arxiv.org/abs/2305.13245) · [MLA/DeepSeek-V2](https://arxiv.org/abs/2405.04434) · *senior*
*In repo:* Qwen3-4B (the A1 base) uses GQA — that's why its KV cache is small.

**13. FlashAttention v2/v3.** IO-aware tiling avoids the N×N matrix; v3
adds Hopper async (warp-specialization, TMA, fp8); recompute-in-backward.
"Why faster despite recomputing softmax stats?" *Learn:* [FlashAttention-3](https://arxiv.org/abs/2407.08608) · *staff*
*In repo:* tinygpt rides MLX's fused attention — the kernel you *don't* hand-write.

**14. Long context: RoPE scaling + sparse attention.** Position
interpolation vs NTK-aware vs YaRN; sliding-window (Mistral), ring/blockwise
for sequence parallelism. "Extend 4k→128k — what changes and why does naive
interpolation degrade?" *Learn:* [Extending RoPE / YaRN](https://blog.eleuther.ai/yarn/) · *staff*
(RoPE itself: [`advanced-ml-systems-eval.md`](advanced-ml-systems-eval.md) §2.)

## Serving architecture

**15. Disaggregated prefill/decode & chunked prefill.** Split phases onto
separate GPU pools (needs fast RDMA KV transfer) vs co-locate + interleave;
goodput tradeoffs. "Prefill spikes blow your decode ITL SLO — disaggregate
or chunk?" *Learn:* [DistServe](https://arxiv.org/abs/2401.09670) · *staff*

**16. Inference parallelism: TP / EP / multi-LoRA.** TP (split matmuls,
all-reduce/layer, NVLink-bound); expert parallel for MoE (all-to-all, load
imbalance); multi-LoRA serving (many adapters on one base, S-LoRA). "Serve
a 671B MoE on 8 GPUs — TP vs EP vs hybrid?" *Learn:* [Inference Handbook: parallelism](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism) · *staff*
*In repo:* `serve --lora` is single-base adapter stacking — the seed of
multi-LoRA serving.

## Suggested order

1–3 first (the mental model). 4–6 + 12–13 are the highest-leverage for a
serving role; 7/9 you can read against the repo anchors. The roofline
survey (§1) covers 1, 11, and parts of 9 in one read.
