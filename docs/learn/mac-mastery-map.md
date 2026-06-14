---
title: Mac-local AI mastery map — what's buildable, what's built, the distributed boundary
description: The learning + build agenda for mastering Mac-local AI. Every capability buildable on a single Mac, annotated with fleet coverage (built / partial / to-learn), plus an explicit single-machine ↔ distributed boundary. The spine for "learn everything + build everything buildable on this Mac."
---

# Mac-local AI mastery map

The north-star ([`AGENTS.md`](../../AGENTS.md)): be best-in-class at Mac-local
AI, learn the whole space like a sponge (including the single-machine ↔
distributed boundary), build everything buildable on a Mac, and position for
future scale. This is the living agenda — the checklist of what to learn/build
and what's already covered.

**Legend:** ✅ built/learned · 🟡 partial / scaffolded · ⬜ to learn or build.
Each area links the canonical doc/source — we don't re-teach here (DRY).

## 1. Foundations
The ground-up curriculum — mostly ✅. See [curriculum](./curriculum.md) and
sessions [01](./session-01-neural-net-basics.md)–[08](./session-08-training-mechanics.md)
+ [llm-mechanics-fundamentals](./llm-mechanics-fundamentals.md).
- ✅ NN basics, gradient descent, non-linearities, ML paradigms, scaling,
  tokenization/embeddings, behavior learning, training mechanics.
- ✅ Modern architecture (RoPE, RMSNorm, SwiGLU, GQA, MoE, attention-as-matmuls)
  — see [advanced-ml-systems-eval](./advanced-ml-systems-eval.md) §1–4.

## 2. Training & post-training
Canonical: [advanced-llm-training](./advanced-llm-training.md).
- ✅ From-scratch pretrain (tiny), SFT, DPO, finetune — `tinygpt train/sft/dpo/finetune`.
- ✅ PEFT zoo: LoRA, DoRA, QLoRA, VeRA, PISSA, LoftQ, AdaLoRA — `SFT.swift`.
- ✅ **Distillation / cost-compression** — validated: a 0.6B matched a 4B on
  tool-calling at 1/7th size. `tinygpt distill`. *This is the live winning lane.*
- ✅ Synthetic data (`magpie`), quality classifier + filter.
- 🟡 RL: GRPO-on-clarify (`docs/GRPO_CLARIFY.md`); reward modeling (B28 composite reward).
- ⬜ **Learn deeper:** full GRPO/PPO loop, reward over-optimization, WSD schedule
  (B11), layer-wise LR decay (B15), micro-automixer (B21).

## 3. Inference & serving
Canonical: [advanced-llm-inference](./advanced-llm-inference.md).
- ✅ Sampling, speculative decoding (B14), KV-quant, ANE/CoreML serve, GGUF load.
- ✅ Browser WebGPU inference (hand-written WGSL kernels) + transformers.js page.
- 🟡 `tinygpt serve` HF path is **unoptimized (~7 tok/s on 4B)** — known gap.
- ⬜ **Learn/build (steal from oMLX):** continuous batching (B34, for eval),
  tiered KV cache RAM→SSD, persistent prefix-KV cache, PagedAttention,
  FlashAttention internals.

## 4. Evaluation & judgment  ← *the differentiated moat*
Canonical: [advanced-ml-systems-eval](./advanced-ml-systems-eval.md) §11–13.
- ✅ Harnesses: BFCL, τ-bench, HumanEval, lm-eval, MTEB, Indic; `eval-compare`.
- ✅ **eval-gate** (B32, CI regression gate) + **mac-assistant-judgment** benchmark
  (novel, unpublished-elsewhere baselines).
- 🟡 LLM-as-judge (E7); perplexity/contamination (theory in the eval doc).
- ⬜ **Learn/build:** judge-bias mitigation, RL environments as reward functions
  (the Prime Intellect angle), agent-eval-protocol (B23).

## 5. Interpretability
- ✅ SAE + SAE-explore, ROME, MEMIT, tuned-lens, linear-probe, causal-trace, patch.
- 🟡 SAE→SAELens/Neuronpedia export (B17); interp-on-checkpoints timeline (B13).

## 6. Modalities
- ✅ Text; embeddings + rerankers (`rerank-train/eval`, MTEB).
- 🟡 VLM ([qwen3-vl-mrope-deepstack](./qwen3-vl-mrope-deepstack.md); `vlm-smoke`).
- 🟡 Speech (Pace): Apple Speech STT + Kokoro TTS shipping; WhisperKit streaming ⬜.
- See [speech-and-systems-topics](./speech-and-systems-topics.md) for the voice stack.

## 7. Agents & tools
Canonical: [advanced-ml-systems-eval](./advanced-ml-systems-eval.md) §9–10,
[agent-context-hierarchy](./agent-context-hierarchy.md), [model-vs-agent](./model-vs-agent.md).
- ✅ Tool-calling, `tinygpt agent`, deferred tools (B26 🟡), Pace's plan-act-observe loop.
- 🟡 Mini-router family (B2–B7).
- ⬜ Trajectory recorder (B22), agent-eval protocol (B23), RAG vector layer.

## 8. Compression & efficiency  ← *the validated lane*
- ✅ Quantization (HQQ, GPTQ, 4-bit), structured/unstructured pruning.
- ✅ **Distillation** (big→small for cost) — proven on this Mac.
- ⬜ **Learn/build:** logit-level (soft) distillation, fp8, KV-quant tradeoffs,
  energy-per-token (B9).

## 9. Distribution & packaging
- ✅ Model gallery + project pins (B31), `quickstart` (B33), browser playground,
  GGUF / CoreML / safetensors export.

---

## The single-machine ↔ distributed boundary
*What one Mac can do, what it can't, and why — the line to own.*
Theory: [advanced-llm-training](./advanced-llm-training.md) (ZeRO/FSDP/3D parallelism),
[advanced-llm-inference](./advanced-llm-inference.md) §15–16 (disaggregation, TP/EP).

| Fully on a Mac | The hard middle (Mac *barely*) | Needs distributed — and why |
|---|---|---|
| LoRA/QLoRA ≤~8B · **distillation** · quantization · pruning · eval · interp · ANE/CoreML serve · small from-scratch · agents/RAG · WebGPU | QLoRA on a 30B MoE (memory-tight) · frontier→small distillation · "many Macs as one" (teale/Petals — hits the **latency wall**) | Pretrain ≥10B from scratch (tensor/pipeline parallel, GPU-weeks) · 1000-GPU async RL + 2,500 envs (Prime Intellect `prime-rl`) · million-QPS serving — all **compute/bandwidth-bound, not cleverness-bound** |

**Case studies to learn from** (boundary-mapping, not detours):
- **Prime Intellect** — globally *distributed RL training* (INTELLECT-2/3, 32B→100B+); training tolerates latency, so the swarm works *here*. [primeintellect.ai](https://www.primeintellect.ai/)
- **teale / Petals** — *decentralized inference*; the sharded-swarm dies on per-token latency, the whole-model-marketplace survives. [teale.com](https://teale.com/)
- **oMLX** — single-Mac serving done right (RAM↔SSD KV, continuous batching). [omlx.ai](https://omlx.ai/)
- **Why decode resists distribution:** memory-bandwidth-bound + sequential → network round-trips dominate. (advanced-llm-inference §1.)

## Learning gaps — the ⬜ shortlist (what to learn/build next)

1. **Inference depth** — batching + KV-SSD paging + prefix cache (B34); read the
   PagedAttention + FlashAttention papers, then build the batched eval-runtime.
2. **RL/reward** — close the full GRPO loop on a small model; understand reward
   modeling + over-optimization; RL environments as reward functions.
3. **Distributed boundary** — read the HF Ultra-Scale Playbook end-to-end; be
   able to whiteboard ZeRO stages, 3D parallelism, and why decode won't shard.
4. **Compression** — logit-level distillation; fp8; the cost/quality frontier.
5. **Agents** — trajectory recording (B22) + agent-eval protocol (B23).

> Living doc — correct coverage markers as reality changes. New learning lands
> as a focused page elsewhere in `docs/learn/`; this map just indexes + tracks.
