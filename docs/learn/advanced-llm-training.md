---
title: Advanced LLM training & post-training — interview-grade map
description: Senior/staff interview topics for LLM training at scale (parallelism, precision, MoE, data) and post-training (RLHF/DPO/GRPO/reward modeling/distillation), each mapped to the best external source and to where this repo touches it.
---

# Advanced LLM training & post-training — interview-grade map

Staff-level training topics, mapped the house way: **what interviewers
probe**, the **single best external source** (we don't re-teach), and
**in the repo** where there's a real anchor. Distributed-parallelism
basics (FSDP2/ZeRO/TP/PP menu) live in
[`speech-and-systems-topics.md` §8](speech-and-systems-topics.md) — this
doc adds the interview depth on top and the post-training half.

## Distributed training (depth beyond the §8 menu)

**1. ZeRO stages / FSDP2 sharding.** Probed: what each stage shards
(1: optimizer state, 2: +grads, 3: +params) and FSDP2's per-parameter
reshard-after-forward / all-gather-on-demand. "A 70B won't fit in DDP —
how does ZeRO-3 change per-rank memory and what comms does it cost?"
*Learn:* [HF Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) · *staff*

**2. 3D parallelism decision tree.** TP intra-layer (NVLink-bound), PP
cross-node depth (1F1B to shrink the bubble), DP/FSDP outer ring, +
sequence parallel for long context. "Lay out parallelism for a 400B model
on 1024 GPUs and justify each axis." *Learn:* [ZeRO paper](https://arxiv.org/pdf/1910.02054) · *staff*

**3. Comms–compute overlap & MFU.** All-gather/reduce-scatter overlapped
with backward, bucketing, prefetch; diagnosing comms-bound vs compute-bound.
"MFU is 35% on multi-node FSDP — which knobs?" *Learn:* [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) · *staff*
*In repo:* tinygpt is the **single-device counterexample** (`Train.swift`,
unified memory, zero inter-device comms) — know FSDP to know what you're *not* paying for.

## Memory & precision

**4. Gradient checkpointing.** Store O(√n) activations, recompute the
rest (~30% extra FLOPs); selective recompute of cheap ops. "Activations,
not weights, are your OOM — what do you do and what's the cost?"
*Learn:* [Chen et al. 2016](https://arxiv.org/abs/1604.06174) · *senior*
*In repo, today:* the A1 4B QLoRA run OOM'd at batch 4 until
`--grad-checkpoint` brought peak mem to 8.3 GB — this exact tradeoff, live.

**5. Mixed precision (bf16 vs fp8).** Why bf16 is the pretraining default
(wide exponent, no loss scaling) and what breaks in fp8 (per-tensor/delayed
scaling, keep master weights + reductions higher, layernorm/optimizer stay
bf16). *Learn:* [NVIDIA FP8 Formats](https://arxiv.org/abs/2209.05433) · *staff*
*In repo, today:* the A1 run's loss dropped 0.93→0.45 then went **NaN at
iter 100** — a precision/stability spike fixed by dropping LR 1e-4→2e-5;
the canonical "good progress then NaN" story interviewers love.

**6. Optimizer-state sharding & ZeRO-Offload.** Adam state ≈ 2× params in
fp32; shard (ZeRO-1) or offload the step to CPU when GPU memory is the wall
(PCIe-bound). *Learn:* [ZeRO-Offload](https://arxiv.org/pdf/2101.06840) · *senior*

## MoE & data

**7. MoE routing & load balancing.** Top-k token-choice, the aux
load-balancing loss, routing collapse / dead experts, capacity factor +
token dropping, and the newer **aux-loss-free** bias adjustment (DeepSeek).
"Dead + overloaded experts — diagnose and contrast aux-loss vs expert-choice
vs loss-free." *Learn:* [Aux-Loss-Free Balancing](https://arxiv.org/abs/2408.15664) · *staff*
*In repo:* tinygpt is dense; the MoE you actually run is qwen3-30b-a3b
(Pace's planner) — the serving/expert-parallel angle is in
[`advanced-llm-inference.md`](advanced-llm-inference.md).

**8. Pretraining data mixture & curation.** LangID → quality filter
(fastText/classifier top-k%) → safety → dedup → domain upsampling; choosing
mixture weights via proxy-model sweeps / DoReMi. "Decide code:web:books
ratio and validate it without a full run." *Learn:* [DCLM](https://arxiv.org/abs/2406.11794) · *senior*
*In repo:* `tinygpt train-quality-classifier` + `quality-filter` (B10,
FineWeb-Edu-style scorer) and `tinygpt dedupe` are this pipeline in miniature.

**9. Dedup & decontamination.** Exact vs fuzzy (MinHash/LSH) vs semantic
dedup; why it beats just saving tokens; eval-set contamination guards.
*Learn:* [Lee et al. 2021](https://arxiv.org/abs/2107.06499) · *mid*

**10. Curriculum / annealing.** Two-stage (web → high-quality annealing)
and why it interacts badly with cosine LR decay (best data wasted in the
low-LR tail). *Learn:* [DCLM](https://arxiv.org/abs/2406.11794) · *senior*
*In repo:* `Train.swift` ships a WSD schedule (the decay phase *is* the
annealing knob) — `docs/PLAN.md` B11.

## Post-training

**11. RLHF / PPO.** SFT → reward model → PPO; the **four models in memory**
(policy, ref, reward, critic), KL-to-ref penalty, reward hacking. "What's
in GPU memory during PPO and where's it expensive?" *Learn:* [InstructGPT](https://arxiv.org/abs/2203.02155) · *senior*

**12. DPO & offline preference optimization.** No explicit reward model
(implicit reward via policy/ref log-ratio); the closed-form loss; failure
modes (off-policy drift, likelihood displacement) → iterative DPO, IPO,
KTO, SimPO. "Derive why DPO needs no RM; when still prefer PPO?"
*Learn:* [DPO paper](https://arxiv.org/abs/2305.18290) · *senior*
*In repo:* `tinygpt dpo` (`DPO.swift`) — the implicit-reward loss in code.

**13. GRPO & RL-for-reasoning (RLVR).** Drops the critic; advantage from
group-normalized rewards over k samples/prompt; ties to DeepSeek-R1
outcome-based RL. "Why remove the value model, what does it save, what's
the variance tradeoff vs PPO?" *Learn:* [DeepSeekMath/GRPO](https://arxiv.org/pdf/2402.03300) · *staff*
*In repo:* `docs/GRPO_CLARIFY.md` (GRPO-on-clarify PRD) is exactly this on the Pace task.

**14. Reward modeling.** Bradley-Terry pairwise loss, over-optimization /
Goodharting, ORM vs PRM (process reward) for reasoning. "Detect + mitigate
reward over-optimization; when PRM over ORM?" *Learn:* [Lil'Log: Reward Hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) · *staff*
*In repo:* B28 composite-reward framework (`docs/learn/castform-rl-finetune.md`)
— typed multi-dimensional reward, the anti-Goodhart structure.

**15. Constitutional AI / RLAIF + best-of-N.** AI feedback replacing human
labels (critique-revise + AI-preference RL); inference-time alignment via
rejection sampling / best-of-N reranking with an RM. "When is RLAIF > RLHF;
how does BoN trade train vs inference cost?" *Learn:* [Constitutional AI](https://arxiv.org/abs/2212.08073) · *senior*
*In repo:* `tinygpt bon` (`BestOfN.swift`) is the inference-time BoN lever.

**16. Distillation & continued pretraining.** Logit/sequence distillation,
domain-adaptive continued pretraining, and forgetting mitigations (replay,
re-warming, distill-as-regularizer, LoRA isolation). "New domain via
continued pretraining tanks general benchmarks — fix without re-pretraining."
*Learn:* [Scalable Continued Pretraining](https://arxiv.org/abs/2403.08763) · *senior*
*In repo:* `tinygpt distill` (`Distill.swift`); the v1–v11 arc
(`docs/RETROSPECTIVE.md`) is a documented catastrophic-forgetting case
(47pp OOS regression from 38 rows).

## Suggested order

Read the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
once for 1–7 (it's the one-stop systems reference), then the post-training
papers 11–16 in order. 4–5 you can study against today's A1 logs.
