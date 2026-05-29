# Roadmap — recent research (2024-2026 highlights)

The 2024-2026-era body of work that informed Tier 1-3. Each entry is
just enough context to know whether to dig in.

## Alignment / preference

- **DPO** — Rafailov et al., NeurIPS 2023.
  "[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)."
  The paper that displaced PPO/RLHF for most labs.
- **KTO** — Ethayarajh et al., 2024.
  "[Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)."
  Single-label preference; no pairs needed.
- **ORPO** — Hong et al., 2024.
  "[Reference-Free Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691)."
  Merges SFT + DPO into one loss. Drops the reference model.
- **SimPO** — Meng et al., 2024.
  "[Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)."
  Reference-free DPO. Length-normalized objective.
- **IPO** — Azar et al., 2023.
  "[A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)."
  Identity preference; stronger regularization than DPO.
- **CPO** — Xu et al., 2024.
  "[Contrastive Preference Optimization](https://arxiv.org/abs/2401.08417)."
  Combines DPO with a behavior-cloning term.
- **NEFTune** — Jain et al., NeurIPS 2023.
  "[Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)."
  3 lines of code; +5-10 points on AlpacaEval.

## Parameter-efficient fine-tuning

See [`docs/lora_guide.md`](../lora_guide.md) for the mechanics.

- **DoRA** — Liu et al., 2024.
  "[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)."
  Magnitude + direction decomposition; consistently beats LoRA.
- **GaLore** — Zhao et al., 2024.
  "[GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)."
  Full fine-tuning at LoRA memory.
- **LoftQ** — Li et al., ICLR 2024.
  "[LoftQ: LoRA-Fine-Tuning-Aware Quantization](https://arxiv.org/abs/2310.08659)."
  Quantization-aware LoRA initialization.
- **VeRA** — Kopiczko et al., ICLR 2024.
  "[VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454)."
  Shared random matrices; 10× smaller than LoRA.
- **PISSA** — Meng et al., 2024.
  "[PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://arxiv.org/abs/2404.02948)."
  SVD-based LoRA initialization for faster convergence.
- **LoRA+** — Hayou et al., ICML 2024.
  "[LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)."
  Different LRs for A and B.
- **rsLoRA** — Kalajdzievski, 2023.
  "[A Rank Stabilization Scaling Factor](https://arxiv.org/abs/2312.03732)."

## Quantization

- **GPTQ** — Frantar et al., ICLR 2023.
  "[GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)."
- **AWQ** — Lin et al., MLSys 2024.
  "[AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)."
- **HQQ** — Badri & Shaji, 2024.
  "[Half-Quadratic Quantization](https://mobiusml.github.io/hqq_blog/)."
  Calibration-free int4.
- **KIVI** — Liu et al., 2024.
  "[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)."
- **BitNet b1.58** — Ma et al., 2024.
  "[The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)."
  Ternary weights ({-1, 0, 1}) from scratch. Training-from-scratch; not post-training.

## Inference / efficiency

- **Speculative decoding** — Leviathan et al., ICML 2023.
  "[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)."
- **Medusa** — Cai et al., 2024.
  "[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)."
- **EAGLE / EAGLE-2** — Li et al., 2024.
  "[EAGLE-2: Faster Inference of Language Models](https://arxiv.org/abs/2406.16858)."
- **StreamingLLM** — Xiao et al., ICLR 2024.
  "[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)."

## Architecture variants

- **Multi-Token Prediction (MTP)** — Gloeckle et al., ICML 2024.
  "[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)."
  Used by DeepSeek-V3 and Meta. Predict K tokens per position. See
  [`docs/mtp.md`](../mtp.md).
- **Differential Transformer** — Microsoft 2024.
  "[Differential Transformer](https://arxiv.org/abs/2410.05258)."
  Subtract a noise attention pattern from the signal one.
- **Mixture of Depths** — Raposo et al., 2024.
  "[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)."
  Route tokens through fewer layers.
- **Mamba / Mamba-2** — Gu & Dao, 2023/2024.
  "[Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)."
- **LASER** — Sharma et al., ICLR 2024.
  "[The Truth is in There: Improving Reasoning with Layer-Selective Rank Reduction](https://arxiv.org/abs/2312.13558)."
  Counterintuitively beneficial selective rank truncation.

## Optimizers

- **Sophia** — Liu et al., 2023.
  "[Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)."
- **Lion** — Chen et al., NeurIPS 2023.
  "[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)."
- **Muon** — Jordan, 2024.
  Newton-Schulz orthogonalization on gradients.
  [Notes by Keller Jordan](https://kellerjordan.github.io/posts/muon/).
  Adopted by the nanoGPT speedrun community for big wins.
- **GaLore** — see PEFT section above.
- **LISA** — Pan et al., 2024.
  "[LISA: Layerwise Importance Sampling](https://arxiv.org/abs/2403.17919)."

## Distillation

See [`docs/distillation.md`](../distillation.md) for what's shipped.

- **Soft targets distillation** — Hinton et al., 2015. The original.
- **MiniLLM** — Gu et al., ICLR 2024.
  "[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)."
  KL-divergence variants for distilling LLMs.
- **Distilling Step-by-Step** — Hsieh et al., ACL 2023.
  "[Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301)."
  Distill reasoning chains.
- [On-Policy Distillation Survey, April 2026](https://arxiv.org/abs/2604.00626) — confirms distillation is "the dominant technique for transferring frontier capabilities into smaller, deployable student models." Validates our Tier 1.1.
- **MiniPLM** ([Gu et al., NeurIPS 2024](https://openreview.net/forum?id=tJHDw8XfeC)) — distillation for **pre-training**, not just post-training. Distill a small base FROM a big base. Novel.
- **Knowledge Distillation with Training Wheels** ([Feb 2025](https://arxiv.org/abs/2502.17717)) — student can "request help" from teacher at test time.

## Synthetic data + curriculum

- **Self-Instruct** — Wang et al., 2023.
  "[Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)."
- **Evol-Instruct** — Xu et al., 2024 (WizardLM).
  "[WizardLM: Empowering LLMs to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)."
- **DoReMi** — Xie et al., NeurIPS 2023.
  "[DoReMi: Optimizing Data Mixtures](https://arxiv.org/abs/2305.10429)."
  Learns optimal data-domain ratios.
- **TinyStories** — Eldan & Li, 2023.
  "[TinyStories: How Small Can Language Models Be?](https://arxiv.org/abs/2305.07759)."
  Established the sub-100M coherence threshold on a constrained vocabulary.
- **Magpie** — Xu et al., ICLR 2025.
  "[Magpie: Alignment Data Synthesis from Scratch](https://arxiv.org/abs/2406.08464)."
  Generates high-quality SFT data by prompting aligned LLMs with just their
  pre-query templates. No seeds, no prompt engineering, no human labels.
  Models SFT'd on Magpie data surpass Llama-3-8B-Instruct.

## 2025-era — reasoning + RLVR + verifier-based training

The biggest single shift in 2025 was the move from "preference
alignment is the post-training step" to "reasoning + RL on verifiable
rewards is the post-training step."

- **DeepSeek-R1 / R1-Zero** — DeepSeek-AI, Jan 2025.
  "[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning](https://arxiv.org/abs/2501.12948)."
  Showed that RL with verifiable rewards (math/code grading) on a
  strong base produces emergent chain-of-thought reasoning. R1-Zero
  is the variant that skipped SFT entirely and went straight from
  base to RL. Single most-cited result of 2025.
- **GRPO (Group Relative Policy Optimization)** — DeepSeek, 2024-2025.
  PPO variant used by R1 that doesn't need a separate value
  network. Computes advantage from a group of rollouts' relative
  rewards. Less memory than PPO, fits on smaller hardware.
- **DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)** —
  [ByteDance Seed & Tsinghua, March 2025](https://arxiv.org/abs/2503.14476).
  Beat DeepSeek-R1 by 3 points on AIME 2024 with 50% of R1's training
  compute. Four techniques layered on top of GRPO: **clip higher**
  (asymmetric PPO clipping bounds), **dynamic sampling** (rejection
  sample bad rollouts before they enter the gradient), **token-level
  policy gradients** (per-token advantage rather than per-sequence),
  **overlong reward shaping**. Open source at
  [BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO).
  Practical implication: **GRPO is the right Tier 3 mental model;
  DAPO is the right Tier 3 implementation.**
- **RLVR (Reinforcement Learning from Verifiable Rewards)** —
  umbrella for: math → grade with verifier; code → grade by running
  tests; reasoning → grade by deterministic checker. Allen AI's
  Tulu-3 paper was the first open recipe; DeepSeek-R1 scaled it.
- **OpenAI o1 / o3** — proprietary but reframed the field. The
  "test-time compute" thesis: hand the model more inference compute
  (chain-of-thought tokens) and it gets noticeably smarter.
- **Reasoning distillation** — DeepSeek-R1-Distill series: reasoning
  traces from R1 are used as SFT data for smaller models.
- **OpenThoughts** — community open-source dataset of reasoning
  traces. Built to be the open analog of R1's reasoning data.
- **Test-time compute scaling** — separate axis from training
  scaling. Spend more inference tokens (CoT) for better outputs.
  Notable: Snell et al., 2024.
  "[Scaling LLM Test-Time Compute Optimally Can Be More Effective
  Than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)."
- **DeepSeek-V3** — DeepSeek-AI, Dec 2024.
  "[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)."
  671B-MoE (37B active). Multi-Token Prediction during training.
- **Qwen3** — Alibaba, 2025. Latest in the Qwen family (4B-397B).

### What "2025-era post-training" looks like at our scale

The R1-class recipe at the 100M-1B scale:

1. **Pretrain** (still the same) — FineWeb-edu etc.
2. **SFT on reasoning traces** — use OpenThoughts or R1-distill data.
   Trains the model to emit `<think>...</think>` blocks before answers.
3. **RL with verifiable rewards** — GRPO/DAPO on math/code/format
   tasks with programmatic verifiers.

Distillation from R1-Distill-7B / 14B into a 100M model is
plausible at our scale; full RLVR is probably a stretch.

## Evolution Strategies — competitive again (genuinely surprising)

**Evolution Strategies at Scale** —
[Qiu et al., Sept 2025](https://arxiv.org/abs/2509.24372). First
successful application of ES to billion-parameter LLM fine-tuning at
full parameter scale (no dimensionality reduction). Findings:

- **Beats PPO/DPO** on long-horizon and delayed-reward tasks.
- **More robust to reward hacking** than RL.
- **Better training stability** — no value network to tune.
- Tolerates very high-dimensional parameter spaces.
- Open source: [VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper).

**Why this matters specifically for TinyGPT:** ES is parallelizable
across CPU workers (no GPU needed for the rollouts) and has lower
per-step memory than PPO/GRPO. At our resource-constrained scale it
could plausibly out-perform DPO/SimPO for instruction-following at
the same wall-clock budget. Worth a real benchmark.

See [`docs/evolution_strategies.md`](../evolution_strategies.md).

## FP4 training (parked — hardware-blocked)

Three papers in 2025 established that **fully-quantized FP4 training**
(weights + activations + gradients all in FP4) reaches BF16-comparable
quality:

- [Wang et al., Jan 2025](https://arxiv.org/abs/2501.17116) — "Optimizing LLM Training Using FP4 Quantization." Demonstrated at 13B params on 100B tokens.
- [Microsoft & Nvidia FP4-All-the-Way, May 2025](https://arxiv.org/abs/2505.19115) — 200B-token validation.
- [Quartet II / NVFP4, Jan 2026](https://arxiv.org/abs/2601.22813) — improved gradient estimator.

The format that wins is **NVFP4**: blocks of 16 FP4 values share a
scale factor; stochastic rounding on backward+update, round-to-nearest
on forward. Key empirical threshold: when gradient norm drops below
~√3 × quantization noise, FP4 training stops working — caps how deep
into training you can stay in FP4.

**For us:** Mac M-series GPUs don't have native FP4 ops yet (we'd
simulate). bf16 → FP4 is a ~3-4× memory savings on top of bf16's 2×.
But the dependency on hardware FP4 means this is **parked** for
TinyGPT until Apple silicon supports it. Listed in
[`blockers.md`](blockers.md).

## 2026 small-model landscape (relevant peers)

The competitive scale for "small model that's actually useful" has
shifted up since our project started:

- **SmolLM3-3B** — fully open instruct + reasoning model. Beats
  Llama-3.2-3B and Qwen2.5-3B on 12 benchmarks. Trained on
  ~600B tokens (SmolLM corpus = Cosmopedia + FineWeb-edu + Stack).
- **Qwen3.5-0.8B** — multimodal (text + vision) from scratch.
  Apache 2.0. Released Feb-March 2026.
- **Phi-4-mini-instruct** — Microsoft. Data-quality-over-scale
  thesis. Beats GPT-4o on MATH; beats Llama-3.2-3B across all
  benchmarks.
- **Gemma-3n-E2B-IT** — Google. On-device-focused. 2B with
  multi-modal.

Implication: **the leaderboard's "browser-trainable small model"
niche is now distinctively educational + open-process, not
performance-competitive with 2026 commercial small models.** The
leaderboard product narrative should emphasize "every byte of training
code is here" + "trains in a tab" rather than "competes with Phi-4."

## Tools we should know about

- **[Unsloth](https://github.com/unslothai/unsloth)** — fine-tuning
  framework that fits 8B models on 12 GB consumer GPUs via custom
  Triton kernels + memory tricks. Not Mac/MLX-Swift, but worth
  studying for technique transfer.
- **[Argilla Distilabel](https://github.com/argilla-io/distilabel)** —
  Python framework for synthetic SFT/DPO data generation pipelines.
  Wraps Magpie, DEITA, UltraFeedback recipes.
- **DEITA** ([Liu et al., 2024](https://arxiv.org/abs/2312.15685))
  — instruction-tuning data quality framework. Scores complexity ×
  quality × diversity for instruction sets.

## Survey / overview reads

- **State of GPT** (Karpathy, 2023) — still the cleanest mental model
  of pretrain → SFT → RM → PPO. We're skipping RM/PPO for DPO.
- **A Survey of LLMs** (Zhao et al., 2024) —
  "[arXiv 2303.18223](https://arxiv.org/abs/2303.18223)." Continuously updated.
- **HuggingFace Alignment Handbook** —
  [github.com/huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook).
  The reference recipes for SFT/DPO at the 7B scale.
- **AllenAI Tulu-3 paper** — Lambert et al., 2024.
  "[Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)."
- **SmolLM blog post** (Hugging Face, 2024) —
  [blog](https://huggingface.co/blog/smollm). 135M / 360M / 1.7B
  fully-open small models with their training recipe.

## Honest note on the cutoff

Assistant knowledge cutoff is **January 2026**. After that, items above
were folded in via web search; spottier coverage of Feb-May 2026.
If you have specific recent papers, datasets, or techniques in mind
that aren't listed, point me at the URL or name and they can be folded in.
