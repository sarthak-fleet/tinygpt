# Single-machine roadmap — index

A complete inventory of techniques that **run on one Mac (or in a browser
tab)**, ROI-ranked for TinyGPT. The original 1,400-line master plan is
split across the files below — each is short enough to read in one
sitting.

## How to read this

**Filter** — single-machine only. Anything requiring a GPU cluster
(ZeRO/FSDP, tensor parallelism, large RLHF runs) is in
[`tier4_skip.md`](tier4_skip.md).

**Two views of the same landscape:**

- **Tiers 1-4** are the ROI ranking of *training-or-product-shaping*
  techniques (what to build next). Higher tier = better ROI for us.
- **The category sections** ([`categories.md`](categories.md)) are an
  *exhaustive taxonomy* of everything else — optimizers, data,
  interpretability, browser perf, etc. — orthogonal to the main
  pipeline.

**Status legend** (used throughout):

- 🟢 shipped · 🟡 partial · ⬜ not yet built · 🟣 considered + parked

Markers last verified against the codebase: **2026-05-30**.

For current external benchmark landscape, see
[`docs/research/inference_benchmarks_may_2026.md`](../research/inference_benchmarks_may_2026.md)
and [`docs/research/quality_benchmarks_may_2026.md`](../research/quality_benchmarks_may_2026.md).

## The files

| File | What it covers |
|---|---|
| [`tier1.md`](tier1.md) | **High ROI — build next.** Distillation, sequence packing, QLoRA, ORPO, SimPO, NEFTune, gradient checkpointing, speculative decoding, browser benchmark runner. |
| [`tier2.md`](tier2.md) | **Medium ROI.** KTO, IPO, DoRA, GaLore, VeRA, LoftQ, AWQ/GPTQ readers, HQQ, sliding window, ALiBi, KIVI, MTP, MQA, attention sink, prefix caching, prefix tuning. |
| [`tier3.md`](tier3.md) | **Niche / specialized.** RLAIF, GPTQ from-scratch, pruning, LASER, RLHF, Medusa, MoE, differential attention, MoD, LoRA variants (LoRA+, AdaLoRA, ReLoRA, PISSA, etc.). |
| [`tier4_skip.md`](tier4_skip.md) | **Skip — not worth it for us.** fp16, ZeRO at single device, SSMs, etc. |
| [`categories.md`](categories.md) | Orthogonal categories — optimizers, training stability, data, tokenization, interpretability, inference, browser perf, architecture, PEFT taxonomy, infra. |
| [`recommended_order.md`](recommended_order.md) | The top-10 order to build in. |
| [`datasets.md`](datasets.md) | Open-source datasets we'd actually use (pretrain / SFT / DPO / code / math / eval). |
| [`recent_research.md`](recent_research.md) | 2024-2026 highlights with arxiv links (alignment, PEFT, quantization, inference, optimizers, distillation, reasoning RL). |
| [`phased_plan.md`](phased_plan.md) | The executable 10-phase plan, 7 weeks of sequenced work. |
| [`blockers.md`](blockers.md) | What we **can't** build right now and why (hardware, library, budget, integration). Includes the Phase 9/10 detailed status appendix. |
| [`honest_summary.md`](honest_summary.md) | The "what we can / can't / shouldn't build" summary, plus cross-references. |

## One-line answer to "what do I build first?"

Phase 1 of [`phased_plan.md`](phased_plan.md): NEFTune + gradient clipping
+ LoRA+ + persistent tokenized cache + the browser-side benchmark runner.
~3 days, all small wins.
