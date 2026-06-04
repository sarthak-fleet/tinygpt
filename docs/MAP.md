# Doc map — where things moved

This restructure split, merged, and archived a few docs. Use this table
to find where the content of any old path now lives.

If you came here looking for a doc that used to exist, find the old path
in the left column → click the new path.

## Split

The 1,400-line master roadmap was broken into a folder. Every section of
the original lives at exactly one of the new paths:

| Old path | New path |
|---|---|
| `docs/single_machine_roadmap.md` (Part 1, Tier 1) | [`docs/roadmap/tier1.md`](roadmap/tier1.md) |
| `docs/single_machine_roadmap.md` (Part 1, Tier 2) | [`docs/roadmap/tier2.md`](roadmap/tier2.md) |
| `docs/single_machine_roadmap.md` (Part 1, Tier 3) | [`docs/roadmap/tier3.md`](roadmap/tier3.md) |
| `docs/single_machine_roadmap.md` (Part 1, Tier 4) | [`docs/roadmap/tier4_skip.md`](roadmap/tier4_skip.md) |
| `docs/single_machine_roadmap.md` (Part 2, categories) | [`docs/roadmap/categories.md`](roadmap/categories.md) |
| `docs/single_machine_roadmap.md` (Part 3, top-10 order) | [`docs/roadmap/recommended_order.md`](roadmap/recommended_order.md) |
| `docs/single_machine_roadmap.md` (Part 4, datasets) | [`docs/roadmap/datasets.md`](roadmap/datasets.md) |
| `docs/single_machine_roadmap.md` (Part 5, recent research) | absorbed into [`docs/PLAN.md`](PLAN.md) §4; archived at [`docs/archive/recent_research.md`](archive/recent_research.md) |
| `docs/single_machine_roadmap.md` (Part 6, phased plan) | [`docs/roadmap/phased_plan.md`](roadmap/phased_plan.md) |
| `docs/single_machine_roadmap.md` (Part 7, blockers) | [`docs/roadmap/blockers.md`](roadmap/blockers.md) |
| `docs/single_machine_roadmap.md` (honest summary) | [`docs/roadmap/honest_summary.md`](roadmap/honest_summary.md) |
| `docs/single_machine_roadmap.md` (index / TOC) | [`docs/roadmap/index.md`](roadmap/index.md) |

The training-pipeline doc was split by phase:

| Old path | New path |
|---|---|
| `docs/training_phases.md` (Phase 1: Pretrain) | [`docs/training/pretrain.md`](training/pretrain.md) |
| `docs/training_phases.md` (Phase 2: SFT) | [`docs/training/sft.md`](training/sft.md) |
| `docs/training_phases.md` (Phase 3: DPO) | [`docs/training/dpo.md`](training/dpo.md) |
| `docs/training_phases.md` (end-to-end + reading) | [`docs/training/index.md`](training/index.md) |

## Merged (lossless)

Content of these docs now lives as appendices of the canonical home; the
original moved to `docs/archive/`.

| Old path | New home | Archived at |
|---|---|---|
| `docs/evaluation.md` | [`docs/validation_report.md`](validation_report.md) (appendix) | [`docs/archive/evaluation.md`](archive/evaluation.md) |
| `docs/watch_the_model_think.md` | [`docs/interpretability.md`](interpretability.md) (appendix) | [`docs/archive/watch_the_model_think.md`](archive/watch_the_model_think.md) |
| `docs/phase_9_10_status.md` | [`docs/roadmap/blockers.md`](roadmap/blockers.md) (appendix) | [`docs/archive/phase_9_10_status.md`](archive/phase_9_10_status.md) |

## Archived

Moved as-is to `docs/archive/`:

| Old path | New path |
|---|---|
| `docs/annotated_transcript.md` | [`docs/archive/annotated_transcript.md`](archive/annotated_transcript.md) |
| `docs/parked_multi_model.md` | [`docs/archive/parked_multi_model.md`](archive/parked_multi_model.md) |
| `docs/shared_vs_native.md` | [`docs/archive/shared_vs_native.md`](archive/shared_vs_native.md) |

## URL redirects

Old web URLs (`/docs/<old_slug>`) keep working via static redirects in
`browser/astro.config.mjs`. So a link to `/docs/single_machine_roadmap`
on social media still resolves; it just takes one extra hop to land at
`/docs/roadmap`.

## Canonical homes (for DRY)

To avoid duplicating mechanics across docs, these are the canonical
homes for repeated concepts. If you find an explanation of one of these
*outside* its home doc, it should be a 1-2 sentence pointer + link, not
a full explanation.

| Concept | Canonical home |
|---|---|
| bf16 / gradient accumulation / gradient checkpointing | [`docs/memory_tradeoffs.md`](memory_tradeoffs.md) |
| LoRA mechanics | [`docs/lora_guide.md`](lora_guide.md) |
| MoE (mixture of experts) | [`docs/moe.md`](moe.md) |
| Distillation | [`docs/distillation.md`](distillation.md) |
| MTP (multi-token prediction) | [`docs/mtp.md`](mtp.md) |
| ES (evolution strategies) | [`docs/evolution_strategies.md`](evolution_strategies.md) |
| Quantization (precision study) | [`docs/precision.md`](precision.md) |
| Quantization (Phase 9 status appendix) | [`docs/roadmap/blockers.md`](roadmap/blockers.md) |
| Interpretability (logit lens, attention vis, ablation) | [`docs/interpretability.md`](interpretability.md) |
| Training pipeline (pretrain → SFT → DPO) | [`docs/training/`](training/index.md) |
| Single-machine roadmap + research | [`docs/roadmap/`](roadmap/index.md) |
| Open-source datasets (pretrain/SFT/DPO/code/math/eval) | [`docs/roadmap/datasets.md`](roadmap/datasets.md) |
