# Roadmap — top-10 recommended order

Filtering for highest ROI + most likely to surprise-and-delight at our
scale, in order:

1. **NEFTune** ([Tier 1.6](tier1.md), ~half day) — 3-line embedding noise
   +5-10 points on instruction-following. The single highest-impact-per-minute
   item on this entire document.
2. **Magpie synthetic SFT pipeline** ([recent_research.md](recent_research.md),
   ~1 day) — use a public aligned LLM to synthesize SFT data without seeds.
   Models trained on Magpie data surpass Llama-3-8B-Instruct (ICLR 2025).
   Eliminates our dependency on Dolly/no_robots specifically.
3. **Sequence packing** ([Tier 1.2](tier1.md), ~1 day) — already on the
   roadmap.
4. **Knowledge distillation** ([Tier 1.1](tier1.md), ~2 days) — the
   educational + leaderboard play, validated by April 2026 survey.
5. **SimPO** ([Tier 1.5](tier1.md), ~half day) — reference-free preference
   training, half the DPO memory.
6. **Evolution Strategies trainer** ([recent_research.md](recent_research.md),
   ~3 days) — the "genuinely novel for our scale" play. Beats RL on
   long-horizon rewards; lower memory; parallelizable on CPU. May be the
   right alignment recipe for resource-constrained settings.
7. **QLoRA training** ([Tier 1.3](tier1.md), ~1 day) — 6× fine-tune ceiling.
8. **ORPO** ([Tier 1.4](tier1.md), ~1 day) — merge SFT + DPO into one pass.
9. **Browser-side benchmark runner** ([Tier 1.9](tier1.md), ~half day) —
   closes the leaderboard loop.
10. **Gradient clipping** ([categories.md](categories.md) — training-stability
    section, ~half day) — we should already have this; it's standard hygiene.

≈ **7 days of focused work for the entire Tier 1 + top training-stability
items** — and at the end of that we have: a noise-regularized SFT path,
4× SFT speedup, distillation working, ref-free preference training,
combined int4-base + LoRA training, single-pass SFT+DPO, in-browser
benchmark scoring, and proper gradient clipping. That's the "polish
the post-training pipeline" tier.
