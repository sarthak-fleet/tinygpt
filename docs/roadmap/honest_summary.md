# Roadmap — the honest summary

- **What we CAN build but haven't:** everything in
  [Tier 1-3](tier1.md) + the optimizers / data / interpretability /
  browser-perf items in [categories.md](categories.md) + the
  [phased plan](phased_plan.md). ~50 distinct items, ~10 weeks
  of focused work total.
- **What we CAN'T build right now:** the items in
  [blockers.md](blockers.md). The blockers are real, but **none of
  them prevent us from shipping a genuinely useful artifact** at
  the 100M-1B scale on one Mac.
- **What we COULD build but probably shouldn't:**
  [tier4_skip.md](tier4_skip.md) items (fp16, ZeRO at single-device,
  RLHF/PPO, etc.) — superseded by better alternatives.

## Doc readiness checklist

- ✅ Exhaustive landscape ([Tier 1-4](tier1.md) + orthogonal
  [categories](categories.md))
- ✅ Recent research with arxiv links
  ([recent_research.md](recent_research.md), 2024-2026,
  web-verified)
- ✅ Open-source datasets with URLs + licenses
  ([datasets.md](datasets.md))
- ✅ Phased executable plan ([phased_plan.md](phased_plan.md))
- ✅ "What we can't add right now" ([blockers.md](blockers.md))
- ✅ Honest knowledge-cutoff acknowledgment (in
  [recent_research.md](recent_research.md))
- ✅ Cross-references to other docs
  ([`docs/training/`](../training/index.md),
  [`docs/memory_tradeoffs.md`](../memory_tradeoffs.md),
  [`docs/leaderboard.md`](../leaderboard.md),
  [`docs/perf_quest.md`](../perf_quest.md))

**This doc is ready** to be the master reference for "what's worth
building on TinyGPT." Update path: when new research lands or items
ship, edit in place (the file is the source of truth).

## Cross-reference

- [`docs/training/`](../training/index.md) — pretrain → SFT →
  DPO pipeline (current form)
- [`docs/memory_tradeoffs.md`](../memory_tradeoffs.md) — bf16, grad accum,
  grad checkpointing
- [`docs/leaderboard.md`](../leaderboard.md) — benchmark framework
- [`docs/archive/parked_multi_model.md`](../archive/parked_multi_model.md) — MoE park
- [`docs/perf_research.md`](../perf_research.md),
  [`docs/perf_quest.md`](../perf_quest.md) — browser-side performance levers
- [`docs/precision.md`](../precision.md) — the fp32/fp16/bf16 study
