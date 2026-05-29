# Roadmap — the phased plan (sequential, executable)

7 weeks of sequenced work. Each phase is ~1 week of focused effort with
one concrete deliverable. You can stop after any phase and still have
shipped something useful.

## Phase 1 — foundation polish (~3 days of quick wins)

| Item | Days | Bucket |
|---|---:|---|
| NEFTune (noisy embeddings) | 0.5 | quality |
| Gradient clipping | 0.5 | stability |
| LoRA+ (different LR for A vs B) | 0.5 | quality |
| Persistent tokenized corpus cache | 0.5 | dev velocity |
| Browser-side benchmark runner | 0.5 | new option (browser) |
| Investigate Mega-bf16 OOM + add guardrails | 1.5 | stability |

**Deliverable:** every post-training step gets the NEFTune bump;
gallery models can be benchmarked from the browser; resumed runs
don't re-tokenize the corpus.

## Phase 2 — useful post-trained model (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| Magpie SFT pipeline | 1 | new option |
| SimPO (reference-free DPO) | 0.5 | perf |
| ORPO (merge SFT + DPO) | 1 | new option |
| KTO (single-label preference) | 0.5 | new option |
| Sequence packing for SFT | 1 | perf |
| Run Huge → SFT → DPO end-to-end, score | 1 | shipping |

**Deliverable:** a Huge-100M model that follows instructions
reasonably, with concrete leaderboard scores.

## Phase 3 — inference unlock (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| Speculative decoding | 2 | perf |
| KV cache quantization | 1 | perf + long context |
| Prefix / prompt caching | 1 | perf |
| StreamingLLM attention sink | 1 | new option |

**Deliverable:** browser playground feels 2-4× snappier; long-context
sampling becomes practical.

## Phase 4 — knowledge distillation, the educational headliner (~6 days)

| Item | Days | Bucket |
|---|---:|---|
| Knowledge distillation trainer | 2 | learning + new option |
| Distill Mega-instruct → 5M student | 1 | shipping |
| Reasoning distillation from R1-Distill | 2 | learning + new option |
| Compare: distillation vs same-size from scratch | 1 | shipping |

**Deliverable:** a 5M-param model that punches above its weight on
the leaderboard. Case study: tiny models reproducibly competing.

## Phase 5 — MoE + distillation combined (~7 days, the bet you're excited about)

This is its own phase because it's the highest-leverage capability
unlock that fits locally. **More effective capability per gigabyte
of memory than the dense equivalent.**

| Item | Days | Bucket |
|---|---:|---|
| MoE architecture: router + expert MLP + load-balance loss | 3 | learning + new option |
| Train a tiny MoE from scratch on a known-good corpus (sanity) | 1 | learning |
| Distill from open-MoE teacher (DeepSeek-V3-Distill family or Mixtral-class) into our 2B / 8-expert MoE | 2 | new option |
| Compare: 2B MoE (4 GB at bf16) vs dense 500M baseline at same per-token compute | 1 | shipping |

**Deliverable:** a 2B-total / ~500M-active MoE that fits in ~4 GB
of memory and outperforms a 500M dense model at the same per-token
compute cost. The "we run a much-bigger-effective model locally"
artifact.

## Phase 6 — new training paradigms + bigger models (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| Evolution Strategies trainer | 3 | new option |
| Multi-Token Prediction | 2 | perf + new option |
| Gradient checkpointing | 2 | new option (bigger models) |

**Deliverable:** real benchmark numbers comparing ES vs DPO; the
first Behemoth/Titan training run that fits in memory.

## Phase 7 — browser-side performance frontier (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| WebGPU subgroups | 2 | perf (browser) |
| WebGPU cooperative matrix | 2 | perf (browser) |
| WebNN integration as fallback | 1 | new option (browser reach) |

**Deliverable:** speedup curve extends from 12.1× into 15-20× on
Chrome; capability pills advertise active perf paths.

## Phase 8 — interpretability tools (~3 days, browser playground)

| Item | Days | Bucket |
|---|---:|---|
| Logit lens visualization | 1 | learning |
| Attention heatmap UI | 1 | learning |
| Per-layer ablation tool | 1 | learning |
| Activation patching | 1.5 | learning |

**Deliverable:** playground gets an "Inspect" tab alongside Sample /
Train / Fine-tune. See [`docs/interpretability.md`](../interpretability.md)
for what's now shipped.

## Phase 9 — quantization + small-model story (~5 days)

| Item | Days | Bucket |
|---|---:|---|
| QLoRA training (int4 base + LoRA) | 1 | new option |
| DoRA | 1 | quality |
| AWQ reader | 1 | new option |
| HQQ quantization | 1.5 | perf |
| LASER selective rank reduction | 0.5 | perf + new option |

**Deliverable:** every gallery model ships in three sizes (fp32 /
bf16 / int4) with documented quality tradeoffs. See
[`blockers.md`](blockers.md) for the per-item Phase 9 status.

## Phase 10 — architecture menu (~6 days, educational)

| Item | Days | Bucket |
|---|---:|---|
| Sliding window attention | 1 | new option |
| ALiBi position bias | 1 | learning |
| Mixture of Depths | 2 | learning |
| Differential attention | 1.5 | learning |
| YOCO cross-layer KV sharing | 1 | new option |

**Deliverable:** five attention variants implemented alongside the
standard one — the "every modern architectural idea has a one-file
implementation here" story.

## The cut-points that matter

- **Stop at Phase 2 (~8 days):** you have a useful 100M
  instruction-following model + real leaderboard numbers. V1 done.
- **Stop at Phase 5 (~26 days):** add inference perf + distillation
  + the MoE-distill big-model-locally artifact. The story is
  complete and compelling. **This is where I'd cut for the HN launch.**
- **Stop at Phase 7 (~36 days):** add ES + browser perf frontier.
  Everything that's both novel and at-our-scale.
- **Everything after Phase 7 is polish + educational deepening.**
