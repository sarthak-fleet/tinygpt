---
name: Factory completeness tracker — meta-PRD
status: tracking
owner: maintainer
created: 2026-06-07
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md
---

# Factory completeness — meta-PRD

Tracks the "make the factory complete" thrust spawned 2026-06-07.

User goal (verbatim): _"to build a model there are multiple steps
involved. There are multiple paths that take you from scratch. There
are few parts that build a brother and few parts that build a child.
For each path we have a best process so I want all the best process in
there. Whatever thing we make, it needs to be evaluated so we need an
eval pipeline."_

## Factory primitives — status

### Parent path (from-scratch)

| Primitive | Shipped | Gap PRD |
|---|---|---|
| Data acquisition (HF / parquet) | ✅ | — |
| Data dedupe (MinHash) | ✅ | — |
| Data quality filter | ✅ | — |
| Data PII / toxicity filter | ✅ | `factory-pii-toxicity-filter.md` |
| Tokenizer trainer | ✅ | `factory-tokenizer-trainer.md` |
| Pretrain (WSD, AdamW, RoPE) | ✅ | — |
| Loss-spike detect/rollback | ✅ | — |
| Determinism harness | ✅ | — |
| Multi-checkpoint analysis | ✅ | — |

### Child paths (descend from parent)

| Primitive | Shipped | Gap PRD |
|---|---|---|
| SFT / LoRA | ✅ | — |
| QLoRA (4-bit + LoRA) | ✅ v0 | `factory-qlora.md` |
| Continued pretrain | ✅ (via `--resume`) | — |
| Domain-adapt mode | ✅ | `factory-domain-adapt.md` |
| Distillation (hard) | ✅ | — |
| Distillation (soft / KL) | ✅ | `factory-soft-distill.md` |
| **Synthesize teacher labels** | ✅ | `factory-synthesize.md` |
| DPO | ✅ CLI; app stub | — |
| GRPO / reasoning RL | ⬜ Tier 3 | (defer) |
| MEMIT / patching | ✅ | — |
| SAE | ✅ | — |
| Pruning | ✅ | — |

### Quantization / compression

| Primitive | Shipped | Gap PRD |
|---|---|---|
| GGUF Q5_K / Q6_K / Q8_K | ✅ | — |
| GGUF Q4_K_M | ✅ | `factory-q4-km-quant.md` |
| GPTQ | ✅ | — |
| HQQ | ✅ | — |
| AWQ | ⬜ Tier 3 | (defer) |
| Unstructured prune | ✅ | — |
| Structured prune | ✅ | — |

### Brother paths (parallel)

| Primitive | Shipped | Gap PRD |
|---|---|---|
| TIES + DARE merging | ✅ | `factory-ties-merge.md` |
| LoRA hot-swap | ⬜ | (queued — not blocking) |
| Best-of-N | ✅ (`bon`) | — |
| Routing / escalation | ✅ (`cloud` / `escalate`) | — |
| Spec dec | ✅ | — |

### Inference

| Primitive | Shipped | Gap PRD |
|---|---|---|
| KV cache (sample) | ✅ | — |
| KV cache (serve) | ✅ | `factory-serve-prompt-cache.md` |
| Constrained decoding (GBNF) | ✅ | — |
| ANE inference | ⬜ Tier 3 | (defer) |
| Continuous batching | ⬜ Tier 3 | `serve-vllm-mlx-wrap.md` |

### Eval (universal)

| Primitive | Shipped | Gap PRD |
|---|---|---|
| E0 schema + eval-compare | ✅ | — |
| lm-eval-harness wrapper | ✅ | — |
| Custom user eval | ✅ | — |
| LLM-as-judge | ✅ | — |
| BFCL | ✅ | — |
| τ-bench | ✅ | — |
| HumanEval / MBPP | ✅ | — |
| MTEB (retrieval) | ✅ | `factory-mteb-eval.md` |
| Train-time eval hook | ✅ | — |
| Cross-checkpoint emergence | ✅ | — |

## Priority order for elf pickup

P0 (shipped):
1. `factory-synthesize.md` (shipped 2026-06-07)

P1 (high-leverage, no dependencies):
2. `factory-mteb-eval.md` (unblocks KB validation)
3. `factory-ties-merge.md` (brother path)
4. `factory-qlora.md` (opens 13-30B fine-tune)

P2 (refinement, lower urgency):
5. `factory-soft-distill.md`
6. `factory-q4-km-quant.md`
7. `factory-reranker.md` (shipped v1 2026-06-07)
8. `factory-tokenizer-trainer.md` (shipped 2026-06-07)
9. `factory-pii-toxicity-filter.md`

P3 (nice-to-have):
10. `factory-domain-adapt.md`

Deferred (Tier 3):
- AWQ, GRPO/reasoning-RL, ANE, vllm-mlx wrap

## When the factory is "complete"

- Every primitive above marked ✅
- Each has a CLI surface + sensible defaults
- DPO + distill + QLoRA + merge have first-class App Train-tab UIs (not stubs)
- Eval surface for every artifact type (LLM, embedder, reranker)

Estimated total: ~4-6 weeks of focused work across all gap PRDs.
