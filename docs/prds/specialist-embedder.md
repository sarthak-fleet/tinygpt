---
name: specialist embedder — RECLASSIFIED to strategy doc (not a PRD)
status: superseded-by-strategy-doc
owner: n/a — see strategy doc
created: 2026-06-06
superseded: 2026-06-06 (same day — category error: this is a Tier 2 modality entry, not a single elf-shippable task)
---

# Not a PRD — Tier 2 modality tracked elsewhere

This file existed briefly as a PRD but was reclassified the same day.
The specialist-embedder track is a **new Tier 2 modality** (encoder
models for retrieval / RAG), not a discrete 1-5 day task. It contains
multiple sub-tasks that each merit their own PRD when work begins.

## Where the content lives now

See **`docs/sessions/2026-06-06-mac-specialist-platform.md`**:
- "Strategic Decision 2: Add 'specialist embedder' as Tier 2 modality" —
  the strategic rationale + wedge framing
- Open question #6 — embedder-vs-A1 sequencing (parked until N02 lands)
- The Tier 2 "Embeddings" line in the modality roster

## What would be PRD-shaped within this modality

When the strategic decision approves and N02 finishes (so we can compare
the embedder vs A1 paths empirically), the work breaks into:

| Sub-PRD | Effort | Why it's PRD-sized |
|---|---|---|
| `embedder-model-arch` | ~2 days | Encoder + mean-pool head + `.tinygpt-embed` format. Single feature. |
| `embed-train-cli` | ~2 days | `tinygpt embed-train` subcommand + distillation loop. Discrete. |
| `embed-train-matryoshka` | ~1 day | Matryoshka loss layered on existing trainer. Single feature. |
| `embed-train-hard-negatives` | ~1 day | Hard-negative mining utility. Reuses dedup primitive. Discrete. |
| `embed-infer-cli` | ~1 day | `tinygpt embed` inference + parquet output. Single feature. |
| `rag-local-embedder-recipe` | ~1 day | Cookbook + `scripts/rag-shim.py`. Docs+glue. |
| `embed-eval-mteb-subset` | ~1 day | Custom-eval task for retrieval metrics (MRR/NDCG/Recall@K). |

Each gets a real PRD when work begins (probably post-N02, when the
sequencing question resolves).

## Why this was a category error

I packed an entire modality's roadmap into one "PRD" because it felt
important enough to PRD. But a modality is a strategic decision +
multi-PRD initiative, not a single elf-shippable task. The right
artifact is a **strategy-doc entry that lists the sub-PRDs**, not a
monolithic PRD pretending to be one.

Lesson (same as the vllm-mlx file): if a "PRD" introduces a new modality,
spans 5+ days, or describes a multi-feature initiative, it's a strategic
plan, not a PRD.
