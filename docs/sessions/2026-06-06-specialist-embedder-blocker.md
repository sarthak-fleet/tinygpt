# Specialist Embedder PRD Blocker

Date: 2026-06-06

PRD: `docs/prds/specialist-embedder.md`

Status: blocked for this pass.

## Why This Is Blocked

The PRD is a real Tier 2 modality, not a small CLI/documentation follow-up.
Shipping it honestly requires:

- A new encoder/pooling architecture in `TinyGPTModel`.
- A `.tinygpt-embed` file variant and loader/inference path.
- Teacher embedding generation from BGE-M3, which is a multi-GB model pull.
- Contrastive/matryoshka training loops and hard-negative mining.
- Parquet or LanceDB-compatible vector output.
- Retrieval evals with MRR@K / recall@K.
- Local training and inference smokes.

Those steps involve long model downloads, package installs, and sustained
training/benchmark loops. The repo's macOS safety rules require explicit user
approval before running that class of workload.

## What Not To Do

Do not add fake `tinygpt embed` or `tinygpt embed-train` commands that produce
hash vectors or placeholders. That would make downstream RAG examples look
shipped while the model and eval path do not exist.

Do not download BGE-M3 or run embedding training without approval.

## Smallest Honest Next Slice

1. Add `EmbedderModel.swift` with an encoder-only forward and mean-pooling
   shape tests, using synthetic token IDs only.
2. Add `embed-train --dry-run` that validates corpus schema and prints the
   planned teacher/student/matryoshka config without downloading weights.
3. Add `embed --dry-run` that validates model metadata once `.tinygpt-embed`
   exists.
4. Only after that, request approval for the BGE-M3 download and the first
   1,000-row training smoke.

## Acceptance Deferred

The PRD acceptance criteria remain deferred:

- No `.tinygpt-embed` file has been produced.
- No BGE-M3 distillation smoke has run.
- No LanceDB/parquet vector output has been verified.
- No domain MRR@10 win has been measured.

This PRD should stay blocked until the owner approves the heavy training and
dependency work.
