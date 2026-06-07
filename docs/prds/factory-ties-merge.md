---
name: tinygpt merge — TIES + DARE model merging
status: shipped-2026-06-07
owner: unassigned (parallel-agent task)
created: 2026-06-07
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (factory primitive — "brother" path)
priority: P1
---

# PRD — `tinygpt merge`

## 2026-06-07 ship note

`tinygpt merge` is implemented in `native-mac/Sources/TinyGPT/Merge.swift` and
wired into the CLI. It supports `ties`, `dare`, and `linear`, validates matching
checkpoint tensor layouts, and writes a valid merged `.tinygpt`.

## Goal

Add a `tinygpt merge` subcommand that combines two or more `.tinygpt`
models (same architecture) into one merged model using TIES + DARE.
Adopt mergekit's algorithm (the consensus best merging method as of 2025).

This is the "brother path" of model building — combining models, not
descending from one.

## Why ship

Multiple training arcs eventually produce sibling models you want to
combine:
- Pace planner specialist + Pace style-tuned LoRA → one deployable
- General SFT specialist + DPO-tuned variant → keep both abilities
- Multiple language-specialist code models → one polyglot

TIES + DARE is the consensus winner (mergekit's primary method).
Implementation is ~100 lines of weight arithmetic per algorithm.

## Scope — in

### CLI surface

```
tinygpt merge \
    --models A.tinygpt B.tinygpt C.tinygpt \   # 2+ models, same architecture
    --weights 1.0 0.5 0.5 \                    # per-model contribution
    --method ties \                            # ties | dare | linear
    --density 0.5 \                            # TIES drop-bottom fraction (param)
    --out merged.tinygpt
```

### Algorithm

For TIES:
1. Compute task vectors: `delta_i = model_i - base` (if base provided)
   or `model_i` (no base)
2. Keep top-K% of each delta by magnitude (drop the rest)
3. Per-parameter elect sign by majority vote across models
4. Disjoint-merge: average non-zero entries among models that agreed

For DARE: like TIES but use random Bernoulli mask + rescale (preserves
expected weights).

For linear: just weighted-average all weights.

### Acceptance criteria

1. `tinygpt merge --models data/gallery/code.tinygpt data/gallery/chat.tinygpt --method ties --out merged.tinygpt` produces a valid `.tinygpt`
2. Loading the merged model + sampling works (no NaN, no crash)
3. Test: merging the same model with itself = identity (within fp noise)
4. PR includes a smoke comparison: sample from each input model + merged model on the same prompt

## Scope — out

- LoRA-into-base merge (separate PRD; that's "bake LoRA into base")
- Cross-architecture merge (only same-arch for v1)
- More than 8 models at once (loop is fine but skip stress test)

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/Merge.swift` |
| **modify** | `native-mac/Sources/TinyGPT/TinyGPT.swift` — dispatch for `merge` |

## Estimated effort

**~1-2 days.** TIES algorithm is well-documented; main work is the
weight-walking + safetensors round-trip.

## Source

- TIES: Yadav et al. 2023 (https://arxiv.org/abs/2306.01708)
- DARE: Yu et al. 2023 (https://arxiv.org/abs/2311.03099)
- mergekit reference impl: https://github.com/cg123/mergekit
