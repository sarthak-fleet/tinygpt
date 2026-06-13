---
name: B19 Group-SAE (layer-group SAE training)
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B19)
related_prds: B13-interp-on-checkpoints.md, B17-saelens-interop.md
---

# PRD — Train one SAE per layer-group instead of per-layer

## Goal

Add `tinygpt sae --group-layers SPEC` so a single SAE is trained on
the concatenated activations of a contiguous block of layers — e.g.
`--group-layers "0-3,4-7,8-11,12-15"` trains 4 SAEs for a 16-layer
model instead of 16. Cuts SAE training cost ~4× at this group size
([Wang et al. 2024](https://arxiv.org/abs/2410.21508)) with modest
feature-recovery loss.

Layered onto the existing `tinygpt sae` path — no new model class,
just a different activation source.

## Why now

- We ship per-layer SAEs today (`tinygpt sae --layer L`). At 24 layers
  × ~hours-per-SAE on a small model, "train SAEs across the timeline"
  (B13) costs days. Group-SAE makes that interp-on-checkpoints arc
  practical.
- The Wang et al. paper validates the cost/quality tradeoff. Their
  finding: groups of 3–4 layers retain ~90% of the per-layer feature
  recovery at ~25% of the compute.
- We have the infra (the SAE training loop) and the eval (SAE
  reconstruction MSE + L0). The PRD is a small new code path, not a
  redesign.

## Scope — in

- `--group-layers SPEC` flag on `tinygpt sae`, parsed as
  comma-separated layer ranges (same parser style as `tinygpt memit
  --layers`).
- New activation gather path: for a group `[lo, hi]`, concatenate the
  residual-stream tensors across `lo..hi` *along the d_model axis*
  (not the token axis — that would explode batch size). The SAE's
  input dimension becomes `(hi - lo + 1) × d_model`.
- Sidecar format extends `.sae` with a `group: [lo, hi]` field
  (back-compat: absent = single layer, the existing semantics).
- `tinygpt sae-explore` learns the group format — for a group SAE,
  features are decomposed back to per-layer via a simple column
  slice when visualizing per-layer attribution.
- Re-use the existing trainer hyperparameters; the only width change
  is the SAE encoder's input dim.

## Scope — out

- **Cross-layer feature alignment** — interesting research question
  but outside this PRD's "cheaper training" goal.
- **Multi-resolution groups** (some groups of 2, others of 4). V1
  uses uniform groups per the paper.
- **Group-SAE on attention activations**. V1 sticks to residual stream
  (the existing path).

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/Sae.swift` | parse `--group-layers`; assemble group activation tensor |
| `Sources/TinyGPTModel/SaeReader.swift` | add `group` field to `.sae` sidecar; reader keeps back-compat default |
| `Sources/TinyGPT/SaeExplore.swift` | layer-attribution decomposition for group SAEs |
| `Tests/TinyGPTTests/GroupSaeTests.swift` | new — train 100 steps on 4-layer group of a tiny model, assert MSE non-increasing |
| `docs/interpretability.md` | "Group-SAE for cheaper timeline interp" subsection |

## Acceptance criteria

- [ ] `tinygpt sae --group-layers "0-3,4-7" --corpus shakespeare.txt
  --steps 500 --out /tmp/g.sae` produces a single sidecar with two
  group SAEs.
- [ ] Reconstruction MSE on a held-out shakespeare batch is within
  20% of the per-layer SAE for the same configuration.
- [ ] Training wall-clock for `--group-layers "0-3"` is within ±10%
  of *one* per-layer SAE, not 4× (the whole point).
- [ ] `tinygpt sae-explore` opens a group SAE without errors and
  reports per-layer attribution for the top-K features.
- [ ] Smoke test passes.

## Reference patterns

- `Sources/TinyGPT/Sae.swift` — direct template.
- [Wang et al. 2024](https://arxiv.org/abs/2410.21508), §4 — the
  group concatenation strategy + per-layer decomposition recipe.
- `tinygpt memit --layers SPEC` parser — same grammar; lift the helper
  into a shared `LayerSpec.parse(_:)`.

## Open questions

- Default group size when `--group-layers` is missing. **Recommendation:**
  no default — fall through to per-layer behavior. Users opt into the
  cost cut explicitly.
