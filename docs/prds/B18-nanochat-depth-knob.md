---
name: B18 nanochat-style `--depth` single-knob HP derivation
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B18)
related_prds: B11-wsd-schedule.md (the LR-schedule half; --depth derives that too)
---

# PRD — Single `--depth N` knob auto-derives every pretrain HP

## Goal

`tinygpt train --depth 12 ...` should auto-derive `d_model`, `n_heads`,
`d_mlp`, `peak_lr`, `batch_size`, and `total_steps` from compute-
optimal scaling laws. UX win — most users don't want to tune 6
hyperparameters when the depth alone implies a reasonable point.

[karpathy/nanochat](https://github.com/karpathy/nanochat) ships this
exact pattern. We borrow the surface.

## Why now

- B6 (Mac app demo) wants a 1-click train flow. Six hyperparameter
  fields in the app are six bugs waiting to happen. `--depth` is
  the right abstraction.
- TinyGPT users currently hand-set d_model / n_heads / etc. via
  `ModelConfig` presets. Presets work for ~5 named sizes; everything
  in between is hand-tuning.
- Cheap to add — a single function that maps depth → ModelConfig +
  TrainSchedule.

## Scope — in

- `Sources/TinyGPTModel/DepthDerivation.swift` — `deriveHP(depth: Int)
  -> (config: ModelConfig, schedule: TrainSchedule)` per the scaling
  laws documented in nanochat's recipe + the Chinchilla compute-
  optimal corner.
- `tinygpt train --depth N` flag — when set, overrides individual
  HP flags (with a warning if conflicting flags are also passed).
- Documented table in `docs/training_guide.md` mapping depth →
  derived HPs, so users can verify what they're getting.

## Scope — out

- **Width independent of depth.** V1 is depth-driven; aspect-ratio
  tuning is V2.
- **µ-Transfer (µP)** for HP transfer across scales. Bigger arc;
  defer.
- **Recipe presets** for non-Chinchilla regimes (longer-than-
  compute-optimal training, etc.). V1 = Chinchilla corner.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/DepthDerivation.swift` | new — the derivation fn |
| `Sources/TinyGPT/Train.swift` | parse `--depth`; warn on conflicts; resolve and log the derived HPs in the banner |
| `Tests/TinyGPTModelTests/DepthDerivationTests.swift` | new — assert derived HPs match a hand-checked table for depth ∈ {4, 12, 24, 36} |
| `docs/training_guide.md` | the depth-HP table |

## Acceptance criteria

- [ ] `tinygpt train --depth 12 --corpus shakespeare.txt --steps
  1000` runs with auto-derived HPs and reports them in the banner.
- [ ] Derivation matches nanochat's curve to ±10% on the chinchilla-
  optimal points.
- [ ] Existing recipe paths (explicit d_model / n_heads / etc.)
  unchanged.

## Reference patterns

- [karpathy/nanochat](https://github.com/karpathy/nanochat) — direct
  inspiration.
- [Chinchilla, Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)
  — the compute-optimal curve. Cite.

## Open questions

- Whether the derivation hits the Chinchilla corner or a slightly
  over-trained corner for small models. **Recommendation:** add a
  `--regime {chinchilla,overtrained}` flag, default chinchilla.
