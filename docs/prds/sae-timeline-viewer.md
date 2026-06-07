---
name: Browser SAE-timeline viewer (B13 visualization)
status: shipped-2026-06-05
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier B (B13 v2)
---

# PRD — `/sae-timeline.astro` viewer

## Goal

Ship `browser/src/pages/sae-timeline.astro` — a publishable web page
that consumes a JSONL file produced by `tinygpt sae --checkpoint-dir
… --timeline-out …` and renders SAE feature-emergence across a
training run.

## Why now

- B13 v1 shipped the CLI: `tinygpt sae --checkpoint-dir <dir>
  --timeline-out <jsonl>` trains an SAE per checkpoint and emits
  rows of `{step, layer, d_model, d_features, mse, l0_per_sample,
  l0_frac, sae_path, ckpt_path}`. The data exists. There's no viewer.
- This is the unique-to-TinyGPT differentiation story. No other local
  AI tool ships "watch feature emergence across training" — even a
  basic viewer materializes the differentiation publicly.
- Pairs with `/training-dashboard.astro` (already shipped): one shows
  loss over time, this shows interp signal over time. Sibling pages.

## Scope — in

- Single new file `browser/src/pages/sae-timeline.astro`
- File-upload picker (drop a `.jsonl` from `tinygpt sae --timeline-out`)
- Three charts side-by-side:
  1. **MSE over steps** — SVG line chart, x = step, y = reconstruction
     MSE. Lower is better — should trend down.
  2. **L0 over steps** — x = step, y = mean L0 (active features per
     sample) / d_features. Capability proxy — % of features actively
     firing.
  3. **Feature usage histogram** — for the latest checkpoint, per-feature
     activation count (need to load the actual `.sae` file to compute;
     v1 can skip this and show only the timeline charts).
- Visual language matches `/training-dashboard.astro` — same Theme CSS
  vars, drop-zone, dark mode.

## Scope — out (v2)

- Loading the actual `.sae` files for feature-level introspection
  (would need a Swift parser → safetensors viewer; v1 = JSONL only)
- Feature-alignment-across-checkpoints (the hard interp question:
  is "feature 47 at step 10K" the same as "feature 47 at step 50K"?
  Requires matching post-hoc — v2)
- Cross-run comparison (multiple training runs side-by-side)

## Inputs the agent has

| Resource | Location |
|---|---|
| Sample JSONL | Run `tinygpt sae --checkpoint-dir <existing checkpoints> --timeline-out /tmp/sae-timeline.jsonl` against `/tmp/huge-smoke-30min.step-*.tinygpt` files (exist by the time the agent runs) |
| Pattern to copy | `browser/src/pages/training-dashboard.astro` (full mirror) |
| JSONL schema | `tinygpt sae` source: `native-mac/Sources/TinyGPT/SAE.swift` (search for "Schema kept stable") |
| Theme CSS | Already exported by `training-dashboard.astro` |

## The JSONL schema (read these per line)

```ts
type SaeTimelineRow = {
  step: number;          // training step from the .tinygpt header
  layer: number;         // single-layer; "layers" for group-SAE
  layers?: number[];     // group-SAE only
  d_model: number;
  d_features: number;
  mse: number;           // reconstruction MSE on a held-out batch
  l0_per_sample: number; // mean active features per sample
  l0_frac: number;       // l0_per_sample / d_features, in [0, 1]
  sae_path: string;      // path to the .sae sidecar
  ckpt_path: string;     // path to the .tinygpt checkpoint
};
```

## Acceptance criteria

1. `cd browser && npm run build` produces `dist/sae-timeline.html`
2. Drop a SAE timeline JSONL → all three charts render
3. The MSE chart shows the expected trend (down → flat as training
   progresses)
4. The L0 chart shows the rising/stabilizing pattern documented in
   B13's smoke (24% → 37% across 5 ckpts)
5. Empty state explains the page + describes the input shape

## File paths

| Action | Path |
|---|---|
| **create** | `browser/src/pages/sae-timeline.astro` |
| **read** | `browser/src/pages/training-dashboard.astro`, `native-mac/Sources/TinyGPT/SAE.swift` |
| **don't touch** | Other pages, Swift sources, plan docs |

## Estimated effort

**~1 day.** Mostly mirror of `/training-dashboard` with different
charts.

## Coordination

PR description must include screenshots of all three charts with
real data. Maintainer adds a link from `/training-dashboard.astro`
to `/sae-timeline.astro` (sibling pages).

## Known risks

- B13 has both single-layer AND group-SAE rows. The viewer should
  show single-layer by default; group SAEs (where `layers` is set)
  should aggregate or be filterable.
- The `.sae` file format (per `SAE.swift`'s `SaeWriter`) is custom
  binary (TGSA magic). v2 feature-level introspection requires a
  Swift-side parser; out of scope here.

## Source links

- B13 source: `native-mac/Sources/TinyGPT/SAE.swift` (runTimeline + Row schema)
- Sibling viewer: `browser/src/pages/training-dashboard.astro`
- Bricken et al. 2023 (SAE theory background, for the empty-state copy):
  https://transformer-circuits.pub/2023/monosemantic-features
