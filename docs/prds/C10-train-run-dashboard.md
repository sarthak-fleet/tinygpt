---
name: C10 training-run dashboard
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier C (C10)
related_prds: train-controls-gap-closure.md, app-train-controls-thermal.md
---

# PRD — Live-streaming training-run dashboard (browser viewer)

## Goal

Ship a browser page (`/train-viewer.astro`) that drag-drops a
`tinygpt train` run-history directory (or watches it live via OPFS on
the same machine) and renders the canonical live-training charts:
loss + grad-norm + LR + tok/s over steps. Drag-drop pattern mirrors
`eval-leaderboard.astro` and `sae-timeline.astro` — no server, the
browser parses the on-disk artifacts.

Removes the "guess what's happening at hour 6 of a run" tax. Today
`tinygpt train` prints a line per step to stdout; you scroll the
terminal hoping you saw the right thing.

## Why now

- Every other interpretability surface has a viewer page now
  (eval-leaderboard, sae-timeline). Training-run is the one with no
  visual companion, and it's the one users stare at the longest.
- The shipped train hook already writes per-step JSONL (loss, lr,
  grad-norm, tok/s) — there's a stream to plot. The viewer reads what
  exists; no instrumentation change.
- Building this on the existing Astro + Chart.js-or-similar stack is
  ~1 day; building a W&B integration would be 3–5 days of auth +
  per-run-key wrangling for a project that explicitly wants
  self-contained tooling.

## Scope — in

- `web/src/pages/train-viewer.astro` — drag-drop a directory (Chrome's
  `webkitdirectory`); browser-side ingest of the run-history JSONL
  files.
- Charts (single page, vertical stack):
  - Loss vs step (train + val on the same plot; val is dotted)
  - Grad norm vs step (log scale)
  - LR vs step
  - Tok/s vs step (steady-state throughput)
- Optional sidecar: if the run was started with `--eval-every N`
  (E8 hook), the eval-leaderboard JSONL is read and overlaid as
  star-markers on the loss chart at the eval steps.
- Live mode (same-machine only): if the path is under
  `/runs/<name>/history.jsonl`, set an OPFS file-handle and poll
  every 5s. Works in Chromium-class browsers; documented as a Chrome-
  only feature.
- A "what to look for" panel below the charts: short tooltip text
  ("grad-norm > 3× rolling mean = spike risk; see B12") + links to
  the relevant docs.

## Scope — out

- **W&B / TensorBoard integration**. Out of scope by design — this
  is self-contained tooling.
- **Multi-run overlay** (compare two runs on the same chart). Useful
  but adds complexity; defer to V2.
- **Mobile-responsive layout**. Desktop-only — you're watching a 4-
  hour run, you're at a desk.

## Files to touch

| File | Change |
|---|---|
| `web/src/pages/train-viewer.astro` | new — the page |
| `web/src/lib/train-jsonl.ts` | new — parser for run-history files |
| `web/src/lib/charts.ts` | extend if a shared chart helper exists; else inline (one page) |
| `docs/training_guide.md` | "Watching a run" section pointing at the viewer |
| `docs/prds/README.md` | add row under "Browser viewers" |

## Don't touch

- `tinygpt train` itself — viewer reads what it already writes.

## Acceptance criteria

- [ ] Page loads at `/train-viewer.astro` in dev (`pnpm dev`) and prod
  (`pnpm build && pnpm preview`).
- [ ] Drag-drop a recent `runs/<name>/` directory; loss / grad-norm /
  LR / tok/s charts render within 500ms.
- [ ] If an eval JSONL is present, star-markers overlay on loss at
  every eval step.
- [ ] Live mode: edit history.jsonl with a manual append; charts update
  within 5 seconds.
- [ ] Page is reachable via the existing `/docs/[slug]` route + linked
  from the training guide.

## Reference patterns

- `web/src/pages/eval-leaderboard.astro` — exact template for drag-drop
  ingest + render. Steal the dropzone + the chart wrapper.
- `web/src/pages/sae-timeline.astro` — secondary template (time-series
  rendering).
- `tinygpt train`'s existing history.jsonl format — already documented
  in `train-controls-gap-closure.md`. No new format.

## Open questions

- Chart library. **Recommendation:** whatever `eval-leaderboard.astro`
  already uses (don't add a new dependency). If it's Chart.js, stay
  on Chart.js.
- Live-mode polling vs file-system observer API. **Recommendation:**
  5s polling. The observer API is Chrome-only AND requires permission
  prompts; polling is simpler and works on first load.
