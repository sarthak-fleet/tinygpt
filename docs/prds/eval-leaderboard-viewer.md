---
name: Browser eval-leaderboard viewer
status: shipped-2026-06-05
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier E (sibling of E0)
---

# PRD — `/eval-leaderboard.astro` viewer

## Goal

Ship `browser/src/pages/eval-leaderboard.astro` — a publishable web page
that consumes any number of E0-schema eval JSONL files and renders the
three views from `tinygpt eval-compare` (--by model / --by step / --by
task) as interactive charts + tables.

## Why now

- The CLI `tinygpt eval-compare` works but produces ASCII tables. For
  sharing on the website / social / a launch post, a browser view is
  the right surface.
- Pairs with the **emergence story**: the cross-checkpoint view (--by
  step) becomes a real chart showing capability emergence over training.
  Loss curve is one chart in `/training-dashboard`; this is its sibling.
- Once shipped, the README + landing can link to a live leaderboard
  using TinyGPT-trained models.

## Scope — in

- Single new file `browser/src/pages/eval-leaderboard.astro`.
- Mirror the visual language of `browser/src/pages/training-dashboard.astro`
  (same Theme CSS vars, same drop-zone, same dark-mode aesthetic).
- File-upload picker (drag-drop + click) accepting one or more `.jsonl`
  files. Multi-file = merged view, not per-file panels.
- Three view-mode toggles in the header: **Model · Step · Task**. Default:
  Model.
- **Model view**: pivot table — rows are tasks, columns are model names.
  Baselines visually distinguished (e.g., gray text or a "(baseline)"
  pill). Cells show `score (n=N)`. Sort tasks alphabetically.
- **Step view**: pick one `model_name` from a dropdown (default: most
  populated). Render an SVG line chart — x = `model_step`, y = score,
  one line per task. Pairs with `/training-dashboard` visually.
- **Task view**: per-task ranked-best-to-worst list. For each task, show
  all (model_name, model_step?) rows with their scores. Baselines marked.
- Robust JSONL parser — skip malformed lines, tolerate partial last
  lines, ignore rows missing required schema keys.

## Scope — out (v2)

- Persistence to localStorage. v1 re-uploads on every visit.
- Filtering by date / harness_version.
- Live polling of a server endpoint. v1 is static / upload-driven.
- Per-task `subtask` drilldown (BFCL has subtasks). v1 collapses them
  via `aggregate by task` when --by model is selected; v2 adds an
  expand-subtask UI.
- Statistical significance / error bars (lm-eval emits `*_stderr,none`
  alongside `acc,none`; v1 ignores stderr fields — v2 plots them).

## The E0 schema row to consume

Defined in `native-mac/Sources/TinyGPT/EvalCompare.swift` — every E*
harness wrapper emits rows of this shape:

```ts
type EvalRow = {
  run_id: string;             // UUID per eval invocation
  model_path: string;         // absolute path on disk
  model_name: string;         // "tinygpt-huge-base-v1", "SmolLM2-135M", ...
  model_step?: number;        // training step (nil for foreign models)
  baseline: boolean;          // true = reference model
  task: string;               // "mmlu", "gsm8k", "bfcl", ...
  subtask?: string;           // "mmlu/physics", "bfcl/parallel", ...
  metric: string;             // "accuracy", "exact_match", "acc", "acc_norm"
  score: number;              // 0..1 typically
  n_examples: number;
  wall_seconds: number;
  timestamp: string;          // ISO8601
  harness_version?: string;
};
```

Read these from JSONL — one row per line — into an in-memory array,
then group/render per the selected view mode.

## Inputs the agent has

| Resource | Location |
|---|---|
| Pattern to copy | `browser/src/pages/training-dashboard.astro` (~300 lines: file picker, parser, SVG charts, dark theme, all hand-rolled with no chart libs) |
| Theme CSS | already exported by `training-dashboard.astro`; reuse the same `--accent`, `--panel`, `--base`, `--fg`, `--muted`, `--faint`, `--warn`, `--danger` vars |
| Sample JSONL | `/tmp/eval-smoke.jsonl` and `/tmp/eval-tinygpt-logprobs.jsonl` from today's session (or any synth via the Python snippet in [the schema test below](#schema-test-data)) |
| Schema source | `native-mac/Sources/TinyGPT/EvalCompare.swift` — `EvalCompare.Row` struct |

## Acceptance criteria

1. `npm run build` from `browser/` exits 0; produces
   `browser/dist/eval-leaderboard.html`.
2. Drag `/tmp/eval-smoke.jsonl` onto the page → see the SmolLM2 row
   in the model-view table with arc_easy 0.533 / piqa 0.700.
3. Drop the synthetic emergence JSONL ([schema test data
   below](#schema-test-data)) → step-view chart shows
   tinygpt-huge-base-v1 climbing from 0.001 → 0.018 on gsm8k across
   steps 20000 / 50000 / 100000.
4. Switch view modes via the header toggle without re-uploading.
5. Empty state explains the schema + has the drop-zone visible.

### Schema test data

Same data the CLI was smoke-tested with:

```python
import json
ts = "2026-06-05T17:00:00Z"
rows = [
    {"run_id":"r1","model_path":"/tmp/huge-base-v1.tinygpt","model_name":"tinygpt-huge-base-v1","model_step":100000,"baseline":False,"task":"mmlu","metric":"accuracy","score":0.247,"n_examples":14042,"wall_seconds":312.4,"timestamp":ts,"harness_version":"lm-eval-0.4.4"},
    {"run_id":"r2","model_path":"~/.cache/hf/SmolLM2","model_name":"SmolLM2-135M","baseline":True,"task":"mmlu","metric":"accuracy","score":0.351,"n_examples":14042,"wall_seconds":289.1,"timestamp":ts,"harness_version":"lm-eval-0.4.4"},
    {"run_id":"r3","model_path":"/tmp/huge-base-v1.tinygpt","model_name":"tinygpt-huge-base-v1","model_step":100000,"baseline":False,"task":"gsm8k","metric":"exact_match","score":0.018,"n_examples":1319,"wall_seconds":42.1,"timestamp":ts,"harness_version":"lm-eval-0.4.4"},
    {"run_id":"r4","model_path":"~/.cache/hf/SmolLM2","model_name":"SmolLM2-135M","baseline":True,"task":"gsm8k","metric":"exact_match","score":0.061,"n_examples":1319,"wall_seconds":39.8,"timestamp":ts,"harness_version":"lm-eval-0.4.4"},
    {"run_id":"r5","model_path":"/tmp/huge-base-v1.step-20000.tinygpt","model_name":"tinygpt-huge-base-v1","model_step":20000,"baseline":False,"task":"gsm8k","metric":"exact_match","score":0.001,"n_examples":1319,"wall_seconds":41.2,"timestamp":ts,"harness_version":"lm-eval-0.4.4"},
    {"run_id":"r6","model_path":"/tmp/huge-base-v1.step-50000.tinygpt","model_name":"tinygpt-huge-base-v1","model_step":50000,"baseline":False,"task":"gsm8k","metric":"exact_match","score":0.008,"n_examples":1319,"wall_seconds":41.8,"timestamp":ts,"harness_version":"lm-eval-0.4.4"},
]
with open("/tmp/eval-demo.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
```

## File paths

| Action | Path |
|---|---|
| **create** | `browser/src/pages/eval-leaderboard.astro` |
| **read for pattern** | `browser/src/pages/training-dashboard.astro` (full mirror), `native-mac/Sources/TinyGPT/EvalCompare.swift` (schema source) |
| **don't touch** | Other Astro pages, `Sources/`, `docs/PLAN.md`, `HANDOFF.md` |

## Implementation notes (mirroring training-dashboard)

The whole page is a single `.astro` file:

- `pageStyle` string with the same CSS vars `training-dashboard.astro` uses
- `pageBody` string with HTML for the drop-zone + view-mode toggle + chart
  container
- Inline `<script type="module" is:inline>` block with:
  - JSONL parser (`parseJsonl(text)` returning `EvalRow[]`)
  - Grouping helpers (`groupByModel`, `groupByStep`, `groupByTask`)
  - SVG renderers (one canvas-free SVG per view) — copy the
    `makeAxes` / `drawLine` helpers from `training-dashboard.astro`
- File-upload + drag-drop event wiring (clone from
  `training-dashboard.astro`'s `addFile` function)

No chart library. No new npm deps.

## Estimated effort

**~2-3 days focused.** Breakdown:

- 1 hr: read `training-dashboard.astro`, copy the scaffold
- 2-3 hrs: JSONL parser + grouping logic in vanilla JS
- 3-4 hrs: three SVG views (model pivot table is easy, step chart
  needs the line-rendering bits, task ranking is simple lists)
- 2 hrs: view-mode toggle + URL state (?view=model)
- 1-2 hrs: empty state + schema-mismatch error UX
- 1 hr: smoke against the test JSONL above

## Coordination

PR description must include:
1. Screenshot of each of the three views with the test JSONL loaded
2. `npm run build` output showing `[build] Complete!`
3. Any CSS-var additions to `Theme` (should be none — reuse what
   `training-dashboard.astro` already exports)

Maintainer will:
- Add a sidebar link from `browser/src/pages/index.astro` to
  `/eval-leaderboard`
- Mark Tier E sibling task done in PLAN.md / HANDOFF.md

## Known risks

- **Chart layout for many tasks**: if a leaderboard accumulates 20+ tasks,
  the pivot table needs horizontal scroll. Mirror what `speedup.astro`
  does for long tables.
- **Mixed-metric rows**: lm-eval emits both `acc` and `acc_norm` for the
  same task. v1 should display them as separate rows, OR collapse via
  a "primary metric" rule (prefer `acc_norm` if present, else `acc`).
  Pick one and document the choice.
- **Subtask explosion** (BFCL has ~10 subtasks per model run). v1
  aggregates by `task` only; if a row has `subtask`, sum its score
  contribution into the parent task's mean, or display as
  `task / subtask` cells. Pick one — document.

## Source links

- E0 schema (canonical): `native-mac/Sources/TinyGPT/EvalCompare.swift`
- Sibling viewer pattern: `browser/src/pages/training-dashboard.astro`
- The CLI this page renders: `tinygpt eval-compare`
- PLAN entry: `docs/PLAN.md` §3 Tier E preamble (talks about the
  "shared schema → comparison view" architecture)
