---
name: B13 interp-on-checkpoints methodology
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B13)
related_prds: sae-timeline-viewer.md (visualization sibling — ships the chart, this PRD ships the infra producing the data)
---

# PRD — Interpretability across a training-run timeline

## Goal

Replay the shipped interp tools (`tinygpt sae`, `tinygpt memit`,
`tinygpt rome`, `tinygpt patch`, `tinygpt causal-trace`) across the
multi-checkpoint history a single `tinygpt train` run produces, so we
can see *when* a feature emerges, not just *that* a final model has it.
Pythia ([Biderman et al. 2023](https://arxiv.org/abs/2304.01373)) and
OLMo ([Groeneveld et al. 2024](https://arxiv.org/abs/2402.00838)) made
this the standard small-model interp protocol; no competitor ships it
at TinyGPT's "every-byte-of-code-here" scale.

The `/sae-timeline.astro` viewer (already shipped) consumes the
output of this PRD's tooling. Without this PRD, the viewer renders
empty.

## Why now

- E8 train-time eval hook shipped already (writes `<step>-evals.jsonl`).
  This is the *interp* analogue — replay shipped interp probes per
  saved checkpoint instead of per-eval task.
- Save-every-N is the natural hook point. Already implemented for
  the eval-hook PRD; we attach to the same callback list.
- Decisive value for A1 specialist work: training a tool-caller and
  not knowing when features emerge means we're working blind. With
  this PRD, we can pinpoint "tool-call refusal" feature appearance.

## Scope — in

- `Sources/TinyGPT/InterpReplay.swift` — orchestrator. Walks a
  history directory, loads each checkpoint, runs the requested probe,
  writes one JSONL row per (probe, layer, checkpoint).
- New CLI: `tinygpt interp-replay <history-dir> --probe {sae,memit,
  rome,patch,causal-trace} [--layer L | --layers SPEC] --out timeline.jsonl`
- Row schema: `{step: Int, ckpt_hash: String, probe: String, layer:
  Int, metric: String, value: Double, extra: [String: Any]}`. The
  `extra` blob carries per-probe specifics (SAE MSE + L0, MEMIT
  per-fact residual, etc.). The shared `metric` axis lets the timeline
  viewer plot any probe.
- Default behavior: when only `--out` is given (no probe), replay
  all probes that have inputs (a `.sae` config, a `--memit-facts` file,
  etc.) — turns this into "give me the full interp timeline for this
  run".
- Wire into `tinygpt train`'s optional `--interp-every N` hook (same
  shape as `--eval-every`). Non-blocking; skip if previous interp
  pass is still running.

## Scope — out

- **New interp probes.** This PRD only orchestrates the shipped
  ones. New probes (e.g. concept bottlenecks, activation patching at
  scale) go in their own PRDs.
- **Live overlay** of in-progress checkpoint data while training. The
  shipped train viewer (C10) already shows real-time loss; the interp
  side is fast enough as post-hoc.
- **Cross-run timeline** — comparing the same feature across two
  separate training runs. Useful but adds 200 LOC of UI; defer.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/InterpReplay.swift` | new — orchestrator |
| `Sources/TinyGPT/Train.swift` | optional `--interp-every N` flag; spawn background interp pass per checkpoint |
| `Sources/TinyGPT/TinyGPT.swift` | `case "interp-replay"` |
| `Sources/TinyGPTModel/CheckpointBatchLoader.swift` | new — iterate a history directory, yield `(step, modelHandle)` pairs efficiently (mmap-friendly) |
| `evals/interp-replay-smoke.sh` | new — run an SAE replay across 3 checkpoints of a 200-step shakespeare run; assert MSE non-increasing |
| `docs/interpretability.md` | "Timeline view" section + invocation example |

## Don't touch

- Individual probe CLIs (`Sources/TinyGPT/Sae.swift`, `Memit.swift`,
  etc.) — extend, don't fork. The orchestrator calls them as library
  functions, not as subprocesses.
- `/sae-timeline.astro` — separate PRD (already shipped). Output
  format here must match what the viewer expects.

## Acceptance criteria

- [ ] `tinygpt interp-replay <history> --probe sae --out timeline.jsonl`
  on a 5-checkpoint history produces a JSONL with 5 × n_layers rows.
- [ ] `--interp-every 1000` integrated into `tinygpt train` doesn't
  block the training loop more than 100 ms per checkpoint (the heavy
  interp work happens in a separate process).
- [ ] `/sae-timeline.astro` drag-drops the produced JSONL and renders
  MSE + L0 curves correctly.
- [ ] Smoke script `interp-replay-smoke.sh` passes in CI on the
  shipped 22M-class checkpoint.
- [ ] `docs/interpretability.md` invocation example reproduces.

## Reference patterns

- `Sources/TinyGPT/RunLmEval.swift` — the post-checkpoint multi-task
  orchestrator pattern. Same shape: load checkpoint, run probe, write
  row, repeat.
- `tinygpt sae`'s existing `.sae` sidecar — the per-checkpoint output
  artifact this PRD batches.
- [Pythia paper](https://arxiv.org/abs/2304.01373) — the methodology
  is theirs; cite. Their interp-on-checkpoints findings (feature
  emergence + reversal points) are the kinds of analysis this enables.

## Open questions

- Whether to pre-materialize SAE features per checkpoint (a few MB
  per checkpoint × 50 checkpoints = manageable) or compute on-demand
  in the viewer. **Recommendation:** pre-materialize; the viewer's
  drag-drop UX requires it.
- Default `--layers` selection — every layer (expensive) vs every
  4th (Pythia's choice). **Recommendation:** every 4th + 0 + last;
  override with `--layers all`.
