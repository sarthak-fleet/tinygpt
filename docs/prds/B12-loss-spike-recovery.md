---
name: B12 loss-spike recovery + replay
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B12)
related_prds: adam-state-persistence.md (state-persistence half — shipped sibling),
              C9-determinism-harness.md (debug-replay sibling)
---

# PRD — Auto-rollback on training spikes (the controller half of B12)

## Goal

Add a grad-norm tracker that triggers an automatic `--resume <step-K>
--lr-drop F` if step N's grad-norm exceeds a rolling-median multiplier
(default 3×) — the "loss spiked, wait, where do we go from here?"
question turned into an automatic action. Saves hours of wasted
compute on long pretrain runs.

The on-disk state is already there (`adam-state-persistence.md`
covered the state-persistence half). C9 (determinism harness) is the
debug-tool half. B12 is the **automatic policy** that uses both.

## Why now

- Long pretrains spike. Recovering manually requires watching the
  training; the platform pitch involves overnight runs.
- Both prerequisites have shipped (Adam state persist) or are
  filed (C9 determinism harness — same wave as this PRD).
- A clean spike-recovery policy is the difference between "this
  training run is unattended" and "babysit it."

## Scope — in

- `Sources/TinyGPT/SpikeMonitor.swift` — rolling-window grad-norm
  tracker. Configurable window (default 50 steps), multiplier
  (default 3×), trigger action.
- On trigger:
  1. Pause the optimizer step
  2. Roll back state to step `N - K` (K configurable; default
     10 — far enough to bypass the spike-causing batch, close
     enough not to waste much progress)
  3. Drop LR by a factor (default 0.5)
  4. Resume; log the rollback to the history JSONL
- `--auto-rollback {off,warn,on}` flag (default `warn` — log but
  don't roll back; user opts in to automatic).
- The history JSONL gains a `rollbacks: [{step, reason, lr_after}]`
  field.

## Scope — out

- **Detecting non-grad-norm spikes** (NaN-loss-but-low-grad-norm,
  etc.). V1 = grad-norm only.
- **Rolling back through multiple spikes** in a single run.
  Bounded at 3 consecutive rollbacks; further fails the run.
- **Replay-and-modify** ("rerun with a different LR from step K"
  as a one-off). That's C9's territory.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/SpikeMonitor.swift` | new — controller |
| `Sources/TinyGPT/Train.swift` | wire spike monitor into the step loop; flag parsing |
| `Sources/TinyGPT/HistoryRow.swift` | (or wherever the history JSONL schema lives) — add `rollbacks` field |
| `Tests/TinyGPTTests/SpikeRecoveryTests.swift` | new — inject a synthetic spike, assert rollback fires + LR drops |
| `docs/PLAN.md` | B12 ⬜ → ✅ on ship |

## Acceptance criteria

- [ ] Synthetic spike test: a training run with a `tinygpt train
  --spike-at 200 --auto-rollback on` (test-only flag that injects a
  10× grad-norm spike at step 200) rolls back to step 190, drops LR,
  resumes, and the rest of training proceeds normally.
- [ ] Without `--auto-rollback on`, the spike still gets logged in
  the history JSONL but training continues.
- [ ] On a 3-spike-in-a-row scenario, the third rollback fails the
  run with a clear "training is unstable; investigate" message.

## Reference patterns

- `adam-state-persistence.md` — the state we roll back to.
- C9 determinism-harness — the inspection tool to figure out WHY
  the spike happened. B12 reverses; C9 explains.

## Open questions

- Rollback distance K. **Recommendation:** 10 steps. Far enough
  that the spike-causing batch isn't sampled again immediately
  (assumes data shuffling), close enough that the rollback cost
  is small.
