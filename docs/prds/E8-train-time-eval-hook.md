---
name: E8 train-time eval hook
status: not-started
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier E (E8)
task_tracker: #238
---

# PRD — `tinygpt train --eval-every N`

## Goal

Add an `--eval-every N` flag to `tinygpt train`. Every N steps (or at
every `--save-history` checkpoint), the trainer kicks off a lightweight
subset of E3 evals on the current model state, appending rows to the
run's eval JSONL. The browser `/training-dashboard.astro` already
plots loss vs step; this gives us eval-score vs step **on the same
timeline**.

## Why now

- The multi-checkpoint comparison view the user explicitly asked for
  ("see how it improves") needs evals to be RUN per checkpoint, not
  manually re-fired after the run ends.
- E0 schema + E3 lm-eval wrapper + B13 multi-checkpoint primitives all
  shipped. This task is the wire-up.
- Pairs with the "interp-on-checkpoints" story: B13 v1 shows SAE
  features emerging; E8 shows downstream eval scores emerging. Two
  lenses on the same training trajectory.

## Scope — in

- New flag `--eval-every N` in `tinygpt train` (default: off; when
  set, runs evals every N save-history ticks OR every N steps if
  `--save-history` is off)
- New flag `--eval-tasks <csv>` (default: `arc_easy,gsm8k` — small
  + fast, covers loglikelihood and generation modes)
- New flag `--eval-limit N` (default: 50 — keeps per-checkpoint eval
  under ~30s)
- Implementation: after each save-history checkpoint write, spawn
  `tinygpt-cli run-lm-eval --tinygpt-model <ckpt-path> --tasks
  <eval-tasks> --limit <eval-limit> --model-name <run-name>
  --model-step <step> --out <run-eval-jsonl> --serve-port <port>` as
  a background subprocess so training doesn't wait
- The training loop continues immediately; the eval runs in parallel
- If a previous eval is still running when the next checkpoint lands,
  skip the new eval (don't queue, don't block training)

## Scope — out (v2)

- Blocking eval (wait for eval before continuing training). Not useful;
  training time dominates. Background-only is correct.
- Multi-task parallel eval (multiple tasks at once). v1 = one
  serial eval per checkpoint covering all listed tasks.
- Eval-time numerics gate (auto-rollback on regression). Beyond v1.
- Eval-time dashboard streaming. The browser viewer reads the JSONL
  on demand; live polling is a v2 viewer change.

## Inputs the agent has

| Resource | Location |
|---|---|
| Train loop | `Sources/TinyGPT/Train.swift` (the `for step in startStep..<steps` loop, around line ~720) |
| Save-history hook | Same file — the `if saveHistory { … cp to step-N.tinygpt }` block, around line ~898 |
| E3 wrapper | `Sources/TinyGPT/RunLmEval.swift` → CLI invocation `tinygpt run-lm-eval --tinygpt-model …` |
| Tokenizer | Read from the model config (`cfg.tokenizerSource`) when invoking the eval subprocess |
| Serve port | Default 8089; for E8 invocations, pick something different per checkpoint to avoid collisions (e.g., 8200 + step % 100) |
| EvalCompare schema | `Sources/TinyGPT/EvalCompare.swift` |

## Acceptance criteria

1. `tinygpt train --help` shows the three new flags
2. End-to-end smoke:
   ```
   tinygpt train --preset tiny --steps 200 --warmup 20 \
     --save-every 50 --save-history \
     --tokenizer <SmolLM2 dir> \
     --corpus data/examples/tiny-corpus.txt \
     --out /tmp/e8-smoke.tinygpt \
     --log-jsonl /tmp/e8-smoke.jsonl \
     --eval-every 100 --eval-tasks arc_easy --eval-limit 5
   ```
   - Training completes in <1 min
   - 4 history checkpoints written
   - 2 eval invocations fired (at step 100 + 200; first comes from
     the step-100 save-history)
   - `tinygpt eval-compare /tmp/e8-smoke-evals.jsonl --by step`
     shows a 2-row table
3. Background eval doesn't block training — step-rate during eval
   should be unchanged within noise (the trainer continues immediately
   after spawning the eval subprocess)
4. Build passes

## File paths

| Action | Path |
|---|---|
| **modify** | `Sources/TinyGPT/Train.swift` — add 3 flags + 1 invocation block after the save-history hook |
| **read** | `Sources/TinyGPT/RunLmEval.swift` (for the exact subprocess args) |
| **don't touch** | `RunLmEval.swift` itself (its behavior shouldn't change), `EvalCompare.swift`, `TinyGPT.swift` (no new subcommand), `PLAN.md`, `HANDOFF.md` |

## Estimated effort

**~1 day.**

- 1 hr: flag parsing + plumbing
- 2-3 hrs: figure out how to safely spawn a non-blocking subprocess
  from inside a Swift training loop without leaking PIDs
- 1-2 hrs: avoid serve-port collisions (multiple in-flight evals)
- 2 hrs: smoke + verify training step-rate doesn't dip during eval

## Coordination

PR description must include:
1. Smoke command + the resulting `eval-compare --by step` table
2. Step-rate measurement during eval vs no-eval baseline
3. Build confirmation

Maintainer merges, marks E8 done, adds a sentence to HANDOFF.md.

## Known risks

- **Serve port collisions**: if eval at step N hasn't finished and
  step N+save-every triggers another eval, two `tinygpt serve`
  processes try to bind. v1 should detect "previous eval still
  running" via a lock file + skip.
- **Subprocess pid leak**: training may exit (Ctrl-C) while an eval
  is in flight. Need a cleanup hook or accept that the eval process
  gets orphaned (macOS reaps it on exit).
- **Resource contention**: running eval on the SAME GPU while training
  continues will slow training. Document this; user picks
  `--eval-every` large enough that the slowdown is acceptable.

## Source links

- E3 wrapper: `Sources/TinyGPT/RunLmEval.swift`
- B13 save-history: `Sources/TinyGPT/Train.swift` (--save-history flag)
- Training dashboard plot target: `browser/src/pages/training-dashboard.astro`
