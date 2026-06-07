---
name: B12 v2 — Adam optimizer state persistence across save/resume
status: shipped-2026-06-06
owner: unassigned (parallel-agent task — MLX-Swift internals + checkpoint format)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md + docs/prds/app-train-controls-thermal.md (precondition for low-friction frequent pausing)
---

# PRD — persist Adam optimizer state in `.tinygpt` checkpoints

## Goal

Save AdamW optimizer state (first + second moments per parameter) alongside
the model weights in `.tinygpt` checkpoints. On `--resume`, restore the
optimizer state so training continues with *zero* warm-up wobble — same
gradient dynamics as if the run had never stopped.

This makes "pause and resume frequently" (the natural pattern for users
who don't want to thermally hammer their Macs) actually free, rather than
costing 50-100 step wobble per pause.

## Why now

The newly-shipped pause/resume UI (`docs/prds/app-train-controls-thermal.md`)
encourages frequent pausing — thermal safety, machine-shared use, interactive
control. But every pause currently restarts Adam from zero momentum, causing
a small loss reseat on resume. For one pause per run: negligible. For ten
pauses per run: meaningful cumulative signal loss.

Persisting Adam state closes the loop on the pause/resume UX promise:
"pause whenever you want, no cost."

Also unblocks:
- True bit-exact replay (currently `--seed` reproduces init, not full
  trajectory; Adam state is the missing piece for full determinism)
- Loss-spike rollback (B12 v1 detects spikes but can't recover; v2 would
  rollback to a known-good Adam state)
- Multi-job training queues (queue manager can checkpoint between jobs
  without losing optimizer momentum)

## Scope — in

### 1. Checkpoint format extension

Decide between two layouts (PRD recommends option A):

**Option A — sibling file** (recommended)

```
/tmp/run.tinygpt              ← model weights (current)
/tmp/run.tinygpt.opt          ← optimizer state (new)
/tmp/run.step-2000.tinygpt    ← history (current)
/tmp/run.step-2000.tinygpt.opt← history optimizer state (new)
```

Pros: backward compatible (old `.tinygpt` files still load); separate file
makes opt-state optional (resume works without it, just with the wobble);
easier to gitignore selectively (opt files are bigger).

**Option B — embedded** (extends `.tinygpt` format)

Add a section to the existing `.tinygpt` container. Backward-compatible
readers ignore the new section; resume readers check for it.

Pros: one file, simpler bookkeeping.
Cons: requires version-bump of the container format; bigger files always.

**Pick A.** Reasoning: opt-state is large (~2× params) and often
discardable (e.g., for distribution as a final model, you don't want
opt-state). Keeping it separate is cleaner.

### 2. Optimizer state extraction

MLX-Swift's `AdamW` holds `m` (first moment) and `v` (second moment)
buffers per parameter, plus a `step` counter. The agent needs to:

1. **Audit MLX-Swift's `AdamW` public surface** — can we read `m` / `v`
   via the existing `OptimizerBase` API? If yes, great. If not, may need
   to subclass / wrap or contribute upstream.
2. **Serialize** moments to disk in bf16 (matches the model weights' dtype
   for our default bf16 training). Use safetensors format for cross-tool
   compatibility.
3. **File shape**: a safetensors with one tensor per parameter, named
   `<param_path>.m` and `<param_path>.v`, plus a scalar `step` metadata.

### 3. Resume path

On `--resume <path.tinygpt>`:

1. Load model weights as today
2. Check for sibling `<path>.tinygpt.opt`
3. If present:
   - Construct AdamW with same hyperparameters
   - Inject `m`, `v`, `step` from the .opt file
   - Validate: parameter shapes match between weights and opt state
   - On mismatch, fall back gracefully (warn + fresh Adam state)
4. If absent: fall back to fresh Adam state (current behavior) + log
   `[warn] no optimizer state found, resuming with fresh Adam — small loss wobble expected`

### 4. Save path

On every `--save-every` checkpoint write (both canonical and history if
`--save-history`):

1. Write `.tinygpt` as today
2. Atomically write the sibling `.tinygpt.opt`
3. Use `.tmp` suffix + rename for atomicity (don't leave partial files)

### 5. New flag (optional escape hatch)

`--no-save-opt-state` — for users who explicitly want smaller checkpoints
and don't care about clean resume. Default: opt-state is saved.

## Scope — out (v2 of this PRD)

- AdamW variants beyond what's currently supported (Lion, Sophia, Muon)
  — those use different state shapes; extend later if those optimizers
  get shipped (currently they're "explored, not shipped" per PLAN.md)
- Cross-precision resume (fp32 train state from bf16 opt state, etc.)
  — match precision, fail loudly on mismatch
- Adam state validation beyond shape-check (we don't verify the moments
  are "sensible" — trust the file)
- Sharded opt state for distributed training — TinyGPT is single-Mac
  for now; no need

## Acceptance criteria

### Functional

1. **Save smoke**: train a small run (`--preset tiny --steps 200
   --save-every 50 --save-history`). Confirm:
   - `/tmp/<run>.tinygpt` exists
   - `/tmp/<run>.tinygpt.opt` exists
   - `/tmp/<run>.step-50.tinygpt.opt`, `.step-100.tinygpt.opt`, etc. exist
2. **Resume smoke**: pause the run at step 100. Resume via
   `--resume /tmp/<run>.tinygpt`. Confirm:
   - Training continues from step 100
   - **Loss at step 105 is within 5% of loss at step 100** (proves Adam
     state is restored; without it, you'd see a clear wobble)
3. **Fallback smoke**: delete the `.opt` file. Resume. Confirm:
   - Resume works (model weights still load)
   - Warning logged about fresh Adam state
   - Larger wobble visible (this is the current behavior, just gracefully
     fallen-back-to)

### Determinism (bonus)

4. With `--seed 42`, pause at step 100, resume. Compare loss curve from
   steps 100-200 between (a) uninterrupted run and (b) paused-and-resumed
   run. Should match within ~1% (gradient sampling has slight non-determinism;
   true bit-exact is a separate issue).

### Performance

5. Save overhead: adding opt-state should add <500ms per checkpoint
   write on M5 Pro for Huge preset.
6. File size: `.opt` should be roughly 2× the `.tinygpt` size (m + v
   per param). Verify and document.

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPTModel/Trainer.swift` — save/resume opt state |
| **modify** | `native-mac/Sources/TinyGPTModel/Optimizers.swift` — accessor for `m`/`v`/`step` of AdamW |
| **create** | `native-mac/Sources/TinyGPTIO/OptStateIO.swift` — serialize/deserialize opt state safetensors |
| **modify** | `native-mac/Sources/TinyGPT/Train.swift` — wire `--no-save-opt-state` flag |
| **modify** | Existing tests in `Tests/TinyGPTModelTests/` — add round-trip test for opt state |
| **don't touch** | Model weight format, eval pipeline, serve, app code, `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Inputs the agent has

| Resource | Location |
|---|---|
| Current AdamW impl | `native-mac/Sources/TinyGPTModel/Optimizers.swift` |
| Checkpoint save/load | `native-mac/Sources/TinyGPTModel/Trainer.swift` (look for `saveCheckpoint`) |
| Safetensors writer (already shipped) | `native-mac/Sources/TinyGPTIO/Safetensors.swift` (or wherever the writer lives — grep for it) |
| MLX-Swift Optimizer API | https://github.com/ml-explore/mlx-swift docs; check `Sources/MLXOptimizers/AdamW.swift` for moment accessor patterns |
| Resume path today | `Train.swift` `--resume` handler |

## Estimated effort

**~3-5 days focused work.**

- 1 day: audit MLX-Swift AdamW public API; figure out moment access
  (this is the highest-risk step — may need subclass / upstream contribution)
- 1 day: write `OptStateIO.swift` (serialize/deserialize safetensors)
- 1 day: wire save path in Trainer (atomic writes, history support)
- 1 day: wire resume path (load + validate + inject + fallback)
- 1 day: tests + smoke + PR

The risk concentration: step 1. If MLX-Swift hides the moments behind
private API, this becomes a 1-2 week project (upstream PR + wait + adopt).

## Coordination

PR description must include:
1. Smoke output: loss-curve comparison of uninterrupted vs paused-resumed
   runs (the proof Adam state actually restored cleanly)
2. File size breakdown (`.tinygpt` vs `.opt`)
3. Save-overhead timing on a real run
4. Build + existing tests passing

Maintainer marks B12 v2 as shipped in PLAN.md.

## Known risks

- **MLX-Swift may not expose AdamW moments via public API.** Highest-risk
  unknown. Mitigation: if public API doesn't allow it, either subclass
  AdamW with our own moment-exposing variant OR file an upstream PR.
  Workaround in worst case: maintain our own AdamW variant in
  `TinyGPTModel/AdamWWithExposedState.swift`.
- **Checkpoint size doubles.** Opt-state is 2× weights (m + v).
  Documented; the `--no-save-opt-state` flag is the escape hatch.
- **Resume across MLX-Swift version bumps** — if MLX changes its
  optimizer internals between save and resume, the opt-state file may
  not load cleanly. Mitigation: embed MLX version in the .opt file
  metadata; fall back with warning on mismatch.
- **Adam moments at bf16 may lose precision** vs fp32 moments. For most
  training this is fine; users training in fp32 should get fp32 opt state.
  Match the model's dtype.

## Source links

- B12 v1 (loss-spike detector — observe only, deferred auto-rollback):
  PLAN.md §3
- HANDOFF.md notes Adam state persistence as a known gap; this PRD
  closes it
- Companion PRD: `docs/prds/app-train-controls-thermal.md` (pause/resume
  UI — precondition for caring about Adam state)
- MLX-Swift: https://github.com/ml-explore/mlx-swift
