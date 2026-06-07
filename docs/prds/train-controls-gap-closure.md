---
name: Train-controls gap-closure (close 3 partial-ship items from 2026-06-06 audit)
status: shipped-2026-06-06
owner: unassigned (parallel-agent task — small, focused follow-ups)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (today's PRD ship-list audit found these gaps)
related_prds: app-train-controls-thermal.md, adam-state-persistence.md, cleanup-extractordata-datatrove.md
---

# PRD — close 3 gaps from today's audit

## Goal

The 2026-06-06 EOD audit of four "shipped" PRDs found three real gaps
between PRD spec and shipped code. This PRD closes those three gaps in
one focused pass. Each is small. Together they take the platform from
"mostly shipped" to "spec-complete" on today's batch.

## Why now

The four PRDs the elf shipped today are mostly real but have edge gaps:

| PRD | What shipped | What didn't |
|---|---|---|
| `app-train-controls-thermal` | Pause/resume UI, `ProcessInfo.thermalState`, `PowerMonitor` auto-pause | `--throttle` CLI flag; `--max-step-rate` flag; throttle slider in app |
| `adam-state-persistence` | `--no-save-opt-state` flag, `.tinygpt.opt` save/load logic inline | `OptStateIO.swift` as separate file (cosmetic — logic shipped) + verification smoke test |
| `cleanup-extractordata-datatrove` | `scripts/data-prep/`, `prep_data.py`, `prep-data` subcommand wired | `GitHubCorpus.swift` not yet deleted (waiting for parity testing) |

The PowerMonitor auto-pause is arguably *better* than the throttle slider
for thermal management — but the throttle slider remains real product
value for "pre-emptive load reduction" (user wants to use their Mac for
other things while training stays gentle). Closing the gap gives users
both knobs.

## Scope — in

### Gap 1: `--throttle <0.0-1.0>` CLI flag (+ Train-tab slider)

**Backend (Train.swift):**
- Add `--throttle <Float>` flag, default `1.0` (no throttle)
- Add `--max-step-rate <Int>` flag, default `0` (no cap)
- After each completed step, compute desired sleep:
  ```
  step_time = now() - step_start
  throttle_sleep = step_time * (1.0/throttle - 1.0)  // if throttle < 1.0
  rate_sleep = max(0, (1.0/max_step_rate) - step_time)  // if max_step_rate > 0
  sleep_duration = max(throttle_sleep, rate_sleep)
  Thread.sleep(sleep_duration)
  ```
- Log effective load level (e.g., `[throttle] effective rate: 3.5 steps/sec (50%)`) every 100 steps

**App UI (TrainView.swift):**
- Add a 4-step segmented control: `100% / 75% / 50% / 25%` (maps to `--throttle` values)
- Default to **75%** (per the laptop-thermal-aware-defaults principle in the strategy doc)
- Slider works WHILE training is running — writes value to a control file (`~/.cache/tinygpt/<run>.throttle`) that train loop polls every 100 steps and applies live

**Auto-throttle binding to PowerMonitor:**
- When `ProcessInfo.thermalState` changes:
  - `.nominal` → 100% (no auto-throttle, respect user setting)
  - `.fair` → 75% (suggest, don't force)
  - `.serious` → 50% (auto-throttle on)
  - `.critical` → 25% (auto-throttle aggressive)
- UI shows "auto-throttled to X% due to heat" with override checkbox

### Gap 2: Adam-state persistence end-to-end verification

**Add a smoke test** that proves the persistence works:
- `Tests/TinyGPTModelTests/AdamPersistenceTests.swift`
- Train tiny preset for 200 steps with `--save-every 50 --save-history`
- Confirm `.tinygpt.opt` files exist for every history checkpoint
- Restart at step 100 via `--resume`
- Compare loss at step 105 in (a) uninterrupted run vs (b) resumed-from-100 run
- Assert: difference < 5% (proves Adam state restored correctly)
- Document the test in PR

This is small but critical — without verification, we don't know if Adam
state restore actually works clean. The "small loss wobble" symptom was
the original motivation for the whole PRD.

### Gap 3: Delete `GitHubCorpus.swift` after parity testing

**Verify parity** between Swift impl and Python shim, then delete:
1. Pick 3 test repos (one small, one medium, one large)
2. Run BOTH `tinygpt fetch-github` (Swift, legacy) and the new Python
   shim (`python scripts/data-prep/prep_data.py --github ...`) against each
3. Compare output JSONL: row count within 5%, schema identical, sample
   rows look equivalent
4. If parity holds: delete `native-mac/Sources/TinyGPTData/GitHubCorpus.swift`,
   update `tinygpt fetch-github` dispatch to route directly to the Python
   shim with a "[migrated]" notice
5. If parity fails: document the divergences; do NOT delete; leave both

## Scope — out (later)

- A full continuous slider (0-100% in 1% increments) — 4-step is enough
  for v1. Continuous slider is v2.
- Per-task throttle scheduling (e.g., 100% during work hours, 25%
  overnight) — out of scope; just expose the control
- Renaming the `OptStateIO.swift` location (the elf integrated it inline;
  fine; not worth refactoring)

## Acceptance criteria

### Gap 1 (throttle)

1. `tinygpt train --throttle 0.5 --steps 100 --preset tiny ...` runs and
   completes ~2× slower than baseline (verify with wall-time)
2. `tinygpt train --help` lists `--throttle` and `--max-step-rate`
3. App Train tab has 4-step segmented control; default position is 75%
4. Live adjustment: change slider mid-run, observe effective step rate
   change within ~10 seconds (the polling window)
5. Auto-throttle: simulate thermal state change (or use a debug flag);
   slider value reflects auto-decision, "auto" indicator shows

### Gap 2 (Adam verification)

1. New test in `Tests/TinyGPTModelTests/AdamPersistenceTests.swift`
2. Test passes locally and in Xcode
3. PR includes loss-curve comparison plot proving Adam state restore works

### Gap 3 (GitHubCorpus cleanup)

1. Parity test output included in PR
2. Either GitHubCorpus.swift deleted (if parity holds) OR documented
   reason for keeping it
3. Dispatch updated accordingly; build passes

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPT/Train.swift` — add `--throttle` + `--max-step-rate` flags + sleep logic + control-file polling |
| **modify** | `native-mac/Sources/TinyGPTApp/TrainView.swift` — 4-step throttle segmented control + auto-throttle indicator |
| **modify** | `native-mac/Sources/TinyGPTApp/TrainController.swift` — control-file writer; auto-throttle binding to PowerMonitor / thermalState |
| **create** | `native-mac/Tests/TinyGPTModelTests/AdamPersistenceTests.swift` — round-trip test |
| **modify** (or delete) | `native-mac/Sources/TinyGPTData/GitHubCorpus.swift` |
| **modify** | `native-mac/Sources/TinyGPT/TinyGPT.swift` — dispatch route update if GitHubCorpus deleted |
| **don't touch** | EvalCompare.swift, RunLmEval.swift, Serve.swift, the eval CLI surface, `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Inputs the agent has

| Resource | Location |
|---|---|
| Existing throttle PRD spec | `docs/prds/app-train-controls-thermal.md` (throttle-slider section) |
| Existing PowerMonitor integration | grep `PowerMonitor` in `native-mac/Sources/TinyGPT/Train.swift` and `TrainSupport.swift` |
| ProcessInfo.thermalState usage example | already in TrainController.swift / TrainView.swift (elf shipped) |
| Train-loop polling pattern (control file) | Existing precedent: `TrainSupport.stopRequested.set()` — similar polled state |
| Adam test data | `/tmp/huge-base-v1.tinygpt.opt` should exist; can verify |
| Python shim location | `scripts/data-prep/prep_data.py` (just shipped) |

## Estimated effort

**~2-3 days focused work.**

- Gap 1 (throttle CLI + UI + live adjustment + auto-binding): ~1.5 days
- Gap 2 (Adam test + verification): ~0.5 day
- Gap 3 (parity test + decision + cleanup): ~0.5 day

## Coordination

PR description must include:
1. Wall-time verification of throttle (baseline vs `--throttle 0.5`)
2. Screenshot of Train-tab slider in 4 positions
3. Adam persistence test output (loss curve comparison)
4. GitHubCorpus parity table or kept-because-X note
5. Build + existing tests passing

Maintainer marks all 3 source PRDs as "fully shipped" in their status fields.

## Known risks

- **Live throttle adjustment via control file** could have read-while-write
  race conditions. Mitigation: atomic write + polling with short retry.
  Worst case: throttle reverts to previous value for one polling cycle.
- **PowerMonitor auto-throttle conflict with user manual setting** —
  what wins? Design choice: user setting is a *cap* (auto can lower it
  but not raise it above user's choice). Document this clearly in UI.
- **GitHubCorpus parity may show real divergences** (different default
  filtering, different field order in JSONL). Document; don't force-delete
  if parity fails.

## Why now (re-prioritization context)

The user's "results first; architecturally interesting later" principle
(strategy doc 2026-06-06) means we ship spec-complete tools before
exploring new architectures. These three gaps are spec-completion of
today's batch — they unblock the "frequent pausing without state loss"
UX and the "explicit user thermal control" feature both.

Should ship before any new Tier 2 modality work begins.
