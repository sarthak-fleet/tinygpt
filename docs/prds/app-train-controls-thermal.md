---
name: App Train tab — pause/resume controls + thermal-safety warning
status: shipped-2026-06-06
owner: unassigned (parallel-agent task — Mac app UX)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (platform polish)
---

# PRD — Train tab pause/resume + thermal-safety UX

## Goal

Two related additions to the Mac app's Train tab:

1. **Pause/Resume controls** — let users interrupt a long training run
   without losing progress, then continue from the last checkpoint
2. **Thermal-safety warning** — surface a clear, persistent warning that
   sustained training is hard on Macs, with concrete best-practice
   guidance (hard surface, laptop stand, clamshell mode)

Both are pure Mac-app UX additions on top of CLI capabilities that
already exist. No new training code required.

## Why now

- Long training runs (multi-hour to multi-day) are the platform's core
  use case. Lacking pause/resume forces users to commit a multi-hour
  block; lots of friction for users who want to free up the GPU for
  other tasks.
- The maintainer's own MacBook fan failed earlier this year from running
  training on a bed (vents blocked). Real user safety issue — users will
  damage hardware if they don't know this. Better to warn proactively
  than apologize after.
- All underlying CLI capabilities exist (`--save-history`, `--resume`,
  process-management in the Server tab). This is pure GUI integration.

## Scope — in

### Pause/Resume controls

| Component | Behavior |
|---|---|
| **"Pause" button** | When training is running: send `SIGTERM` to the training PID. Wait for graceful exit (the trainer's existing checkpoint-on-shutdown handler writes a final checkpoint). UI transitions to "Paused at step N" state. |
| **"Resume" button** | When paused: re-spawn `tinygpt train` as a subprocess with `--resume <last-history-checkpoint>` and all other flags from the previous run. UI transitions to "Running at step N+1". |
| **State display** | Always shows one of: "Idle" / "Running — step N / total" / "Paused at step N" / "Crashed at step N (last log: ...)" |
| **State persistence** | Persist the "paused but want to resume" state in `UserDefaults` so closing and reopening the app doesn't lose the training config. On app launch, if state == "paused", show a "Resume previous run?" prompt. |
| **Progress bar** | Pull from the training's `--log-jsonl` stream (already-shipped feature). Show step / total + ETA + current loss. Updates every ~5s. |

### Throttle slider (continuous "slow down" control between pause and full-tilt)

Pause is 0% load; default training is 100%. Users may want intermediate
load to reduce heat / fan noise / shared-machine impact without stopping.

| Component | Behavior |
|---|---|
| **Backend: `--throttle <0.0-1.0>` flag** | Add to `tinygpt train` CLI. After each gradient step, sleep for `step_wall_time × (1/throttle - 1)`. e.g. throttle=0.5 → run a step, sleep for the same wall time → ~50% sustained GPU load. Default 1.0 = no throttle. |
| **Backend: `--max-step-rate N` flag** | Alternative absolute cap: never exceed N steps/sec. Pick whichever is more limiting between `--throttle` and `--max-step-rate`. |
| **UI: throttle slider in Train tab** | 4-step slider with labels: `100% (full speed)` / `50% (warm)` / `25% (cool)` / `paused`. Maps to `--throttle 1.0` / `0.5` / `0.25` / pause. |
| **UI: ETA recalculation** | When throttle changes, recalculate ETA based on effective step rate and update the progress bar |
| **Live adjustment** | Throttle slider works WHILE training is running — sends the new value to the training process via a control file (`~/.cache/tinygpt/<run>.throttle`) that the train loop polls every N steps. No restart needed. |

This gives users a continuous control surface — pause / 25% / 50% / 100% — instead of just on/off. Pairs with the auto-pause-on-critical thermal logic above: if `.serious` thermal state is reached, drop throttle to 0.5 automatically; if `.critical`, auto-pause.

### Thermal-safety warning

| Component | Behavior |
|---|---|
| **One-time onboarding banner** | First time user opens the Train tab, show a dismissible info banner with the thermal best-practices list (see below). Dismiss persists in `UserDefaults`. |
| **Persistent thermal indicator** | A small always-visible chip in the Train tab footer: "💨 Training is intensive — ensure good airflow [info]". Tapping/hovering shows the full guidance popover. |
| **Pre-launch confirmation** for runs >30 min | If the user starts a training run with `--steps N` that's estimated > 30 min wall, show a single-checkbox confirm: "I understand this will run the GPU at full load for ~X hours. [✓] I'm on a hard surface with good airflow." |
| **Thermal-state badge** | Use **`ProcessInfo.thermalState`** (Apple-blessed, no sudo, no entitlement). Subscribe to `NSProcessInfoThermalStateDidChange` for instant updates. Map states → color: `.nominal` green, `.fair` yellow, `.serious` orange, `.critical` red. Always visible in Train-tab status. |
| **Auto-throttle on thermal** (PRIMARY thermal response) | When thermal state changes, auto-adjust the throttle slider: `.nominal` → 100%, `.fair` → 75%, `.serious` → 50%, `.critical` → 25%. Throttling has **zero effect on training results** — identical model trajectory, just slower wall-clock. So this is safe to do aggressively and silently (with a small UI indicator: "auto-throttled to 50% due to heat"). User can override the auto-throttle decision via the slider. |
| **Auto-pause on sustained critical** (last-resort safety net) | If `.critical` persists for >60 seconds even at 25% throttle, auto-pause. Show clear banner "🛑 Auto-paused — your Mac is still too hot at minimum throttle." User triggers resume manually. |

### Thermal best-practices content (copy for banner + popover)

```
Training your model puts sustained 100% load on the GPU + CPU for
the duration of the run. To avoid damaging your Mac:

✅ Use a hard, flat surface — NOT a bed, couch, or lap.
   The bottom vents need clear airflow.

✅ Better: laptop stand or riser ($20) — air flows under
   and around the chassis.

✅ Best for long runs: clamshell mode with external monitor.
   The closed-lid chassis is the most efficient heat sink Apple
   has designed.

⚠️ Don't cover the keyboard during training — it's a heat sink too.

⚠️ Cooler ambient room temperature = more thermal headroom.

⚠️ Multi-day training? Consider a Mac mini or Mac Studio.
   Desktop cooling is fundamentally better than laptop cooling.

📊 Monitor temps with Stats (free menu-bar app) or
   `sudo powermetrics --samplers smc -i 1000 -n 1`.
```

## Scope — out (v2)

- Persisting Adam optimizer state across pause/resume (so resume has
  zero warm-up wobble). This is a separate, larger work item — B12 v2
  in HANDOFF. For v1, accept the ~50-100 step momentum loss on resume.
- Multi-job queue ("queue 5 training runs, run them in sequence"). Out
  of scope for v1.
- Resume-from-arbitrary-checkpoint UI (pick any `.step-N.tinygpt` from
  a list). v1 just resumes from latest.
- True throttle-aware "pause if too hot" auto-pause. Useful but requires
  sudo for temp access — defer.

## Acceptance criteria

1. **Pause smoke**: start a small training (`--preset tiny --steps 500`),
   wait until ~step 200, hit Pause. Confirm:
   - Process exits gracefully within 30s
   - `/tmp/<run>.tinygpt` updated with step-200 state
   - UI shows "Paused at step 200"
2. **Resume smoke**: from the paused state, hit Resume. Confirm:
   - New `tinygpt train --resume ...` process spawns
   - UI transitions to "Running"
   - Training continues from step 200 (visible in log)
3. **App-restart persistence**: pause a run, quit the app, reopen.
   Confirm a "Resume previous run?" prompt appears with the right
   checkpoint path.
4. **Thermal banner**: fresh app install, open Train tab. Confirm
   banner appears with the full guidance. Confirm dismiss persists.
5. **Pre-launch confirm**: start a >30-min run. Confirm checkbox
   appears, blocks launch until checked.
6. **Build passes; existing Train-tab functionality unaffected.**

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPTApp/ContentView.swift` (Train tab section) |
| **modify** | The trainer controller that spawns the subprocess (find via grep for the existing "Start Training" button) |
| **create** | `native-mac/Sources/TinyGPTApp/ThermalSafetyView.swift` — the reusable banner/popover view |
| **create** | `native-mac/Sources/TinyGPTApp/TrainingProgressParser.swift` — small JSONL stream parser for the progress bar |
| **don't touch** | `Train.swift` (the actual training loop), `Serve.swift`, `Sample.swift`, `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Inputs the agent has

| Resource | Location |
|---|---|
| Existing process-spawn pattern | `ContentView.swift` Server tab — uses `Process()` to spawn `tinygpt serve`, manages PID, sends signals |
| Training subcommand surface | `tinygpt train --help` — shows `--resume`, `--save-history`, `--log-jsonl` already exist |
| Trainer's graceful-shutdown handler | `Train.swift` already handles SIGTERM and writes a final checkpoint — verify by grepping for `SIGTERM` handler / signal handling |
| JSONL log shape | `Sources/TinyGPT/TrainLog.swift` defines the schema — read this for parser fields |
| `UserDefaults` persistence pattern | Used elsewhere in `ContentView.swift` for sampler settings; copy the pattern |

## Estimated effort

**~half-day to 1 day focused work.**

- ~1 hr: pause button + SIGTERM + state display
- ~1 hr: resume button + subprocess re-spawn with `--resume`
- ~1 hr: state persistence across app restarts
- ~1-2 hr: progress bar JSONL parser + UI update
- ~1 hr: thermal banner + popover + dismiss persistence
- ~1 hr: pre-launch confirmation modal
- ~1 hr: testing + polish

## Coordination

PR description must include:
1. Screen recording or screenshots of: pause → paused state → resume → running
2. Screenshot of the thermal warning banner (default + dismissed states)
3. Screenshot of the pre-launch confirm for a long run
4. Confirmation that app-restart resume prompt works
5. Build passes

Maintainer merges; updates HANDOFF.md with the new Train-tab capability.

## Known risks

- **SIGTERM timing**: if the trainer is mid-batch when SIGTERM arrives,
  the graceful exit may take 10-30s. UI should show "Pausing..." with a
  spinner; don't let user spam-click Pause.
- **Resume Adam-state wobble**: documented in the popup. Don't oversell
  "pause is free" — there's a small (~50-100 step) loss reseat.
- **Multiple training runs at once**: if user starts a new training while
  one is paused, the "Resume previous run?" state gets confusing. v1
  handle: simple — only allow one training session at a time; gray out
  "Start" if a paused session exists.
- **Thermal warning fatigue**: don't over-show warnings. Once per onboarding
  + once per long run. Footer chip is the always-on reminder.

## Source links

- Existing Server tab process management:
  `native-mac/Sources/TinyGPTApp/ContentView.swift`
- Training CLI flags:
  `native-mac/Sources/TinyGPT/Train.swift` (`--resume`, `--save-history`, `--log-jsonl`)
- JSONL log schema:
  `native-mac/Sources/TinyGPT/TrainLog.swift`
- macOS-specific signal handling: standard `Foundation` patterns; see
  `Process.terminate()` and `Process.interrupt()`
