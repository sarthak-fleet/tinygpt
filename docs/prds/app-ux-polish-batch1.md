---
name: App UX polish batch 1 — sidebar, Train-tab discovery, throttle row
status: shipped-2026-06-07
owner: unassigned (parallel-agent task — Mac app UX, no new backend)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (laptop-thermal-aware-defaults — extending to UI polish from in-app observation)
---

# PRD — App UX polish batch 1

## 2026-06-07 ship note

Implemented in the current app tree:

- collapsible Gallery sidebar state via `@AppStorage`
- Train tab existing-run detection through `RunLockFile` and controller attach
- active-run lock file written/cleared by `tinygpt train`
- throttle row layout and larger chip controls
- seed field uses a placeholder instead of literal `rand`

## Goal

Four targeted Mac app UX fixes spotted during the first real in-app
session 2026-06-06 EOD. Pure UI work, no backend changes. Closes the
gap between "shipped" and "actually feels like a finished app."

## Why now

User opened the app for the first real walkthrough today. Surfaced four
visible issues that make the app feel either cluttered or disconnected
from the user's actual training state. These are small individually,
collectively damaging to the "this is a polished platform" pitch.

## Scope — in

### Issue 1: Gallery shouldn't always be visible in the sidebar

**Current**: gallery items (corpora + downloaded models — Shakespeare,
Stories, code, chat, HF-downloaded models) always show in the left
sidebar, regardless of which tab is active. Visually noisy. Takes
sidebar space away from tab-relevant content.

**Fix**: collapse Gallery into a single sidebar item (`▸ Gallery`) that
expands on click. When collapsed, only the row name + count badge shows.
Default state: collapsed. Persists in `UserDefaults`.

**Acceptance**: gallery row is a single line when collapsed; click
expands; persists across app restarts.

### Issue 2: Train tab doesn't surface existing training runs

**Current**: Train tab shows "Press Start to train. Loss will appear
here." even when a training run is ALREADY in progress on the machine
(e.g., user fired `tinygpt train` from the CLI, or a previous app
session started one). User can't see real-time progress of their actual
training without staring at a terminal.

**Fix**: on Train tab open, detect existing training runs and surface
them:

1. **Detection strategy** (do all of these):
   - Read `~/.cache/tinygpt/runs/active.json` if it exists — a lock
     file the train loop writes (PID + log JSONL path + canonical
     output path + start time). Train CLI should write/clear this
     file in `Train.swift` `startUp` / `shutDown`.
   - Run `pgrep -f "tinygpt train"` to detect orphan training runs
     not tracked by the lock file (e.g., CLI-spawned with nohup).
   - Scan `/tmp/*.tinygpt.jsonl` files modified in the last 24h with
     no recent termination — likely an active run.
2. **UI surface**:
   - If a run is detected: replace the empty-state ("Press Start...")
     with a live status panel: PID, step / total, val loss, eta, last
     log line, pause/resume controls that target the detected PID.
   - Parse the JSONL log to plot the existing loss curve.
   - Update controls to show "Attached to existing run" instead of
     "Start training".
3. **Multiple runs detected**: show a picker — "2 training runs detected,
   which to attach?"

**Acceptance**:
- Detection works for runs started via app's Start button (lock file
  written) AND CLI-spawned runs (pgrep)
- Loss curve renders from the existing JSONL within ~2 seconds of tab
  open
- Pause/resume/throttle controls on the existing-run UI actually
  pause/resume/throttle that run

### Issue 3: Throttle row is cluttered and labels truncate

**Current** (visible in 2026-06-06 screenshot):
- `throttle` label truncates to `throttl` (next button "100%" is
  touching it)
- 100% / 75% / 50% / 25% / auto buttons are crammed together with no
  visual separators
- `auto` checkbox touches the last percentage button
- Whole row looks like a wall of dense text

**Fix**:
- Stack the label above the buttons (label is now `Throttle:` on its
  own line)
- Add `Spacer()` between the percentage buttons + `auto` checkbox
- Increase horizontal padding around each button
- Use SF Symbols icons next to percentage buttons (optional: snowflake
  for 25%, etc.) for quick scan

**Acceptance**: row reads cleanly at default app width; no truncation;
percentages and auto have clear visual separation.

### Issue 4: Seed field shows literal "rand" instead of placeholder

**Current**: seed text field shows `rand` as visible text (looks like
literal user input).

**Fix**: use a SwiftUI `TextField` placeholder (`.placeholder("random
— auto-pick at runtime")`) or italicize / grey-out the text to show it's
default, not user-typed.

**Acceptance**: when seed field is empty, placeholder is visually
distinct from typed text.

### Issue 5: Click targets are too small across the Train tab

**Current**: percentage buttons (100% / 75% / 50% / 25%), checkbox
toggles, dropdown chevrons, and inline text labels all have tight click
hitboxes. Users have to be precise to land clicks. Visible in the
2026-06-06 screenshot — the throttle row controls look ~16-18pt tall,
below comfortable macOS click-target sizes.

**Fix**: enforce minimum click hitboxes across Train-tab controls:
- macOS HIG / Apple usability guidance: interactive controls should
  have at least ~22pt × 22pt hit area for compact controls,
  ~44pt × 44pt for primary buttons
- Apply `.contentShape(Rectangle())` + minimum frame size to all
  Train-tab buttons / toggles / chips
- Add 8-12pt internal padding so visual button + invisible hit area
  expand together
- Toggle / checkbox: increase tappable region beyond just the checkbox
  square — wrap the entire label+checkbox row in a `.onTapGesture`
- Dropdown chevrons: ensure the whole dropdown area (not just the
  small arrow) is clickable

**Acceptance**:
- Every interactive Train-tab control has a hit area ≥ 22pt × 22pt
- Primary buttons (Start, Pause, Resume) ≥ 44pt height
- Manual test: click anywhere visually within a control's bounding box
  → registers the click; no "dead zone" inside the visible button
- Apply same audit to Sample, Fine-tune, Interp, Server, Eval tabs (sweep)

## Scope — out (later)

- Reflowing the entire Train-tab layout — these are point fixes, not a
  redesign
- Customizable sidebar (drag-reorder, hide categories) — v2
- Multi-run attach with simultaneous monitoring — v1 picks one
- Detecting non-tinygpt training (e.g., Python `accelerate` runs) — out
  of scope

## Acceptance criteria (rollup)

1. Gallery collapses to single sidebar row by default
2. Train tab detects + attaches to existing runs (both app-spawned and
   CLI-spawned via pgrep)
3. Loss curve renders from existing JSONL
4. Throttle row reads cleanly; no truncation; clear separation
5. Seed field shows distinct placeholder
6. Build passes; other tabs unaffected
7. Screenshots in PR show before/after for each issue

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPTApp/ContentView.swift` — Gallery sidebar collapse |
| **modify** | `native-mac/Sources/TinyGPTApp/TrainView.swift` — throttle-row layout + seed-field placeholder + existing-run UI |
| **modify** | `native-mac/Sources/TinyGPTApp/TrainController.swift` — `detectExistingRun()` method + JSONL-log parser for the existing-run case |
| **modify** | `native-mac/Sources/TinyGPT/Train.swift` — write/clear `~/.cache/tinygpt/runs/active.json` lock file on startup/shutdown |
| **create** (or modify) | `native-mac/Sources/TinyGPTIO/RunLockFile.swift` — small typed helper for the lock-file schema |
| **don't touch** | Serve.swift, eval pipeline, eval tab, `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Inputs the agent has

| Resource | Location |
|---|---|
| Current TrainView | `native-mac/Sources/TinyGPTApp/TrainView.swift` (the throttle row is around line 112) |
| Current ContentView (sidebar) | `native-mac/Sources/TinyGPTApp/ContentView.swift` |
| JSONL log schema | `native-mac/Sources/TinyGPT/TrainLog.swift` |
| Active training right now (for testing detection) | PID 37912 (SIGSTOPPED N02 at step 80,400 — perfect test case for "detect orphan CLI-spawned run") |
| Existing log file | `/tmp/n02-resume-*.log` + `/tmp/huge-base-v1.jsonl` |

## Estimated effort

**~2-3 days focused work.**

- 0.5 day: Issue 1 (Gallery collapse) — pure SwiftUI
- 1-1.5 days: Issue 2 (existing-run detection + attach + loss-curve
  render) — most of the work
- 0.5 day: Issue 3 (throttle row layout)
- 0.25 day: Issue 4 (seed placeholder)
- 0.5 day: testing, screenshots, PR

## Coordination

PR description must include:
1. Before/after screenshots of each issue (Gallery collapse / Train
   tab with-and-without existing run / throttle row / seed field)
2. Demo of attaching to the live N02 process (PID 37912 if still SIGSTOPPED,
   or a fresh test run)
3. Build passes

## Known risks

- **Lock file race conditions** — if app crashes mid-run, lock file
  may be stale. Mitigation: include `started_at` timestamp; consider
  stale if PID not running OR timestamp > 7 days.
- **pgrep detection has false positives** — could pick up `tinygpt
  train --help` or similar. Match on a more specific signature like
  `tinygpt train.*--steps`.
- **Multiple runs** — for v1, pick the most-recently-started; show a
  picker if user wants to switch.

## Why these specific four

Each one is small, but each one breaks the "polished product" claim
in a noticeable way. A user opening the app for the first time sees:
- Cluttered sidebar (Issue 1) → "this feels noisy"
- Empty Train tab while they actually have training running (Issue 2)
  → "the app doesn't know what I'm doing"
- Cramped throttle row (Issue 3) → "the UI was rushed"
- Literal "rand" in a field (Issue 4) → "this looks like a bug"

Fixing all four in one PR ships the next batch of polish cohesively.
