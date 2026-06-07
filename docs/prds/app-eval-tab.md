---
name: App Eval tab — bring the eval pipeline into the GUI
status: shipped-2026-06-06
owner: unassigned (parallel-agent task — Mac app UX, no new backend)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (results-first prioritization)
---

# PRD — Mac app Eval tab

## Goal

Add an Eval tab to the Mac app that exposes the existing eval CLI surface
(`run-lm-eval`, `eval-bfcl`, `eval-tau-bench`, `eval-humaneval`, `eval-compare`,
`judge`, `run-bench`) as a guided GUI flow. **No new backend.** Pure UI
on shipped capabilities.

This closes a major product gap: today, scoring a specialist requires
dropping into CLI. The Eval tab makes scoring as first-class as training.

## Why now

- The full eval stack (E0-E8) shipped 2026-06-05 + the elf's batch
  yesterday. CLI surface is complete and battle-tested (multiple runs
  through it today already).
- The browser viewer `/eval-leaderboard.astro` exists but requires
  manual JSONL drop. App users shouldn't have to leave the app.
- The user's product framing is "Mac app for individuals to build and
  upgrade specialists." Scoring is half of that loop; without an app
  UI for it, the platform feels incomplete.
- High-ROI pure-UX work — no new ML code, no new backend, no new
  research. Just compose existing CLI subcommands behind a clean wizard.

## Scope — in

### 1. Eval tab in app sidebar

Sits as a peer to Sample / Train / Fine-tune / Interp / Server.

### 2. Three modes (radio-button or segmented control at top)

| Mode | What | Backend used |
|---|---|---|
| **Quick score** | Pick model + pick task → score it | `run-lm-eval --tinygpt-model X --tasks Y` |
| **Cross-checkpoint emergence** | Pick a run-stem with history → sweep all checkpoints + baseline | `scripts/score-run.sh` pattern |
| **Custom eval** | Drag-drop JSONL of (prompt, expected) → score against model | currently CLI-only; v1 wraps existing `run-lm-eval` with custom-task generation |

### 3. Quick-score mode UX

- **Model picker**: dropdown of `.tinygpt` files from canonical gallery paths + HF dirs that have been downloaded + "Browse..." button
- **Task picker**: multi-select chips for common tasks (arc_easy, hellaswag, piqa, gsm8k, mmlu, truthfulqa, bfcl, humaneval, mbpp); "more..." dropdown for full lm-eval task list
- **Limit slider**: 10 → 50 → 100 → 500 → full (defaults to 30)
- **Add baseline checkbox**: toggle to auto-score SmolLM2-135M alongside (default: on)
- **"Run" button**: spawns `tinygpt run-lm-eval` subprocess, streams stdout to a log pane
- **Results table**: shows JSONL output rendered as `eval-compare --by model` table, live-updating as rows land

### 4. Cross-checkpoint emergence mode UX

- **Run picker**: dropdown of `.tinygpt` files with `.step-*` siblings (auto-detects history)
- **Task + limit + baseline** controls (same as Quick score)
- **"Sweep" button**: spawns the `score-run.sh` pattern internally — iterates through history checkpoints + canonical + baseline
- **Live chart**: line chart of score-vs-step as rows land, X = step, Y = score, separate line per task
- **Compare table**: 3-view tabs (by step / by model / by task) showing the same data the CLI `eval-compare` produces

### 5. Custom eval mode UX

- **Model picker** (same)
- **Drag-drop zone for JSONL**: `{"prompt": "...", "expected": "..."}` rows
- **Scoring metric picker**: exact-match / contains / regex / LLM-as-judge
- **If judge selected**: pick judge model from same dropdown
- **"Score" button**: subprocess that wraps `run-lm-eval` with a generated lm-eval-harness custom task YAML, OR wraps `judge` if the metric is LLM-as-judge
- **Per-row inspector**: clickable table showing input + model output + expected + pass/fail

### 6. Cross-mode features

- **Live process log** pane (collapsible, like Server tab's log) showing subprocess output
- **Cancel button**: SIGTERM the eval subprocess
- **Results JSONL path** displayed + "Reveal in Finder" + "Open in eval-leaderboard browser viewer" buttons
- **History panel**: persist last N evals in UserDefaults, click to re-load results table

## Scope — out (v2)

- Auto-fire evals on save-history checkpoints — that's E8 / train-time hook
  (already shipped via Train tab `--eval-every`)
- Multi-model concurrent scoring (queue + parallelize) — v1 is one model
  at a time
- Eval comparisons across runs from different sessions — v1 shows current
  results; cross-session needs the browser leaderboard
- Per-task fine-grained settings (few-shot count, prompt template) — use
  lm-eval defaults for v1
- New backend: this is pure UI work. If a feature requires new CLI
  surface, it's out of scope.

## Acceptance criteria

1. **Eval tab appears** as 6th tab in the app sidebar.
2. **Quick-score smoke**: pick a checkpoint, select arc_easy, hit Run.
   Confirm:
   - Subprocess spawns, log streams
   - Results table populates within ~30s
   - JSONL file appears at the documented path
3. **Cross-checkpoint smoke**: pick a run with `.step-*` history.
   Hit Sweep. Confirm:
   - All checkpoints scored sequentially
   - Live chart updates as each row lands
   - 3-view comparison renders at completion
4. **Custom eval smoke**: drop a 5-row `{prompt, expected}` JSONL.
   Select exact-match. Run. Confirm per-row pass/fail table renders.
5. **Cancel smoke**: start any eval, hit Cancel mid-run. Subprocess
   terminates within 5s, UI returns to idle.
6. **History smoke**: run 3 evals, close + reopen the app. Confirm last
   3 results still in the History panel.
7. **Build passes; other tabs unaffected.**

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPTApp/EvalView.swift` — main Eval tab view |
| **create** | `native-mac/Sources/TinyGPTApp/EvalController.swift` — subprocess + JSONL parsing |
| **create** | `native-mac/Sources/TinyGPTApp/EvalChart.swift` — small line chart for emergence view |
| **modify** | `native-mac/Sources/TinyGPTApp/ContentView.swift` — wire 6th tab |
| **don't touch** | Train.swift, Serve.swift, RunLmEval.swift, EvalCompare.swift, the eval CLI surface (no backend changes), `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Inputs the agent has

| Resource | Location |
|---|---|
| Existing subprocess-management pattern | `TrainController.swift` and Server tab — spawn `Process()`, track PID, send signals, stream stdout |
| Eval CLI surface | `tinygpt run-lm-eval --help`, `tinygpt eval-compare --help`, `tinygpt judge --help` |
| E0 row schema | `Sources/TinyGPT/EvalCompare.swift` `Row` struct |
| Sample browser viewer for charting cues | `browser/src/pages/eval-leaderboard.astro` (drag-drop pattern, 3-view rendering) |
| Existing run scripts | `scripts/score-run.sh`, `scripts/score-checkpoint.sh` — shells out the same way the GUI will |
| Existing artifacts to use as test data | `docs/artifacts/score-n02-mid-*.jsonl`, `docs/artifacts/baselines-*.jsonl`, `docs/artifacts/emergence-smoke-*.jsonl` |

## Estimated effort

**~3-5 days focused work** for v1 covering all three modes.

- 1 day: Quick-score mode (model picker, task picker, subprocess, results table)
- 1 day: Cross-checkpoint mode (sweep logic, live chart)
- 1 day: Custom eval mode (JSONL drag-drop, custom-task YAML generation, per-row inspector)
- 1 day: History panel + cross-mode polish (cancel, log streaming, reveal-in-Finder)
- 1 day: Testing + screenshots for PR

## Coordination

PR description must include:
1. Screenshot of all 3 modes
2. Live chart screenshot from cross-checkpoint sweep
3. Confirmation that existing tabs work unchanged
4. Build + existing tests passing

Maintainer merges, updates HANDOFF.md noting Eval tab as the 6th tab.

## Known risks

- **Lm-eval-harness subprocess slow to spin up** (Python import time ~3-5s). Show a spinner during init; don't make user wait without feedback.
- **Custom task YAML generation** may have edge cases (escaping, prompt templates). v1 should fall back to simple exact-match if YAML generation fails; document the workaround.
- **Large JSONL drag-drop** may freeze UI during parse. Stream-parse off the main thread; show progress.
- **Subprocess zombie on app crash** — defer to existing TrainController pattern for process cleanup on app exit.

## Why this is the highest-ROI next app PRD

- Pure UI work — no new ML code, no new backend, no research uncertainty
- Closes the "scoring is CLI-only" product gap
- Uses already-shipped, already-tested CLI surface
- Makes the platform feel complete: training-tab + eval-tab pair gives users the full build-and-grade loop in-app
- ~3-5 days work; ships before the deeper Tier 1 items (QLoRA, TIES merging, Create Specialist wizard)
