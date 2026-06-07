---
name: Persistent training output — stop defaulting `/tmp` for training artifacts
status: shipped-2026-06-07
owner: unassigned (parallel-agent task — small, focused, post-incident)
created: 2026-06-07
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (post-mortem on 2026-06-07 morning loss)
incident: 2026-06-07 — Mac rebooted overnight, /tmp cleared, lost N02 checkpoint suite (best.tinygpt val_loss 4.434 + canonical + all step-N history + JSONL log)
---

# PRD — training output defaults to persistent path

## 2026-06-07 ship note

Implemented in the current tree:

- `tinygpt train` defaults run artifacts under `~/.cache/tinygpt/runs/`
- explicit `/tmp` outputs warn loudly before continuing
- JSONL logs default alongside the output checkpoint
- run `README.md` metadata is auto-written
- active-run lock file support exists at `~/.cache/tinygpt/runs/active.json`
- nightly and scoring scripts now use persistent run directories
- `scripts/migrate-tmp-runs.sh` migrates surviving `/tmp/*.tinygpt` runs

## Goal

Change the default training output location from `/tmp/` to a persistent
path under `~/.cache/tinygpt/runs/<run-name>/`. Loud warning when user
explicitly passes a path under `/tmp/`. Update all scripts +
documentation + nightly scripts accordingly.

## Why now — incident report

**2026-06-07 morning:** lost ~14 hours of N02 training output. Mac
rebooted overnight (cause unknown — display sleep + lid close, or system
update). macOS default behavior: `/tmp` is cleared on every boot.

What we lost:
- `huge-base-v1.best.tinygpt` (val_loss 4.434 at step 88,500 — the best
  checkpoint of the run, tracked by the watcher)
- `huge-base-v1.tinygpt` (canonical at step 89,901, where PowerMonitor
  cleanly paused)
- All step-N history checkpoints (`.step-2000.tinygpt` through
  `.step-88000.tinygpt`) — meant for B13 interp-on-checkpoints work
- `huge-base-v1.jsonl` — training log
- `huge-base-v1.history.jsonl` — reconstructed sidecar

What survived:
- `~/.cache/tinygpt/nightly/logs/2026-06-05_2253-N02-huge-base-v1.log`
  (text log of first segment — useful for trajectory analysis but no weights)
- Gallery models (`data/gallery/*.tinygpt`) — small browser-class, not
  the Huge base

**Root cause**: training scripts + nightly N02 + docs all default to
`/tmp/huge-base-v1.tinygpt`. Users (including the maintainer) assume
`--save-history` is sufficient for durability. It isn't, because the
directory itself is volatile.

**Lesson**: `/tmp` is a fine path for *intentionally ephemeral* artifacts
(smoke tests, throw-away experiments). It's wrong for anything you'd
miss after a reboot. The fix is to default to a persistent path and make
`/tmp` an explicit opt-in.

## Scope — in

### 1. New default output location

Pattern: `~/.cache/tinygpt/runs/<run-name>/<run-name>.tinygpt`

Where `<run-name>` defaults to a sensible auto-name when `--out` is not
specified: `<preset>-<timestamp>` (e.g., `huge-2026-06-07-145012`). The
directory groups all artifacts of one run:

```
~/.cache/tinygpt/runs/huge-base-v1/
├── huge-base-v1.tinygpt              ← canonical
├── huge-base-v1.tinygpt.opt          ← Adam state
├── huge-base-v1.best.tinygpt         ← best-val watcher output
├── huge-base-v1.best.meta.json
├── huge-base-v1.jsonl                ← training log
├── huge-base-v1.history.jsonl        ← reconstructed sidecar
├── huge-base-v1.step-2000.tinygpt    ← history checkpoints
├── huge-base-v1.step-2000.tinygpt.opt
├── …
└── README.md                          ← auto-generated run metadata
```

The run dir replaces the current convention where all 100+ files live
side-by-side in `/tmp/`. Easier to manage; easier to archive; easier to
delete cleanly.

### 2. `--out` flag behavior changes

- If `--out` is unset: auto-derive `~/.cache/tinygpt/runs/<auto-name>/<auto-name>.tinygpt`
- If `--out` starts with `/tmp/`: print loud warning:
  ```
  [warn] --out points at /tmp — this path is wiped on Mac reboot!
  [warn] If you intend long training, use --out ~/.cache/tinygpt/runs/<name>/<name>.tinygpt
  [warn] Continuing in 3s... (Ctrl-C to abort)
  ```
- If `--out` is anywhere else: respect it (user knows what they want)

### 3. Companion file routing

All sidecar files follow `--out`:
- `<out>.opt`
- `<out>.best.tinygpt` + `.best.meta.json`
- `<out>.jsonl` (when `--log-jsonl` is unset — log defaults to `<out>.jsonl`)
- `<out>.step-N.tinygpt` (when `--save-history` is set)

Currently some of these are hardcoded or share a stem assumption that
breaks if `--out` and `--log-jsonl` disagree. Audit + unify.

### 4. README.md auto-write

When the run dir is created, write a `README.md` listing the actual CLI
flags used + start timestamp + model config + corpus hash. So a year from
now, opening the dir tells you exactly what was trained without
greping shell history.

### 5. Update all scripts + docs

- `scripts/nightly/N02-huge-base-v1.sh`: change default to
  `~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt`
- `scripts/score-run.sh`, `score-checkpoint.sh`, `sae-run.sh`: update
  their default checkpoint-path assumptions
- `docs/PLAN.md` + `HANDOFF.md`: search for `/tmp/` references and
  update
- `docs/recipes/*.md`: same
- Any nightly scripts under `scripts/nightly/`: same

### 6. Migration helper

`scripts/migrate-tmp-runs.sh`: scans `/tmp/*.tinygpt`, prompts user to
move each surviving run to the new structure. One-time cleanup tool.

## Scope — out (v2)

- Cross-machine run sync (cloud / iCloud / Drive) — out of scope; users
  who want this can symlink `~/.cache/tinygpt/runs/` anywhere
- Run-management TUI / GUI — separate work (could live in app's Train tab
  history)
- Automatic cleanup of old runs — manual for v1; document the `du -sh
  ~/.cache/tinygpt/runs/*` command

## Acceptance criteria

1. `tinygpt train --steps 100 --preset tiny --tokenizer <SmolLM2 dir>
   --corpus data/examples/tiny-corpus.txt` (no `--out` flag) creates
   `~/.cache/tinygpt/runs/tiny-<timestamp>/tiny-<timestamp>.tinygpt`
2. Same command with `--out /tmp/foo.tinygpt` prints the warning then
   continues
3. README.md auto-generated in the run dir with model config + flags
4. `scripts/nightly/N02-huge-base-v1.sh` writes to
   `~/.cache/tinygpt/runs/huge-base-v1/`
5. Existing `--resume /tmp/old-run.tinygpt` still works (resume reads
   from wherever it's told)
6. Build passes; existing tests pass; smoke train completes successfully

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPT/Train.swift` — `--out` default + `/tmp` warning + auto-name |
| **modify** | `scripts/nightly/N02-huge-base-v1.sh` |
| **modify** | `scripts/score-run.sh`, `scripts/score-checkpoint.sh`, `scripts/sae-run.sh` |
| **modify** | `docs/PLAN.md`, `HANDOFF.md`, `docs/recipes/*.md`, `docs/sessions/*.md` (string replace `/tmp/huge` etc.) |
| **create** | `scripts/migrate-tmp-runs.sh` |
| **don't touch** | Anything not directly about persistent paths; this is a focused PRD |

## Estimated effort

**~1 day focused work.**

- 2-3 hrs: Train.swift default + auto-name + /tmp warning
- 1 hr: companion file routing audit
- 1 hr: README.md auto-write
- 2-3 hrs: update scripts + docs (mostly grep + replace)
- 1 hr: migration helper + testing

## Why this matters strategically

This is a "platform credibility" issue. A platform that loses ~14 hours
of training overnight from a default-path choice is not a platform
people trust for serious work. Fixing this is small but critical.

Also: the lesson generalizes. **Audit all "ephemeral by default" choices
in the platform** — `/tmp` for outputs, in-memory state without a sync
hook, anything that depends on a process staying alive. Make persistence
the default; make volatility the explicit opt-in.
