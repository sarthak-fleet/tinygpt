---
name: AX-derived screen-capture pipeline for VLM training data
status: shipped-v1-2026-06-07-runtime-capture-needs-macos-permissions
owner: unassigned (parallel-agent task — Swift CLI)
created: 2026-06-08
priority: P1 — feeds VLM M6 Stage B; can run for days before VLM is ready to consume
parallel-safe: yes (no overlap with ANE or VLM elves)
unblocks: factory-vision-specialist.md M6 Stage B
---

# PRD — AX-derived screen capture for VLM training

## Why this PRD exists

## Ship note — 2026-06-07

V1 is implemented as `tinygpt ax-capture`:

- Capture loop writes foreground-window `.png` + AX-derived `.json` pairs
- `index.jsonl` appends one row per successful non-duplicate capture
- Privacy excludes are loaded from
  `~/.config/tinygpt/ax-capture-excludes.json`, with password-manager /
  auth defaults merged in
- Daemon controls are available through `--daemonize`, `--stop`, and
  `--pid-file`
- Output schema is documented at `docs/schemas/ax-capture.schema.json`

Verification in this agent session is limited to build/help checks. A real
capture run still requires macOS Screen Recording and Accessibility grants
for the terminal or `tinygpt` binary.

The VLM PRD's M6 calls for ~500-2000 Mac-specific (screenshot, AX
tree) training pairs as **Stage B** of the three-stage data plan.
These deserve to be collected NOW, in the background, while the VLM
arc finishes its architectural decisions and milestones.

This is a small standalone CLI tool. Independent of every other arc.

## Why not just run it manually

Manual capture is tedious and produces unrepresentative data. A
background daemon that captures opportunistically while the owner
uses the Mac normally gives:
- Larger sample sizes
- More diverse app coverage (whatever the owner actually uses)
- No "I have to remember to capture" friction
- Ground truth labels for free (AX tree is deterministic)

## Goal

Ship `tinygpt ax-capture` as a long-running CLI that:

1. Periodically (every ~10s, configurable) captures a screenshot of
   the foreground window
2. Walks the AX tree of that window, extracts interactive elements
   (buttons, links, fields, etc.) with `(label, role, bbox)`
3. Filters out duplicates (same screen + same AX tree as last capture)
4. Writes `(screenshot.png, ax_tree.json)` pairs to a configurable
   output directory
5. Runs in the background unobtrusively (no UI, minimal CPU)
6. Respects an exclude list (e.g., don't capture password manager
   windows)

## Scope — in

### 1. Capture loop

```swift
while running {
  let screenshot = captureFrontmostWindow()
  let axTree = walkAXTree(frontmostApp)
  if !isDuplicate(screenshot, axTree, vsLast) {
    write(screenshot, axTree, to: outputDir)
  }
  sleep(intervalSeconds)
}
```

- Screenshot: use CGWindowListCreateImage targeted at the frontmost
  window (not full screen — keeps file sizes manageable and
  privacy-respectful)
- AX tree walk: use AXUIElement APIs (`AXUIElementCopyAttributeValue`
  for kAXChildrenAttribute, recurse). Extract:
  - kAXLabelAttribute / kAXTitleAttribute / kAXValueAttribute
  - kAXRoleAttribute
  - kAXFrameAttribute (gives x, y, width, height)
  - kAXEnabledAttribute (skip disabled elements)

### 2. Deduplication

Hash the (screenshot pixel data downsampled to 64×64, AX tree
structure as canonical JSON). If hash matches the last capture, skip.

Optional: also skip if NONE of the visible AX elements changed since
the last successful capture (covers the case where the screenshot
differs by cursor position but the underlying UI is the same).

### 3. Privacy / exclude list

Read `~/.config/tinygpt/ax-capture-excludes.json`:
```json
{
  "bundle_ids": ["com.1password.1password", "com.apple.keychainaccess"],
  "window_titles_matching": ["password", "secret", "auth"],
  "process_names": ["1Password"]
}
```

Skip capture if frontmost app matches.

### 4. Output format

Per capture:
- `<outputDir>/<timestamp>-<bundleID>.png` (screenshot)
- `<outputDir>/<timestamp>-<bundleID>.json`:
  ```json
  {
    "timestamp": "2026-06-08T14:32:15Z",
    "bundle_id": "com.apple.Safari",
    "window_title": "GitHub - sarthakagrawal927/tinygpt",
    "window_frame": {"x": 100, "y": 50, "width": 1400, "height": 900},
    "elements": [
      {
        "id": 0,
        "role": "AXButton",
        "label": "Star",
        "frame": {"x": 1200, "y": 200, "width": 80, "height": 30},
        "enabled": true
      },
      ...
    ]
  }
  ```

Also emit a rolling `<outputDir>/index.jsonl` with one line per
capture for quick consumption by training data scripts.

### 5. CLI invocation

```
tinygpt ax-capture \
  --out ~/.cache/tinygpt/datasets/vlm-ax-mac \
  --interval-sec 10 \
  --max-captures 10000 \
  [--excludes path/to/excludes.json]
```

Defaults reasonable so a curious user can just run
`tinygpt ax-capture` and it captures sanely.

### 6. Daemon mode

`tinygpt ax-capture --daemonize` runs in the background, writes a
PID file, can be stopped with `tinygpt ax-capture --stop`.

## Scope — out

- Capturing during fullscreen apps (skip those — usually games or
  full-screen video; AX tree is sparse)
- Capturing image content INSIDE elements (we capture coords only,
  not the rendered icon)
- Video capture (frame-by-frame screen reading)
- Cross-Mac sync (run on each Mac independently)
- Online learning / labeling (purely a capture tool; consumers process
  later)
- Anonymization (privacy is per-exclude-list — if the user wants
  PII redaction in screenshots, that's a separate post-process step)

## Acceptance

1. `tinygpt ax-capture --out /tmp/ax-test --max-captures 5 --interval-sec 2`
   runs to completion in ~10s
2. /tmp/ax-test contains 5 (.png, .json) pairs
3. Each .json validates against a documented schema
4. AX tree extraction handles edge cases without crashing:
   - apps that don't expose AX (system apps may refuse)
   - very deep AX trees (hundreds of elements)
   - missing labels (use AXValue or fall back to role-only)
5. Privacy excludes work: capturing while frontmost = excluded app
   produces no output file
6. Daemon mode survives display sleep / wake without runaway resource use
7. CPU usage <1% of one core during steady-state idle waiting

## Data consumption

Once ~500-2000 pairs are captured:
1. Convert to VLM training format (M6 Stage B in VLM PRD)
2. Train: each row becomes `(screenshot, "list all interactive
   elements", AX-derived-label-list)` SFT example
3. Optional: synthesize harder tasks by hand-writing intents per
   screenshot ("click X", "what's the value of Y") with deterministic
   answers from the AX tree

## Files involved

- New: `native-mac/Sources/TinyGPT/AXCapture.swift` — main capture loop
- New: `native-mac/Sources/TinyGPT/AXTreeWalker.swift` — extract elements
- New: `native-mac/Sources/TinyGPT/ScreenCapture.swift` — CGWindow APIs
- Modified: `native-mac/Sources/TinyGPT/TinyGPT.swift` — register `ax-capture` subcommand

## Estimated effort

**1-2 days** for the elf. CGWindow + AXUIElement APIs are
well-documented; the loop is straightforward; deduplication and
excludes are small additions.

## Won't conflict with other elves

- New files only in TinyGPT (executable target)
- No touch on TinyGPTModel, TinyGPTServe, TinyGPTApp
- The ANE elf, VLM elf, and any future elves are unaffected

## Why this should ship before the VLM specialist trains

The bigger the Stage B dataset, the better the VLM specialist
generalizes. Starting capture now means by the time the VLM elf
reaches M7 (SFT), there's already a substantial real-world dataset
ready. This is pure leverage — sequential capture during owner's
normal Mac use is "free" data.
