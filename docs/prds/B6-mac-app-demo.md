---
name: B6 Mac app demo
status: not-started (blocked-by A1)
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B6)
related_prds: app-eval-tab.md, app-train-controls-thermal.md, app-ux-polish-batch1.md
              (existing Mac app PRDs; B6 ties them into the user-facing demo flow)
---

# PRD — End-to-end Mac app demo with A1 specialist

## Goal

The product-shaped artifact that the platform's pitch hinges on: a
Mac app the user can open, point at their data, pick a base model,
train a LoRA, and run inference on the specialist they just trained
— all without leaving the app. Today the Mac app ships training
controls + an eval tab but no "specialist factory" wiring; B6 closes
that loop end-to-end.

Visible-to-user product story: "open the app → drop in some data
→ click train → 30 minutes later, here's your specialist running
on your machine."

## Why now

- A1 (CLI specialist) ships first. B6 is the GUI wrapper around
  the recipe A1 proves works. Order: A1 validates the science;
  B6 packages the experience.
- Existing app shells (`app-eval-tab.md`, `app-train-controls-thermal.md`)
  already cover the supporting tabs. The missing piece is the
  "factory" tab that chains: data import → recipe pick → train →
  eval → deploy.
- The platform's "Mac specialist factory" framing in PLAN.md needs
  a product-shape artifact, not just a CLI.

## Scope — in

- New tab in the existing app: **Factory** — wizard-style flow:
  1. **Pick your task** (preset: tool-calling, shell, SQL, custom)
  2. **Drop in your data** (drag JSONL/JSON/CSV; the app validates
     against the recipe's expected schema; shows a preview)
  3. **Pick your base** (from the model picker; shows
     size + capability tags; default = the curated A1 base)
  4. **Recipe summary** (read-only view of the SFT recipe; "Start
     training" CTA)
  5. **Live training view** (loss + grad-norm + LR — embeds the
     C10 train viewer; cancel + checkpoint controls)
  6. **Eval gate** (auto-runs the recipe's domain eval; passes
     or fails the ship gate; ships an adapter file on pass)
  7. **Try it now** (opens a chat with the new specialist;
     before/after compare against the base 0-shot)
- The wizard is a single SwiftUI view tree calling the existing
  `tinygpt train` / `eval-*` subprocesses through the existing
  ServerController + ProcessRunner.
- A "Save recipe" button so a working recipe becomes a sharable
  `.tinygpt-recipe` file users can ship to others.

## Scope — out

- **Custom recipe authoring** in-app (multi-stage pipelines,
  conditional steps). V1 = presets + load `.tinygpt-recipe` files.
- **Cloud training** as a button. The Mac is the training surface
  by design.
- **App Store distribution.** TestFlight + direct download for V1.

## Files to touch

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTApp/FactoryTabView.swift` | new — the wizard |
| `native-mac/Sources/TinyGPTApp/FactoryRecipe.swift` | new — recipe model |
| `native-mac/Sources/TinyGPTApp/FactoryDataset.swift` | new — drop-in + validate logic |
| `native-mac/Sources/TinyGPTApp/AppView.swift` | add the Factory tab to the tab bar |
| `recipes/factory/*.tinygpt-recipe` | new — preset recipes (tool-call, shell, SQL) |
| `docs/specialists/build-your-own.md` | new — user-facing how-to |
| `docs/PLAN.md` | B6 ⬜ → ✅ on ship |

## Don't touch

- The existing Train + Eval tabs — Factory is a new sibling, not
  a refactor.
- `tinygpt train` / eval subcommands — V1 calls them as
  subprocesses through the existing ServerController.

## Acceptance criteria

- [ ] User can open the app, run the tool-call preset against a
  test JSONL, train + eval + deploy a specialist in one session
  on M5 Pro in < 1h wall-clock.
- [ ] The eval gate fires correctly (passes when the trained
  adapter beats the gate; fails-with-actionable-message otherwise).
- [ ] "Try it now" tab opens chat against the new specialist
  with a baseline side-by-side comparison.
- [ ] Save-recipe produces a `.tinygpt-recipe` file that another
  install can load to reproduce.
- [ ] User-doc walkthrough reproducible from a clean install.

## Reference patterns

- `app-eval-tab.md` — existing tab pattern.
- `app-train-controls-thermal.md` — existing live-training view.
- C10 train-run-dashboard — the browser-side companion; the in-app
  view can share the chart components if useful.

## Open questions

- Whether to support custom-base import inside Factory (drag a
  `.tinygpt` / `.safetensors` from disk). **Recommendation:** yes —
  the existing model picker already supports it; pipe through.
