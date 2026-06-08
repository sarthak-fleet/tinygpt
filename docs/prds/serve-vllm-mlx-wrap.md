---
name: vllm-mlx wrap — RECLASSIFIED to strategy doc (not a PRD)
status: superseded-by-strategy-doc
owner: n/a — see strategy doc
created: 2026-06-06
superseded: "2026-06-06 (same day — category error: this is a strategic adoption decision, not an elf-shippable task)"
---

# Not a PRD — strategic decision tracked elsewhere

This file existed briefly as a PRD but was reclassified the same day.
The vllm-mlx adoption question is a **strategic product decision** for
the maintainer, not a discrete task an elf can pick up and ship in
1-5 days.

## Where the content lives now

See **`docs/sessions/2026-06-06-mac-specialist-platform.md` → "Strategic
Decision 1: How to handle vllm-mlx serve threat"**. That section captures
the three options (match / wrap / status-quo) and the recommended path
(wrap, per "results-first" principle).

## What would be PRD-shaped within this decision

If/when the strategic decision lands on "proceed with wrapping," the
work breaks into discrete PRD-sized sub-tasks:

| Sub-task | Effort | Why it's PRD-sized |
|---|---|---|
| `vllm-mlx-phase1-investigation` | ~2-3 days | Specific deliverable: a 1-page memo + benchmark output. Elf-shippable. |
| `serve-backend-flag` | ~1-2 days | Add `--backend native\|vllm-mlx` flag to `tinygpt serve`. Discrete. |
| `tinygpt-to-mlx-weights-converter` | ~1-2 days | Adapter from `.tinygpt` → MLX weight dir. Single feature. |
| `server-tab-backend-toggle` | ~1 day | UI addition. Discrete. |

Each gets a real PRD when the work begins.

## Why this was a category error

PRDs should be for **discrete tasks an elf builds in 1-5 days with
specific acceptance criteria**. Strategic adoption decisions are for the
maintainer to discuss and ratify (with research input). They don't
belong in the same file format.

Lesson: when a "PRD" has phases, sub-tasks, or "if approved, then..."
structure, it's a strategic decision, not a PRD.
