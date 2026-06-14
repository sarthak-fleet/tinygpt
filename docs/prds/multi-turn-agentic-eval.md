---
title: Multi-turn / agentic tool-calling eval (PRD)
description: A Mac-local, frontier-gated, STATEFUL multi-turn eval — the single
  biggest blind spot in our tool-calling work. Single-turn numbers (88.7 for our
  best 4B) overstate agentic ability; small models are known to cliff hard on
  multi-turn. This is the prerequisite for judging any model as an agent/planner.
status: proposed
---

# Multi-turn / agentic tool-calling eval

## Why (the blind spot)

Everything we've measured is **single-turn**. Our headline (Qwen3-4B-2507 bf16 =
**88.7**, frontier 98) says nothing about *holding a conversation* — observing a
tool result and acting on it across turns, tracking state. The literature is blunt
about the gap:

| Model | single-turn | multi-turn |
|---|---|---|
| Command-R7B | 69% | **5%** |
| Llama-3.1-8B | 61% | **9.6%** |
| xLAM-2-8b-fc (purpose-tuned) | — | ~69% |

So our 88.7 likely **overstates** these models as agents by a wide margin — and we
have **zero** measurement of the real number. This is the prerequisite capability
for everything next: a Jarvis-style assistant (Pace) and self-improving game NPCs
are *both* multi-turn agents. See
[`docs/learn/small-model-tool-calling-playbook.md`](../learn/small-model-tool-calling-playbook.md) §4.

## Goal

A Mac-local, **stateful** multi-turn tool-calling eval that, for a given model:
1. runs a multi-turn task where the model's calls **execute against a backend** and
   results feed back across turns,
2. scores **end-to-end task completion** (final backend state matches gold) — not
   just per-call AST,
3. is **frontier-gated** (per our eval philosophy: a sound eval → a frontier model
   aces it; if not, fix the eval before trusting Mac-model numbers),
4. reports the **single→multi-turn drop** for our best 4B vs frontier.

## Data + matching

- **Source:** BFCL-v4 `multi_turn_base` (200 examples, ~4 turns each) — already on
  disk at `~/.cache/tinygpt/datasets/_external/gorilla-bfcl/.../data/`. Each example
  has `question` (turn-list), `initial_config` (backend state), `involved_classes`
  (the stateful Python backends, e.g. `GorillaFileSystem`), and a gold action `path`.
- **Matching:** state-based — instantiate the backend from `initial_config`, execute
  the model's calls turn by turn, feed results back into the transcript, and at the
  end compare the **resulting state** (and/or the executed-call path) to gold. This
  is BFCL's own multi-turn protocol; we replicate the executor.
- **Stretch:** a τ-bench-style slice (user-simulator + DB-state check + Pass^k
  reliability) once the BFCL executor works.

## Design / scaffolding

`scripts/bfcl_multiturn_eval.py` (skeleton in this PR):
1. Load example → instantiate `involved_classes` with `initial_config`.
2. For each turn: render transcript (system tools + prior turns + tool results) →
   model emits call(s) → **execute** against the backend instances → append results
   → next turn.
3. After the last turn: compare final state / executed path to gold → pass/fail.
4. Backends: `local` (MLX) + `frontier` (`claude -p`), reusing `bfcl_ast_eval.py`'s
   call parser.

The hard part is faithfully executing calls against BFCL's backend classes (import
or vendor them from the gorilla repo). Skeleton marks these as `TODO(executor)`.

## Acceptance criteria

- [ ] Frontier (`claude -p`) scores high (~90%+) on the slice — if not, the eval /
      golds are broken; fix before reporting Mac numbers (frontier-ceiling gate).
- [ ] Report our best 4B (Qwen3-4B-2507 bf16) multi-turn vs its single-turn 88.7 —
      quantify the cliff.
- [ ] Run ≥1 small model + frontier; land the numbers in
      `docs/learn/tool-calling-frontier-parity.md`.

## Risks / notes

- **Executor fidelity** is the main cost — replicating BFCL's stateful backends.
  Mitigation: vendor the gorilla `multi_turn` executor rather than reimplement.
- **Contamination:** BFCL multi-turn backend Python overlaps training data (known
  issue); note it, don't over-claim.
- This is a **build**, not a quick win — hence a PRD, not an inline change.

## Next after this

If small models cliff hard (expected), the lever is **multi-turn RL** in an
open-ended environment — which is the sister PR (game-as-RL-environment PoC).
