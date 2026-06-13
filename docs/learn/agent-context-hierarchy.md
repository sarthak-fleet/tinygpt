---
title: Agent context as a memory hierarchy
description: The L1/L2/L3 framing for agent context engineering — what we stole from the Shortcut vertical-agents essay and where each steal lives in this repo.
---

# Agent context as a memory hierarchy

**Source:** "Vertical agents" — engineering essay by the builder of Shortcut,
the spreadsheet agent deployed in multistrategy hedge funds (unpublished
draft, read 2026-06-12; no public URL yet — check shortcut.ai's blog later).

**What:** a good agent is a *faithful compression of its task distribution*.
With the model fixed, accuracy is a function of context quality — and since
task frequency is long-tailed, context should be tiered like a CPU cache:

```
L1  always resident      the 80% case — lives in the system prompt
L2  on demand            curated specs, one cheap fetch when needed
L3  escape hatch         the raw complete reference + a skill to grep it
```

Every placement trades token cost on *every* task (L1) against discovery
cost on the *rare* task (L3). Put each capability at the tier that
minimizes total cost across the distribution.

**Why it matters to THIS project:** our planner models are 4–12B, not
Claude-class — they degrade *faster* under context bloat, so the tiering
discipline binds harder here than in the essay. And we already paid for
the lesson once: the v9 regression was the compose system prompt (an L1
resident) biasing non-compose intents. That is exactly the essay's
"L1 bloat" failure mode, named.

## The steal map

### 1. Diff triage — "here's what changed, and here's the part you probably got wrong" — SHIPPED 2026-06-12

The essay's best artifact: after a bulk action, don't dump raw results —
group them by pattern, sample a few, and pull anomalies into a flagged
"needs review" section. We applied it to eval reporting:

- `scripts/eval_pace_unhappy.py` now emits a `failure patterns` legend
  (failures grouped by normalized reason, count + example fixtures) and
  writes `failure_patterns` into the suite JSON.
- `scripts/eval_planner_report.py` prints the candidate's top 3 patterns
  per suite under the champion verdict — the n=130 drilldown loop no
  longer requires mining 130 raw rows to learn *how* a model fails.

The normalization trick is the essay's formula-aliasing move: collapse
instance-specific detail (quoted strings, numbers) so identical failure
modes group, but keep intent-mismatch reasons verbatim because the
confusion direction (`got X, expected Y`) is the signal.

### 2. L1 bloat biases behavior — REFUTED on this surface (2026-06-13)

Ran the A/B on google/gemma-3-12b across the n=130 unhappy drill.
v11-compact (name-only action index) **regressed every dim**: ambig
-10pp, oos -6.7pp, destructive -3.3pp vs v11 with full schemas. Steal
#2's prediction was *strictly wrong* here. See PLAN.md E9 for the
numbers and the post-hoc interpretation.

What the negative result says about the framing, not just this run:

- **Schemas are evidence, not just instructions.** At 12B/12-action
  scale, the per-action `args:` lines plus example calls help the
  model decide what pace *can't* do as much as what it *can*. Strip
  them and "play a tune" stops sharply rejecting and starts
  hallucinating into the nearest action.
- **L1 budget is small here.** The full v11 prompt is ~2K tokens; a
  12B model doesn't degrade meaningfully under that load. The
  essay's "L1 bloat" effect is real at much larger catalogs (the
  essay was written for a much bigger action surface) — the
  mistake was assuming the effect scales down.
- **The compact prompt isn't dead.** It's archived at
  `grammars/pace-system-prompt-v11-compact.txt` for re-test if the
  action surface grows past ~50 entries (e.g. App Intents catalog,
  generic MCP front-door). At that scale the L1 budget argument may
  flip back.

The triage (Steal #1) was still the right tool — the A/B diff fits in
one screen because the failure-pattern view tells you *which* intent
shifts, not just that something moved. That's how this turned into a
decisive negative result in 30 minutes instead of an inconclusive
"the numbers moved a little, hard to tell."

### 3. Deferred tool schemas — feature filed

The essay's meta-tool wall (`get_tool_info` → fetch schema once,
session-cached) maps onto `tinygpt serve --tools`, which today injects
the full catalog into every request. Filed as **B26** in
[PLAN.md](../PLAN.md): serve answers `get_tool_info` itself, so any
OpenAI-compatible client gets deferred tools for free. Gated on BFCL
parity per the no-quality-regression rule.

### 4. KV prompt cache is the literal L1 — already shipped

`serve --prompt-cache-dir` (see `docs/agent_runtime.md`) caches the
system prompt's attention state to disk, SHA-256 keyed. The essay pays
L1 cost in *tokens* on every call; we pay it once per prefix. This is
why a moderately large L1 is cheaper for us than for an API-priced
agent — but it does **not** rescue accuracy: resident tokens still
occupy attention even when prefill is free. Cache solves cost, not bias.

## What we deliberately did not steal

- **The single `execute_code` tool.** Shortcut owns its agent loop;
  `tinygpt serve` is an OpenAI-compatible *server* — the client owns the
  loop, so tool-surface design belongs to the caller. B26 is the piece
  of this idea that lives server-side.
- **The 70k-line L3 tome + grep skill.** No equivalent surface here; our
  action registry is 12 entries. Revisit if a specialist ever fronts a
  large API (e.g. App Intents catalog).

## The addition the essay is missing (our note, not theirs)

A CPU cache has hit-rate counters; the essay never measures which tier
tasks resolve in, so placement stays vibes. Our equivalent discipline:
the failure-pattern triage (steal #1) is the telemetry — when one
pattern dominates a suite, that's a misplaced-context signal, and E9 is
the first experiment that consumes it.
