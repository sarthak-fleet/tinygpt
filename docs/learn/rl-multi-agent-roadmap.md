---
title: RL + multi-autonomous-agents roadmap — the AI game as a Mac-local RL environment
description: Parked blueprint. The fleet's AI game (autonomous characters in a world) is already a multi-agent RL environment — the scarcest RL ingredient. This is the plan to turn it into a self-improving-agents testbed on the Mac, and the staged roadmap toward multi-agent dynamics. Revisit after distillation + the first GRPO loop.
---

# RL + multi-autonomous-agents roadmap

**Status: parked — revisit later** (after the distillation thread closes and a
first GRPO loop exists). Captured so we can pick it up cold.

## The realization

The fleet's **AI game** (a bunch of characters living in a world autonomously)
*is* a multi-agent RL environment. Environments are the **scarcest** ingredient
in RL — Prime Intellect built a 2,500-environment Hub precisely because that's
the bottleneck. We already have a live one. That's a real, ownable asset.

## What RL concretely does here

A game produces the RL loop for free: character observes world → acts → world
updates → outcome = a **trajectory**. RL needs three things, all present:

| RL ingredient | In the game |
|---|---|
| **Policy** | the character's model brain — the **distilled small local model** (cost-compression lever) |
| **Environment** | the world |
| **Reward** | a *measurable in-game outcome* (goal reached, need satisfied, task done, survived, won an interaction) — clean/verifiable = **RLVR**, the tractable kind |

Wire them together → **self-improving NPCs**: log trajectories (B22 recorder:
`observation → action → reward` per turn), run **GRPO** on the policy, and the
character gets better at living in the world over time — trained on a Mac, in
our own environment.

## First experiment (the PoC to run when we return)

1. Pick **one** measurable behavior (a character completing a specific goal /
   winning a negotiation).
2. Instrument the game to log each turn `(observation → action → outcome→reward)`.
3. Run **GRPO** on a small local model (the distilled specialist) as that
   character's brain; reward = the in-game outcome.
4. Measure: does that character's success rate climb over RL iterations vs the
   base brain?

A "yes" = a self-improving agent, trained on a Mac, in our own world — the
"after autonomous agents" thesis demonstrated, not theorized.

## Staged roadmap (multi-autonomous-agents)

- **Stage 0 (have):** autonomous characters = model + loop + world.
- **Stage 1:** instrument trajectories + rewards (B22) → the game becomes an RL *environment*.
- **Stage 2:** single-character self-improvement (GRPO on in-game reward) → self-improving NPC.
- **Stage 3:** multi-agent dynamics — characters co-adapt, specialize, learn from
  each other; emergent social behavior (generative-agents / AI-town, but
  **local + RL-trained**, not just prompted).
- **Stage 4:** the game as a **general RL gym** — agents trained in-game transfer
  *out* to other fleet projects; the eval/judgment layer verifies them.

## Why this fits the north-star

It fuses *everything* into one Mac-runnable system, on an asset we already own:
- **distillation** (cheap local character brains) + **RL** (self-improvement from
  in-game rewards) + **multi-agent** (the world) + **eval** (judge the agents).
- It's "build everything Mac-buildable" + "what comes after autonomous agents
  (self-improving + multi-agent)" + "an environment most people have to build."

## What's needed to make it runnable

The game's repo specifics:
- How does a character pick an action each turn? (a model call? an
  OpenAI-compatible endpoint? what observation/action format?)
- What in-game signals can serve as rewards?

With those, scope the trajectory-instrumentation + the first GRPO experiment.

## Related

- [model-vs-agent](./model-vs-agent.md) — the static picture (what makes a model an agent).
- [advanced-llm-training](./advanced-llm-training.md) §13 — GRPO mechanics.
- [agent-context-hierarchy](./agent-context-hierarchy.md) — context discipline for the loop.
- Distillation result (cost-compression: 0.6B ≈ 4B on tool-calling) — the cheap local brains.
