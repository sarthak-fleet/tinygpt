---
title: Game-as-RL-environment PoC (PRD)
description: Turn the fleet's AI game (autonomous characters in a world) into a
  Mac-local RL environment and train a self-improving NPC with GRPO. Environments
  are the scarcest RL ingredient — we own a live one. The parked trigger
  (distillation closed + a working GRPO loop) is now MET.
status: proposed
---

# Game-as-RL-environment PoC

Makes actionable the parked blueprint in
[`docs/learn/rl-multi-agent-roadmap.md`](../learn/rl-multi-agent-roadmap.md).
**Trigger condition is now met:** the distillation thread closed and a from-scratch
MLX GRPO loop exists and is validated (GSM8K + tool-calling — see
[`tool-calling-frontier-parity.md`](../learn/tool-calling-frontier-parity.md) §5).

## Why now (and why this, not more single-turn RL)

- **Environments are the scarcest RL ingredient** (Prime Intellect built a
  2,500-environment Hub precisely because of that). The fleet's AI game *is* a live
  multi-agent environment — an ownable asset most can't get.
- **Single-turn RL saturated** on a strong base (GRPO on the 4B was neutral: 87.3→86.7,
  because a fixed verifiable task has a ceiling). **An open-ended game environment
  won't saturate** — there's always a better policy. That's exactly why the game, not
  another single-turn benchmark, is the right next RL target.
- It's the on-mission direction: *learn everything about agentic/multi-agent RL, on a
  Mac, in an environment we own.*

## Goal (PoC scope — deliberately one behavior)

Show **one** distilled small model, as **one** NPC's brain, measurably improve at
**one** in-game behavior via GRPO, trained entirely on the Mac.

| RL ingredient | In the PoC |
|---|---|
| Policy | a distilled small local model (the cost-compression lever), LoRA |
| Environment | the game world (one scenario) |
| Reward | a measurable in-game outcome (goal reached / interaction won) — clean ⇒ RLVR |

## Design

1. **Trajectory recorder** (game side): log per turn `(observation → action →
   outcome → reward)`. This is the roadmap's "B22 recorder." Define the
   observation serialization + the reward signal for the one chosen behavior.
2. **Reward** = a verifiable in-game outcome (binary or graded), so it's RLVR — the
   tractable kind. Avoid LLM-judge rewards for the PoC.
3. **GRPO trainer**: reuse the validated loop (group-normalized advantage, KL-to-ref,
   grad-accumulation — the same structure as our tool-calling GRPO). Generalize it to
   take an `(env.reset, env.step, reward)` interface instead of the BFCL prompt set.
   Skeleton: `scripts/game_rl_poc.py` (this PR).
4. **Loop**: sample K rollouts of the NPC acting in the scenario → reward each →
   group-normalize → policy-gradient on the LoRA → repeat. Watch the behavior's
   success rate trend up.

## Acceptance criteria

- [ ] One scenario instrumented with a clean reward; trajectories logged.
- [ ] GRPO run on a distilled small model as the NPC brain; the chosen behavior's
      success rate **rises vs the pre-RL baseline** over training.
- [ ] Stable (KL-bounded, no blowup) — the tool-calling GRPO already proved the loop.

## Risks / notes

- **Game integration** is the dependency — needs the recorder + a programmatic
  step/reset/reward interface from the game. Scope the PoC to one scenario to bound it.
- **Reward hacking / degenerate policies** — keep the reward verifiable and the
  scenario simple; KL-to-ref guards against drift.
- Sample efficiency: GRPO is rollout-hungry; one Mac limits scale. PoC targets a
  *trend*, not a finished agent.

## Relationship to the other PRs

- The **multi-turn eval PRD** measures *whether* small models can hold agentic
  conversations; this PRD is *how to improve them* via RL in an open-ended env.
  Eval first (measure the cliff), then this (train the climb).
