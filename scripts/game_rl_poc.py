"""Game-as-RL-environment PoC — SKELETON (see docs/prds/game-rl-environment-poc.md).

Generalizes the validated tool-calling GRPO loop (group-normalized advantage +
KL-to-reference + grad-accumulation) to an Env interface, so a distilled small model
can be trained as a self-improving NPC brain on the Mac.

STATUS: scaffolding. TODO(game) blocks need the game's recorder + a programmatic
reset/step/reward interface (the roadmap's B22 recorder).
"""
import os
# NOTE: the GRPO mechanics (rollout, group-normalized advantage, KL-to-ref, grad
# accumulation for big-vocab logits) are proven in the tool-calling loop. This file
# adapts that structure to an Env instead of a fixed prompt set.

class Env:
    """TODO(game): wrap ONE game scenario behind this interface.
    The reward must be a VERIFIABLE in-game outcome (RLVR), not an LLM judge."""
    def reset(self):
        """Return the initial observation (serialized to text the policy consumes)."""
        raise NotImplementedError("game scenario reset")
    def step(self, action_text):
        """Apply the NPC's action; return (observation, reward, done)."""
        raise NotImplementedError("game scenario step")

def rollout(policy, tok, env, max_turns):
    """One trajectory: obs -> action -> (obs, reward, done), accumulating reward.
    Returns (token_ids of the policy's actions, total_reward)."""
    raise NotImplementedError("TODO: render obs -> generate action -> env.step")

def grpo_step(policy, ref, opt, env, K, batch):
    """Reuse the validated loop: sample K rollouts/scenario-seed, group-normalize the
    advantage (skip zero-variance groups = dynamic sampling), KL-to-ref penalty, grad
    accumulation. See scripts/bfcl_ast_eval.py's sibling GRPO for the exact mechanics."""
    raise NotImplementedError("port grpo loop with Env reward in place of AST reward")

def main():
    print("Game-RL PoC (SKELETON) — see docs/prds/game-rl-environment-poc.md")
    print("Needs: (1) game recorder + Env(reset/step/reward), (2) a distilled policy,")
    print("       (3) port the proven GRPO loop. Targets a rising success-rate TREND.")

if __name__ == "__main__":
    main()
