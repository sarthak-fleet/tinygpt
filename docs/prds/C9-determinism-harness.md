---
name: C9 determinism harness — bit-exact replay of step N
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier C (C9)
related_prds: B12 loss-spike recovery (covered partly by adam-state-persistence.md) — this is the debug counterpart
---

# PRD — Bit-exact replay of `tinygpt train` step N

## Goal

Add `tinygpt train --replay-step N --replay-from <ckpt>` that runs
exactly step N of an earlier training run, deterministically. The
output is a logged forward + backward trace identical to what the
original step produced — same input batch, same weights at step N-1,
same RNG, same intermediate tensor norms.

The single tool that turns "training spiked at step 4218" from a 4-hour
investigation into a 4-minute one: load the saved state, replay 4218,
inspect the activations / gradients / Adam state directly.

## Why now

- Loss-spike recovery (B12) and the existing AdamState persistence
  (`adam-state-persistence.md`) already write the state needed to
  enable this. The infra exists; the CLI doesn't.
- Long pretrain runs are increasingly the norm and "wait until it
  spikes, re-run with `--debug` from scratch" doesn't scale past a few
  hours of compute.
- Determinism harness is also the validation gate for the data-mixture
  experiments (B21 micro-AutoMixer): proving step N is bit-exact across
  reruns is what lets you trust the mixer's A/B comparisons.

## Scope — in

- `tinygpt train --replay-step N --replay-from <history-dir>` flag set.
  Loads the checkpoint at step N-1, the AdamState at step N-1, the RNG
  state at step N-1, and the data-loader cursor at step N-1; runs one
  forward + backward + optimizer step; emits a JSON trace.
- New file `Sources/TinyGPTModel/RNGSnapshot.swift` — serializable RNG
  state for MLX random + the data sampler. Tiny (RNG state is a few
  longs).
- Snapshot writer in the main train loop: every `--save-every`
  checkpoint also writes `<step>.rng` + `<step>.cursor`, sharing the
  history dir's lifecycle.
- Replay-trace JSON format: `{step, batch_hash, loss, grad_norm,
  per_layer_act_norms: [...], per_layer_grad_norms: [...], adam_m_norms:
  [...], adam_v_norms: [...]}`.

## Scope — out

- **Replay across different hardware** — MLX's `softmax` and
  `scaledDotProductAttention` are deterministic given fixed input on
  the same device, but cross-device replay needs upstream guarantees
  we don't have. Document the limitation; replay only on M5 Pro (or
  the same machine class).
- **Replay-and-modify** ("what if at step 4218 we had lowered LR by
  10×") — that's a fork, not a replay. Defer.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/RNGSnapshot.swift` | new — serialize/restore MLX RNG + sampler cursor |
| `Sources/TinyGPT/Train.swift` | snapshot RNG + cursor every save; `--replay-step` / `--replay-from` flag handlers |
| `Sources/TinyGPT/ReplayTrace.swift` | new — emit + reload trace JSON |
| `Tests/TinyGPTModelTests/DeterminismTests.swift` | new — run 10 steps, replay step 5, assert byte-equality of weights at step 5 |
| `docs/training_guide.md` | "Debugging a loss spike" runbook section |

## Acceptance criteria

- [ ] `tinygpt train --replay-step 10 --replay-from <hist-dir>` on a
  fresh shakespeare run produces a trace whose `loss` matches the
  original step 10 loss to ε=0 (byte-equal float).
- [ ] DeterminismTests.swift passes — 5-step replay byte-equal on the
  M5 Pro CI box.
- [ ] The replay trace's `per_layer_grad_norms` correlates with the
  original training's grad-norm telemetry (any divergence is a bug).
- [ ] Storage cost of `<step>.rng + <step>.cursor` is < 10 KB per
  checkpoint (RNG state is small).
- [ ] Runbook entry in `training_guide.md` walks through a worked
  example of finding a contrived spike.

## Reference patterns

- `Sources/TinyGPT/Train.swift` lines around the SIGINT-atomic-save
  pattern — that's where replay snapshot writes attach.
- `adam-state-persistence.md` — the design partner; that PRD ships the
  Adam state; this PRD adds RNG + cursor + the replay CLI.
- MLX random API docs — `MLXRandom.split` and `seed` are the
  determinism handles we need to save.

## Open questions

- Whether to default replay on for every save (the snapshot is ~10 KB,
  basically free) or opt-in via `--replay-record`. **Recommendation:**
  default on; cheap insurance.
- Whether the trace JSON gets versioned. **Recommendation:** yes —
  add a `replay_trace_version: 1` field, the same way checkpoints do.
