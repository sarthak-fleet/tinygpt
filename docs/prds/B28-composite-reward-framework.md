---
name: B28 composite reward framework
status: scaffolding-shipped-2026-06-13 (training-loop integration pending)
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B28)
parent_learn: docs/learn/castform-rl-finetune.md (Steal #1)
related_prds: 5.1-reasoning-on-22M.md (GRPO consumer), B22-trajectory-recorder.md (the rollouts each Reward scores)
---

# PRD — Composite reward with named dimensions

## Goal

Ship a typed `CompositeReward` abstraction: a reward is a *bag of
named scalar dimensions* + weights → total. Every dimension is
independently logged, plotted, and inspectable. DPO / ES / GRPO
(5.1) all consume the same struct.

Stolen from Castform's reward composition pattern (see
`docs/learn/castform-rl-finetune.md` §1). Different from existing
ES/DPO single-scalar paths.

## Why now

- A1 specialist's eventual DPO-refinement pass benefits immediately.
  A specialist's reward is rarely "one thing" — it's "tool-call is
  right" + "spoken_text is concise" + "didn't hallucinate". Hand-
  rolling that combination per recipe is the friction this removes.
- 5.1 GRPO PRD calls out RLVR for math + code with verifiable
  outcome rewards. Once that lands, "verifiable outcome" is one
  dimension; "did the rollout use the cheap tool first?" is
  another. Same composite abstraction supports both.

## Scope — in

- `Sources/TinyGPTModel/CompositeReward.swift` — types + aggregation.
  Already shipped in this PR:
  - `RewardDimension` (name, score, weight)
  - `CompositeReward` (dimensions, total computed property, logging)
  - JSONL round-trip for per-rollout dashboards
- Training-loop integration (the remaining work):
  - `tinygpt dpo --reward-fn <path>` — load a user-defined reward
    fn from a Swift-shaped specification file (JSON with weights;
    individual scorers are still Swift code per recipe). The shipped
    `.swift` recipe files do the actual scoring; the `--reward-fn`
    arg just routes the aggregation.
  - ES same flag.
  - `Sources/TinyGPT/Grpo.swift` (5.1) consumes natively.
- Train-time logging: per-step JSONL row gains `composite_reward:
  {dim_a: …, dim_b: …, total: …}`.

## Scope — out

- **Pluggable Python reward fns** — Castform's `BaseEnv.compute_reward`
  is in Python. We keep scoring in Swift for type safety + perf;
  Python harness scoring (existing `eval_pace_unhappy.py` style)
  emits its own JSONL that the Swift training loop reads as input.
- **Auto-weighting** (learn the weights from data). Manual weights
  for V1.
- **Reward shaping over time** (decay one dimension's weight as
  training progresses). V2.

## Files (already shipped in this PR)

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTModel/CompositeReward.swift` | new — types + aggregation + JSON I/O |
| `native-mac/Tests/TinyGPTModelTests/CompositeRewardTests.swift` | new — aggregation + JSON round-trip tests |

## Files still to touch (remaining B28 work)

| File | Change |
|---|---|
| `Sources/TinyGPT/Dpo.swift` | accept `--reward-fn` path; route the chosen/rejected scalar into a CompositeReward block |
| `Sources/TinyGPT/Es.swift` | same |
| `Sources/TinyGPT/Grpo.swift` | uses CompositeReward natively when 5.1 ships |
| `web/src/pages/train-viewer.astro` (C10) | render per-dimension lines when present in history JSONL |
| `docs/recipes/composite-reward.md` | new — recipe |

## Acceptance criteria

### Scaffolding (shipped this PR)

- [x] `CompositeReward` aggregates weighted dimensions to a total
  matching a hand-checked formula.
- [x] JSON round-trip preserves dimension names + weights + scores.
- [x] Unit tests pass under `xcrun swift test --filter CompositeRewardTests`.

### Full B28 ship (remaining)

- [ ] `tinygpt dpo --reward-fn weights.json ...` runs end-to-end on
  a fixture dataset; per-step history JSONL carries the composite
  reward block.
- [ ] Train viewer (C10) renders per-dimension reward curves.
- [ ] One published recipe in `docs/recipes/composite-reward.md`.

## Reference patterns

- `Sources/TinyGPT/Es.swift` lines 187–227 — the existing
  centred-reward pattern; CompositeReward stacks above it.
- `Sources/TinyGPT/Dpo.swift` SimPO/IPO loss paths — the
  margin-reward shape composite replaces with named dims.
- [Castform site](https://castform.com/) — the public reward-
  composition copy.

## Open questions

- Whether to require weights to sum to 1.0 (normalize) vs allow
  arbitrary weights. **Recommendation:** don't normalize — the
  user knows what they mean; the total is what's plotted.
