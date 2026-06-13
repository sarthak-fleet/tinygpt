---
name: B5 cloud-escalation training signal
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B5)
related_prds: A1-first-specialist-tool-caller.md (V1 specialist doesn't yet learn to escalate),
              B22-trajectory-recorder.md (substrate for the labeled rollouts this PRD trains on)
---

# PRD — Teach the specialist when to defer to cloud

## Goal

Train a specialist to emit `{"defer_to_cloud": true, "reason": "..."}`
when it shouldn't answer itself — too uncertain, off-domain, or
explicitly destructive — instead of fabricating. Today's
`tinygpt agent --cloud-escalate` is a *runtime* policy: a regex over
the response that triggers cloud retry. B5 makes escalation a
*trained behavior*: the model learns to emit the signal.

## Why now

- The cloud-escalate runtime path already ships (`AgentLoop.swift`).
  It works but is brittle (the regex misfires; the model doesn't
  know when it's wrong). Trained escalation is the next-quality
  step.
- B22 ships the trajectory recorder — the substrate for the labeled
  rollouts this PRD needs (each rollout knows whether the cloud
  fallback fixed it, which becomes the train-time label).
- For Pace specifically: trained escalation lets the local-first
  story stay honest. The local specialist gets harder tasks pushed
  to cloud cleanly, not via a string-match hack.

## Scope — in

- **Data generation:** take a rollout dataset (`.atraj` files from
  B22), label each turn:
  - "answered correctly locally" → negative example (don't escalate)
  - "answered wrong locally; cloud retry fixed it" → positive example
    (would have been right to escalate)
  - "answered correctly locally; cloud agreed" → ambiguous; drop
- **Training:** SFT on the labeled set with a new output channel.
  Schema gains a `defer_to_cloud: bool` + `reason: string` field
  on the existing planner response shape.
- **Eval:** `tinygpt eval-escalate` — measures (a) escalation
  precision (when the model says defer, was it actually wrong
  locally?), (b) recall (of the cases where local was wrong, what
  fraction did the model flag?), (c) over-escalation rate (defer
  when local was right).
- **Runtime:** `tinygpt agent` honors the trained signal as the
  *primary* trigger; the regex stays as a fallback for safety.

## Scope — out

- **Learning the cost-benefit** of escalating (cloud is more
  accurate but slower + costs money). V1 is binary classifier;
  cost-aware is V2.
- **Multi-tier escalation** (local → bigger-local → cloud). V1 is
  two-tier.
- **Self-judging without a cloud reference.** This PRD assumes a
  cloud teacher for labeling. Self-improvement is a different arc.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/BuildEscalateData.swift` | new — labels `.atraj` rollouts using the existing cloud-escalate path |
| `Sources/TinyGPT/EvalEscalate.swift` | new — eval harness |
| `Sources/TinyGPTServe/AgentLoop.swift` | recognize the trained `defer_to_cloud` field; fall back to regex only when missing |
| `docs/recipes/cloud-escalate-train.md` | new — recipe |
| `docs/PLAN.md` | B5 ⬜ → ✅ on ship |

## Acceptance criteria

- [ ] Labeling pass on 1K rollouts produces a balanced (positive +
  negative) dataset.
- [ ] Trained specialist + escalation head reduces over-escalation
  (false defer) to < 10% on a held-out 200-rollout eval, while
  catching ≥ 70% of true defer cases.
- [ ] Runtime: `tinygpt agent --cloud-escalate` honors the trained
  signal first, regex second. Token-level test confirms.

## Reference patterns

- `Sources/TinyGPTServe/CloudEscalate.swift` — the existing
  runtime path.
- `factory-pace-planner-v6_1.md` — the response-schema pattern;
  add the new field there.
- B22 trajectory format — input substrate.

## Open questions

- Whether to add the `reason` field as a generation target or just
  a control flag. **Recommendation:** generate it; helps debug
  + provides interpretability.
