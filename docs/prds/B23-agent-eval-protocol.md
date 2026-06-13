---
name: B23 agent eval protocol hardening
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B23)
related_prds: B22-trajectory-recorder.md, B21-micro-automixer.md (Poolside-discipline siblings),
              E1-bfcl-eval.md, E2-tau-bench-eval.md (the harnesses this PRD hardens)
---

# PRD — Repeated-pass@1 + fixed resource budgets for agent evals

## Goal

Replace single-shot agent eval runs with the Poolside-class protocol:
each task is run K times under fixed budget constraints, results are
averaged with confidence intervals, and the exact resource envelope
(max steps, sandbox CPU/RAM, sampling params, model temperature) is
logged with every score. Applies to BFCL, τ-bench, the Pace unhappy-
paths suite, and any future SWE-mini / Terminal-mini benchmarks.

The result the leaderboard reports stops being "the model got 47/60
on OOS this run" and becomes "the model gets 47.3 ± 2.1/60 on OOS
under (max_steps=8, T=0, sandbox=1cpu+512mb)".

## Why now

- E9 just showed how noisy n=130 evals are — dims at n=40 carry ~±15pp
  CI. Single-run numbers in the SLM leaderboard are misleading, and
  cross-model comparisons amplify the noise.
- The leaderboard v0 just shipped (B27 in PLAN.md); rigor on the
  protocol is the difference between "internally interesting" and
  "publishable / linkable" outputs.
- Cheap to add — each harness already has a per-sample loop; the
  change is wrapping it in a per-task repeat loop with seed control
  and a budget config.

## Scope — in

- `Sources/TinyGPTModel/AgentEvalProtocol.swift` — shared types:
  ```swift
  struct AgentEvalBudget {
      let max_steps: Int
      let sandbox_cpus: Double
      let sandbox_ram_mb: Int
      let temperature: Float
      let top_p: Float
      let sampling_seed: Int  // base seed; pass+i is the per-pass seed
      let infra_patches: [String]  // e.g. "fix for bfcl#127 applied"
  }
  struct AgentEvalRunSummary {
      let task: String
      let model_fingerprint: String
      let budget: AgentEvalBudget
      let k: Int                         // number of pass@1 trials
      let per_trial_scores: [Double]
      let mean: Double
      let stdev: Double
      let ci95: (low: Double, high: Double)
  }
  ```
- Add `--passes K --budget budget.json` to `tinygpt eval-bfcl`,
  `tinygpt eval-tau-bench`, `scripts/eval_pace_unhappy.py`.
- Output JSON gains the `protocol` block carrying the budget + the
  per-trial scores. Existing `passed`/`total` stays for back-compat.
- `eval-compare` learns to render error bars (use ±1.96σ if `ci95`
  present).
- Default K=1 with budget loaded from a sensible per-task default
  (so the old `tinygpt eval-bfcl` invocation works the same; opt
  into repeats via `--passes 3`).

## Scope — out

- **Sandbox enforcement** (actually capping CPU/RAM in the rust
  sandbox). The budget recorded here is *declarative* — the eval
  scripts pass the right flags into the existing sandbox-runner (see
  E5 HumanEval sandbox PRD). Hard enforcement is its own infra PRD.
- **Per-task default budgets baked into a config file**. Defer; this
  PRD ships a CLI flag, callers can build their own config.
- **Automatic flake detection** (find tasks where K=10 trials swing
  >20%). Useful follow-up; defer.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/AgentEvalProtocol.swift` | new — shared types |
| `Sources/TinyGPT/EvalBFCL.swift` | `--passes K --budget` wiring |
| `Sources/TinyGPT/EvalTauBench.swift` | same |
| `scripts/eval_pace_unhappy.py` | `--passes K`; aggregate per-trial via mean ± stdev |
| `Sources/TinyGPT/EvalCompare.swift` | render ci95 as `score ± σ` columns |
| `evals/sample-budget.json` | new — fixture budget config |
| `docs/research/mac_slm_leaderboard_v0.md` | "Protocol" subsection citing the budget format |

## Don't touch

- The harness scoring logic itself — we wrap, not rewrite.

## Acceptance criteria

- [ ] `tinygpt eval-bfcl <model> --passes 3 --budget evals/sample-budget.json`
  runs 3 passes per BFCL category and emits the `AgentEvalRunSummary`
  shape.
- [ ] Default K=1 with default budget reproduces today's exact output
  format (back-compat).
- [ ] `tinygpt eval-compare` renders error bars when `ci95` is present.
- [ ] The SLM leaderboard page updates its column headers to reflect
  the per-task pass count.
- [ ] Pace unhappy-paths gains `--passes` and the runbook in
  `docs/recipes/eval_planner.md` (if it exists; create if not) walks
  through the K=3 protocol.

## Reference patterns

- `Sources/TinyGPT/RunLmEval.swift` — already loops over tasks; the
  outer per-pass loop sits one level above that.
- [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive)
  — the protocol-hardening rationale (their pass@1-with-fixed-budget
  is the discipline this PRD copies).
- Pace planner-champion JSON already records `date` per result —
  add the budget block alongside.

## Open questions

- Default K. **Recommendation:** K=1 for back-compat; the user-facing
  recipe in `docs/recipes/` advocates K=3. CI runs at K=1 to keep CI
  fast; manual publication runs at K=3+.
- Whether to fold the existing planner-champion JSON into the new
  schema or keep a separate "leaderboard champion" format.
  **Recommendation:** evolve in place — add the budget block, keep
  the existing fields. Migration is a one-time tool.
