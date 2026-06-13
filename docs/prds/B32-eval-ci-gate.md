---
name: B32 `tinygpt eval` as a CI / pre-commit gate
status: scaffolding-shipped
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B32)
parent_strategy: docs/sessions/2026-06-13-market-landscape-mac-first.md (move #4)
related_prds: E1-bfcl-eval.md, E2-tau-bench-eval.md (the shipped harnesses this reframes), B23-agent-eval-protocol.md (the rigor B32 invokes)
---

# PRD — Reframe the eval harnesses as a developer-workflow gate

> **Status (2026-06-13): scaffolding shipped.** Pure gate logic
> (`TinyGPTModel/EvalGate.swift`: direction heuristic, pp thresholds,
> per-suite override, missing-baseline handling, K-pass mean) with 13 unit
> tests in `EvalGateTests.swift`. CLI (`Sources/TinyGPT/EvalGate.swift`):
> `--spec` / `eval-gate.json` / `tinygpt.project.json` `eval` block
> resolution, `--candidate` (no-GPU path), `--baseline`, `--threshold`,
> `--passes`, `--update-baseline`, `gate-result.json`, exit 0/1. Action at
> `.github/actions/tinygpt-eval-gate/`, recipe `docs/recipes/eval-gate.md`,
> smoke `evals/eval-gate-smoke.sh` (asserts both exit codes against committed
> fixtures). **Remaining to flip to ✅:** run a real specialist's suites end
> to end through the gate (the `command`-driven path) on a self-hosted Mac
> runner — exercised by the user, not by CI.

## Goal

Ship `tinygpt eval-gate` — a single command that runs a project's
declared eval suites against a model + adapter, compares to a stored
baseline, and **exits non-zero when any suite regresses past a
threshold**. Plus a GitHub Action wrapper and a pre-commit hook recipe.

The landscape finding (`docs/sessions/2026-06-13-market-landscape-mac-first.md`):
BFCL and τ-bench are academic leaderboards with no product wrapping them
into "gate my SLM in CI." We already wrapped both (E1/E2 shipped). B32 is
the *framing* work that turns a benchmark wrapper into a workflow
primitive — near-zero new model code, high product leverage.

## Why now

- E1 (BFCL), E2 (τ-bench), E3 (lm-eval), E5 (HumanEval), the unhappy-path
  suite, and B27 leaderboard all ship. The infrastructure to *run* evals
  is done; the missing piece is a one-command, exit-code-driven *gate*.
- Competitors sell eval as a SaaS that ingests your traces. A local
  exit-code gate that never phones home is the structural counter — it
  runs in *your* CI on *your* runner.
- Pairs with the B31 project file: the gate reads which models the
  project pins + which suites it declares, so `tinygpt eval-gate` is
  zero-config in a project that already has a `tinygpt.project.json`.

## Scope — in

- `Sources/TinyGPT/EvalGate.swift` — orchestrator:
  - Reads an eval-gate spec (a `[eval]` block in `tinygpt.project.json`,
    or a standalone `eval-gate.json`): which suites, which model/adapter,
    which baseline file, per-suite regression thresholds (default: any
    drop > 2pp fails; configurable per suite).
  - Runs each declared suite via the shipped harness (E1/E2/E3/E5/unhappy).
  - Compares to the baseline (the same `EvalCompare.Row` JSONL format).
  - Prints a human summary + writes a machine `gate-result.json`.
  - **Exit code: 0 if all suites pass their thresholds, 1 otherwise.**
- `--update-baseline` flag — re-stamp the baseline after an intentional
  improvement (the "accept the new numbers" path).
- `--passes K` — delegates to B23 protocol (K-pass mean ± σ) so the gate
  isn't fooled by single-run noise. Default K=1 for CI speed; recipe
  recommends K=3 for release gates.
- GitHub Action: `.github/actions/tinygpt-eval-gate/action.yml` — thin
  wrapper that runs the binary on a self-hosted Mac runner and annotates
  the PR with the pass/fail table.
- Pre-commit recipe: `docs/recipes/eval-gate.md` documents both the
  `.pre-commit-config.yaml` hook and the Action.

## Scope — out

- **Hosted dashboard** for gate history. The leaderboard (B27) +
  train-viewer (C10) cover visualization; the gate just emits JSONL they
  can read.
- **Cloud CI** — the gate is designed for a self-hosted Mac runner
  (that's the point: the model never leaves the device). GitHub-hosted
  Linux runners can't run the MLX path; documented as a constraint.
- **Auto-bisect** (find which commit regressed the eval). Useful
  follow-up; defer.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/EvalGate.swift` | new — orchestrator + exit-code logic |
| `Sources/TinyGPT/TinyGPT.swift` | `case "eval-gate"` |
| `Sources/TinyGPTModel/ProjectManifest.swift` | extend with an optional `[eval]` block (B31 schema addition) |
| `.github/actions/tinygpt-eval-gate/action.yml` | new — Action wrapper |
| `evals/eval-gate-smoke.sh` | new — pass + fail cases assert correct exit codes |
| `docs/recipes/eval-gate.md` | new — pre-commit + Action recipe |
| `docs/PLAN.md` | B32 ⬜ → ✅ on ship |

## Don't touch

- The individual eval harnesses (E1/E2/E3/E5) — the gate calls them as
  library functions, no changes.
- `Sources/TinyGPT/TinyGPT.swift` beyond the one switch case.

## Acceptance criteria

- [ ] `tinygpt eval-gate` in a dir with a valid spec runs the declared
  suites and exits 0 when all pass, 1 when any regresses past threshold.
- [ ] `--update-baseline` re-stamps the baseline JSONL.
- [ ] The GitHub Action annotates a PR with the suite table + pass/fail.
- [ ] `eval-gate-smoke.sh` asserts both exit codes (a deliberately
  regressed fixture fails; a matching fixture passes).
- [ ] The recipe reproduces on a clean checkout with a self-hosted Mac
  runner.

## Reference patterns

- `Sources/TinyGPT/EvalCompare.swift` — the baseline-vs-candidate diff
  logic; the gate adds the threshold + exit-code layer on top.
- `scripts/eval_planner_report.py` — the existing champion-vs-candidate
  verdict pattern (the E9 gate's `PASS/HOLD` logic is the template for
  B32's per-suite verdict).
- B23 agent-eval-protocol — the K-pass rigor the gate invokes.

## Open questions

- Default threshold direction. **Recommendation:** per-suite, default
  "fail if score drops > 2pp from baseline"; suites where lower-is-better
  (PPL, latency) invert automatically via the `EvalCompare` direction
  metadata.
- Whether the gate spec lives in `tinygpt.project.json` or a separate
  `eval-gate.json`. **Recommendation:** an optional block in the project
  file (one config surface), with the standalone file as an override.
