---
name: B21 Micro-AutoMixer for specialist data mixes
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B21)
related_prds: B22-trajectory-recorder.md, B23-agent-eval-protocol.md (Poolside-discipline sibling PRDs),
              factory-synthesize.md (synthetic data pipeline)
---

# PRD — Micro-AutoMixer for specialist pretrain ratios

## Goal

Replace the current "hand-wave a 50/30/20 code/web/math split" pattern
with a small AutoMixer-style ratio search. Train ~6–12 short proxy
runs across candidate ratios, score them on a fixed capability suite
(BFCL + GSM8K + HumanEval + Pace unhappy-paths), fit a simple
quadratic surrogate, propose the next mix. Stop when proposed gains
fall below a threshold.

Scaled down from Poolside's Laguna recipe ([Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive))
to a single Mac.

## Why now

- A1 specialist training is the upcoming work; current ratios are
  guesses. Even a 3% absolute improvement on BFCL from a better mix
  is bigger than most of our other Tier-B knobs.
- Each proxy run is small (~30 min on a 22M model, ~3 hours on the
  flagship-class). The full search is ~12 × 30 min = 6 hours wall-
  clock for the small-proxy regime — overnight runs return a ratio
  recipe.
- The eval surfaces all exist (BFCL, τ-bench wired in E1/E2). The
  search loop and the surrogate fit are the new pieces.

## Scope — in

- `Sources/TinyGPT/AutoMix.swift` — orchestrator. Inputs:
  - A list of corpora (`--corpus name1=path1.txt --corpus
    name2=path2.txt ...`) — these are the variables to ratio.
  - A list of eval tasks (`--task bfcl --task gsm8k ...`) — uses the
    existing `tinygpt run-lm-eval` + `tinygpt eval-bfcl` paths.
  - Search budget: `--proxy-runs N --proxy-steps S` (e.g. 8 × 2000
    steps).
- Search loop:
  1. Sample N initial mixes (Dirichlet over corpora) including the
     "uniform" anchor.
  2. Train each proxy run.
  3. Score on the tasks.
  4. Fit a quadratic surrogate on (ratio, score).
  5. Propose the next mix to evaluate (Expected Improvement over the
     current best).
  6. Stop when EI < threshold or budget exhausted.
- Output: `automix-report.jsonl` with one row per proxy run + a
  final `automix-recommendation.json` carrying the best ratio +
  predicted full-scale lift.

## Scope — out

- **Full-scale AutoMix** (replacing the small proxy with a near-full
  run). Defer — proxy-to-full transfer is the open research question.
  V1 uses small proxies and trusts the small/full correlation;
  documented as a caveat.
- **Multi-objective fronts** (Pareto across speed vs accuracy).
  V1 uses a fixed weighted sum.
- **In-the-loop AutoMix** (mix changes during a single run). Way
  bigger scope; defer.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/AutoMix.swift` | new — orchestrator |
| `Sources/TinyGPT/TinyGPT.swift` | `case "automix"` |
| `Sources/TinyGPTModel/MixSampler.swift` | new — Dirichlet sampler over corpora |
| `Sources/TinyGPTModel/SurrogateFit.swift` | new — quadratic fit + EI proposer |
| `evals/automix-smoke.sh` | new — 3 corpora × 4 proxy runs × 200 steps on tiny model; assert final recommendation is sensible |
| `docs/recipes/automix.md` | new — recipe + worked example |
| `docs/PLAN.md` | flip B21 ⬜ → ✅ when shipped |

## Don't touch

- `tinygpt train` itself — automix orchestrates `tinygpt train`
  subprocesses. No coupling.

## Acceptance criteria

- [ ] `tinygpt automix --corpus code=stack.txt --corpus math=meta.txt
  --corpus web=fineweb.txt --task bfcl --proxy-runs 6 --proxy-steps
  2000 --out auto.jsonl` runs end-to-end on the M5 Pro.
- [ ] The surrogate's EI estimate at step N+1 correlates positively
  with the actual measured improvement (a sanity check that the fit
  isn't degenerate).
- [ ] `automix-smoke.sh` passes in CI on tiny corpora.
- [ ] `docs/recipes/automix.md` walks through interpreting an
  `automix-report.jsonl`.

## Reference patterns

- `Sources/TinyGPT/RunLmEval.swift` — the post-checkpoint
  subprocess-orchestration pattern. Same shape: spawn child, parse
  JSONL, aggregate.
- [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive) —
  the automixing-before-pretrain motivation. Cite, don't redocument.
- [DoReMi, Xie et al. 2023](https://arxiv.org/abs/2305.10429) — the
  formal mixture-optimization treatment. Our V1 uses the simpler EI/
  quadratic surrogate; cite as the rigor target for V2.

## Open questions

- Whether to use Bayesian optimization (Gaussian process) over the
  quadratic surrogate for the EI step. **Recommendation:** quadratic
  for V1 — 200 LOC; GP is 500+ and a dependency.
- Proxy-to-full transfer assumption. **Recommendation:** document as
  a caveat; cite Poolside's claim that 10x-smaller proxies retain
  rank-correlation on capability evals.
