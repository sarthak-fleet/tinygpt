---
title: Eval gate (CI / pre-commit)
description: Gate a TinyGPT specialist in CI — `tinygpt eval-gate` exits non-zero when any declared eval suite regresses past threshold, on a self-hosted Mac runner so the model never leaves the device.
---

# Recipe — `tinygpt eval-gate` as a CI / pre-commit gate

`tinygpt eval-gate` runs your project's declared eval suites against a
stored baseline and **exits non-zero when any suite regresses past a
threshold**. It's the developer-workflow framing of the shipped eval
harnesses (E1 BFCL, E2 τ-bench, E3 lm-eval, E5 HumanEval, the unhappy-path
suite): a benchmark wrapper becomes a gate you can put in front of a merge.

Everything runs on *your* runner — the gate never phones home.

## The spec

The gate reads an `eval-gate.json` in the cwd, an `eval` block in
`tinygpt.project.json` (B31), or a path you pass with `--spec`:

```json
{
  "baseline": "evals/baseline.jsonl",
  "default_threshold": 2.0,
  "suites": [
    { "name": "bfcl", "task": "bfcl",
      "command": ["tinygpt", "eval-bfcl", "model.tinygpt", "--out", "$TINYGPT_EVAL_OUT"] },
    { "name": "tau",  "task": "tau", "threshold": 3.0,
      "command": ["tinygpt", "eval-tau-bench", "model.tinygpt", "--out", "$TINYGPT_EVAL_OUT"] }
  ]
}
```

- `baseline` — a JSONL of `EvalCompare.Row`s (the shared schema every
  `tinygpt eval-*` command emits). Generate it once with `--update-baseline`.
- `default_threshold` — max allowed regression in **percentage points**
  (default 2.0). `threshold` on a suite overrides it.
- `command` — the argv to produce candidate rows; the gate sets
  `TINYGPT_EVAL_OUT` to the JSONL the suite should write. Omit `command`
  and pass `--candidate <jsonl>` to gate a run you already have (the
  no-GPU path used in tests).

Direction is inferred from the metric name: accuracy / pass@1 / f1 are
higher-is-better; ppl / loss / latency_ms are lower-is-better and invert
automatically.

## First run — stamp the baseline

```bash
tinygpt eval-gate --update-baseline      # runs the suites, writes baseline.jsonl, exits 0
```

Re-run `--update-baseline` whenever you *intentionally* move the numbers
(the "accept the new scores" path).

## Gate a change

```bash
tinygpt eval-gate                 # exit 0 = all suites within threshold; 1 = a regression
tinygpt eval-gate --passes 3      # run each suite 3× and gate on the mean (noise guard)
```

It prints a per-suite table and writes `gate-result.json` (machine-readable).

## GitHub Action

The gate runs the MLX path, so it needs a **self-hosted Apple-silicon
runner** — GitHub-hosted Linux runners can't run MLX.

```yaml
# .github/workflows/eval-gate.yml
name: eval-gate
on: pull_request
jobs:
  gate:
    runs-on: [self-hosted, macOS, ARM64]
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/tinygpt-eval-gate
        with:
          spec: tinygpt.project.json
          passes: "3"
```

The action builds `tinygpt` release, runs the gate, annotates the PR with
the suite table in the job summary, and fails the check on a regression.

## Pre-commit hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tinygpt-eval-gate
        name: tinygpt eval-gate
        entry: tinygpt eval-gate
        language: system
        pass_filenames: false
        stages: [pre-push]   # too slow for every commit; gate on push
```

Use `pre-push` (not `pre-commit`) — running real suites per commit is too
slow. For a fast local guard, point the hook at `--candidate` with a cached
JSONL and reserve the full run for CI.

## Verify

```bash
bash evals/eval-gate-smoke.sh    # asserts exit 0 (match) + exit 1 (regression) with fixtures
```

## See also

- `docs/prds/B32-eval-ci-gate.md` — the PRD + scope boundaries.
- `docs/prds/E1-bfcl-eval.md`, `E2-tau-bench-eval.md` — the harnesses this gates.
- `docs/sessions/2026-06-13-market-landscape-mac-first.md` — why a local,
  exit-code gate is the structural counter to eval-as-a-SaaS.
