---
name: E2 τ-bench eval wrapper
status: shipped-2026-06-05
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier E (E2)
task_tracker: #232
---

# PRD — `tinygpt eval-tau-bench`

## Goal

Ship `tinygpt eval-tau-bench <model.tinygpt>` — a Swift subcommand that
runs the τ-bench multi-turn-agent harness against a TinyGPT model and
emits results in the shared E0 `EvalCompare.Row` schema.

## Why now

- τ-bench is the **multi-turn** complement to E1 BFCL's per-call
  function-calling evaluation. A tool-caller specialist (A1) needs
  both: BFCL says "can it pick the right function?" and τ-bench says
  "can it complete a multi-turn workflow that calls tools in sequence?"
- Harness source already on disk at
  `~/.cache/tinygpt/datasets/_external/tau-bench/`. `run.py` is the
  CLI entrypoint.
- Pattern is the same as E1 BFCL — once that lands, this is a sibling
  copy with the harness invocation swapped.

## Scope — in

- New file `Sources/TinyGPT/EvalTauBench.swift`
- Add `case "eval-tau-bench"` to dispatch — agent submits diff line, I merge
- Spawn `tinygpt-cli serve` in background (same pattern as
  `RunLmEval.runViaServe`)
- Spawn `python run.py --model <our model name> --env <task-env> --max-steps N`
  against the serve endpoint (τ-bench drives an OpenAI-compatible client)
- Parse τ-bench's per-task pass@1 / pass^k output
- Emit one `EvalCompare.Row` per (env, metric) — `task = "tau-bench"`,
  `subtask = <env>` (e.g., `"retail"`, `"airline"`)

## Scope — out (v2)

- Pass^k variance reporting (need multiple seeds; v1 = pass@1 single run)
- Custom user-simulator config (τ-bench has GPT-4-driven users by default;
  v1 hardcodes their default user model, fed via env vars)
- Sandbox for the tool side (τ-bench's task envs are stateful Python
  classes — v1 trusts them; not user input)
- Parallel multi-env runs

## Inputs the agent has

| Resource | Location |
|---|---|
| τ-bench source | `~/.cache/tinygpt/datasets/_external/tau-bench/` (read `README.md` + `run.py` first) |
| Pattern to copy | `Sources/TinyGPT/RunLmEval.swift` → `runViaServe` AND the just-shipped `Sources/TinyGPT/EvalBFCL.swift` (E1) — if E1 lands first, copy from it |
| E0 schema | `Sources/TinyGPT/EvalCompare.swift` |
| Smoke target | `/tmp/huge-smoke-30min.tinygpt` |
| User-simulator API key | τ-bench needs `OPENAI_API_KEY` for the user side. v1 docs that limitation; v2 swaps in a local-model user via E7 judge |

## Acceptance criteria

1. `tinygpt eval-tau-bench --help` prints clean usage
2. End-to-end smoke runs without crashing:
   ```
   tinygpt eval-tau-bench /tmp/huge-smoke-30min.tinygpt \
     --tokenizer <SmolLM2 dir> \
     --env retail --limit 3 --serve-port 8100 \
     --out /tmp/tau-bench-smoke.jsonl
   ```
3. Output JSONL has rows with `task="tau-bench"` and `subtask="retail"`,
   `metric="pass_at_1"`, `score=<0..1>`
4. `tinygpt eval-compare /tmp/tau-bench-smoke.jsonl --by task` renders it
5. Build passes

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/EvalTauBench.swift` |
| **read** | `RunLmEval.swift`, `EvalBFCL.swift` (when it lands), `EvalCompare.swift` |
| **don't touch** | `TinyGPT.swift` (submit dispatch diff in PR), `PLAN.md`, `HANDOFF.md` |

## Estimated effort

**~1 day** (slightly less if E1 BFCL ships first — same pattern, second time).

## Coordination

PR must include the dispatch-line diff for `TinyGPT.swift`, the
specific τ-bench CLI invocation used, the smoke JSONL output, and
build confirmation.

## Known risks

- **τ-bench expects `OPENAI_API_KEY`** for the user-simulator side
  (default user is GPT-4). Document this; v1 should fail loudly if
  the env var is missing.
- **τ-bench envs are stateful**: a "retail" episode mutates a sqlite
  DB. Make sure the harness's working directory is isolated per run
  so concurrent invocations don't trash each other.
- Same long-prompt risk as E1: τ-bench's system prompts are long;
  pass `--max-context 4096` to serve when needed.

## Source links

- τ-bench paper + repo: https://github.com/sierra-research/tau-bench
- Local snapshot: `~/.cache/tinygpt/datasets/_external/tau-bench/`
- Existing pattern: `native-mac/Sources/TinyGPT/RunLmEval.swift`
