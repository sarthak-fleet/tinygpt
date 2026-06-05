# docs/prds/ — Product Requirement Briefs

Self-contained briefs for parallel-agent tasks. Each PRD is written
so an agent can pick it up cold, ship the task, and submit a PR without
loading the full session context.

## How to use

If you're a fresh agent: pick a `status: not-started` PRD, set the
`owner:` field on your branch, work the acceptance criteria, submit
a PR with the deliverables the PRD asks for.

**Coordination rule**: every PRD names a small set of "don't touch"
files. The maintainer merges those changes (typically a switch-case
in `Sources/TinyGPT/TinyGPT.swift`) after reviewing your PR.

## Index

### Eval pipelines (Tier E)

| PRD | What | Effort | Blocks |
|---|---|---|---|
| [E1 BFCL](E1-bfcl-eval.md) | wrap BFCL function-calling harness | ~1d | A1 specialist scoring |
| [E2 τ-bench](E2-tau-bench-eval.md) | wrap τ-bench multi-turn agent harness | ~1d | A1 multi-turn score |
| [E5 HumanEval + sandbox](E5-humaneval-sandbox.md) | Rust-isolated Python sandbox + scorer | ~1-2d | code specialist eval |
| [E7 Local judge](E7-judge-shim.md) | LLM-as-judge via local Qwen/SmolLM | ~1d | preference evals (AlpacaEval, MT-Bench) |
| [E8 train-time hook](E8-train-time-eval-hook.md) | `--eval-every N` flag for emergence view | ~1d | training-dynamics dashboard |

### Browser viewers

| PRD | What | Effort |
|---|---|---|
| [Eval leaderboard](eval-leaderboard-viewer.md) | `/eval-leaderboard.astro` — 3-view comparison page | ~2-3d |
| [SAE timeline](sae-timeline-viewer.md) | `/sae-timeline.astro` — B13 feature-emergence chart | ~1d |

### Rust performance tools

| PRD | What | Effort |
|---|---|---|
| [Parquet decoder](rust-parquet-decoder.md) | replace pyarrow with a 2 MB static binary | ~half-day |
| [HF downloader](rust-hf-downloader.md) | parallel shard fetches with progress + retry + resume | ~1d |

## File-touching protocol

Every PRD's **"don't touch"** section names files that, if multiple
agents edited in parallel, would conflict. The canonical list:

- `Sources/TinyGPT/TinyGPT.swift` — dispatch table (one new
  case per E* task; agents submit the diff line, maintainer merges)
- `docs/PLAN.md` — single source of truth for status; maintainer
  updates after each merge
- `HANDOFF.md` — single source of truth for next-session pickup
- `Package.swift` — only the maintainer adds new targets
- `Package.resolved` — auto-generated; never hand-edit

Everything else (new files, new tests, new scripts, new browser pages)
is fair game per the per-PRD scope.

## When to write a new PRD

- A task is independent enough that another agent can ship it without
  reading the current session
- The scope is well-bounded (single feature, single file or small set)
- Acceptance criteria can be stated in <10 bullets
- There's a concrete pattern in the repo the agent can copy

Don't write a PRD for:

- Exploratory research / "figure out the right approach"
- Anything requiring tight coordination across multiple files
- Bugfixes — those are too small / the fix is usually obvious from the
  symptom
