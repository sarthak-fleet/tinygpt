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

### Specialists (Tier A integration + Tier B follow-ons)

| PRD | What | Effort | Blocks / blocked-by |
|---|---|---|---|
| [A1 first-specialist-tool-caller](A1-first-specialist-tool-caller.md) | qwen3-4b + LoRA tool-calling specialist; +3pp BFCL ship gate | 3-5d | north-star validator; consumes E1/E8 + B23 |
| [B25 scaledown-specialist](B25-scaledown-specialist.md) | extractive context-compression specialist; submit to ScaleDown leaderboard | 3-5d | needs E6 harness |

### Training-quality + optimizer (Tier B)

| PRD | What | Effort |
|---|---|---|
| [B10 quality-classifier](B10-quality-classifier.md) | FineWeb-Edu-style scorer + corpus filter | ~2d |
| [B11 wsd-schedule](B11-wsd-schedule.md) | warmup-stable-decay LR; replaces cosine default | ~half-day |
| [B14 speculative-decoding](B14-speculative-decoding.md) | Mini-Llama draft → Mega target; T=0 byte-equality gate | ~2-3d |
| [B15 layerwise-lr-decay-sft](B15-layerwise-lr-decay-sft.md) | `--llrd γ` on sft/dpo/finetune; γ^k depth decay | ~half-day |
| [B21 micro-automixer](B21-micro-automixer.md) | Dirichlet+EI ratio search before specialist training | ~2-3d |

### Interpretability (Tier B)

| PRD | What | Effort |
|---|---|---|
| [B13 interp-on-checkpoints](B13-interp-on-checkpoints.md) | `tinygpt interp-replay` across a history dir → timeline JSONL | ~1-2d |
| [B17 saelens-interop](B17-saelens-interop.md) | one-way `.sae` → SAELens/Neuronpedia exporter | ~2d |
| [B19 group-sae](B19-group-sae.md) | one SAE per layer-group; ~4× cheaper training | ~2-3d |

### Agent protocol (Tier B — Poolside discipline)

| PRD | What | Effort |
|---|---|---|
| [B22 trajectory-recorder](B22-trajectory-recorder.md) | `.atraj` files preserve input_ids/output_ids/rewards | ~2d |
| [B23 agent-eval-protocol](B23-agent-eval-protocol.md) | repeated pass@1 with fixed budgets; mean ± σ + ci95 | ~1d |
| [B26 deferred-tools](B26-deferred-tools.md) | `--tool-mode {full,deferred}` + `get_tool_info` meta-tool | shipped, BFCL gate pending |

### Polish + harness (Tier C)

| PRD | What | Effort |
|---|---|---|
| [C3 dora-ondisk-format](C3-dora-ondisk-format.md) | persist trained DoRA adapters (closes the gap with LoRA roundtrip) | ~1d |
| [C4 tool-extractor-bpe](C4-tool-extractor-bpe.md) | BPE tokenizer path for the mini-router trainer | ~2d |
| [C5 decode-jitter-thermal](C5-decode-jitter-thermal.md) | 30-min sustained decode bench + powermetrics sidecar | ~1d |
| [C9 determinism-harness](C9-determinism-harness.md) | bit-exact replay of step N (uses Adam-state-persist) | ~2d |
| [C10 train-run-dashboard](C10-train-run-dashboard.md) | `/train-viewer.astro` drag-drop live charts | ~1d |

### Tier 5 — research frontier

| PRD | What | Order |
|---|---|---|
| [5.1 reasoning-on-22M](5.1-reasoning-on-22M.md) | GRPO/DAPO at 22M; publishable negative-result-shaped artifact | 5-7d |
| [5.2 testtime-compute-scaling](5.2-testtime-compute-scaling.md) | Snell quality-vs-FLOPs curve at 22M (+ stretch cross-size) | 3-5d |
| [5.3 vision-language-toy](5.3-vision-language-toy.md) | LLaVA-style from-scratch VL on consumer hardware | ~2w |
| [5.5 sparse-moe-kernels](5.5-sparse-moe-kernels.md) | Metal kernels for hard MoE routing (blocked upstream) | 2-3w when unblocked |
| [5.6 tts-toy](5.6-tts-toy.md) | EnCodec + autoregressive over codebook IDs (after 5.3) | 2-4w |
| [5.7 explainer-video-model](5.7-explainer-video-model.md) | structured DSL + renderer; visual-planner specialist (after A1-B8 + 5.3) | 3-6w |

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
