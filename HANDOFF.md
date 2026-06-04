# Session handoff — pick this up cleanly

A fresh-context agent should read this first, then `docs/PLAN.md`.

## Snapshot (top of session-end 2026-06-04 PM)

**Branch:** `main` · **Working tree:** clean save for `.claude/` (session state) and `default.profraw` (test artifact, gitignore-eligible).

**Most recent commits** (newest first — confirm via `git log --oneline -10`):

```
<sha> train: --seed flag for deterministic model init (C9 v1)
<sha> browser: training-run dashboard viewer (C10 frontend)
<sha> deps: bump mlx-swift 0.31.3 → 0.31.4 (B16)
<sha> train: WSD scheduler + --depth + spike detector + JSONL log
<sha> docs: PLAN.md — roadmap + 2026-06-04 competitor sweep
```

**Nothing is running.** No background `tinygpt train` process, no pending shell jobs. `pgrep -fl 'tinygpt train'` should return nothing.

## What just shipped (this session)

In commit order, oldest → newest:

1. **PLAN.md sweep** — pretrain + runtime quality lane (B10–B20, C9, C10) + 2026-06-04 competitor sweep (nanochat, mlx-lm, Ollama+MLX, EXO, SAELens, Apple M5 NA, modded-nanogpt, Group-SAE). Docs only.
2. **B11 WSD scheduler** — `--lr-schedule wsd` + `--decay-steps N`. MiniCPM/SmolLM-style warmup-stable-decay. Decay phase doubles as annealing window.
3. **B18 nanochat `--depth N`** — single-knob HP override. Sets nLayers=N, dModel=64·N, nHeads=N (nKvHeads=N — full multi-head, no GQA), dMlp=4·dModel. Preset still supplies ctx/vocab/dtype.
4. **B12 v1 loss-spike detector** — observe-only; logs `[spike]` to stderr when loss > F × moving-average. Flags: `--no-spike-detect`, `--spike-window N`, `--spike-factor F`. v2 (auto-rollback) deferred.
5. **B20 cross-stream attention** — research only, folded into PLAN.md §4. Verdict: speedrun-empirical, modest 5ms gain, not worth porting at our scale.
6. **B16 mlx-swift 0.31.3 → 0.31.4** — bumped, bench at small preset showed no signal (workload too small to be compute-bound). Real M5 NA verification deferred until a Mega/Huge checkpoint exists.
7. **C10 training-run dashboard** — `--log-jsonl <path>` appends JSON-lines stream (meta/step/val/done events). Viewer at `/training-dashboard.astro` (zero-dep, multi-run overlay, drag-drop).
8. **C9 v1 determinism** — `--seed UInt64` seeds MLXRandom before model construction. Init reproducible; batch sampling still uses non-seedable `Int.random` (v2 follow-up). See `docs/determinism.md`.

**Files touched this session** (use `git diff 5143366..HEAD --stat`):

- New: `Sources/TinyGPT/TrainLog.swift`, `Sources/TinyGPTModel/TrainSchedHelpers.swift`, `Tests/TinyGPTModelTests/TrainSchedHelpersTests.swift`, `browser/src/pages/training-dashboard.astro`, `docs/determinism.md`
- Modified: `Sources/TinyGPT/Train.swift` (many flags + plumbing), `Sources/TinyGPT/TrainSupport.swift` (refactor: helpers moved to TinyGPTModel), `Package.swift` (mlx-swift bump), `docs/PLAN.md`

## Tests

The durable env limitation still applies: **`swift test` doesn't work outside Xcode** (no Testing.framework + no MLX default.metallib in SPM build). All tests written this session compile but aren't run.

16 new unit tests added in `Tests/TinyGPTModelTests/TrainSchedHelpersTests.swift` cover the WSD scheduler math (warmup, stable, 1−√(t) decay, endpoints, clamps) and the loss-spike detector (warmup never-fires, threshold, debounce, NaN safety). Run them via Xcode (Product → Test) before merging anything that touches the helpers.

`swift build -c release` is the build verifier in this env. All commits this session pass.

## Where to pick up

`docs/PLAN.md` §3 is the source of truth for what's next. The current pending TODO items, ranked roughly by ROI:

### Tier 1 — no training required (ship today)

| Task | Effort | Files to touch |
|---|---|---|
| **B13 Interp-on-checkpoints infra** | ~half-day | New multi-checkpoint loader for `tinygpt sae`, `tinygpt memit`, `tinygpt patch`. Save-every (already exists) + a glob loader. |
| **B17 SAELens / Neuronpedia format export** | ~2 days | Read our SAE artifact → write SAELens-compatible safetensors + config. Look at `decoderesearch/SAELens` for the schema. |
| **B19 Group-SAE** | ~2-3 days | Layer-group variant of existing SAE trainer. See arxiv 2410.21508. |
| **B10 Quality classifier** | ~2 days build + ~1 hr classifier training | FineWeb-Edu-style fastText scorer; tiny scaffold first, the corpus filter use is a separate ~days job. |

### Tier 2 — light training (~30 min to a few hrs)

| Task | Training cost | Notes |
|---|---|---|
| **B14 Speculative decoding** infra | 0 to build | Algorithm in inference path. Needs two trained models — tiny + small smoke is fine for a demo. |
| **B13 demo** (after Tier 1 infra) | ~10 min training (huge, 500 steps, save-every 50) | Produces 10 checkpoints to run SAE/MEMIT/patch across. |

### Tier 3 — serious training (the A-track gate)

- **A2-A6 dataset pulls** (~3-4 hrs wall, mostly network). Then **A1 specialist** (3-5 days wall) unblocks #193 ANE experiment, B6 Mac app, B7 routing. See PLAN.md §3 Tier A.

### Deferred

- **B12 v2 auto-rollback** — needs full Adam-state persistence first (currently restart-only on `--resume`)
- **B15 layer-wise LR for SFT** — pretrain has it; SFT uses a different optimizer path (LoRA-tagged params); LoRA-aware port is bigger than half-day
- **B16 v2 M5 NA verification** — meaningful only with a Mega/Huge checkpoint
- **C9 v2 full bit-exact replay** — replace stdlib `Int.random` in `sampleBatchRaw` with a seeded host RNG (see `docs/determinism.md` for the plan)

## How to verify nothing's broken before claiming a phase done

1. `cd native-mac && swift build -c release` — must complete with `ok (build complete)`
2. **For training paths**: run a smoke train, e.g.
   ```bash
   ./native-mac/.build/arm64-apple-macosx/release/tinygpt train \
     --preset small --steps 30 --warmup 5 --lr-schedule wsd --decay-steps 10 \
     --val-split 0.1 --val-every 10 --seed 42 \
     --corpus data/examples/tiny-corpus.txt --out /tmp/smoke.tinygpt \
     --log-jsonl /tmp/smoke.jsonl --sample-every 100
   ```
   Should finish in <5 s, write a valid `.tinygpt`, and emit a 35-ish-line JSONL.
3. **For browser changes**: `cd browser && npm run build` — must complete with `[build] Complete!`. Then `npm run dev` and visually check the affected page.
4. **For new helpers in TinyGPTModel**: add an XCTest to `Tests/TinyGPTModelTests/`. It won't run here (Xcode-only) but compiles into the test target on a developer machine.

## Files NOT to touch without asking

- Any trained checkpoint: `/tmp/*.tinygpt`, `data/gallery/*.tinygpt`, `browser/public/gallery/**/*.bin`
- Tokenizers under `/tmp/smollm2/`, `/tmp/*-tokenized/`
- Secrets / env files / cloud configs (per global CLAUDE.md)
- `Package.resolved` — let SPM regenerate it on `swift package update`
- `browser/src/content/docs/` — regenerated from `docs/**/*.md` by `browser/scripts/copy_docs.mjs` at build time

## User constraints (from CLAUDE.md + auto-memory)

- Ask before install / migration / network-heavy commands
- Prefer small reviewable diffs
- Never use `--no-verify` on git hooks
- Don't edit secrets / env files / cloud configs
- Solo dev, budget-constrained — Tinker / cloud-training off the table
- **No quality regression** — every perf path needs an automated numerics gate
- **Opportunistic edge** — best perf for latest-Chrome users, graceful degradation
- **PLAN.md is canonical** — older planning docs stub-link to it
- **No launch optics** — recent guidance: "there is no launch; just a good product." Optimize for users-of-the-thing, not narrative.

## Where to read more

- `docs/PLAN.md` — canonical roadmap (§1 shipped, §2 skipped, §3 TODO, §4 research catalogue, §5 appendix)
- `docs/MAP.md` — old-path → new-path index, canonical-home list for shared concepts (LoRA, MoE, quantization, etc.)
- `docs/determinism.md` — C9 contract + v2 roadmap (new this session)
- `docs/training/{pretrain,sft,dpo}.md` — phase-specific guides
- `docs/interpretability.md` — SAE / MEMIT / patch / logit lens canonical home
- `CLAUDE.md` and `AGENTS.md` at project root — global agent instructions

Good luck. Read PLAN.md before doing anything substantive; this file just gets you oriented.
