# Session handoff — pick this up cleanly

A fresh-context agent should read this first, then `NIGHTLY.md` for the
training cadence, then `docs/PLAN.md` for the long-term roadmap.

## 🌙 Nightly training cadence

Project shape changed: every night the Mac produces a training artifact.
Run `./scripts/nightly.sh` before bed; it picks the next pending job
from `scripts/nightly/N*.sh`, wraps it in `caffeinate -di`, logs to
`~/.cache/tinygpt/nightly/logs/`, and posts a Mac notification on
completion. See `NIGHTLY.md` for the full plan and queue state.

**State as of 2026-06-05 PM (after daytime audit + parquet unblocking):**
- N01 ✅ done — Gutenberg combined corpus + SmolLM2 tokenizer + hermes-fc
  (50 MB SFT data) verified. N01 is now a *verifier*, not a downloader;
  parquet/HF_TOKEN-gated stuff is tracked under "Known blockers" in
  NIGHTLY.md.
- N02 ⏳ queued — repointed at FineWeb-Edu (241 MB educational text,
  decoded from parquet via the new `scripts/parquet_to_txt.py`). 200K
  steps, ~11 hrs. Smoke-tested at 100 steps: loss 11.4 → 7.4 → still
  decreasing. **Fire `./scripts/nightly.sh` before bed to start it.**
- N03-N05 — not scripted yet; written after N02 produces a base.

**Parquet decoder unlocked**: FineWeb-Edu (50K rows = 241 MB text) and
UltraFeedback (187K rows = 1.2 GB JSONL) both decoded and ready.
`scripts/parquet_to_txt.py` handles arbitrary parquet shards — see its
header docstring. Requires `pyarrow` (pip-installed today).

**Group-SAE v1 (B19)** — `tinygpt sae --layer-group A,B,C` trains one
SAE on the union of residuals from N layers. ~3× fewer SAE trainings
at ~16% higher MSE. Smoke-tested. Combines with --checkpoint-dir for
group-aware timeline. AMAD-driven group selection deferred to v2.

**SAELens export (B17)** — `tinygpt sae-to-saelens <in.sae> --out <dir>`
emits sae_weights.safetensors + cfg.json in the on-disk shape SAELens
and Neuronpedia consume. Provenance preserved in cfg.json metadata
(hook_name, hook_layer, base shapes, group layers for B19 SAEs).
Smoke-verified: safetensors loads in Python.

**Stochastic spec-dec (B14)** — `tinygpt sample --draft <m> --temperature T`
now picks the right algorithm: T==0 → existing greedy step (lossless wrt
argmax); T>0 → new rejection-sampling step (Leviathan 2023 Thm 3.5 —
distributionally lossless wrt sampling from p_target). Smoke-tested with
self-speculation.

**Quality classifier (B10)** — `tinygpt train-quality-classifier` +
`tinygpt quality-filter`. Bag-of-ngrams + logistic regression, pure
Swift no MLX. The FineWeb-Edu architecture, agnostic to labels — user
supplies positive/negative for their quality dimension. .tgfq format
(~262 KB binary). Smoke-tested: 97% accept on coherent text vs 2%
accept on word-shuffled text after 5 epochs on a 1000-doc training
set.

**Mac app — full sprint** (2026-06-05 PM session 2, "decent standard" + "exceptional")

Sample tab:
- Sampler inspector (right rail): temperature + top-K + repetition
  penalty + max-tokens, all persisted via @AppStorage; toggleable
  show/hide
- Completion history: each Generate run appends a card (timestamp,
  prompt, output, sampler recipe, perf, Copy button). **Persisted to
  UserDefaults** — relaunch the app and your last 200 generations
  are still there.
- "+" button in sidebar — `NSOpenPanel` for opening arbitrary
  `.tinygpt`/`.bin` files outside the gallery search paths

Train tab (unique vs LM Studio/Ollama — they don't train):
- LR-schedule picker: cosine / **wsd** / constant (calls the same
  `lrAtWSD` primitive `tinygpt train` uses)
- `--seed` text field (UInt64; blank = random) — `MLXRandom.seed`
  before model construction
- Spike-detector toggle (uses `LossSpikeDetector` from TinyGPTModel)
- Sticky spike-alert banner with the offending step + MA + threshold

Interp tab (unique — no other local-AI Mac app has this):
- Pickers for model + corpus + output sidecar
- Steppers for layer / d_features / steps / batch / ctx
- "Train SAE" runs `tinygpt-cli sae …` as a subprocess; stdout
  streams live into the right pane; MSE + L0 pills update inline
  while the run is in flight
- Reveal-in-Finder on the saved sidecar

App-wide:
- `.app` bundle via `scripts/build_macapp.sh` — proper
  Contents/{MacOS,Resources} layout, Info.plist, ad-hoc codesigned,
  bundled `tinygpt-cli` next to the app binary so Interp works
  regardless of where the .app is installed
- Branded icon via `scripts/make_icon.sh` — pipes
  `browser/public/favicon.svg` through qlmanage → sips → iconutil →
  `native-mac/Resources/TinyGPT.icns`
- Welcome pane with Sample/Train/Fine-tune/Interp three-feature pitch
- Gallery discovery walks `data/gallery/` (added), `browser/public/
  gallery/`, `public/gallery/`, plus the system Application Support
  fallback. Empty-state lists the paths + has a Reload button.
- `File → New` dropped (no document model to instantiate)

App bundle: ~380 MB (mostly the MLX metallib + a CLI binary copy).
Launch: `open build/TinyGPT.app` or `cp -r build/TinyGPT.app /Applications/`.
Both binaries verified by smoke-launching the .app + invoking the
bundled CLI to produce a real `.sae` sidecar (MSE 5.52e-02, L0 34.45%).

## What's deferred (with reasons)

These items need either training to materialize, or non-trivial
infrastructure that wasn't worth blocking the lane on:

- **B12 v2 auto-rollback** — needs MLX-Swift AdamW state persistence
  across save/load. Currently `--resume` restarts Adam (~100-step
  loss warm-up). Without preserved Adam state, "rollback to step N"
  isn't a clean restore. Real work; skip until either MLX upstreams
  state-readable Adam, or someone does a clean manual port.
- **B13 v2 (memit/patch on checkpoints + browser viewer)** — the
  `--checkpoint-dir` pattern from `tinygpt sae` ports mechanically
  to `tinygpt memit` and `tinygpt patch`. ~1 day. The browser
  viewer to plot SAE feature emergence over time is the third
  piece. None of this blocks training; all of it is post-N02
  follow-up work.
- **B15 layer-wise LR for SFT** — pretrain already supports
  `--lr-layer-decay`. SFT uses `makeOptimizer` directly on
  LoRA-tagged params; needs a LoRA-aware layer-block index map.
  Bigger than half-day, so explicitly punted.
- **C9 v2 full bit-exact replay** — current v1 seeds MLXRandom
  (init reproducible); batch sampling still uses Swift stdlib
  `Int.random`. Full coverage needs a seeded host RNG threaded
  through ByteCorpus/TokenizedCorpus.sampleBatchRaw. See
  `docs/determinism.md` for the v2 plan.
- **KV-cache for spec-dec path** — `tinygpt sample --draft` (both
  greedy + new stochastic) bypasses the KV cache. Wiring KV
  through to target's parallel verification forward is a
  separate task.
- **AMAD-driven group selection for B19** — the paper's group-
  picking algorithm (Average Maximum Angular Distance). v1 takes
  hand-specified groups; AMAD auto-grouping is queued.

## Snapshot (top of session-end 2026-06-04 PM)

**Branch:** `main` · **Working tree:** clean save for `.claude/` (session state) and `default.profraw` (test artifact, gitignore-eligible).

**Most recent commits** (newest first — confirm via `git log --oneline -10`):

```
<sha> train + sae: interp-on-checkpoints v1 (B13)
<sha> docs: HANDOFF.md — refresh for 2026-06-04 session-end state
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
9. **B13 v1 interp-on-checkpoints** — `tinygpt train --save-history` writes per-step `<stem>.step-N.tinygpt` copies; `tinygpt sae --checkpoint-dir <dir> --timeline-out <jsonl>` trains an SAE per checkpoint and emits a JSONL timeline. Smoke: tiny preset · 5 ckpts · MSE 0.019→0.013 trajectory + L0 24%→37% — feature emergence visible. v2 = same pattern for memit/patch + browser viewer.

**Files touched this session** (use `git diff 5143366..HEAD --stat`):

- New: `Sources/TinyGPT/TrainLog.swift`, `Sources/TinyGPTModel/TrainSchedHelpers.swift`, `Tests/TinyGPTModelTests/TrainSchedHelpersTests.swift`, `browser/src/pages/training-dashboard.astro`, `docs/determinism.md`
- Modified: `Sources/TinyGPT/Train.swift` (many flags + plumbing), `Sources/TinyGPT/TrainSupport.swift` (refactor: helpers moved to TinyGPTModel), `Package.swift` (mlx-swift bump), `docs/PLAN.md`

## Tests

The durable env limitation still applies: **`swift test` doesn't work outside Xcode** (no Testing.framework + no MLX default.metallib in SPM build). All tests written this session compile but aren't run.

16 new unit tests added in `Tests/TinyGPTModelTests/TrainSchedHelpersTests.swift` cover the WSD scheduler math (warmup, stable, 1−√(t) decay, endpoints, clamps) and the loss-spike detector (warmup never-fires, threshold, debounce, NaN safety). Run them via Xcode (Product → Test) before merging anything that touches the helpers.

`swift build -c release` is the build verifier in this env. All commits this session pass.

## Where to pick up

`docs/PLAN.md` §3 is the source of truth for what's next. The current pending TODO items, ranked roughly by ROI:

### Tier 1 — no training required

All shipped today (2026-06-05). Nothing left on this tier.

| Task | Status |
|---|---|
| **B13 v1 Interp-on-checkpoints** | SHIPPED — `tinygpt train --save-history` + `tinygpt sae --checkpoint-dir` + JSONL timeline. |
| **B17 SAELens / Neuronpedia export** | SHIPPED — `tinygpt sae-to-saelens`. |
| **B19 Group-SAE** | SHIPPED — `tinygpt sae --layer-group A,B,C`. |
| **B10 Quality classifier** | SHIPPED — `tinygpt train-quality-classifier` + `tinygpt quality-filter`. |

### Tier 2 — light training (~30 min to a few hrs)

| Task | Training cost | Notes |
|---|---|---|
| ~~**B14 Speculative decoding**~~ | SHIPPED — greedy was already there; stochastic (T>0) added today via SpeculativeDecode.stepStochastic. |
| ~~B13 demo~~ | DONE — smoke-tested at 5 ckpts. Real-scale demo runs on tonight's N02 base. |
| **B13 v2 — port pattern to memit + patch + browser viewer** | ~1-2 days | Mechanical: `tinygpt memit --checkpoint-dir <dir> --timeline-out <jsonl>`. Then `/sae-timeline.astro` viewer mirrors `/training-dashboard.astro`. Cross-checkpoint feature alignment is the harder follow-up. |

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
