# Session handoff — pick this up cleanly

A fresh-context agent should read this first, then `NIGHTLY.md` for the
training cadence, then `docs/PLAN.md` for the long-term roadmap.

---

## ⚡ State as of 2026-06-10 — read this FIRST (overrides everything below)

**Pace handed off + released.** Pace v0.2.0 merged to main (PR #3) and tagged:
agent loop, RAG layer, MCP, watch mode, voice pipeline, dictation Stage A.
Canonical state doc: `docs/pace-handoff-2026-06-10.md`. Memory:
`project-pace-handoff-2026-06-10`. Key reframe: **Pace's runtime planner is
qwen3-30b-a3b via LM Studio, not our specialist** — any future planner eval
needs a 30B comparison row.

**Planner freeze**: v11 is ONE command away (`bash scripts/v11_pipeline.sh
[--amplify]`, supersedes v10_pipeline.sh). One run, ship or fail, then the
planner freezes for a month regardless. Gate: `docs/prds/pace-planner-v11-ship-gate.md`.

**tinygpt is now a local-LLM research lab.** Four threads mapped with
file:line plans (2026-06-10 research agents):
1. int8 ANE handoff (#306) — Phase A: fp16 IOSurface outputBackings ping-pong
   in `Qwen3ANEChunked.swift:188-203` (1d, macOS 15+); Phase B: int8 weights
   via `linear_quantize_weights` + macOS26 target (the real 1.8×); Phase C
   (int8 activations) gated/optional.
2. VLM M4 port (#266) — UI-Venus is ALSO Qwen3-VL arch, one port serves both;
   7 milestones w/ parity gates, 8-10d; run the A/B in Python (mlx_lm) first.
3. Quantized inference (#305) — everything dequants to dense fp32 today
   (`HFModel.swift:287-353`, GGUF too); serve `--quantize` is 0.5d; QLoRA
   wrapper needed for LoRA-on-quantized (`Lora.swift:53-71`).
4. Spec-decode in serve (#262) — B14 transfers (`SpeculativeDecode.swift`)
   but is KV-cache-free; grammar masks must apply to draft tokens via FSM
   clone (`Serve.swift:1782`); 3-5d.

Fixed: `eval_bfcl.py` now exits nonzero on a missing category (was silently
SKIPping — could have dropped Dim2 from the v11 ship gate).

---

## State as of 2026-06-09 morning (superseded)

North Star locked: `result = (speed × accuracy) / cost`. ROI bar = ≥5% on
formula per 2-week investment. Research-first doctrine: knowledge cutoff
Jan 2026 is stale; verify current state before recommending any
model/library.

**Memory to load first**:
- `feedback-tinygpt-north-star` (formula + ROI bar)
- `feedback-research-first-doctrine` (verify before recommending)
- `feedback-pace-doctrine-2026-06-08` (manifesto)
- `project-dora-fix-shipped-2026-06-09` (TGLA v1→v2 with magnitudes)
- `project-embedding-swap-2026-06-09` (mxbai → Qwen3-Embedding-0.6B)
- `project-anemll-vs-m8-2026-06-09` (REJECTED, port macOS 26 int8 in-house)
- `project-tinygrad-rejected-2026-06-09` (REJECTED — Ollama itself switched to MLX Mar 2026)
- `project-competitive-teardown-2026-06-09` (Dottie attack vectors + landing-page lines)

**Shipped this morning**:
- DoRA serialization v1→v2 with magnitudes (Swift, validated end-to-end)
- Embedding swap to Qwen3-Embedding-0.6B (faster on batches, same family as planner)
- `scripts/score_formula.py` — canonical (speed × accuracy / cost) measurement
- Pace landing-page draft at `pace/docs/landing/v1-draft.md`
- v10 cascade at `scripts/v10_pipeline.sh` (auto-runs train+bake+eval+score)
- PRDs: scope-narrowing, quantized-inference-swift, macos26-int8-ane-port

**Models qualified**:
- WhisperKit large-v3-turbo (voice, 2026-06-08)
- Qwen3-Embedding-0.6B (RAG, 2026-06-09 — swapped from mxbai)

**Rejected this morning** (don't re-propose):
- anemll — no LoRA path + open Qwen3 bugs; port macOS 26 int8 ANE in-house instead (#306)
- tinygrad — no measurable Pace benefit; Apple silicon island converged on MLX
- Apple Foundation Models adapter — restricted to Apple's model, not Qwen

**In-flight (next-steps if you pick this up cold)**:
- v10 planner training pending — multiplier running OR ready to fire via `scripts/v10_pipeline.sh`
- Qwen3-VL-2B downloaded (~2.8GB) for the UI-Venus A/B test on #266

**Open eval mystery** (not blocking): v8 LoRA scores 33.3% reproducible on
fm-fixtures-v2 in current env, not the documented 73.3%. Files unchanged.
Treat 33% as working baseline. v9-LoRA + tightened v9-compose-v2 prompt
matches v8 on non-compose AND adds 70% on new compose fixtures — ships.

---

## ⚡ State as of 2026-06-08 (overnight session — read this first)

The strategic picture changed last night. Three load-bearing
findings that override anything older in this doc:

### Finding 1 — ANE is energy-only

The ANE moat is real but **narrow**: ANE = small-model inference at
~3W, not a speedup over MLX at any scale. Validated empirically:

- ANE chunked Qwen3-0.6B (Pace specialist): 25 tok/s (Swift orchestrator)
- ANE chunked Qwen3-14B: 1.2 tok/s — unusable for foreground
- MLX Qwen3-14B-4bit: 30 tok/s
- MLX Qwen3-30B-A3B-4bit (MoE): 98 tok/s

ANE's role is **concurrent low-power background workloads** (Pace
specialist running on ANE while the user's GPU runs other things).
Not a general speed story. Stop chasing tok/s on ANE.

Shipped artifacts (all in tree, committed):
- M6 bisect findings: `docs/learn/ane-research/m6-findings.md`
- M8 per-block ANE export: `scripts/ane/m8_block_export.py`
- M8 chained decode: `scripts/ane/m8_chained_decode.py`
- Swift orchestrator: `native-mac/Sources/TinyGPTModel/Qwen3ANEChunked.swift`
- CLI: `tinygpt coreml-chunked-smoke --chunked-dir <dir> --hf-dir <dir>`
- Pace-baked bundle: `~/.cache/tinygpt/ane-pace-v5/m8-block-{0..27}.mlpackage`
- Base Qwen3-0.6B bundle: `~/.cache/tinygpt/ane/m8-block-{0..27}.mlpackage`
- M9 spec-dec experiment (negative result): `scripts/ane/m9_spec_decode.py`

Critical precision fix learned the hard way: chunked ANE needs
`compute_precision=ct.precision.FLOAT32` with `state=fp16`. fp16
compute alone drifts to garbage at 28-layer chained depth. See
m6-findings.md for the bisect.

### Finding 2 — Spec dec doesn't beat the baseline with our model pairs

Tested ANE 0.6B draft + MLX 14B verify (#9 milestone). Math doesn't
work: draft per-token (~40ms) is SLOWER than verify per-token
(~33ms with KV cache). Spec dec only helps when draft << verify.

Tested vs Qwen3-30B-A3B-MoE verify: even worse — MoE makes verify
~100 tok/s, so spec dec goes more backwards.

Spec dec is **shelved for now**. Could be revived with: (a) a
much faster draft (MLX 0.6B at ~50 tok/s), (b) a genuinely slow
verify (dense 32B+ at low quantization), (c) tree-based "speculative
speculative" decoding with custom MLX attention masks.

### Finding 3 — THE GATE: evaluations may be measuring framework, not model

The most important finding of the session. Owner ran an experiment
showing that a deterministic endpoint (grammar enforcement + lookups
+ regex) passes fm-fixtures comparably to a "trained model" —
suggesting the eval was testing format compliance, not capability.

**This invalidates the comparisons we've been celebrating** between
v3 / v5 / v6 / v6.1 / Pace specialist / teacher. Until evaluation
methodology can isolate model contribution, every model-training
decision is operating on noise.

This is the **gate task #270**. Until it ships, do not:
- Train new LoRAs (#265 v6.1, #268 specialist — both blocked on #270)
- Claim any specialist beats any teacher
- Compare v* artifacts
- Ship Pace models based on fm-fixture scores

What an honest eval looks like:
1. Build a rule-based "fake Pace" endpoint (tokenizer + grammar +
   element-list deterministic lookup + regex intent routing)
2. Score it against fm-fixtures → `baseline_score`
3. Score each LoRA artifact the same way → `lora_score`
4. **Model contribution = lora_score - baseline_score**
5. If ≤ 0, the model isn't doing useful work
6. Design new fixtures where rule-based scores ≤50% but a real model
   should reach ≥85% — held-out behavioral diversity, open-ended
   generation, capability checks the grammar can't fake

### Tasks updated to reflect overnight findings

- #263 ANE moat for Pace: **completed** (M8 Swift orchestrator shipped)
- #269 ANE M8 drift fix: **completed** (FLOAT32 compute solved it)
- #270 Eval methodology overhaul: **new, P0, gates #265 and #268**
- #265 v6.1 quality block: pending, **blocked on #270**
- #268 specialist quality block: pending, **blocked on #270**
- #266 VLM M4 full Qwen3-VL port: pending (real engineering, not eval-blocked)
- #267 v7 SFT: pending, blocked on #270 (don't train without eval)

### Honest read on what's worth working on tomorrow

1. **#270 eval methodology overhaul** — the gate. Until it lands,
   nothing else in the factory is meaningful.
2. **#266 VLM M4 Qwen3-VL port** — real engineering, not eval-blocked
   because it's about getting Qwen3-VL working at all, not comparing
   to a baseline.
3. **HTTP wrap of coreml-chunked-smoke** — `tinygpt coreml-chunked-serve
   --chunked-dir <dir> --hf-dir <dir> --port N`. Makes the ANE moat
   usable from Pace. Mechanical port of CoreMLServe.swift.

Anything else (more LoRA training, more spec dec, more ANE work) is
either pretend or eval-blocked.

### Anomalies the next session should know about

- **Remote URL outdated**: pushes work but redirect from
  `sarthakagrawal927/tinygpt` → `sarthak-fleet/tinygpt`. Update with
  `git remote set-url origin https://github.com/sarthak-fleet/tinygpt.git`
- **v6.1 four-way SFT collapse** (10 → 8 → 2 → 0 of 19) — almost
  certainly an eval-methodology artifact. Don't retry SFT.
- **190 uncommitted files** caught up in 5 commits on 2026-06-08
  (5 commits up to `ce86eca`). Heterogeneous; if something downstream
  breaks, suspect the "session catch-up" commit first.

---

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

HF model browser (cloud-icon button in sidebar):
- Sheet-based downloader. Text-input for `owner/repo`; pulls
  config.json + tokenizer.json + safetensors shards into
  `~/Library/Application Support/TinyGPT/hf/<owner>__<repo>/`.
- URLSession + per-chunk progress callbacks. `HF_TOKEN` env for
  gated models. Atomic .part rename on completion.
- Downloaded models list with Reveal-in-Finder + Copy-CLI-cmd +
  Delete per row.
- v2 follow-ups: search via /api/models?search=, in-app sampling
  (needs ModelController to learn `TinyGPTModelHF` alongside the
  existing `TinyGPTModel`).

Server tab (5th tab, OpenAI-compatible HTTP endpoint):
- Wraps `tinygpt-cli serve <model> --host 127.0.0.1 --port N` as a
  subprocess. Model picker accepts files OR HF dirs. Start/Stop
  toggle with live status. Endpoint card shows the URL + the three
  OpenAI routes (/v1/chat/completions, /v1/completions, /v1/models).
- Live log tail of the server's stdout/stderr. Request count parses
  "POST /v1/" lines from the log.
- Bound to 127.0.0.1 unconditionally — LAN exposure is a one-line
  change but deferred pending a security decision. Cmd-Q reaps the
  subprocess via OS process tree cleanup; explicit
  applicationWillTerminate hook is queued.

The app now hits every table-stakes item from the LM Studio / Ollama /
Jan comparison + ships two things they don't have (live training,
interp).

App bundle: ~380 MB (mostly the MLX metallib + a CLI binary copy).
Launch: `open build/TinyGPT.app` or `cp -r build/TinyGPT.app /Applications/`.
Both binaries verified by smoke-launching the .app + invoking the
bundled CLI to produce a real `.sae` sidecar (MSE 5.52e-02, L0 34.45%).

## 2026-06-05 PM — eval-first session (training-prep)

**Premise**: user has a 2-day training window starting now. Before
firing the next real training run, get the **evals + data + parallel-
work briefs** ready so the training day produces a model you can
*score*, not just a model that finishes.

**What landed (eval + data side, in order):**

1. **PLAN.md Tier D + Tier E** — split data gaps (Tier D) from eval
   pipelines (Tier E). E0–E8 enumerated, A1 specialist now gated on
   E1+E3.
2. **E0 shared schema + `tinygpt eval-compare`** —
   `Sources/TinyGPT/EvalCompare.swift`. Codable `Row` (snake_case
   JSON) so every harness writes one JSONL shape. Three view modes:
   `--by step` (training emergence), `--by model` (cross-model), `--by
   task` (which task scored what). Fixed dup-key crash + cell-truncation
   bug during smoke.
3. **E3 `tinygpt run-lm-eval`** — `Sources/TinyGPT/RunLmEval.swift`
   wraps `lm-eval-harness` as a subprocess. Two modes:
   - `--hf-model <id>` (canonical baseline scoring via HF transformers)
   - `--tinygpt-model <ckpt>` (boots `tinygpt serve` and routes lm-eval
     via `local-completions` — uses our actual forward pass; no
     .tinygpt→HF semantic conversion). Self-invocation fallback chain:
     `CommandLine.arguments.first ?? Bundle.main.executableURL ??
     resolveExecutable("tinygpt"|"tinygpt-cli")`.
4. **`tinygpt serve` logprobs path** — added `scoreLogprobs(prompt:)`
   in `Sources/TinyGPTServe/Serve.swift` for echo+logprobs requests
   (teacher-forced log_softmax). Required by lm-eval's loglikelihood
   tasks. Trigger condition is `logprobsRequested && echo` (any
   max_tokens).
5. **Smoke training** — 10K steps Huge bf16 on FineWeb-Edu w/
   `--save-history`, completed in 1627s (6.1 step/s). 5 checkpoints
   at 2K/4K/6K/8K/10K written under `/tmp/huge-smoke-30min.*`.
6. **Emergence sweep** — `tinygpt run-lm-eval` against all 5 checkpoints
   + `SmolLM2-135M` baseline (limit=10, arc_easy). 12-row JSONL preserved
   at `docs/artifacts/emergence-smoke-2026-06-05.jsonl`. Real numbers:

   | Model | Step | arc_easy (n=10) |
   |---|---|---|
   | SmolLM2-135M | baseline | **0.500** |
   | tinygpt-huge-smoke | 2000 | 0.300 |
   | tinygpt-huge-smoke | 4000 | 0.300 |
   | tinygpt-huge-smoke | 6000 | 0.300 |
   | tinygpt-huge-smoke | 8000 | 0.300 |
   | tinygpt-huge-smoke | 10000 | 0.300 |

   Honest read: 0.300 across all our checkpoints is statistically
   indistinguishable from random (0.25 + ~0.15 stderr at n=10) — our
   10M-param model has 13× fewer params and ~0.00014% the data of
   SmolLM2 and hasn't learned anything ARC-relevant yet. **Pipeline
   works**; the model needs more training. This is exactly what A1
   needs to ship: a runnable score path.

**Parallel-agent PRDs drafted** — `docs/prds/` now has 10 self-contained
briefs an elf can pick up cold. Each names its "don't touch" files so
multiple agents can ship in parallel without merge conflict:

| PRD | What | Estimated |
|---|---|---|
| `E1-bfcl-eval.md` | wire Berkeley Function Calling Leaderboard | ~1d |
| `E2-tau-bench-eval.md` | multi-turn agent eval | ~1d |
| `E5-humaneval-sandbox.md` | Rust-isolated Python exec scorer | ~1-2d |
| `E7-judge-shim.md` | LLM-as-judge via local Qwen/SmolLM | ~1d |
| `E8-train-time-eval-hook.md` | `tinygpt train --eval-every N` | ~1d |
| `eval-leaderboard-viewer.md` | `/eval-leaderboard.astro` (3 views) | ~2-3d |
| `sae-timeline-viewer.md` | `/sae-timeline.astro` (B13 viz) | ~1d |
| `rust-parquet-decoder.md` | replace pyarrow with Rust binary | ~half-day |
| `rust-hf-downloader.md` | parallel HF shard fetches | ~1d |
| `dataset-decode-verify.md` | low-skill data plumbing | ~1hr |

Coordination protocol in `docs/prds/README.md` — every PRD names a
small "don't touch" set; agents submit a diff line for shared dispatch
files, maintainer merges.

**Verify pipeline still works (quick smoke before training):**

```bash
./native-mac/.build/arm64-apple-macosx/release/tinygpt \
    eval-compare docs/artifacts/emergence-smoke-2026-06-05.jsonl --by step
```

Should print the 5-row TinyGPT trajectory table above.

**Training-day state (2026-06-05 PM):**

- **N02 fired** — `./scripts/nightly.sh` started N02 (Huge bf16,
  FineWeb-Edu, 200K steps, ~11 hrs). PID was 18877; `caffeinate -di`
  wrapping the job. Output goes to
  `~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt` +
  `huge-base-v1.step-*.tinygpt` (every 2K steps) +
  `huge-base-v1.jsonl` (dashboard log).
- N02 script already has `--save-history --log-jsonl --val-every 500`
  wired — no patches needed; emergence view applies as-is.
- Smoke loss curve was healthy: 11.34 → 5.11 over 10K steps, no spikes,
  no NaN. N02 (20× longer) should comfortably continue descending.
- **Huge preset is 22M params, not 10M** (I was misquoting earlier).
  Body 12L · d=256 · dMlp=1024 · with the SmolLM2 49K-vocab embedding
  the total is ~22M. Still ~6× smaller than SmolLM2-135M.

**When N02 finishes — fire-and-forget runbooks:**

```bash
# Score every checkpoint against the eval suite + SmolLM2 baseline.
./scripts/score-run.sh ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt
#   → writes docs/artifacts/score-huge-base-v1-<date>.jsonl
#   → prints --by step / --by model / --by task tables

# Train SAEs across the same checkpoints → feature-emergence timeline.
./scripts/sae-run.sh ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt
#   → writes docs/artifacts/sae-huge-base-v1-<date>/timeline.jsonl
```

Both scripts are CPU-light glue around already-shipped CLI surface.
`score-checkpoint.sh` is the per-checkpoint primitive; `score-run.sh`
wraps it for the full sweep.

**Parallel agents:** 10 PRDs under `docs/prds/` are out for elves
(E1, E2, E5, E7, E8 + eval-leaderboard / sae-timeline viewers + Rust
parquet/HF tools + dataset-verify). Don't observe their progress;
maintainer merges when PRs land.

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
