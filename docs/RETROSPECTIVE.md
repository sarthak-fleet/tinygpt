# TinyGPT — retrospective and forward plan

Written 2026-06-11 after the clarify-v1 experiment failed and the
small-specialist training bet closed. This doc replaces no other; it
records what we learned at high cost so future work doesn't relearn it.

## What we tried to do

Train small models (originally 0.6B, later 4B) locally on a curated
corpus to beat larger / cloud models on a well-defined task — first
Pace's planner, later a public clarify-discipline benchmark. The
factory (corpus generation, amplifier, ship gate, eval matrix, training
loop, ANE serve, grammar-constrained decoding) was built to support
that bet.

## What we found out

Each finding below is backed by a numbered run in
`~/.cache/tinygpt/runs/` or a fixture suite in `evals/`.

### 1. A 0.6B model memorizes behaviors; it does not learn rules.

Eleven training versions (v1–v11) on Qwen3-0.6B never beat zero-shot
larger models on Pace's gate. v11 reached train loss 0.001 (full
memorization of 709 rows) and still refused **0/30** held-out
out-of-scope prompts. Surface-regular triggers (e.g. "delete all" →
destructive) generalized; judgment (refuse, clarify) did not.

### 2. Zero-shot Qwen3-4B beats every trained 0.6B we shipped.

On the same h2 held-out suites: 80% OOS / 0% clarify / 70% destructive
/ 66.7% happy-path, untrained, at 2.3 GB int4. Competence comes from
scale; the right move is fine-tuning a model that already has the
capability you want, not creating capability through training.

### 3. Small-corpus LoRA on a competent base causes catastrophic
### interference — and we have numbers.

clarify-v1: 38 contrastive rows trained on Qwen3-4B regressed the
**untrained** OOS dimension from 80% → 33% (−47pp), while moving the
trained ambig dimension only from 0% → 5%. Training on a slice of
behaviors moved the whole intent distribution. If you fine-tune a
strong base, the corpus must cover every dimension you don't want to
regress, with enough density per dimension.

### 4. The clarify-not-guess failure is universal.

Every model we tested on the held-out clarify suite — trained 0.6B
specialists, Apple Foundation Models, Qwen3-4B zero-shot, **Claude via
CLI** — scored 0–15%. All of them confidently picked one option
instead of asking. This is a real product gap in local-and-cloud
assistants generally, not a TinyGPT-specific failure.

### 5. Apple Foundation Models is the refusal champion, action-blind.

OOS 97%, destructive 70%, happy-path 13%, clarify 5% (guided @Generable
required — plain text scored 0/60 by refusing correctly in prose that
the runner couldn't parse). Worth using as a free, zero-footprint
refusal pre-filter on Apple Intelligence Macs; not a Pace planner.

### 6. Eval infrastructure is a research instrument, not a chore.

Across the run we caught:
- v8's "73% → 33% reproducibility crisis" was config drift (default
  system prompt vs as-shipped prompt) — same model, same fixtures, 40pp
  swing.
- v11's 0/30 OOS verdict was real, not a harness bug — verified by
  reading raw responses before declaring.
- LM Studio mid-suite contention produced "instant-failure" rows with
  ~1 ms p50 latency. Without that signature check we'd have read
  capability failures where there were only socket failures.
- Per-block numerical parity (cosine ≥ 0.9998) is insufficient — the
  28-block chain still drifted by step 5 under per-channel int8.
  per-block(32) required for the chain gate to pass.
- Held-out fixtures derived from training seeds leak — fresh h2 suites
  were required for honest measurement.

### 7. The runtime infrastructure outperformed the training bet.

Things that work and are reusable: `tinygpt serve` with grammar-
constrained JSON masking (119 ms warm TTFW), `--quantize int4|int8`
in-memory (int8 = 2.3× decode at zero quality loss on v9), the 28-block
ANE chain (17 tok/s decode, int8 per-block weights, fp32-compute /
fp16-state, numerics gate green), the eval system that caught every
finding in §6, the `fm_bridge` + `fm_shim` harness for benchmarking
Apple FM, the contamination-checked h2 suites (60 fixtures).

## What we now believe

- **Don't train when zero-shot is close enough.** Apply the rule
  *before* committing compute: measure the base on the same gate; train
  only when the gap is too wide for prompt engineering to close.
- **If you do fine-tune a 4B+ base, the corpus must cover every
  dimension you don't want to regress.** Catastrophic interference is
  not a corner case at small data scale; it is the default.
- **The 0.6B specialist track is closed.** Not "park" — closed. Future
  small-model work needs a different premise.
- **Pace's planner is qwen3-4b-instruct-2507 zero-shot.** Already
  wired (pace commit 247fb32).
- **Pace's success metric is daily use, not eval percentages.** What
  Pace needs now is engineering, not models.

## Forward plan (where Sarthak's time goes)

| Track | What | Status |
|---|---|---|
| Pace, daily-use | Use Pace daily, fix the things that break first. Voice loop, AX actions, retrieval, dictation — all wired. | Active |
| TinyGPT, paused | Active development stops. The runtime + eval + harnesses are reference assets. | Paused |
| Write-up | One public post on the small-corpus interference finding (this doc + clarify-v1 numbers + h2 suites + cloud baselines as evidence). Optional but cheap. | Optional |
| QLoRA on 4B+, deferred | Only if Pace usage surfaces a *specific* capability gap that's clearly worth fine-tuning *and* that fine-tuning won't regress the dimensions Pace already relies on. Documented in `docs/prds/qlora-large-model-finetune.md`. | Deferred |
| Core AI / M9 | Apple's new framework is the right vehicle for whatever ANE work resumes. The M8 chain is reference code for that. | Deferred |

## What you should NOT do

- Train v12. The dimension that needs improvement (clarify) is universal across all
  tested models, including Claude — *no fine-tuning evidence shows
  small corpora fix it without breaking the rest*.
- Reopen Pace integration in tinygpt. The collaboration is closed
  (memory `pace-divorce-2026-06-11`).
- Optimize the eval system further without a model worth gating.

## Artifacts inventory

- Runtime: `native-mac/Sources/TinyGPTServe/`,
  `native-mac/Sources/TinyGPTModel/Qwen3ANEChunked.swift`,
  `scripts/ane/m8_*.py`, `scripts/ane/m8_numerics_gate.py`
- Eval: `scripts/eval_pace_v2.py`, `scripts/eval_pace_unhappy.py`,
  `scripts/eval_bfcl.py`, `scripts/score_formula.py`,
  `scripts/fm_bridge.swift`, `scripts/fm_shim.py`,
  `scripts/cloud_shim.py`
- Held-out fixtures: `evals/fm-fixtures-{oos,ambig,destructive}-h2/`
  (60 prompts, zero overlap with any training corpus)
- Training runs (numbers in `~/.cache/tinygpt/runs/`): v1 through v11
  (0.6B), clarify-v1 (4B), h2-{qwen3-4b,apple-fm,claude} (zero-shot
  baselines)
- Docs that are still current: `docs/wwdc-2026-impact.md`,
  `docs/v11-baselines-2026-06-09.md`,
  `docs/prds/quantized-inference-swift.md`,
  `docs/prds/qlora-large-model-finetune.md`,
  `docs/prds/tinygpt-product-thesis.md`. Read those alongside this one;
  they predate the final findings.

## One-line summary

We tried to train small models to beat big ones. We learned, with
numbers, that a strong base zero-shot is a better bet than a small
model trained — and that fine-tuning a strong base on a thin corpus
breaks more than it fixes. Pace ships on the strong base; TinyGPT
training pauses.
