# Session — eval-first prep before the 2-day training window

**Date:** 2026-06-05 (afternoon → evening)
**Premise:** user has a 2-day window starting now during which they will *only*
be training the model and working on site projects. Before firing the long
training run (N02), close every gap that would block scoring the resulting
model — schema, harness, baseline comparison, multi-checkpoint emergence view.

## Question we walked in with

> "Models we are trying to train are specialised. Do we have the evals and
> the data set to verify their quality?"

The honest answer was *no, not really.* We had Tier A data plans (xlam,
hermes-fc, BFCL source code) and Tier B / C model work shipped, but no
end-to-end **score this checkpoint → number** path. So the next training
run would produce an artifact nobody could grade against anything.

This session closed that gap. The training run that fires after this session
will produce a checkpoint that gets graded automatically against three
comparison axes.

## What landed (in order)

### 1. Plan restructure — Tier D + Tier E

Split data gaps (Tier D — pull / decode / verify) from eval pipelines
(Tier E — wrap harness → score JSONL). E0–E8 enumerated. A1 specialist
shipping criterion now includes E1+E3 wired.

### 2. E0 — shared eval JSONL schema + `tinygpt eval-compare`

`Sources/TinyGPT/EvalCompare.swift`. One Codable `Row` (snake_case JSON
keys) that every harness emits. Three view modes:

- `--by step` — same model across training checkpoints (emergence view)
- `--by model` — multiple models at one snapshot (head-to-head leaderboard)
- `--by task` — which task scored what (cross-section)

This was the unblocking architectural choice. Every E* harness writes the
same shape so `eval-compare` can aggregate across families without per-
harness adapters.

### 3. E3 — `tinygpt run-lm-eval` wraps EleutherAI lm-eval-harness

`Sources/TinyGPT/RunLmEval.swift`. Subprocess-out to the
canonical loglikelihood harness, two modes:

- `--hf-model <id>` — score any HF transformers model (baseline scoring)
- `--tinygpt-model <ckpt>` — boots `tinygpt serve` and routes lm-eval via
  the `local-completions` backend. Uses our actual forward pass; no
  semantic conversion from `.tinygpt` → llama-architecture HF dir.

The `local-completions` route was the right choice: a `.tinygpt`→HF
adapter would have to map our architectural choices (tied embeddings,
GQA configuration, custom byte fallback) into a Llama-shaped folder.
Lossy and bug-prone. Serving the model and letting lm-eval treat it as
"some OpenAI-compatible server" is the cleaner separation.

### 4. `tinygpt serve` — log-prob scoring path

`Sources/TinyGPTServe/Serve.swift`. Added `scoreLogprobs(prompt:)` for
echo + logprobs requests. Teacher-forced `log_softmax`. Triggered when
`logprobsRequested && echo` (any `max_tokens`). Required by lm-eval's
loglikelihood tasks.

### 5. Smoke training — validate the whole pipeline

10K-step Huge run on FineWeb-Edu with `--save-history` (5 checkpoints
at 2K/4K/6K/8K/10K). Wall time 1627s, 6.14 step/s.

Loss curve was healthy: 11.34 → 5.11 over 10K steps, no spikes, no NaN.
Whether the model learned anything *useful* is a separate question — and
the next item was about answering it.

### 6. Cross-checkpoint + cross-model sweep

Scored all 5 TinyGPT checkpoints + SmolLM2-135M baseline on arc_easy
(limit=10). Emitted 12 rows. Three view modes rendered.

The numbers were honest and informative:

| Model | Step | arc_easy (n=10) |
|---|---|---|
| SmolLM2-135M (135M params, ~7T tokens) | baseline | **0.500** |
| tinygpt-huge-smoke (22M params, 10K steps) | 2000 | 0.300 |
| tinygpt-huge-smoke | 4000 | 0.300 |
| tinygpt-huge-smoke | 6000 | 0.300 |
| tinygpt-huge-smoke | 8000 | 0.300 |
| tinygpt-huge-smoke | 10000 | 0.300 |

0.300 across all our checkpoints is statistically equivalent to random at
this sample size (0.25 baseline + ~0.15 stderr at n=10). The smoke model
hadn't learned anything ARC-relevant. **Expected** — it has 6× fewer params
and 0.00014% the training data of SmolLM2.

**What this proved**: the pipeline produces a real number end-to-end.
A1 specialist will ship with a real number. That was the gate.

Preserved at `docs/artifacts/emergence-smoke-2026-06-05.jsonl`.

### 7. Ten PRDs for parallel agents

`docs/prds/` indexes 10 self-contained briefs an elf can pick up cold:
E1/E2/E5/E7/E8 evals, eval-leaderboard + sae-timeline viewers, Rust
parquet decoder + HF downloader, dataset decode-verify. Each PRD names
its "don't touch" files so multiple elves work without merge conflict.
Coordination rule in `docs/prds/README.md`.

### 8. Fire-and-forget runbooks for N02

`scripts/score-run.sh` — when N02 finishes, scores every checkpoint +
SmolLM2 baseline + renders all 3 view modes. One command.

`scripts/sae-run.sh` — same checkpoints, trains an SAE per checkpoint
for the feature-emergence timeline.

`scripts/score-checkpoint.sh` — single-ckpt primitive.

## Things we learned, by surprise

### lm-eval doesn't fail loudly on weird input

The first end-to-end smoke gave `acc=0.3` on every checkpoint. Looked
like a bug — same number too consistent. Turned out to be: 4-choice ARC,
random baseline 0.25, stderr ~0.15 at n=10, so 0.3 ± 0.15 covers the
random region exactly. The model genuinely hadn't learned anything; the
0.3 was random-walk-around-baseline at small N.

**Takeaway**: when N is small enough that stderr ≈ score, do not
interpret the score as signal. The fix is N=500+, not "is the
implementation broken."

### `local-completions` is the right adapter

We considered a `.tinygpt`→HF-dir conversion (so lm-eval could use
`--hf-model` against us). The architectural-mapping cost was high:
embedding-tying convention, GQA config, byte-fallback handling all had
to line up. Serving the model + treating it as "some OpenAI-compatible
server" sidesteps all of that, and uses our actual forward pass.

### Self-invocation needs `CommandLine.arguments.first`

`tinygpt run-lm-eval --tinygpt-model` spawns `tinygpt serve` as a child.
Finding the right binary path failed when running from
`.build/arm64-apple-macosx/release/tinygpt` because `resolveExecutable("tinygpt")`
only searches `PATH`. Fallback chain that works:

```swift
let selfPath = CommandLine.arguments.first.map { URL(fileURLWithPath: $0) }
    ?? Bundle.main.executableURL
let tinygptCLI = selfPath ?? resolveExecutable("tinygpt") ?? resolveExecutable("tinygpt-cli")
```

### lm-eval extras are not optional

`pip install lm-eval` doesn't pull tenacity (needed for `[api]` extras),
torch (needed for `--hf-model`), or accelerate. Install command that
actually works:

```bash
pip install 'lm-eval[api]' torch transformers safetensors accelerate
```

## What didn't get done (deliberately)

- **N02 itself.** That's the training run this session prepped for. Fires
  in a separate shell so the GPU doesn't compete with the prep work.
- **Real lm-eval generation-task validation.** arc_easy is loglikelihood-
  only. gsm8k (generation) hasn't been exercised end-to-end via
  `local-completions`. Will surface during the post-N02 sweep.
- **E4 standalone GSM8K scorer.** If gsm8k via E3 works, E4 may not be
  needed. Decided to discover via N02 sweep rather than pre-build.
- **Cross-checkpoint feature alignment in B13.** "Is feature 47 at step
  10K the same feature as 47 at step 50K?" is the hard question — needs
  post-hoc Hungarian matching. v2 work.

## Where N02 picks this up

`scripts/nightly.sh` fires N02 (Huge bf16, FineWeb-Edu, 200K steps,
~11 hrs) with `--save-history --log-jsonl --val-every 500` already wired.
When it finishes:

```bash
./scripts/score-run.sh ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt    # full eval sweep
./scripts/sae-run.sh   ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt    # SAE feature timeline
```

Outputs land under `docs/artifacts/`. The browser viewers (eval-leaderboard
+ sae-timeline) are in flight via the elves.

## Why this session matters

Before today, the next training run was a model someone would have to
grade by hand and squint at. After today, the same training run produces
a model that gets graded automatically against multiple baselines and
plotted as an emergence curve across its own checkpoints.

The deliverables were small individually — a JSONL schema, a serve route,
a subprocess wrapper, three runbooks. The integration is what mattered:
one schema → every harness emits the same row → one comparator rolls
them up → three view modes from the same artifact.
