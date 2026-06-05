# TinyGPT — nightly training queue

Project shape: **every night the Mac produces a training artifact.** Daytime
work fills the queue and polishes infrastructure; nighttime runs through it.

## How it works

1. Each job is a shell script under [`scripts/nightly/N*.sh`](scripts/nightly/),
   numbered lex-sortably.
2. The runner [`scripts/nightly.sh`](scripts/nightly.sh) picks the
   lowest-numbered job that doesn't have a matching `.done` file in
   `~/.cache/tinygpt/nightly/done/`, runs it under `caffeinate -di`, logs
   to `~/.cache/tinygpt/nightly/logs/<ts>-<name>.log`, and posts a
   completion notification.
3. Before bed: `./scripts/nightly.sh &`. Wake up to a done job + a log
   + a `/training-dashboard` JSONL ready to drop.
4. To add a job: copy an existing `scripts/nightly/N*.sh` to the next
   number, edit the commands, commit. To re-run a job: delete its
   `.done` marker.
5. To pause the queue: just don't run `./scripts/nightly.sh` that night.

## Tonight (next job the runner will pick)

The runner picks the topmost line in the **Queue** below whose `.done`
file is absent. Move lines to **Done** by hand when you want a visible
audit trail (the runner doesn't edit this file — it touches `.done`
markers).

## Queue

The path to **A1 (first specialist · tool-caller)** end-to-end. Honest
state as of 2026-06-05 — parquet decoder + HF_TOKEN unblocking is
v2 work (see "Known blockers" at bottom):

- [x] **N01 · pull-datasets** · DONE — Gutenberg corpus (32 MB clean
      text) + hermes-function-calling-v1 + SmolLM2-135M tokenizer all
      verified in place. FineWeb-Edu, xlam, UltraFeedback, BFCL still
      blocked (see "Known blockers"). N01 now a verifier, not a
      downloader.
- [ ] **N02 · huge-base-v1** · ~9 hr · `scripts/nightly/N02-huge-base-v1.sh`
      Pretrain Huge preset (12L · d=256 · ctx=256, SmolLM2 BPE) on
      Gutenberg combined (~32 MB · ~8M tokens · ~6 epochs at 200K
      steps). WSD schedule (warmup 1000, decay 20000). save-every
      2000 + save-history (100 checkpoints, ~11 GB). --seed 42, spike
      detector on, JSONL log on. Output: `/tmp/huge-base-v1.tinygpt`
      + `/tmp/huge-base-v1.step-*.tinygpt` (history) +
      `/tmp/huge-base-v1.jsonl`. Smoke-tested at 100 steps: 11.4→6.5
      loss, 6.2 step/s steady state.
**Known blockers (limit our recipe for now):**

- `tinygpt download-dataset` saves parquet files as-is — no parquet→txt
  decoder yet. Blocks FineWeb-Edu (richer pretrain, 2.4 GB on disk),
  UltraFeedback (DPO data), and any other parquet-only HF dataset.
  **Daytime task**: add parquet decoding to `tinygpt download-dataset`
  (or write a small `tinygpt parquet-to-txt` subcommand).
- `Salesforce/xlam-function-calling-60k` is gated — needs `HF_TOKEN`.
  Either `export HF_TOKEN=hf_…` or `huggingface-cli login`.
- BFCL flat-emit hit a bug — `bfcl.jsonl` is 0 bytes. The cached HF
  dataset is fine; needs a re-emit pass once we know what fixed it.

**Planned (scripts written after N02 results land):**

- N03 · sft-toolcaller-v1 · ~8 hr · LoRA SFT on huge-base-v1 against
  xlam+hermes concat (~70K examples · 3 epochs · rank 8 · chatml).
  Output: `/tmp/sft-toolcaller-v1.lora`. **Recipe finalized after we
  see N02 base PPL + sample quality** — bad base → adjust SFT steps/lr.
- N04 · dpo-toolcaller-v1 · ~6 hr · DPO LoRA on huge-base-v1 against
  UltraFeedback (β=0.1, 1 epoch, rank 4). NOTE: tinygpt dpo's
  reference model is the BASE, not the SFT-tuned base, so this is a
  separate adapter from N03 rather than chained on top. v2 = adapter
  composition at inference + a `dpo --init-from-sft` flag.
- N05 · eval-toolcaller-v1 · ~2 hr · generate on BFCL sample with
  base + each adapter; emit JSON with accuracy + per-category breakdown.
  Built after N03+N04 produce real adapters to score.

After N05 we have the **first evidence the thesis holds (or doesn't).**
Next queue depends on the numbers:
- If A1 beats some baseline on BFCL → N06+ trains a second specialist
  (code, math, or debugger) on the same base.
- If A1 doesn't beat the baseline → N06 is a debug-and-retrain cycle
  with adjusted recipe (more SFT steps, different LoRA rank, etc.).

## Done

(Touch a `~/.cache/tinygpt/nightly/done/<name>.done` marker to "complete"
a job manually. Or run the job via the runner and the marker is created
automatically on exit-0.)

## Disk budget

Each Huge checkpoint is ~115 MB. `--save-history` with save-every 1000
on a 50K-step run = 50 history checkpoints × 115 MB = **~5.8 GB per
base pretrain**. Plan to keep:

- Latest base + its dashboard JSONL
- Last 5 history checkpoints from the latest base (for B13 interp work);
  archive the rest off-disk OR delete after the run finishes if not
  doing interp on it
- Every shipped specialist's final adapter (~10 MB) + DPO adapter +
  eval JSON. These are the deliverables, keep forever.

Tonight's projected disk use: ~7 GB (datasets + base + ckpts). 473 GB
free → 60+ nightly runs of headroom before cleanup is a hard constraint.

## Caveats

1. **Mega is OOM-blocked** at ctx=1024 batch=16 on 48 GB. All queue
   entries use Huge. Until gradient checkpointing ships, Huge is the
   ceiling for our base pretrain.
2. **Spike rollback is v2.** If a long run spikes hard at step 25K,
   you lose ~5 hr of compute. Spike *detector* logs the warning but
   doesn't auto-rollback. Worth shipping spike rollback if a real run
   loses time to this.
3. **The Mac is committed for the full night.** `caffeinate -di`
   wrapping prevents sleep + idle; you can still use the laptop, but
   GPU contention with other workloads (Xcode builds, browser
   workers, etc.) will slow training.
4. **First run is the riskiest.** A flag-name typo, a numerics gotcha
   in WSD on a long horizon, a memory creep — all undetected until
   the first overnight. Watch the first hour of N02 before going to
   bed; if step rate looks healthy and spike detector is silent, walk
   away.
