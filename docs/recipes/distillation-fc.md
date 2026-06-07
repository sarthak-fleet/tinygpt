# Recipe — distill function-calling from Phi-3-mini → TinyGPT-Huge

**Goal:** produce a 22M-param specialist that matches Phi-3-mini-4k-instruct
(3.8B) on function-call accuracy while running 15-30× faster and using
150× less memory on the same Mac.

**Premise:** with distillation in the toolkit, "small beats large at
quality" reduces to "small *matches* large at quality on the specific
task" — and we win on speed/memory by construction. Speed-and-memory
dominance is automatic; quality parity is the lever.

**Status:** queued to fire when N02 base lands. All inputs are cached
or downloadable today.

## Why function-calling first

| Reason | Why it matters |
|---|---|
| Narrow output space (JSON matching a schema) | Easier for a small model to learn |
| Abundant cached data | hermes-fc (50 MB / 11K rows), function-calling-chatml (333 MB / 113K rows) |
| Established benchmark | BFCL — eval pipeline already wired (E1 shipped) |
| Phi-3-mini already on disk | From the 2026-06-05 baseline-score sweep |
| Output format gives us a hard correctness signal | parse-and-execute > LLM-as-judge |

## The 3-axis win condition

Compare three models on the same BFCL benchmark + the same Mac:

| Model | Params | What we expect |
|---|---|---|
| **Phi-3-mini-4k-instruct** (teacher) | 3.8B | quality reference; the bar we want to match |
| **TinyGPT-Huge-FC** (our distilled student) | 22M | ≥ 90% of teacher's BFCL accuracy; **15-30× faster**, **150× less memory** |
| **SmolLM2-135M-Instruct** (size-class peer baseline) | 135M | should LOSE to us on BFCL — it's instruct-tuned, not FC-specialized |

If TinyGPT-Huge-FC hits the middle row spec, that's the win. Performance
optimizations (quantization, KV cache, ANE, spec-dec) come after.

## Pipeline

### 1. Build the distillation dataset

Source pool (cached):

```
hermes-fc.jsonl                  ~11K rows   (system + user + tool calls)
function-calling-chatml.jsonl    ~113K rows  (multi-turn dialogues w/ calls)
```

Process:

```bash
# Sample ~10K diverse (system_prompt, user_query) pairs from cached data.
python3 scripts/distill-prep.py \
    --task fc \
    --sources hermes-fc,fcc \
    --target-rows 10000 \
    --out ~/.cache/tinygpt/datasets/distill-fc-inputs.jsonl
```

(Need to write `scripts/distill-prep.py` — it's a thin reformatter; the
existing `scripts/scaledown-prep.py` is the pattern.)

### 2. Label with the teacher

Run Phi-3-mini-4k locally via `tinygpt serve` or HF transformers:

```bash
# Spin up Phi-3 as an OpenAI-compatible endpoint.
# Easiest path: use existing `tinygpt run-lm-eval --hf-model` machinery
# with a relabel mode (TODO: add `--relabel` flag to RunLmEval).

# OR — simpler — use llama-cpp-python / HF transformers from a Python
# script that consumes distill-fc-inputs.jsonl and appends teacher_output:

python3 scripts/distill-label.py \
    --teacher microsoft/Phi-3-mini-4k-instruct \
    --input ~/.cache/tinygpt/datasets/distill-fc-inputs.jsonl \
    --out ~/.cache/tinygpt/datasets/distill-fc-labeled.jsonl
```

Expected wall time: ~30 ms/row × 10K = ~5 min on Mac (M5 Pro), but Phi-3
will hog 8 GB; do this when N02 finishes.

### 3. SFT TinyGPT-Huge on the labeled data

```bash
tinygpt train \
    --base ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt \
    --preset huge \
    --tokenizer <SmolLM2 dir> \
    --corpus ~/.cache/tinygpt/datasets/distill-fc-labeled.jsonl \
    --task sft-distill \
    --steps 20000 \
    --batch 4 --accum 4 \
    --lr-schedule wsd --warmup 200 --decay-steps 2000 \
    --max-lr 1e-4 --min-lr 1e-5 \
    --save-every 1000 --save-history \
    --val-split 0.02 --val-every 200 \
    --log-jsonl ~/.cache/tinygpt/runs/huge-fc-distill/huge-fc-distill.jsonl \
    --out ~/.cache/tinygpt/runs/huge-fc-distill/huge-fc-distilled.tinygpt
```

Expected wall time: ~45 min on M5 Pro (estimate from N02's 7 step/s).

### 4. Score all three on BFCL + measure performance

```bash
# Quality.
tinygpt eval-bfcl /tmp/huge-fc-distilled.tinygpt \
    --out docs/artifacts/distill-fc-quality.jsonl
tinygpt eval-bfcl microsoft/Phi-3-mini-4k-instruct \
    --baseline --out docs/artifacts/distill-fc-quality.jsonl
tinygpt eval-bfcl HuggingFaceTB/SmolLM2-135M-Instruct \
    --baseline --out docs/artifacts/distill-fc-quality.jsonl

# Speed + memory.
scripts/bench-perf.sh /tmp/huge-fc-distilled.tinygpt
scripts/bench-perf.sh microsoft/Phi-3-mini-4k-instruct
# (TODO: write bench-perf.sh; needs tokens/sec + peak RSS)
```

Render:

```bash
tinygpt eval-compare docs/artifacts/distill-fc-quality.jsonl --by model
```

## Variants for "even better"

Hard distillation as above is v1. If quality parity isn't reached, escalate:

| Variant | Cost | Likely gain |
|---|---|---|
| **Soft distillation** (KL on teacher logits) | Need teacher logits exported per token; heavier infra | +2-5% quality |
| **Multi-teacher ensemble** (Phi-3 + Qwen3 + Claude Haiku for label) | More label time, more disk | +1-3% quality, more diversity |
| **Constrained decoding** at student inference (force valid JSON) | Free — no retraining | +5-15% on format-compliance, ~0 on semantic correctness |
| **More data** (50K+ examples from full fcc corpus) | Longer SFT | +1-5% quality |
| **Test-time best-of-N** with student | Slower inference, may eat speed budget | +3-8% quality at 4-8× latency hit |

Constrained decoding is the highest-ROI v2 — it's free (no retraining)
and the model's output space is a JSON schema we already have.

## Performance optimizations queued for v3 (after quality lands)

Quality first. **Don't optimize a model that doesn't work.** When the
quality bar is hit, fire these in order:

1. **GGUF quantization** (B17, already shipped) — Q8 → Q5 → Q4 step-down
   gives 4× more memory headroom; usually <1% quality drop. Use existing
   `gguf-extract` toolchain.
2. **KV cache for `tinygpt serve`** — currently disabled per
   `serve.swift:986` ("KV cache is built fresh each call"). For batched
   eval and long sessions, KV cache cuts per-token cost dramatically.
3. **Speculative decoding** (B14, already shipped) — use the student as
   its own draft model with `--draft self` mode, or use a smaller draft.
   Lossless, ~2× speedup.
4. **ANE (Apple Neural Engine) inference** (queued #193) — 22M-param
   model is small enough to fit on the NE; potential 5-10× speedup on
   M5 hardware, ~0 power draw.
5. **MLX compile + fusion** — `tinygpt sample --compile` already
   exists; ensure serve uses it.

Stacked: distilled student + Q5 GGUF + KV cache + spec-dec + ANE
could plausibly hit 100-500× the teacher's tokens/sec on the same Mac
while using 1/300th the memory.

## Acceptance criteria for "v1 done"

1. `docs/artifacts/distill-fc-quality.jsonl` shows TinyGPT-Huge-FC
   within 10 percentage points of Phi-3-mini-4k on BFCL.
2. `tokens/sec` for TinyGPT-Huge-FC ≥ 10× Phi-3-mini-4k on the same
   prompts, same Mac.
3. Peak RSS for TinyGPT-Huge-FC < 200 MB; Phi-3-mini-4k > 4 GB.
4. Three-row JSONL renders cleanly via `eval-compare --by model`.
5. A short writeup in `docs/sessions/<date>-distill-v1.md` covering the
   recipe + the numbers + what didn't work.

## Risks

- **N02 base is too undertrained for SFT to land.** Mitigation: train
  N02 longer OR shrink to Mega preset which saturates earlier.
- **Phi-3's output format doesn't match BFCL's expected schema.** Mitigation:
  add a format-coercion post-processor on the teacher's outputs before
  using them as labels.
- **22M is too small to learn the function-call distribution well.**
  Mitigation: use a bigger student (Mega preset, 76M) — same recipe,
  more headroom.
- **Constrained decoding mid-stream confuses tokenizer.** Mitigation: only
  apply at JSON-block boundaries, not per token.

## File layout

| File | Role |
|---|---|
| `docs/recipes/distillation-fc.md` | this doc |
| `scripts/distill-prep.py` | dataset sampler (TODO) |
| `scripts/distill-label.py` | teacher inference loop (TODO) |
| `scripts/bench-perf.sh` | tokens/sec + RSS benchmark (TODO) |
| `~/.cache/tinygpt/datasets/distill-fc-inputs.jsonl` | unlabeled inputs (TODO) |
| `~/.cache/tinygpt/datasets/distill-fc-labeled.jsonl` | (input, teacher_output) pairs (TODO) |
| `/tmp/huge-fc-distilled.tinygpt` | trained student |
| `docs/artifacts/distill-fc-quality.jsonl` | 3-model BFCL comparison |

## Links

- Hinton, Vinyals, Dean 2015 — *Distilling the Knowledge in a Neural Network* (https://arxiv.org/abs/1503.02531). The original.
- Phi-1 (Gunasekar et al. 2023) — *Textbooks Are All You Need* (https://arxiv.org/abs/2306.11644). Beat GPT-3 at HumanEval with 1.3B model + curated data — closest analogue to what we're attempting.
- Phi-3 technical report (Abdin et al. 2024) — https://arxiv.org/abs/2404.14219
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html
- Companion: `docs/recipes/b25-scaledown.md` (sibling specialist track).
