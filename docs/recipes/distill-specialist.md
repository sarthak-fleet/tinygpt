# Recipe: distill a big model into a small local specialist

Compress a big model's capability *on your task* into a small, cheap, **local**
model on the Mac. This is the validated cost-compression lane — the lever that
*wins* (unlike fine-tuning-to-beat-the-base, which loses on tasks the base
already does well).

## Validated result (tool-calling, n=120)

| Model | fn-name | full (name+args) | size vs teacher |
|---|---|---|---|
| 4B teacher | 73.3% | 55.8% | 1× |
| **1.7B distilled** | **74.2%** | **53.3%** | **~2.3× smaller** |
| **0.6B distilled** | 72.5% | 45.8% | ~7× smaller |
| (1.7B / 0.6B zero-shot) | ~4–6% | ~4–6% | — |

**The frontier:** **1.7B** recovers the teacher on *both* metrics; **0.6B**
matches *function selection* but loses ~10pp on *argument precision* (that
precision is what needs the extra capacity). Pick your point: 1.7B for full
fidelity, 0.6B for max compression.

## Use it

```bash
scripts/distill-specialist.sh <data-dir> <student-model> <out-dir> [iters]

# e.g. full-fidelity:
scripts/distill-specialist.sh ./mydata Qwen/Qwen3-1.7B ./out/my-specialist 400
# e.g. max-compression:
scripts/distill-specialist.sh ./mydata Qwen/Qwen3-0.6B ./out/my-specialist-tiny 400
```

Produces a **standalone fused model** (LoRA merged in) — serve it via
`mlx_lm.server`, oMLX, or LM Studio. No adapter wiring downstream.

## Data format (and the one gotcha)

`<data-dir>/train.jsonl`, one example per line, mlx_lm chat format:
```json
{"messages":[{"role":"system","content":"…"},{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}
```
- The **assistant turn is the target** (`--mask-prompt` trains only on it).
- **Keep each example ≤ ~5500 chars** (~<2000 tokens). If an example's *prompt*
  alone exceeds `--max-seq-length`, truncation removes the response tokens →
  loss over zero tokens → **NaN**. (This is exactly what bit the first A1 run.)
- For a focused specialist, filter to the rows that exercise the target
  behavior (we filtered hermes-fc to the `<tool_call>` rows).

## Method notes

- It's **SFT-on-task** distillation: the student learns the correct behavior
  from gold labels. Gold is a *perfect* teacher, so this beats hard
  teacher-output distillation (a teacher that's only ~56% correct gives noisier
  labels than gold).
- The only deeper lever left is **logit-level (soft) KD** (`/tmp/a1_kd.py`
  prototype) — match the teacher's full distribution; can squeeze the 0.6B's
  argument precision higher. Not needed for full fidelity — just use 1.7B.
- **Always eval the student vs the teacher** on a held-out slice with a
  task-specific scorer before trusting it (the function/arg-match scorer in
  `/tmp/a1_eval.py` is the tool-calling example).

## Why this matters

This is the per-project specialist factory: any project's data → a small,
private, Mac-runnable model that recovers a 4B-class capability for its task.
The cost-compression bet, validated and repeatable.
