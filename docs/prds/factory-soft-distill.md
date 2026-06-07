---
name: Soft distillation — KL divergence on teacher logits
status: shipped-2026-06-07
owner: unassigned
created: 2026-06-07
priority: P2
---

# PRD — Soft (logit-level) distillation

## 2026-06-07 ship note

The core KL + NLL distillation path already existed in `tinygpt distill`. This
pass completed the PRD-facing CLI contract:

- `--mode soft|hard`
- `--data` alias for `--corpus`
- `--kl-weight` alias for `--alpha`
- hard mode skips teacher loading/forwarding when the KL weight is zero

## Goal

Add a soft-distillation mode to `tinygpt distill` that trains the
student on the teacher's full logit distribution (KL divergence loss),
not just the teacher's argmax outputs. Quality bump of 2-5% over hard
distillation on most tasks.

Hard distillation (current) is shipped: student trains on teacher's
generated tokens via cross-entropy. Soft distillation: student trains
to match teacher's *probability distribution* via KL divergence — uses
the teacher's full uncertainty, not just its picks.

## Why ship

Hard distillation is 90% of the value at 10% of the cost; that's why we
have it first. Soft distillation closes the remaining gap when the
teacher is strong + the student small (the 30B → 1B regime we target).

Per Hinton 2015: soft distillation typically yields 2-5pp better
student quality on classification + structured-output tasks.

## Scope — in

### CLI

```
tinygpt distill \
    --teacher Qwen2.5-32B.tinygpt \
    --student qwen3-0.6b.tinygpt \
    --data inputs.jsonl \                    # only inputs, NOT (input,output) pairs
    --mode soft \                            # NEW: soft|hard (default hard)
    --temperature 4.0 \                      # softening — higher = softer distribution
    --kl-weight 1.0 \                        # mix with standard CE on labels (0 = pure soft)
    --steps 5000 \
    --out student.tinygpt
```

### Implementation

Per training step:
1. Forward teacher on batch → logits T (no grad on teacher)
2. Forward student on same batch → logits S
3. Loss = KL(softmax(T/τ) || softmax(S/τ)) × τ²
4. Backprop only through student

Requires teacher logits available — either:
- **Local teacher**: load teacher model in same process (RAM-heavy but
  exact)
- **Remote teacher**: requires teacher endpoint that exposes logprobs
  for ALL tokens (rare; most APIs only return top-K)

v1 = local teacher only.

### Acceptance criteria

1. `tinygpt distill --mode soft` runs without OOM on Mac with a
   reasonable teacher/student pair (tiny + small presets)
2. Loss curve descends; final loss < hard-distill baseline on same data
3. Document: "soft distillation requires teacher in RAM; if teacher
   doesn't fit, use hard mode + synthesize JSONL via `tinygpt synthesize`"

## Scope — out

- Remote teacher with logprob streaming (most APIs don't support full
  vocab; skip)
- Mixed teacher ensemble (soft over N teachers)
- Online distillation (teacher generates labels mid-training)

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPT/Distill.swift` |
| **don't touch** | Synthesize (separate primitive), hard distill (preserve), eval |

## Estimated effort

**~3-5 days.** Risk: simultaneous teacher + student forward in MLX
without OOM. Mitigation: gradient checkpointing on student, no_grad on
teacher.

## Source

- Hinton, Vinyals, Dean 2015: https://arxiv.org/abs/1503.02531
