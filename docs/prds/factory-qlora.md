---
name: QLoRA recipe — 4-bit base + LoRA delta fine-tuning
status: shipped-v0-2026-06-07-packed-base-deferred
owner: unassigned
created: 2026-06-07
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (child path — opens 13-30B fine-tune on Mac)
priority: P1
---

# PRD — QLoRA recipe wired into `tinygpt sft`

## Goal

Wire QLoRA (4-bit quantized base + LoRA delta) into the existing `sft`
CLI. The primitives already exist (`gptq`, `hqq`, LoRA training); this
PRD composes them into the canonical QLoRA recipe: load base at 4-bit,
keep frozen, train LoRA on top.

Effect: opens **13-30B parameter fine-tuning on a 48GB Mac** without
needing the full bf16 weights in memory.

## 2026-06-07 implementation note

`tinygpt sft` now accepts `--qlora` and `--qlora-bits 4|8`. This wires the
recipe through the existing LoftQ-style path: wrapped base weights are
quantize-then-dequantized in memory, LoRA targets expand to attention + MLP
projections, and the adapter trains against that frozen quantized
approximation.

This is the shipped QLoRA v0 factory path. The current quantization stack still
stores train-time base weights as floating-point dequantized tensors; packed
int4/int8 train-time storage and a matching inference composition path are
deferred until the MLX-Swift quantized/autograd path is real enough to support
the memory-saving claim honestly.

## Why ship

The "fine-tune a 30B teacher locally on your Mac" pitch requires QLoRA.
Without it, Mac fine-tuning ceiling is ~7B at bf16. With it: 30B.

This is what unblocks the platform's "fine-tune any model up to ~30B
locally" story.

## Scope — in

### CLI surface — augment `tinygpt sft`

```
tinygpt sft \
    --base qwen3-7b.tinygpt \
    --data sft.jsonl \
    --qlora \                          # NEW: load base at 4-bit, train LoRA
    --qlora-bits 4 \                   # 4 (default) or 8
    --rank 16 --alpha 32 \             # LoRA hyperparams
    --steps 2000 \
    --lr 1e-4 \
    --out spec.lora
```

### Behavior

1. Load base via `gptq` / `hqq` quantization to 4-bit (in-memory)
2. Wrap target modules (q_proj, k_proj, v_proj, o_proj, plus optional
   MLPs) with LoRA adapters
3. Forward pass: dequantize base on-the-fly per layer (standard QLoRA
   trick), apply LoRA delta, full bf16 gradient through LoRA only
4. Save LoRA adapter only (frozen 4-bit base stays on disk; user
   composes at inference)

### Acceptance criteria

1. `tinygpt sft --base qwen3-7b.tinygpt --data sample.jsonl --qlora --steps 100 --out test.lora` completes on Mac without OOM
2. Memory usage stays ≤ 1/3 of bf16 fine-tune of the same base
3. Loss curve descends comparably (within ~10%) to a bf16 LoRA run on
   tiny preset (validate algorithm correctness)
4. Saved `.lora` adapter loads + applies cleanly via existing inference
   path

## Scope — out

- AWQ (use existing gptq/hqq for v1)
- Per-layer mixed precision (everything 4-bit + LoRA bf16 for v1)
- 30B+ model loading (test on 7B; extrapolate)

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPT/SFT.swift` — add `--qlora` flag |
| **modify** | `native-mac/Sources/TinyGPTModel/LoRA.swift` — wrap quantized base |
| **don't touch** | Quantization sources (already shipped); `tinygpt train`; eval pipeline |

## Estimated effort

**~3-5 days.** Risk concentration: MLX-Swift's gradient flow through
the dequant op. May need a custom op or stop-grad pattern.

## Source

- QLoRA: Dettmers et al. 2023 (https://arxiv.org/abs/2305.14314)
- Reference impl: bitsandbytes-style `Linear4bit` + PEFT
