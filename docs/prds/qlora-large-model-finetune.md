# QLoRA on Mac — fine-tuning larger open models on 48 GB

Status: PRD (2026-06-10). Phase-3 centerpiece of the roadmap (specialist
close-out → VLM/Pace → **this**). Implements the 2026-06-10 strategic shift:
focus moves from training small models to fine-tuning/distilling larger open
models. See `tinygpt-product-thesis.md` for the positioning this serves.

## Problem

tinygpt's trainer applies LoRA against an **fp16-materialized base**. Fine at
0.6B; impossible at the sizes that now matter:

| Base | fp16 weights | 4-bit weights | LoRA-trainable on 48 GB today? |
|---|---|---|---|
| Qwen3-0.6B | 1.4 GB | 0.4 GB | yes (current path) |
| Qwen3-4B | ~8 GB | ~2.3 GB | marginal fp16, easy at 4-bit |
| Qwen3-8B | ~16 GB | ~4.5 GB | no fp16 (grads+activations blow past 48 GB), yes at 4-bit |
| Qwen3-14B | ~28 GB | ~8 GB | impossible fp16, feasible at 4-bit |
| Qwen3-30B-A3B | ~60 GB | ~18 GB | teacher-only, never a student |

QLoRA = frozen quantized base + fp16/fp32 LoRA A/B (and DoRA m) as the only
trainable params. Optimizer state covers only the adapter (tens of MB at
r=32), activations stay fp16. This is the single unlock that makes 8–14B
adaptation fit the M5 Pro.

## What exists already (leverage inventory)

- `MLXNN.quantize(model:groupSize:bits:mode:filter:)` — per-layer filter
  callback; `QuantizedLinear` IS a `Linear` subclass
  (mlx-swift `Source/MLXNN/Quantized.swift:57,238`).
- `serve --quantize` shipped 2026-06-10 (int8 = 2.3× decode at zero fixture
  loss on the v9 planner) — the inference half is proven.
- LoftQ scaffolding in `PeftVariants.swift` already teaches adapters
  quantization error — directly reusable as the QLoRA init.
- Python mlx_lm trains LoRA on quantized models routinely — the underlying
  `quantized_matmul` primitives support the needed gradients (w.r.t.
  activations; base weights stay frozen). mlx-swift wraps the same core.
- `mlx_lm convert -q` produces packed-quantized safetensors; the native
  loader for these is `quantized-inference-swift.md` Phase 2 (#305) — a
  prerequisite shared with this work.

## Design

### Q-LoRA layer

New `QLoraLinear` (and `QDoraLinear`) in `Lora.swift`:

- wraps a `QuantizedLinear` base (packed weight + scales + biases, frozen)
- holds fp32 `loraA`/`loraB` (+ `m` for DoRA), the only unfrozen params
- forward: `quantized_matmul(x, base) + (alpha/r) · (x · Aᵀ) · Bᵀ`
- save path: existing `.lora` v2 format unchanged — adapters stay
  base-precision-agnostic, so a QLoRA-trained adapter loads on an fp16 base
  and vice versa (gate verifies this parity).

### Injection order (the invariant that bit us in serve)

quantize FIRST, inject SECOND. `LoraInjectionHF.makeAdapterLinear` gains a
`QuantizedLinear` branch; the existing mutual-exclusion warning in
serve/sample flips to the supported path once this lands.

### Loading

Two entry points, same model afterwards:
1. fp16 HF dir + `--quantize-base 4` → quantize in-memory, then inject
   (works today-ish; peak RAM = fp16 size during load).
2. Pre-quantized MLX safetensors (the #305 native packed load) → no fp16
   materialization at all; peak RAM = packed size. Required for 14B.

## Milestones (each gated, no silent regressions)

- **Q0 — go/no-go spike (0.5 d):** one training step through a
  `QuantizedLinear`-wrapped block in mlx-swift; verify grads reach A/B and
  loss decreases on a toy task. If grads don't flow, file upstream and stop.
- **Q1 — native packed load (1 d):** `mlx_lm convert -q` safetensors load
  without dequant (#305 design). Gate: logit cosine ≥ 0.999 vs Python mlx_lm
  on 5 prompts.
- **Q2 — QLoraLinear + trainer wiring (1–1.5 d):** train the v11 corpus on
  Qwen3-0.6B-4bit; gate = final loss within 5% of the fp16-LoRA run AND
  fixture score within 1 fixture. This validates the math cheaply before any
  big-model spend.
- **Q3 — first real target, Qwen3-4B (compute-bound):** QLoRA on the v11
  corpus + 30B-A3B-distilled data. **Zero-shot rule applies: measure raw
  Qwen3-4B on the full v11 gate FIRST; train only if it falls short, and
  report both columns.**
- **Q4 — scale to 8B** only if Q3's formula score says the next size up is
  worth the speed cost.

## Decision rules (carried from the specialist era)

1. Zero-shot base measured on the same gate before any training — the rule
   that would have saved v1–v10.
2. Formula = (speed × accuracy) / cost decides size: a 4B at int4 serves
   ~50–80 tok/s vs the 0.6B's 212 — accuracy must pay for that gap
   (accuracy > speed per the Pace decision framework, but measure, don't
   assume).
3. Every step has an automated gate (numerics or eval) or it doesn't ship.
4. Distillation signal: teacher = qwen3-30b-a3b (no thinking) or qwen3-14b
   (thinking) via LM Studio; data-level distillation first (v11 amplifier
   pattern), logit distillation only if data-level stalls.

## Risks

- **mlx-swift gradient gap through quantized ops** — Q0 exists to find this
  in half a day, not after building the stack. Fallback: train via Python
  mlx_lm (it has QLoRA today), keep Swift for inference only — less elegant,
  same product outcome.
- **Memory spikes at 14B** even packed (8 GB weights + activations + KV at
  ctx 1024) — start at 4B, measure RSS at each size; Mega-bf16 OOM lesson
  applies.
- **Adapter/base precision mismatch quality drift** — LoftQ init mitigates;
  the Q2 parity gate catches it.

## Done when

- Q2 parity gate passes on 0.6B-4bit
- One ≥4B model QLoRA-trained on-machine, evaluated against its own
  zero-shot baseline on the v11 gate, formula scores recorded
- Memory `feedback-focus-finetune-distill-large` updated with measured
  reality

## Related

- `docs/prds/quantized-inference-swift.md` — shares Q1; serving half shipped
- `docs/prds/tinygpt-product-thesis.md` — why this is the centerpiece
- Memory: `feedback-focus-finetune-distill-large`,
  `project-mega-bf16-oom`, `feedback-no-quality-regression`
