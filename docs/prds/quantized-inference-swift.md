# Quantized inference in tinygpt's HFModel — Swift-side QuantizedLinear

Status: PRD (2026-06-09). Phase 1 SHIPPED 2026-06-10 (`serve --quantize`); native packed load (the design below) still open.

## 2026-06-10 Phase 1 findings — `serve --quantize int4|int8`

Shipped (commit d3b7879): in-memory `MLXNN.quantize` after load, same mechanic as
`sample --quantize`. Skipped under `--lora`; for adapters, bake first
(`bake-lora` → `serve --quantize`). Quantize config folded into the KV
prompt-cache fingerprint.

Measured on the shipping planner (pace-planner-v9-lora/baked-hf, Qwen3-0.6B,
M5 Pro, release build, fm-fixtures-v2 with v9 compose-v2 prompt + v9 schema +
serve-side `--grammar`):

| config | fm-fixtures-v2 | decode (best of 3) | verdict |
|---|---|---|---|
| fp16 | 9/15 (60.0%) | 92 tok/s | baseline |
| **int8 g64** | **9/15 (60.0%)** | **212 tok/s (2.3×)** | **zero quality loss — recommended** |
| int4 g64 | 8/15 (53.3%) | 282 tok/s (3.1×) | -1 fixture: fails no-regression gate |

- **int8 is the free lunch**: 2.3× decode, identical fixture score. int4 is
  faster still but drops a fixture — per the no-quality-regression rule, int8
  is the planner default and int4 needs a per-model gate run before use.
- RSS is flat (~3 GB): load is still fp16-materialize-then-quantize. The real
  memory/disk win needs the native packed load below — that remains the open
  Phase 2.
- Two eval-setup traps (cost ~30 min): per-request `response_format` does NOT
  trigger constrained decoding — the grammar must be passed serve-side via
  `--grammar`, or the model free-runs `<think>` and scores 0%. And the frozen
  33.3% Dim1 baseline appears to have used a different prompt/schema config:
  with v9 compose-v2 prompt + v9 schema + grammar, fp16 v9 scores 60.0% —
  worth re-examining the Dim1 baseline before the v11 run.

## Problem

`mlx_lm convert -q --q-bits 4` produces MLX-quantized safetensors:
- Weights stored as `uint32` (8 values per uint32 for 4-bit)
- Companion tensors per quantized weight: `<name>.scales`, `<name>.biases`
- `config.json` has `quantization` block with `bits`, `group_size`, `mode`

Tinygpt's `HFModel.swift::makeMLXArray` (line 442) fatal-errors on `U32` dtype. The loader has no concept of quantized weights.

Result: v9-LoRA baked-hf is 1.4 GB fp16. Quantized to Q4 is 331 MB (4.2× smaller). But tinygpt serve cannot load it → Pace ships fp16.

## Target outcome

Tinygpt serve loads `mlx_lm convert -q` output natively. Pace bundles 331 MB instead of 1.4 GB. Inference uses MLX's `quantized_matmul` ops directly on packed weights — no dequant-at-load.

Estimated formula payoff:
- **cost**: 4× lower disk + RAM
- **speed**: 1.2-1.5× faster on M-series (quantized matmul is bandwidth-bound; fewer bytes to move)
- **accuracy**: ~1-2% drop on Q4 (per published Qwen3 quant benchmarks)
- Net formula: ~30-40% boost — clears the 5%/2-week bar 6-8× over

## Design

### 1. Detect quantization at load

In `HFModelLoader.load()`, parse `config.json["quantization"]` (or `quantization_config`). If present, set a `quantized: QuantizationSpec` on the load result with `{bits, groupSize, mode}`.

### 2. Detect quantized tensors during weight load

`HFModel.swift::loadShard`, walking each tensor:
- If tensor name ends in `.weight` AND there exist `.scales` + `.biases` peers in the shard's index → this is a quantized weight.
- Load the `.weight` as `MLXArray` of dtype `uint32` (don't upcast).
- Load `.scales` and `.biases` as their declared dtypes (typically bf16).

### 3. Extend `makeMLXArray` for U32

Add a `U32` case that builds `MLXArray(uint32-buffer, shape)` without conversion.

### 4. Wrap Linears as QuantizedLinear at model construction

The hardest part. Today `TinyGPTModelHF` creates standard `Linear` instances. Need:

```swift
// At construction time, if quantizedSpec present:
class HFAttention {
    @ModuleInfo(key: "q_proj") var qProj: Linear  // → QuantizedLinear
    ...
}
```

Two paths:
- **A**: Always declare as protocol/base `Linear`, swap concrete instance at load. mlx-swift's `QuantizedLinear` IS a `Linear` subclass, so this works as a value type, but @ModuleInfo wants a concrete type for reflection.
- **B**: Parameterize the model class on quantization spec, conditionally use `QuantizedLinear` at construction. Cleaner but requires a second model variant or generics.

Recommended: **A**, with a post-load step that replaces each targeted `Linear` with a `QuantizedLinear` via `model.update(modules:)`. mlx-swift's `Module.update(modules:)` allows substituting child modules.

### 5. Wire scales + biases into the QuantizedLinear's params

After substituting `Linear` → `QuantizedLinear`, write the loaded uint32 weight + bf16 scales + bf16 biases into the new module's params via `model.update(parameters:)`.

### 6. Verify forward pass

mlx-swift's `QuantizedLinear.callAsFunction` does `quantized_matmul` automatically. No changes needed in attention/MLP code — they call `qProj(x)` which dispatches polymorphically.

### 7. Smoke + accuracy test

- Load v9-LoRA-baked → quantize via `mlx_lm convert -q` → load via tinygpt serve.
- Run fm-fixtures-v2 + compose suite.
- Compare to fp16 baseline: accuracy delta should be ≤2% absolute (vs current 33% + 70%).
- Measure: disk size, RSS during inference, TTFW, tok/s.
- Compute formula = (speed × accuracy) / cost; expect ~30% improvement over fp16.

## Scope estimate

~1 full day of focused Swift:
- 1 hr: dtype handling + quantization-detection in loader
- 2 hr: model-substitution pattern with mlx-swift Module.update
- 1 hr: scales/biases parameter wiring
- 2 hr: tests + debugging (mlx-swift quirks)
- 1 hr: end-to-end smoke + formula comparison

## Compatibility

- v8/v9 fp16 baked-hf: untouched, load same way (no quantization block in config)
- New Q4/Q8 baked-hf: detected via config.json, routed through quantized path
- LoRA on quantized base: separate concern — DoRA/LoRA wrapping over QuantizedLinear. Defer to v2 of this work; v1 supports quantized base only.

## Alternative — dequant-at-load (rejected)

Could use `dequant_mlx_generic.py` to dequant Q4 → fp16 before loading. Saves disk only, no RAM/speed wins. Defeats the formula payoff. Reject.

## Alternative — switch Pace to mlx_lm.server (rejected)

mlx_lm has its own serving with native quantized model support. But Pace's plan per `tinygpt/docs/prds/scope-narrowing-tinygpt-as-factory.md` is embedded MLX-Swift inference (no HTTP at runtime). mlx_lm is Python — doesn't fit. Reject.

## Done when

- Tinygpt serve loads `mlx_lm convert -q --q-bits 4` output without error
- Forward pass produces sensible output (smoke test: "open my email" → valid JSON)
- fm-fixtures-v2 score within 2pp of fp16 baseline
- Formula score for Q4 > formula score for fp16 by ≥20%

## Related

- `feedback-tinygpt-north-star` — the formula
- `feedback-pace-decision-framework` — accuracy > speed > footprint
- mlx-swift `QuantizedLinear` reference: github.com/ml-explore/mlx-swift `Sources/MLXNN/QuantizedLinear.swift`
- Existing dequant tooling: `scripts/ane/dequant_mlx_generic.py` (the inverse — can re-use for testing)
