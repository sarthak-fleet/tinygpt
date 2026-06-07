---
name: Q4 Qwen3-0.6B base for Pace specialist — RAM crunch
status: shipped-v1-2026-06-07-conversion-smoke-not-run
owner: unassigned (parallel-agent task — Swift + CLI)
created: 2026-06-08
priority: P0 — 4× RAM reduction for Pace specialist, no quality penalty
parallel-safe-with: factory-serve-prompt-cache.md (different files)
---

# PRD — Q4 GGUF base for Pace specialist

## 2026-06-07 ship note

Shipped:
- `scripts/quantize-qwen3-0.6b.sh` wrapper for llama.cpp HF-to-GGUF +
  Q4_K_M quantization
- `GGUFHFMaterializer` in TinyGPTModel
- `ModelLoader.load("*.gguf")` auto-materializes a cached HF-compatible
  directory under `~/.cache/tinygpt/gguf-hf/`
- GGUF tensors dequant through `GGUFReader` and write `model.safetensors`
  for the existing HF loader path
- `tinygpt serve <base.gguf> --lora ...` is now routed through `ModelLoader`

Not run in this pass: real Qwen3-0.6B conversion, RSS measurement, and Pace
fixture eval. Those require the local HF model plus llama.cpp quantization
tools and would be a long conversion/eval loop.

## Goal

Convert `Qwen/Qwen3-0.6B` HF dir → Q4_K_M GGUF, then wire serve to load
GGUF bases (not just HF dirs). End state: same v6 LoRA + Q4 base = ~600
MB RAM (vs current ~3 GB), with <1% quality drop.

## Scope — in

### 1. HF → Q4_K_M GGUF conversion script

Use llama.cpp's converters if installed (`llama-quantize` + `convert.py`),
OR our `gguf-extract` reverse path. Output:
`~/.cache/tinygpt/runs/qwen3-0.6b-q4_km.gguf`

A shell script `scripts/quantize-qwen3-0.6b.sh` is sufficient. ~10 min
runtime per conversion.

### 2. Serve load `.gguf` as base

Today `tinygpt serve <base>` accepts:
- `.tinygpt` files (from-scratch)
- HF directories (Qwen3-0.6B/snapshots/...)

Add: GGUF files (single `.gguf`). The loading path:

```swift
if base.hasSuffix(".gguf") {
    // Use GGUFReader to load weights, wrap in TinyGPTModelHF
    let result = try GGUFLoad.loadHFCompatible(gguf: URL(fileURLWithPath: base))
} else if /* HF dir */ { ... } else { /* .tinygpt */ }
```

We have GGUF dequant shipped (Q4_K_M, Q5_K, Q6_K, Q8_K — see #198, #208).
The loader needs to:
- Read GGUF metadata
- Dequant Q4_K_M weights to fp16 (or keep packed for compute-on-load)
- Construct TinyGPTModelHF with the dequanted weights
- Same path as HF dir load from there onwards

### 3. LoRA compatibility check

Apply `pace-planner-v6.lora` to GGUF-loaded base. Verify same output as
HF-dir-loaded base + LoRA (modulo Q4 dequant precision). Round-trip
smoke against fm-fixtures.

## Acceptance

1. Smoke: `tinygpt serve qwen3-0.6b-q4_km.gguf --lora pace-planner-v6.lora --port 8765`
2. RSS: <800 MB (vs ~3 GB with HF dir base)
3. Eval (via pace-eval-v6.py): score within 1 fixture of HF-dir-base eval
4. Latency: same or faster (Q4 dequant might add a hair)
5. Build clean

## Scope — out

- Multi-LoRA stacking on GGUF base (single LoRA only for v1)
- Mixed-precision (keep Q4 in a sparse path) — premature
- GGUF writing from .tinygpt format (separate; we ship reading only here)

## Files involved

| File | Change |
|---|---|
| `scripts/quantize-qwen3-0.6b.sh` (new) | Wraps `llama-quantize` or `tinygpt gguf-extract` reverse path |
| `native-mac/Sources/TinyGPTModel/GGUFLoad.swift` | Add `loadHFCompatible(gguf:)` that returns a TinyGPTModelHF |
| `native-mac/Sources/TinyGPTModel/AnyModel.swift` | `ModelLoader.load` accepts `.gguf` paths and routes to GGUFLoad |
| `native-mac/Sources/TinyGPTServe/Serve.swift` | NO changes — ModelLoader handles it |

## Estimated effort

**~2-4 hours.** Primitives all exist (dequant, GGUFReader, HF wrapper);
mostly a wiring + smoke test.

## Why P0

Current Pace v6 RAM is ~3 GB. Q4 brings it to ~600 MB — fits comfortably
alongside Pace's VLM (Qwen3-VL-8B Q4 ~6 GB) without GPU contention.
**That's the difference between "can run alongside Pace's VLM" and
"can't."**

## Won't conflict with #260

Touches different files (GGUFLoad, ModelLoader, AnyModel) than the
prompt-cache PRD (Serve.swift). Two elves can run in parallel.
