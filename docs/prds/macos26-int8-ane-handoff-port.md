# Port macOS 26 int8 direct ANE array handoff into M8 chain

Status: IN PROGRESS (2026-06-10) — Phase A SHIPPED, Phase B in flight. Spawned out of the 2026-06-09 research sweep + anemll-vs-M8 decision. Target: 17 tok/s → ~30 tok/s on the existing Qwen3-0.6B 28-block ANE chunked pipeline.

## 2026-06-10 implementation findings (override assumptions below)

1. **The "1.8× handoff" claim decomposes into two separate wins.** Inter-chunk
   array handoff (Phase A) and int8 weights (Phase B). Draw Things' 1.8× is
   mostly Phase B — weight bytes/token — not the boundary arrays (the
   boundary tensor is only 4 KB at H=1024).
2. **Phase A SHIPPED** (commit f2687ef): fp16-IO export + IOSurface
   CVPixelBuffer ping-pong via `MLPredictionOptions.outputBackings` (the
   anemll pattern — anemll does NOT use int8 handoff itself, contrary to
   the assumption below). Measured: prefill 14.1 → 17.1 tok/s (+21%),
   decode flat at 22 tok/s, numerics gate PASS (min cosine 0.9999957,
   100% top-1, ' Paris' canary).
3. **macOS26 deployment target BREAKS ANE binding on this machine**
   (macOS 26.0 / coremltools 9): int8-weight blocks converted with
   `minimum_deployment_target=macOS26` fail at predict with "Unable to
   bind buffer to network @ BindBuffers". int8 WEIGHT compression only
   needs macOS13+, so Phase B uses the macOS15 target — works (block 0:
   ANE load OK, cosine 0.999994, 16 MB package, per-channel
   linear-symmetric).
4. **Numerics gate exists**: `scripts/ane/m8_numerics_gate.py` — 5 prompts
   × 8 greedy steps vs saved fp32 baseline; PASS = 100% top-1 + cosine
   ≥ 0.999 + canary. Run on every phase. Baseline:
   `~/.cache/tinygpt/ane/m8-gate-baseline.npz`.
5. Phase C (int8 activation IO) stays gated/optional — it is the only part
   that truly needs the macOS26 target, which is currently broken (#3).

## What "int8 direct handoff" is

Before macOS 26: passing weights/activations into a CoreML model running on ANE required fp16 or int32 buffers. Any int8 quantized weight got promoted to fp16 at the array-binding boundary, eating ANE memory bandwidth.

In macOS 26 (shipped at WWDC 2026 yesterday): CoreML accepts `MLMultiArray` of dtype `int8` natively, and the ANE backend's int8 matmul kernel (~22 TFLOPs on M4, scaling on M5) is wired through. The bandwidth saving is the main win — 4 bytes → 1 byte per weight at the boundary, ~1.8× decode throughput on M4 per Draw Things engineering blog (2026-04-16).

anemll-bench has NOT adopted this primitive yet (per its docs). We can port it directly into our own M8 stack and bypass the whole anemll question.

## What needs to change in M8

Files in scope: `native-mac/Sources/TinyGPTModel/Qwen3ANEChunked.swift` (299 lines), `native-mac/Sources/TinyGPTModel/ANEInference.swift` (409 lines). Both already target `.cpuAndNeuralEngine`. The change is at the model conversion AND prediction-time array layer.

### 1. Block .mlpackage conversion

Today: each `m8-block-i.mlpackage` is converted via the Python `tinygpt to-coreml` exporter with fp16 weights. The fp16 weights get loaded into ANE memory at runtime.

Change: convert with int8 quantized linear weights using `coremltools.optimize.coreml.linear_quantize_weights` (granularity=`per_block`, mode=`linear_symmetric`, bits=8). Output `.mlpackage` is ~2× smaller on disk; ANE loads int8 + per-block scales directly.

The conversion is offline — re-run once per block. Re-bake the 28-block set.

### 2. MLMultiArray boundary types

Today's prediction path (per token):
```swift
// Build [B=1, S=1, hidden] fp16 input
let input = try MLMultiArray(shape: [1, 1, hiddenSize], dataType: .float16)
// fill from Swift Float buffer
```

Activation tensors stay fp16 (intermediate state must be float). The change is at the WEIGHT boundary which is now baked into the .mlpackage at convert time — no Swift-side change needed for static weights.

The optional Swift-side change is for INPUT tensors that the model expects as int8 (e.g., if we add an int8-quantized token embedding lookup). For Qwen3-0.6B the embedding is fp32 (tied with lm_head), so this stays fp16/fp32 at the boundary.

### 3. Compute precision (already correct)

M8 already uses FLOAT32 compute + FLOAT16 state per task #269. That stays — int8 weights get dequantized at-kernel-time inside ANE; activations remain float.

### 4. State buffers (no change)

`MLState` for k_cache / v_cache stays fp16. State buffers are NOT the bandwidth bottleneck on Qwen3-0.6B (small head dim, short context for v1).

### 5. Validation gate

After re-baking blocks with int8 weights, run on a 50-prompt Pace eval set. **Output divergence MUST be < 1pp on fm-fixtures-v2 pass rate** vs the existing fp16-weight M8 chain. If divergence exceeds 1pp, revert and investigate per-block scale calibration.

## Spike plan (2-3 days)

```
Day 1 — Tooling:
  - Verify coremltools 8.x quantize_weights API on a single Qwen3 block
  - Test that the resulting .mlpackage loads on ANE (.cpuAndNeuralEngine)
  - Run end-to-end with 1 block; measure activation correctness vs fp16 block

Day 2 — Production:
  - Re-bake all 28 blocks via the int8 path
  - Update Qwen3ANEChunked.load() if any dtype detection logic needs adjustment
  - Run formula score (scripts/score_formula.py) end-to-end on the new chain

Day 3 — Validation + ship:
  - Compare formula score vs fp16 M8 baseline
  - Smoke test on a sample of 100 prompts from the v9 traces
  - If accuracy delta ≤ 1pp AND speed ≥ 1.5×: merge, mark M8 as int8-v1
  - If not: revert, document failure mode, file bug against coremltools
```

## Acceptance criteria

- Formula score improvement: ≥ 1.5× speed_score, ≤ -2pp accuracy_score, cost_score ≥ 1× (smaller disk, similar RSS)
- All 28 blocks load on ANE without crashes on macOS 26.0.x
- v9-LoRA via the new int8 chain produces JSON outputs that score within 1pp of the fp16 baseline on fm-fixtures-v2

## Risk + mitigation

- **Block-level quantization breaks attention numerics**: per-channel calibration may be needed instead of per-block. Mitigation: re-run with `granularity=per_channel`.
- **macOS 26.0.1 ANE regression** (anemll Issue #39 cited this for chat loop). Mitigation: pin tests to macOS 26.0.0 + 26.0.1 + 26.1; reject any version that diverges.
- **Per-block scales add load-time cost** for 28 blocks. Mitigation: parallelize block loading in `Qwen3ANEChunked.load()`.
- **CoreML int8 may not work for stateful blocks**. Mitigation: bisect — quantize only the weight-heavy matmul layers (qkv + ffn), leave norm + softmax unchanged.

## Why this beats anemll migration

Per anemll-vs-M8 memo (today): anemll has no LoRA path, has open Qwen3 bugs on macOS 26, and doesn't yet exploit int8 handoff itself. We can port the SAME primitive directly into M8 in 2-3 days. Keeps ownership of the LoRA stack. No migration risk.

## Done when

- 28 blocks re-baked with int8 quantized weights
- formula_score.py shows ≥ 1.5× speed and ≤ -2pp accuracy delta
- v9-LoRA outputs match fp16 baseline on fm-fixtures-v2 within 1pp
- Documented in memory as `project-m8-int8-shipped-<date>` with measured deltas

## Related

- `feedback-tinygpt-north-star` — formula gate (5%/2-week bar; this ships ~30%)
- `project-anemll-vs-m8-2026-06-09` — the decision context
- `feedback-research-first-doctrine` — research that found this
- Draw Things engineering blog 2026-04-16 (primary source for the macOS 26 int8 numbers)
- `Qwen3ANEChunked.swift`, `ANEInference.swift` (the files to edit)
