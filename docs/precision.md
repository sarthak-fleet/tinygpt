# Numerics & precision — the gate framework

How TinyGPT ensures every accelerated code path preserves model quality
before it activates for the user. The non-negotiable rule from the
opportunistic-acceleration scope (see `docs/perf_quest.md` and decision
log entry 19): **speed only counts if it preserves loss.**

## The principle

Each fast path falls into a tiered design:

1. **Feature-detect at startup.** The browser must expose the underlying
   capability (WGSL extension, navigator API, adapter feature).
2. **Verify against the f32 reference before activating.** Run a
   representative matmul on both the new path and the canonical f32 path,
   compare element-wise. The path activates only if the comparison passes
   tight thresholds.
3. **Surface in the capability pills.** If active, render a pill like
   `+f16 storage`. If detected-but-rejected (gate failed), the pill stays
   off — but the corresponding console log records what happened.
4. **Fall back automatically.** If the gate fails, the user gets the
   slower-but-correct path; nothing breaks, nothing's silently degraded.

## The first gated path: f16-storage matmul

**What it does.** The matmul kernel reads weight tensors from a packed-half
(f16) GPU buffer instead of the usual f32. `pack2x16float` /
`unpack2x16float` are core WGSL built-ins — no `shader-f16` extension
required, works on every WebGPU device. Accumulation stays in f32 for
numerical safety; only storage is packed-half. The B-side of the matmul
(the weight) is the bandwidth bottleneck, and halving its byte footprint
roughly halves the inner-loop K-step bandwidth. Expected gain on
bandwidth-bound shapes: ~1.5-2×.

**Where it lives.**
- Shader: `webgpu/train_f16.wgsl` (`matmul_blocked_f16`, `pack_to_f16`).
- Dispatch + gate: `webgpu/ops.ts` (`matmulF16Weight`, `packToF16`,
  `verifyF16Storage`, `f16Ready`).
- Wiring: `webgpu/gpu_model.ts` (`prepareForInference`, `linear`'s
  opportunistic branch, `invalidateF16Cache` on `trainStep`).
- UI: `+f16 storage` pill, posted via the worker's `gpu_caps` message.

**Activation scope.** Forward-pass matmuls in `GpuModel.linear` (Q/K/V/O
projections and MLP fc_in / fc_out) — those are the matmul-B targets where
the weight is read but not mutated. Training-time matmuls (`matmulAtb`,
`matmulAbt` in backward) stay f32; AdamW updates the f32 weight, so any
cached f16 buffer becomes stale and is invalidated at the top of every
`trainStep` (`invalidateF16Cache`). The tied head logits matmul
(`matmulAbt(lnf, tokEmb)`) also stays f32. So the path benefits sampling
end-to-end and the forward half of training, with full safety on the
mutation path.

## The gate — design and thresholds

The gate runs at `GpuOps.create()` time as a background promise
(`f16Ready`); callers await it before activating the fast path. Test
shape: `M=64, K=128, N=128`. Inputs: deterministic seeded random in
`[-0.2, 0.2]` (typical weight initialization scale).

**Tolerance has to be magnitude-aware.** A naïve `max(|f - r| / |r|)`
relative-error metric explodes when the reference output `r` is near zero
(the dot-product cancels) — a per-element error of 1e-4 against a true
value of 1e-6 reads as 10000% relative, even though the network downstream
won't notice. The framework floors the relative-error denominator at 1%
of the *mean* |reference|, which reflects "would the downstream network
notice this?" rather than "did the bit-exact value match?".

Two thresholds, both must hold:

```
max_abs_err   < 1% of mean |reference|       (catches one-off blow-ups)
mean_rel_err  < 0.5%                         (catches systematic bias)
                where rel = |f - r| / max(|r|, 1% of mean |ref|)
```

The `0.5%` mean threshold is right at the theoretical edge of a
f16-packed-storage dot product (~sqrt(K) × eps_f16 ≈ 5.6e-3 for K=128).
A path that fails this is likely a real bug, not normal f16 drift.

**Measured on M-series WebGPU (May 2026):**

```
mean|ref|   = 1.22e-1
max_abs     = 1.33e-4  (limit 1.22e-3 — 11% of budget)
mean_rel    = 0.075%   (limit 0.500%   — 15% of budget)
max_rel     = 5.40%    (informational only — not gating)
verdict     = PASS
```

End-to-end smoke (`browser/smoke_f16.mjs`): bundled Shakespeare model
generates identical text on the f16 path and the f32 path. No quality
regression observable.

## Extending the framework — pattern for #91, #92, #93

Each future fast path (shader-f16 compute, cooperative matrix, WebNN
inference) follows the same five-step recipe:

1. **Define the kernel / API path.** New WGSL shader (or WebNN graph). Same
   bind-layout convention as `train.wgsl` where possible, so dispatch
   plumbing stays minimal.
2. **Register the pipeline.** Add an entry to the `ENTRIES` const-array in
   `ops.ts`; compile the shader module in `GpuOps.create()`.
3. **Implement the path-specific dispatch method.** E.g.,
   `matmulCoopMatrix(a, b, M, K, N)`. Accept the same input shapes as the
   f32 path so the swap is at the call site.
4. **Implement the gate.** Compute outputs on both the new path and the
   f32 reference at a representative shape; apply the same
   magnitude-aware tolerance; flip the `xxxActive` flag in `GpuOps`. Add
   the path to `f16Ready` / equivalent settling promise so callers can
   await.
5. **Wire the activation.** Add a `useXxx` check in the dispatch chain
   (`linear`, `matmulAbt`, whatever the hot path is). For paths that need
   pre-processing (like weight packing), gate that pre-processing on the
   `xxxReady` promise resolving to true.
6. **Surface in UI.** Worker posts `gpu_caps: { xxx: true }` once the path
   activates. Main thread appends the pill to `#caps .gpu-accel`.

**Anti-pattern to avoid.** Activating a fast path based on capability
detection ALONE, without running the gate. A device exposing the
capability does not mean the numerics will pass for our specific
workload. The gate is what makes opportunistic safe.

## How to read the console output

```
[ops] f16-storage gate: mean|ref|=A, max_abs=B (limit C),
                        mean_rel=D% (limit E%), max_rel=F% — PASS|FAIL
```

- `mean|ref|`: the typical magnitude of f32-reference outputs. Used as the
  baseline for both thresholds.
- `max_abs`: largest single-element absolute error.
- `mean_rel`: average relative error against the floored denominator.
- `max_rel`: the worst-case relative error — informational only because
  near-zero outputs can inflate it without representing real drift.
- Verdict `PASS` → the path activates; `FAIL` → fall back to f32 silently.

Reading two more lines:

```
[ops] f16-storage gate: ... PASS — f16 path active
inference warmed up in 0.5s · f16-storage matmul active — ready to generate.
```

The status bar appends `· f16-storage matmul active` so the user sees
which path their model is running on. The capability pill cluster also
gains a `+f16 storage` chip.

## What the gate does NOT catch

- **Slow drift across many steps.** The gate runs once; it doesn't catch
  precision loss that compounds across a long inference (1000+ tokens).
  Mitigation: f32 accumulation in the kernel (we do this) bounds the
  per-step error.
- **Catastrophic cancellation in unusual inputs.** Random inputs in
  `[-0.2, 0.2]` are representative but not exhaustive. Adversarial inputs
  could theoretically exploit f16 representation gaps. Not a realistic
  concern for our workload.
- **Path-specific bugs that only show on certain shapes.** The gate uses
  K=128. If a path is correct at K=128 but wrong at K=2048, the gate
  passes and the model breaks. Mitigation: when extending the gate to
  new paths, include shape samples that cover the production workload.

## Future work

- **Multi-shape gate.** Test at three representative shapes —
  inference-typical (K=256), training-typical (K=1024), and head-output
  (K=vocab). Reject if any of them diverges.
- **End-to-end sample comparison.** After the gate, run a single
  generation with the same prompt on the f32 path and the fast path;
  compare the first 100 tokens. If divergence, fail.
- **Precision telemetry.** When the path activates, record the gate
  measurements via `trackPlaygroundLoaded` (PostHog) so we can spot
  systematic regressions across users / driver versions.
- **`docs/precision.md` per-lever sections.** When #91 (shader-f16
  compute) ships, add a section here with its measured numerics. Same for
  #92 (cooperative matrix) and #93 (WebNN). One file, all the precision
  truth, in one place.

This doc is the source of truth for the gate framework. Update it as new
levers land — measured numerics, what the gate caught (or didn't), and
the activation status of each path.
