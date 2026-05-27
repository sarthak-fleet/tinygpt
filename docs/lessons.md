# Lessons from building TinyGPT

What didn't work, what surprised me, and what I'd carry forward. The kernel
optimisations are documented in `docs/performance.md` and the milestones in
`docs/status.md`. This is the meta layer: the bugs that taught more than the
features.

## 1. The LR-default bug — config drift hides as a modelling ceiling

The browser default learning rate was `3e-3` for months. The Python reference
uses `3e-4`. Ten times too hot. Training plateaued at loss ~2.45 on real
corpora — a smooth curve, slowly asymptoting, exactly the shape of a model
that's run out of capacity. I spent two days suspecting the GPU kernels
before I noticed.

**Where it was:** `browser/src/types.ts:35` (`DEFAULT_CONFIG.learningRate`)
and the `value="0.003"` on the `#lr` input at `browser/src/pages/index.astro:2621`.

**Why it persisted:** kernel parity tests (`tests/test_webgpu_train.mjs`)
caught every numerical deviation in the maths. Nothing was catching
deviation in the *config defaults*. The reference path is the oracle for
the hyperparameters too — and the defaults need to be parity-checked the
same way the gradients are.

**Fix:** changed both occurrences to `3e-4`. The plateau dropped immediately.

**Lesson:** a "plateau" is a hypothesis, not a fact. Cross-reference the
hyperparameters against the reference path before suspecting the model.

## 2. The Memory64 ABI was untested

The 64-bit WASM module (`browser/public/tinygpt64.{js,wasm}`) compiled with
`-sMEMORY64=1 -sWASM_BIGINT` was claimed-shipped — it allocates a 473M-param
model cleanly in Node when called directly, and the browser feature-detects
and picks it up automatically for big presets. But the browser was hitting
"memory access out of bounds" at d_model ≥ 256 (XL, Massive, Mega, Behemoth).
Same `.wasm`, different host.

The trap: `tests/bench_wasm.mjs` loads the 32-bit `tinygpt.js`, not the
64-bit one. So the 64-bit pthread+Memory64 path had been **completely
untested in Node**. "Node passes XL" was a misleading data point — the test
exercises the 32-bit kernels.

**Reproducer:** `tests/test_wasm64_xl_node.mjs`. Load `tinygpt64.js` in
Node, call `tg_model_create` for XL, call `setData`:

```
TypeError: Cannot convert 80312 to a BigInt
    at ccall (tinygpt64.js:1:24518)
```

`_malloc` returns a Number (because the heap is still below 2³¹), but
`cwrap` with type `"bigint"` for pointer args is strict and won't auto-coerce
a Number argument. The browser was calling into this broken bridge every
time it tried to allocate at scale; the OOB was a downstream consequence of
the malloc never landing where the kernels expected.

**Lesson:** if a build target ships, it needs a test that exercises it on
the same data path the production caller uses. "Same `.wasm`" is not the
same as "same execution path". Parity is per-host, not per-binary.

**Status:** task #66; not yet fixed. Likely fix is to update
`browser/src/backend.ts` to explicitly `BigInt()`-wrap pointer args before
every `cwrap`-bound call into `tinygpt64`, plus an integration test that
runs every preset against `tinygpt64.js` in Node so this can't silently
regress again.

## 3. The speedup is a curve, not a single number

I shipped a flat "9.7× faster" headline based on Small-preset measurements.
Measuring across the full preset table revealed the real picture:

| preset | d_model | speedup (WebGPU vs WASM SIMD multi-thread) |
| ------ | ------- | ------------------------------------------- |
| Small  | 96      | **2.6×** |
| Medium | 128     | **6.8×** |
| Large  | 192     | **9.3×** |
| XL     | 256     | **12.1×** |

The curve trends upward because GPU work amortises better as matmul shapes
grow — at small `d_model` and `ctx`, command-buffer dispatch cost is a
non-trivial fraction of step time; at XL it's noise. The 9.7× number was
true for Medium-class shapes; it sold the bigger presets short and
overstated the smaller ones.

**Lesson:** if a performance number is a function of a tuneable, publish
the function, not one point. "9.7×" reads as a property of the code; "2.6×
→ 12.1×" reads as a property of the problem. The second is more honest and
ages better.

## 4. The default training corpus was 863 bytes

For most of the project, the playground's `#corpus` textarea was
pre-populated with an inline meta-explainer paragraph — ~863 bytes of prose
about how language models learn. New users hitting "Train" with the Huge
preset got 9.6M parameters of model trying to fit ~863 tokens of data.

Inevitable outcome: the model trivially memorised it (train loss 0.14 in
14 minutes), generated verbatim quotations, and convinced nobody.

**Fix:** default corpus is now the full TinyShakespeare (~1.1 MB) fetched
on init from `/shakespeare.txt`. A 9.6M-param Huge model on 1.1 MB of text
is still under-data by classical ratios, but it's enough to learn — not
just to memorise. After 15 min the model produces readable pseudo-Shakespeare.

**Lesson:** the default training data is part of the demo's promise. If the
default doesn't produce readable output, the demo doesn't show what the
project does. Treat data defaults with the same care as model defaults.

## 5. Standalone benchmarks lie; end-to-end parity is the only honest bar

This one came up three times in this project, so it earns its own entry:

- **vec4 matmul** integrated cleanly in standalone benchmarks (1.37× over
  blocked at 2048³), then diverged loss to 88.67 in end-to-end training.
  Root cause: WGSL `var<storage, read>` declared on bindings whose
  bind-group layout was `buffer: { type: "storage" }` (read-write). Apple's
  WebGPU silently returned wrong data instead of erroring.
- **f16-packed weights** were 1.7× faster standalone. Stacked on top of the
  tiled matmul they made the kernel *slower* (17.78 ms vs 16.90 ms at 2048³),
  because tiling had already amortised the global-memory reads it was
  saving.
- **8×8 register blocking** beat 4×4 in the spreadsheet — 4× more arithmetic
  per shared-memory load. Lost at every benchmarked size (0.91× at 1024³,
  0.88× at 2048³), almost certainly because 64 floats per thread for the
  accumulator exceeded Apple's per-thread register budget and forced spill.

**The pattern:** standalone benchmarks measure the kernel in isolation, in
ideal shapes, often square. Real training calls them in awkward
non-square shapes, with state from previous kernels in cache, and inside
a pipeline that interacts with the kernel's compute/memory mix. Anything
that wins standalone has to be re-verified end-to-end before it counts.

**Lesson:** for this project, "shipped" means "passed
`tests/test_webgpu_train.mjs` at 50 steps with drift < 5%". Nothing else.

## 6. Algorithm-in-JS-first for new kernels

The FA2 forward had a non-obvious algorithm — workgroup-cooperative tiling
with online softmax accumulated across K blocks, with a recompute-aware
backward pass that reads only the saved log-sum-exp instead of the full
attention matrix. Writing that directly in WGSL would have been hostile to
debugging.

The pattern that worked: implement the algorithm in vanilla JS first
(`tests/test_fa2_parity.mjs`, `test_fa2_backward_parity.mjs`), get the
parity tests passing against the naive reference, *then* port to WGSL.
The JS version stays as a regression test forever. When the WGSL version
diverges, you have a working oracle to bisect against.

**Lesson:** if you're not 100% sure you understand the algorithm, write it
in JS first. The translation to WGSL becomes a typing exercise instead of
a problem-solving exercise, and you keep the JS as a permanent oracle.

## Where these lessons live

- README.md cites this doc in the Lessons section.
- BLOG.md (the longform post) folds the LR + corpus + curve stories into
  its narrative.
- `browser/src/pages/devlog.astro` keeps the per-entry record.
- `docs/performance.md` documents the *what* shipped; this file is the
  *what didn't* and the *what I learned about the process*.
