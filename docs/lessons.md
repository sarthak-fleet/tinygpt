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

## 2. The Memory64 OOB was a pthread + memory-growth race, not an ABI bug

The 64-bit WASM module (`browser/public/tinygpt64.{js,wasm}`) was hitting
"memory access out of bounds" in the *browser* at d_model ≥ 256 — XL,
Massive, Mega, Behemoth. The exact same `.wasm` ran fine when called
directly in Node. Two days of looking turned up two unrelated findings.

**The misdirection.** The first reproducer
(`tests/test_wasm64_xl_node.mjs`, v1) used `cwrap` with bigint-typed
pointer args and threw `TypeError: Cannot convert 80312 to a BigInt`.
Concluded: ABI bug, cwrap doesn't auto-coerce Number→BigInt. *Wrong
diagnosis.* The browser doesn't use cwrap — `browser/src/backend.ts:118`
already has a `toPtr` helper that BigInt-wraps pointers before every
direct exports call. The cwrap failure was a problem with the test, not
the production path.

**The real diagnosis.** Rewriting the reproducer to mirror exactly how
`backend.ts` calls into the module — direct exports, `BigInt(ptr)` on
every pointer arg — made the Node-side test pass cleanly: 5 XL training
steps, ~2.2 s each, loss descending 5.57 → 3.04. The 64-bit kernels are
correct. The build is correct. The browser path is calling the same
correct kernels with the same correct arguments. The difference must be
*how* the host runs them.

The likely culprit is the well-known interaction between Emscripten's
pthread shim, `SharedArrayBuffer`, and `WebAssembly.Memory.grow`. In the
browser:

```
main thread mallocs → memory.grow → SAB reallocated
worker thread (matmul, mid-call) holds stale HEAPF32 view → reads past
the old view's length → wasm trap: memory access out of bounds
```

Emscripten's pthread shim notifies workers of growth through atomics, but
the notification isn't synchronous with the worker's currently-running
kernel; if the worker is inside a matmul tile loop when growth happens,
the next `HEAPF32[i]` read can hit the stale view. Node's
`worker_threads` shim has a different update path that doesn't race the
same way, which is why the Node test passes.

**The fix.** The simplest reliable fix is to avoid growth entirely during
training: bump `INITIAL_MEMORY` in `wasm/build_wasm64.sh` from 32 MB to
256 MB. XL needs ~350 MB live; Small/Medium/Large/XL all train without
ever triggering `memory.grow`. (Behemoth still needs growth — the eventual
proper fix is `GROWABLE_HEAP_*` helpers in the C++ kernels so stale views
re-resolve, but that's a bigger change.)

**The other trap.** `tests/bench_wasm.mjs` loaded `tinygpt.js` (32-bit),
not `tinygpt64.js`. So the 64-bit module had been entirely *unmeasured*
in Node before this session. Citing "Node passes XL" was meaningless — it
was passing a *different binary*. Per-host parity, not per-binary parity,
is what counts.

**Lesson:** the cheapest reproducer is the one that mirrors the production
path exactly. Substituting cwrap for direct exports — or one host for
another — moves you to a different bug. Build the reproducer that
matches, then trust it.

**Status:** fixed for everything up to and including XL with the
`INITIAL_MEMORY=268435456` bump in `wasm/build_wasm64.sh`. Behemoth still
exercises growth and would benefit from the proper SAB-view-aware path.

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
