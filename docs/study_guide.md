# TinyGPT study guide

The technical material this session touched, organized so you can pick what to learn at your own pace. The chapters are thematic, not chronological. Each one answers the same two questions in a different domain: *what is this thing*, and *why does it matter for TinyGPT specifically*. You don't need to read them in order. If you're comfortable with JavaScript and have shipped a neural net in PyTorch, the prerequisites are met — the rest is the connective tissue between transformer training internals, GPU shading languages, and the particular way the modern web shoves both of those into a static page.

---

## 1. Why the speedup is a curve, not a number

For most of the project's life, TinyGPT advertised "9.7× faster than the Python reference" as a single headline. That number was real, but it was the value of a function evaluated at exactly one point — the Medium preset (d_model = 128). When the same measurement was re-run across the preset table the picture changed shape:

| preset  | d_model | WebGPU vs WASM-SIMD multi-thread |
| ------- | ------- | -------------------------------- |
| Small   | 96      | **2.6×** |
| Medium  | 128     | **6.8×** |
| Large   | 192     | **9.3×** |
| XL      | 256     | **12.1×** |

The cleanest mental model for why it climbs monotonically is to decompose each step into two additive terms:

```
step_time(d) = fixed_overhead + math_cost(d)
```

`fixed_overhead` is roughly the same on both backends: a worker `postMessage` round-trip, a command-buffer or kernel-dispatch setup, a couple of `Atomics.wait` synchronisations. It's a constant in `d_model`. `math_cost(d)` is what changes — the matmuls inside each transformer block scale roughly as `d_model²` per token, and the GPU's arithmetic throughput is many times higher than a CPU's vectorised SIMD.

At Small (d_model = 96), `fixed_overhead` is a meaningful fraction of step time on both backends, so the ratio compresses toward 1. By XL, `math_cost(d)` dominates and the ratio approaches the *intrinsic* GPU-vs-CPU arithmetic gap. Extrapolating to Mega/Behemoth (d_model ≥ 384) you would expect the ratio to keep climbing until it asymptotes around whatever the silicon's true compute-throughput ratio is on the M-series chip.

This pattern generalises far beyond TinyGPT: any time you publish a single "Nx faster" number for a workload whose cost is a function of a tuneable, you are implicitly claiming the value of that function at one point. Publish the function instead.

**Reference**: `docs/lessons.md` §3 (the curve story); `docs/session_retrospective.md` §1 for the longer arc; `browser/measure_curve.mjs` is the actual measurement harness.

---

## 2. Why the LR matters more than people think

The browser shipped with `DEFAULT_CONFIG.learningRate = 3e-3` for months. The Python reference (`python_ref/train.py`) uses `3e-4`. A 10× hot learning rate on Adam-class optimisers is one of those bugs that doesn't look like a bug — it looks like a modelling ceiling.

To see why, picture the loss landscape around the minimum as a long narrow valley. Adam takes adaptive steps scaled by the running second moment of the gradient; the *effective* step size is roughly `lr / sqrt(v)`. With `lr = 3e-3`, the steps near the bottom of the valley overshoot the minimum — the optimiser oscillates across the valley walls instead of descending the floor. The loss curve looks beautiful: smooth, monotonically decreasing for the first few hundred steps, then visibly flattening out at a noise floor (around 2.45 nats on TinyShakespeare-class data). It looks exactly like a model that has run out of capacity.

The session spent two days suspecting GPU kernels — FA2 numerics, matmul tile parity, softmax precision, layernorm gradients — before someone noticed the defaults didn't match. The kernel parity tests (`tests/test_webgpu_train.mjs`) caught every numerical deviation in the maths. Nothing was catching deviation in the *config defaults*. The reference path is the oracle for hyperparameters too, not just for gradients.

Fix: `browser/src/types.ts:35` and the `value="0.003"` on the `#lr` input at `browser/src/pages/index.astro:2621` both changed to `3e-4`. A 5,000-step run on the Huge preset converged to a training loss of 1.30, well into the "samples read as legible Shakespeare" range. Nothing else changed.

The deeper lesson is that "loss has plateaued" is not a fact — it's a hypothesis with two competing explanations: the model has run out of capacity, *or* the optimiser has run out of resolution. Cross-reference your hyperparameters against the reference path before committing to the capacity story.

**Reference**: `browser/src/types.ts:28-41`; `docs/lessons.md` §1; `docs/session_retrospective.md` §2.

---

## 3. Browser-WASM is ~15× slower than Node-WASM

On the same machine, running the byte-identical compiled `.wasm` module, a Small-preset training step takes ~15× longer in Chrome than in Node. The kernels are unchanged; the host pays the entire gap.

The cost is concentrated in the pthread shim. Emscripten's WASM-threads runtime needs a way to spawn workers, share memory between them, and synchronise via atomics. In Node, it lowers `pthread_create` onto `worker_threads`, sharing memory through `SharedArrayBuffer` and synchronising via `Atomics.wait` / `Atomics.notify` — all happening in a single process with a single V8 isolate, so cross-context cost is low. In the browser the same shim spawns Web Workers, each of which is its own isolate with its own event loop. Cross-context synchronisation goes through structured-clone (or SAB pointers), the cross-origin-isolation enforcement (COOP/COEP) gates every shared-memory operation, and every `postMessage` between workers and main thread is at least one event-loop turn.

At Small, the per-call overhead is comparable in magnitude to the per-call work. A single matmul might do 2-3 ms of arithmetic but pay 1-2 ms of orchestration; in Node the orchestration is closer to 100 µs. The ratio collapses at larger model sizes — by XL the kernels are doing tens of ms of work and the orchestration is noise — which is part of why the speedup curve in §1 climbs with model size.

The honest takeaway is that you cannot use a Node-WASM benchmark to predict browser-WASM performance, particularly at small model sizes. Per-host, per-shape numbers are the only ones that mean anything. "WASM is fast" is incomplete; you have to name the host.

**Reference**: `docs/performance.md` (the WASM section); `docs/lessons.md` §7 / `docs/session_retrospective.md` §7.

---

## 4. The Memory64 + SharedArrayBuffer + memory.grow race

XL-and-above presets compile against a 64-bit WASM build (`tinygpt64.wasm` with `-sMEMORY64=1`) to address heaps larger than 4 GB. In the browser this build *out-of-bounds-traps* mid-step on XL, Mega, and Behemoth. The same `.wasm` runs cleanly in Node. The bug isn't in the kernels — it's in the interaction between three browser features.

The story:

1. Main thread allocates a new buffer. The arena overflows; `malloc` calls `memory.grow`.
2. `memory.grow` reallocates the underlying `SharedArrayBuffer` backing `WebAssembly.Memory` — old views become detached.
3. A worker thread, currently inside a matmul tile loop, holds a `HEAPF32` typed-array view captured at the start of the kernel. Emscripten's pthread shim sends an atomics notification, but it isn't synchronous with the worker's current kernel.
4. The worker's next `HEAPF32[i]` read past the old length traps. From the user's point of view, "memory access out of bounds" at d_model ≥ 256.

Node's `worker_threads` shim resolves typed-array views differently — it re-grabs the latest `HEAPF32` on each kernel entry — so the race doesn't fire. Same WASM bytes, different host plumbing, different outcome.

The cheap, ship-now mitigation is to *avoid growth entirely*: bump `INITIAL_MEMORY` in `wasm/build_wasm64.sh` from 32 MB to 256 MB so XL (~350 MB live, with growth headroom from the initial pool) never triggers `memory.grow`. Behemoth still needs growth and remains the open case. The proper fix is `GROWABLE_HEAP_*` view helpers in the C++ kernels so stale views re-resolve on each access — that's a bigger refactor.

**Reference**: `docs/lessons.md` §2 (the full diagnosis); `wasm/build_wasm64.sh`; `tests/test_wasm64_xl_node.mjs` (the reproducer that surfaced it).

---

## 5. WebGPU checkpoint serialization, end to end

A trained model needs to leave the GPU's address space so it can be saved to disk, loaded into a different backend, or inspected from Python. `GpuModel.exportState()` (`webgpu/gpu_model.ts:247`) does this in a way that produces byte-compatible output with the WASM `tg_export_state`, so any `.tinygpt` file is portable between backends.

The mechanism:

1. Iterate over `this.params` (the manifest of weight buffers, set up at model construction in the same order as `collect_params` in `wasm/src/model.cpp`).
2. For each parameter `p`, issue three async GPU→CPU readbacks: `await p.w.download()`, `await p.m.download()` (Adam first-moment), `await p.v.download()` (Adam second-moment).
3. Concatenate into one `Float32Array` in interleaved order: `[w₀, m₀, v₀, w₁, m₁, v₁, …]`.
4. Prefix the whole thing with a 4-byte little-endian int32 carrying the step counter.

The readbacks happen serially, not in parallel. That's deliberate — `mapAsync` on a `GPUBuffer` requires the GPU to flush all pending work that touches that buffer, and overlapping many readbacks at once can hold dozens of staging buffers live. Serial is slower but bounded.

The manifest (built by `browser/src/main.ts:1149`) describes what's in the flat buffer in terms of named tensors with shapes — token embedding, position embedding, final layernorm, then per-layer ln1, q/k/v/o projections, ln2, MLP fc_in/fc_out. The Python reader (`python_ref/load_tinygpt.py`) uses the manifest to reconstruct a `state_dict`, so the saved file round-trips cleanly to PyTorch.

The key invariant: the byte layout produced by WebGPU `exportState` and the byte layout produced by WASM `tg_export_state` must be identical. Any divergence (different traversal order, different m/v interleaving, different step-prefix endianness) silently breaks Python loads — there's no checksum.

**Reference**: `webgpu/gpu_model.ts:247-263`; `browser/src/main.ts:1149-1182` (manifest); `wasm/src/model.cpp` (the C++ writer); `python_ref/load_tinygpt.py` (the reader).

---

## 6. Why generation feels slow even though training looks fast

The training loop posts ~10-20 step updates per second to the UI in the steady state; new users assume generation will be the same. It isn't. There are two compounding reasons.

**No KV cache.** Every new token does a full forward pass over the entire context window. At decode step `s` with context length `s + prompt_len`, the cost is `O(L · T² · d)` where `T = s + prompt_len` — quadratic in sequence length, not linear. After 200 generated tokens against a 64-token prompt, you've done roughly `Σ T²` ≈ 3.5 million attention work-units, where you "should" have done about 64,000 with a proper cache. The total wall-clock cost of decoding `n` tokens is dominated by the last token; doubling `n` quadruples the work.

A KV cache fixes this by saving the per-layer `K` and `V` tensors from prior decode steps. Each new token then only computes one new row of `Q`, one new row of `K`, one new row of `V`, and does attention against the concatenated cache — linear in `T`, not quadratic.

**No streaming.** The decode loop in `worker.ts` runs to completion and only then posts the finished sequence to the main thread, which runs a typewriter animation over already-finished text. The animation is cosmetic. The GPU is idle by the time the first character appears. To a user it looks like the model is "thinking very fast then typing slowly"; really the model is thinking the entire wall-clock duration and the typewriter is decorating dead time.

The fix is split: `gpu_model.ts:generate` now accepts an optional `onToken` callback (in this session); wiring it through the worker as a `postMessage` stream and updating the DOM on each arrival is task #72; the KV cache itself is a bigger architectural change on the roadmap.

**Reference**: `webgpu/gpu_model.ts:268-309`; `docs/session_retrospective.md` §6.

---

## 7. Flash Attention 2 in WGSL — what shipped and why

The naive attention path materialises a full `[B, H, T, T]` attention matrix to global memory. At Mega (B=8, H=8, T=512, ctx=512) per layer per step that's `8·8·512·512·4` = 67 MB of dead-end memory traffic, just for one layer of one step. FA2 deletes that storage by computing softmax *online* — accumulating `(m, l, O)` per Q row as K is walked block by block.

The forward kernel (`webgpu/attention_fa2.wgsl`, entry point `fa2_forward`) is workgroup-cooperative. Each workgroup owns one `(batch, head, Q-tile)` triple. The 16 threads collectively load `Qtile[Br][hd]` into workgroup-shared memory once; then they walk K and V in blocks of size `Bc = 16`, loading each `(K, V)` block cooperatively, computing scores, doing the online softmax merge, and updating the running output. The math is the standard FA1/FA2 identity:

```
m_new = max(m_old, m_block)
α     = exp(m_old − m_new)
O_new = α · O_old + Σ exp(s_jj − m_new) · V[jj]
l_new = α · l_old + Σ exp(s_jj − m_new)
```

After the loop, divide `O /= l` and write out.

The backward kernel (`webgpu/attention_fa2_backward.wgsl`) does the FA2 trick that's the actual reason to bother. The forward saves `L = m_final + log(l_final)` per Q row — one float per row, not a full matrix. The backward reconstructs `P[i,j] = exp(s[i,j] − L[i])` from raw `Q` and `K` instead of reading a cached attention matrix that no longer exists. Recomputing scores is cheap on a GPU; reading 67 MB from global memory is not.

The shipping forward also dropped its previous "compatibility second pass" that re-walked K to write the attention matrix for the old backward — once the FA2 backward shipped, nothing needed the materialised attention any more. That was the real memory win.

**Reference**: `webgpu/attention_fa2.wgsl`; `docs/fa2_forward_notes.md`; `docs/fa2_backward_notes.md`; the FA2 paper (Dao 2023).

---

## 8. The blocked-4×4 matmul kernel

`webgpu/matmul_blocked.wgsl` is the workhorse. It stacks two well-known wins. First, workgroup-shared tiling: a 16×16 workgroup cooperatively loads a 64×16 slice of A and a 16×64 slice of B into shared memory (`var<workgroup>`), then every multiply-accumulate for the 64×64 output tile reads from shared instead of global. Second, thread blocking: each thread computes a 4×4 sub-block of output, held entirely in registers. The K dimension is walked in tiles of 16; per K-tile the workgroup cooperatively fills shared memory once and reuses it across the whole tile.

The reason this is fast comes down to arithmetic intensity. Without thread blocking, each shared-memory load drives one multiply-add — the kernel is memory-bound on shared bandwidth. With a 4×4 output block per thread, each loaded `A` element is reused across 4 columns and each loaded `B` element across 4 rows — every shared-memory read drives ~16 multiply-adds. That ratio crosses the threshold where the kernel is *compute-bound* on Apple's GPU FP32 units instead of bandwidth-bound on shared memory.

Why not 8×8? On paper, 64 MAC operations per shared load. In practice it lost at every measured size (0.91× at 1024³, 0.88× at 2048³). The accumulator is 64 floats per thread; combined with the loop-state and address-computation registers, the per-thread register budget overflowed Apple's GPU's physical registers and spilled to (much slower) device-local memory. 4×4 keeps the accumulator at 16 floats — comfortably below the register cliff.

There's a real lesson here for anyone tuning GPU kernels: occupancy and register pressure are coupled, and the "more arithmetic per load" knob has a sharp ceiling on consumer GPUs. The best block size is the largest one that fits.

**Reference**: `webgpu/matmul_blocked.wgsl`; the unused `webgpu/matmul_blocked8.wgsl` (kept as a counterexample); Goto & van de Geijn 2008.

---

## 9. vec4 loads — and why the standalone benchmark lied

WebGPU's `vec4<f32>` lets a single load instruction issue a 128-bit memory transaction instead of four 32-bit ones. On the M-series GPU, the standalone matmul benchmark at 2048³ went **1.37× faster** with vec4-packed loads. The temptation to merge was immediate.

The first integration *diverged loss to 88.67* on the parity test. WASM at the same step produced loss 2.94. The kernel was returning garbage that happened to pass the standalone equality check (because the standalone test used a fresh allocator and a clean cache) but produced wrong gradients in an end-to-end pipeline.

The root cause was an invisible bind-group mismatch. The WGSL declared `var<storage, read> A: array<vec4<f32>>` but the bind-group layout in `ops.ts` declared `buffer: { type: "storage" }` — read-write. On Chrome/Apple's WebGPU, the validation layer didn't catch the access-mode mismatch; the driver silently returned wrong data. Other implementations (Linux Vulkan, Firefox, the standalone WGPU reference compiler) emit a validation error. Apple/Chromium didn't.

The fix is one line — `buffer: { type: "read-only-storage" }` in the bind-group layout — and the parity drops to <1% drift. But the deeper lesson is the one that pays off long after the bug is fixed: *standalone benchmarks lie*. They measure the kernel in isolation, in ideal shapes, often square, with fresh allocators. Real training calls them in awkward non-square shapes, with state from previous kernels in cache, inside a pipeline that interacts with the kernel's compute/memory mix. End-to-end parity at 50 steps with drift < 5% is the only number that counts as "shipped." Everything else is wishful thinking.

This pattern repeated three times in the project: vec4 loads, f16-packed weights (1.7× standalone, slower in the real pipeline because tiling had already amortised the global reads), and 8×8 thread blocks (great on paper, register-spill in practice).

**Reference**: `webgpu/matmul_blocked_vec4.wgsl`; `docs/lessons.md` §5.

---

## 10. The .tinygpt file format

A self-describing binary, designed to round-trip between every backend (WASM, WebGPU, Python) without losing fidelity:

```
offset  size       contents
 0      4          magic "TGPT"
 4      4          uint32 LE version (= 2)
 8      4          uint32 LE JSON header length N
12      N          UTF-8 JSON header
12+N    M          raw float32 state
```

The JSON header carries `config` (the `RunConfig` the model was trained with), `manifest` (the named-tensor index built by `buildManifest` in `browser/src/main.ts:1149`), `lossHistory` (last 512 points), `finalLoss`, `sample` (a 320-char generated sample for human sanity-checking), `savedAt`, `bestVal`, and the critical `includesOptimizerState` flag.

The state region's layout is dictated by `tg_export_state` in the C++ side: a 4-byte int32 step-counter prefix, followed by per-parameter triplets `[w, m, v]` in manifest order. If `includesOptimizerState` is true the m and v Adam moments are present (a fresh checkpoint can resume training); if false only `w` is present (smaller, but training resumes with a cold optimiser).

Important fact: the Python reader (`python_ref/load_tinygpt.py`) exists and works. The Python writer does not — you can ingest a browser-trained checkpoint into PyTorch, but you cannot go the other direction without writing a `.tinygpt` from scratch. The asymmetry is deliberate: the browser is the demo surface, Python is the inspection surface.

Version bumps are explicit: v1 was the original (config + flat state), v2 added the manifest, loss history, sample, and the optimizer-state flag. The reader at `decodeModelFile` accepts both versions; the writer always emits v2. Forward-compat is not guaranteed — a v3 with new header keys would not parse on a v2 reader.

**Reference**: `browser/src/main.ts:1112-1237`; `python_ref/load_tinygpt.py`.

---

## 11. Astro + Cloudflare Pages deployment

TinyGPT's playground is an Astro site (`browser/`) built statically — `npm run build` emits `dist/` with one HTML file per route, all CSS inlined per page, and JavaScript bundles fingerprinted for cache-busting. Cloudflare Pages serves the static directory verbatim from edge caches; there is no server-side Astro runtime at deploy time.

The wrinkle is `SharedArrayBuffer`. The multi-threaded WASM build needs it; the browser only exposes it under "cross-origin isolation," which requires two response headers:

```
Cross-Origin-Opener-Policy:   same-origin
Cross-Origin-Embedder-Policy: require-corp
```

In production, those come from `browser/public/_headers` — Cloudflare Pages copies that file verbatim and applies the rules at the edge. In development, the Astro dev server has its own header layer; `browser/astro.config.mjs:35` mirrors the same COOP/COEP headers via `vite.server.headers` so local runs match production behaviour. Without that mirror, `npm run dev` would fail to instantiate the threaded WASM module while `npm run build && cf-pages deploy` would work — confusing.

Important quirk for browser-driven training sessions: `astro dev` has hot-module-reload, which is destructive during a long-running browser training. A 15-minute training run that touches CSS triggers an HMR refresh and loses all state. For Playwright-driven training (`browser/train_demo.mjs`), switch to `astro preview` instead — it serves the same `dist/` Cloudflare Pages would serve, with no HMR, same headers, no surprises.

The deploy path is fully static-first; if you want a server-rendered route (analytics endpoint, model upload form), you'd add a `functions/` directory for Cloudflare Pages Functions, which the current build does not need.

**Reference**: `browser/public/_headers`; `browser/astro.config.mjs`; `docs/deploy.md`.

---

## 12. Why a 15-min in-browser training session converges

The "Train your own from scratch" path promises a working model in 15 minutes. The math works out, just barely, and it's worth seeing why.

On an M-series MacBook with WebGPU, the Huge preset (~9.6M params, d_model 384, 6 layers, ctx 256, batch 8) trains at roughly **150 ms/step** in steady state — call it 6-7 steps/second. Round down to 4 steps/sec to absorb stalls, validation pauses, and rendering overhead.

```
15 min × 60 s × 4 steps/sec = 3,600 steps
3,600 steps × 8 batch × 256 ctx ≈ 7.4M tokens seen
```

TinyShakespeare is 1.1 MB. Byte-level vocab, so that's 1.1M tokens. Total tokens seen across 15 minutes ≈ 6-7 *epochs*. That's enough for a transformer to actually learn the corpus statistics — the first epoch teaches it character frequencies and word lengths; the second teaches it bigram structure; by the fourth or fifth it's reproducing recognisable Shakespearean dialogue rhythm with mostly-real-looking made-up words. The final loss lands around 1.30 nats.

This is a sweet spot, not a coincidence. If you swapped in a 10 MB corpus, the model wouldn't complete one epoch in 15 minutes; convergence would look like a long monotonic descent that never bottoms out — visually identical to "this isn't working." If you used an 0.1 MB corpus (the old 863-byte default was even worse), it would overfit in the first few hundred steps and produce verbatim memorisation, which reads as a broken demo. 1 MB ± a factor of 2 is the band where a 15-min run on this hardware produces samples a human will recognise as "learned, not memorised."

For larger corpora, the cure is more training time, not a bigger model. For smaller corpora, the cure is a hold-out validation split and early stopping.

**Reference**: `docs/performance.md`; `docs/lessons.md` §4.

---

## 13. Compute Pressure API

A relatively new browser API (Chromium 125+) that surfaces OS-level pressure signals into web pages. The OS reports a four-state enum — `nominal`, `fair`, `serious`, `critical` — derived from CPU utilisation, thermal headroom, and (depending on platform) GPU load. A page can subscribe:

```js
const observer = new PressureObserver(records => {
  for (const r of records) {
    setIndicator(r.state); // "nominal" | "fair" | "serious" | "critical"
  }
});
observer.observe("cpu");
```

TinyGPT uses it for the pulse-dot chip in the playground's sticky header (`browser/src/main.ts:2506`). When a training run pushes the system to `fair` or `serious`, the chip turns yellow and pulses; at `critical` it turns red. The intent is purely informational — to give the user a real signal of *why* their machine feels slower while training runs, rather than letting them blame the page for being heavy.

It's a small but instructive piece of UX engineering. The page is doing heavy work; the OS knows; the user notices; without the chip, the user assumes the page is just badly written. With the chip, the user understands "I'm asking my laptop to do a lot, and it's telling me." The same status was always observable in Activity Monitor / Task Manager — the chip just brings it into the user's flow.

Browser support is the catch: Safari and Firefox don't ship it yet, so the chip is hidden on those browsers via feature detection (`'PressureObserver' in window`). Progressive enhancement, not a hard dependency.

**Reference**: `browser/src/main.ts:2506-2570`; `browser/src/pages/index.astro:1131-1180` (the styling).

---

## 14. The Hugging Face datasets-server pattern

To let users train on real corpora beyond Shakespeare, TinyGPT pulls text directly from Hugging Face in the browser via the `datasets-server` HTTP API. This works because the server emits `Access-Control-Allow-Origin: *` and serves public datasets without authentication:

```
GET https://datasets-server.huggingface.co/rows
    ?dataset=roneneldan/TinyStories
    &config=default
    &split=train
    &offset=0
    &length=100
->  { rows: [ { row: { text: "..." }, row_idx: 0, … }, ... ] }
```

`browser/src/datasets.ts` exposes a small curated catalog of datasets that are known to work with this endpoint and are reasonable for a tiny byte-level model: TinyStories, TinyShakespeare, Simple English Wikipedia, full English Wikipedia, plus a few others. Each entry names the dataset path, the config (a HF concept — most public datasets have a `default` config but Wikipedia has per-language configs like `20231101.simple`), the split (`train`), and the *text column* — the row field whose string value is the actual training text. TinyStories uses `text`; Tiny Shakespeare on the `Trelis/tiny-shakespeare` mirror uses `Text` (capital T). Getting the column wrong means a clean fetch that yields empty strings.

`browser/src/main.ts:1656 loadHfDataset` is the pager: it walks `offset` in chunks of 100 rows, concatenating each row's text until it reaches the user's character budget (default 2 MB). Each fetch reports progress to the UI so the user sees a live byte counter. The fetch is cancellable via an incrementing token — if the user picks a different dataset mid-fetch, the in-flight one becomes orphaned and its results are dropped.

For private datasets you can store an HF token in `localStorage` and the loader injects an `Authorization: Bearer …` header. The token never leaves the browser — there's no TinyGPT server to forward it through, by design.

**Reference**: `browser/src/datasets.ts`; `browser/src/main.ts:1656-1745`.

---

## What to read next if you want to keep going

If you want to follow specific threads further, here are the most useful entry points:

- **Inside TinyGPT.** `docs/learn.md` walks the entire codebase top-down: what each file does, how the pieces compose, where to make changes for common modifications. `docs/performance.md` is the canonical perf doc, including the real-device benchmark protocol you'd use if you wanted to add a hardware datapoint. `docs/status.md` is the live "what's shipped, what's open" board. `python_ref/` is the reference implementation — model.py is the cleanest single-file transformer you'll find in this project, and it's the oracle every other backend is validated against.

- **Transformers from scratch.** Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) is the canonical 300-line PyTorch implementation; the architecture in `python_ref/model.py` is a direct descendant. His [makemore](https://github.com/karpathy/makemore) tutorial series builds up to a transformer one notebook at a time and is the gentlest path in.

- **Flash Attention 2.** The [original paper (Dao 2023)](https://arxiv.org/abs/2307.08691). For pedagogical purposes, the [Triton tutorial implementation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) is much easier to read than CUDA. The local notes at `docs/fa2_forward_notes.md` and `docs/fa2_backward_notes.md` are TinyGPT-specific commentary on the same algorithm.

- **Matmul tiling and register blocking.** Goto & van de Geijn's "Anatomy of High-Performance Matrix Multiplication" (ACM TOMS, 2008) is the canonical explanation of the cache-blocking + register-blocking decomposition that every GPU matmul kernel rediscovers. For a modern GPU-oriented walkthrough, [Lei Mao's GEMM optimisation series](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) is excellent — it covers the exact 4×4 vs 8×8 register-pressure tradeoff you'll find in `matmul_blocked.wgsl`.

- **WebGPU.** The [WebGPU spec](https://www.w3.org/TR/webgpu/) is dense but readable. The [WebGPU Fundamentals](https://webgpufundamentals.org/) tutorials are a much friendlier introduction. For WGSL specifically, the [WGSL spec](https://www.w3.org/TR/WGSL/) is more usable than most language specs.

- **The Web platform pieces.** [SharedArrayBuffer and cross-origin isolation](https://web.dev/articles/coop-coep) on web.dev covers the COOP/COEP machinery in §11. The [Compute Pressure API explainer](https://developer.chrome.com/docs/web-platform/compute-pressure) covers §13. The [Hugging Face datasets-server docs](https://huggingface.co/docs/datasets-server) cover the API in §14, including the rate limits you'll hit on heavier pulls.

Read in this order if you're starting fresh: Karpathy's makemore → `python_ref/model.py` → `docs/learn.md` → the relevant kernel notes (`fa2_*`, `online_softmax_in_attention.md`) → the WebGPU shaders themselves. By the time you get to the shaders, every loop will mean something.
