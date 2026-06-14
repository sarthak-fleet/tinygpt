# Performance notes

How fast TinyGPT trains, what has been done to speed it up, and what is left.
All numbers are from an Apple M5 Pro laptop.

## Measuring it

Two benchmarks, so a change can be measured instead of guessed at:

- `python_ref/bench.py` — native training (PyTorch on CUDA / MPS / CPU).
- `tests/bench_wasm.mjs` — the compiled WASM module, which is the browser's
  actual training path, timed from Node.

## The browser path (WebAssembly)

Browser training runs in C++ compiled to WebAssembly, on one thread. Measured
with `bench_wasm.mjs`:

| Build                          | standard (0.37M) | capable (0.48M) |
| ------------------------------ | ---------------: | --------------: |
| baseline (scalar)              |      304 ms/step |     632 ms/step |
| + backward-scratch reuse       |      305 ms/step |     ~640 ms/step |
| + WASM SIMD (`-msimd128`)      |  **191 ms/step** | **391 ms/step** |
| net speed-up                   |        **1.6×**  |       **1.6×**  |

What worked, and what didn't:

- **Allocation reuse — no measurable gain.** Caching the backward pass's scratch
  buffers on the model (instead of allocating ~12 vectors per step) changed
  nothing measurable. `malloc` was not the bottleneck; compute is. The change is
  kept — not allocating in a hot path is still correct hygiene — but it is not a
  speed-up and is not claimed as one.
- **WASM SIMD — ~1.6×.** `-msimd128` lets LLVM autovectorize the matmul,
  layernorm, and attention inner loops (four float32 lanes at a time). WASM SIMD
  is supported in every current browser and costs nothing at deploy time. The
  SIMD build is verified by `tests/smoke_wasm_node.mjs` — it still trains
  correctly (loss falls, greedy generation reproduces the corpus).

Still on the table for the WASM path:

- **Hand-written SIMD intrinsics.** Autovectorization reached 1.6×; explicit
  `wasm_f32x4` intrinsics in the matmul hot loop could push toward 2.5–3×.
- **Threads.** Multi-core training via `SharedArrayBuffer` needs the COOP/COEP
  cross-origin-isolation headers, which a plain GitHub Pages host cannot set.
  Deferred — it is coupled to where the site is hosted.

## The native path, for contrast

`bench.py` on the same laptop trains a 2.7M model at ~10 ms/step on the GPU
(MPS). The browser does a smaller 0.37M model at 191 ms/step. Native is roughly
two orders of magnitude faster per parameter-step. That gap is why anything
past a demo-sized model should be trained locally — and why WebGPU is the real
browser lever.

## Rust?

Considered and set aside. C++ and Rust both compile through LLVM to effectively
the same WebAssembly — the source language is not the bottleneck. Rewriting the
kernels in Rust would be a large change for no speed-up. SIMD and threads are
equally reachable from the current C++/Emscripten setup.

## WebGPU training

The whole training loop now also runs on the GPU. It was built in six verified
stages (`webgpu/`):

1. GPU tensors + matmul forward/backward
2. layernorm, GELU, the elementwise ops
3. causal multi-head attention, forward and backward
4. embeddings, cross-entropy, AdamW
5. `gpu_model.ts` — the orchestrator: a full forward + backward + AdamW loop,
   every tensor resident on the GPU
6. wired into the app as a backend toggle (WASM / WebGPU)

**Correctness** is solid: 24 kernel parity checks against plain-JS references,
the project's overfit gate run on the GPU (loss 5.55 → 0.002 in 150 steps), and
a headless-browser e2e that trains on the WebGPU backend.

### Optimizations done

- **Buffer pool** (`webgpu/tensor.ts`) — per-step activation/gradient buffers
  are returned to a pool and reused; after step 1 a run allocates no buffers.
- **One submit per step** (`webgpu/ops.ts`) — a whole step's dispatches record
  into a single command encoder and submit once, instead of one submit per
  kernel.

Both are real reductions in CPU-side overhead and both keep every parity check
and the overfit gate green.

### Why there is no speed number here — and it matters

WebGPU's speed **cannot be measured in this project's test setup.** The headless
Chromium that runs the e2e exposes a WebGPU adapter whose architecture is
`swiftshader` — Google's *software* renderer. It is a CPU implementation of the
WebGPU API; it never touches a real GPU.

So any headless "WebGPU vs WASM" number is *software-emulated WebGPU vs
SIMD-vectorized WASM* — and WASM wins that, which says nothing about real
hardware. (An earlier version of this file quoted such numbers as a verdict;
that was wrong, and is the reason the buffer-pool and batching optimizations
showed no change — SwiftShader's bottleneck is its own software compute, not
buffer allocation or submit count.)

**To measure the real thing:** open the app in a normal browser on a machine
with a real GPU, pick the WebGPU backend, and read the tokens/sec in the
playground. That is the only valid measurement, and it is not something the
headless CI can do. On a real GPU the matmul-heavy work parallelizes hard;
whether end-to-end training beats WASM depends on how much the small
elementwise kernels' dispatch overhead costs. That number is genuinely unknown
until run on hardware — this doc will not guess it.

### Real-device benchmark protocol

A reproducible, copy-pasteable recipe for posting a hardware datapoint. The
goal is one number — steady-state tokens/sec — that another contributor can
reproduce on the same machine.

**Prerequisites.** Open the live app at
[tinygpt.sarthakagrawal.dev](https://tinygpt.sarthakagrawal.dev) (or run
`cd browser && npm run dev` locally after `bash wasm/build_wasm.sh`). Use a
desktop Chrome 113+ / Edge 113+ / Safari 18+ build. The app probes the WebGPU
adapter on load and displays its vendor/device name; check that the displayed
name is not `swiftshader` / `SwiftShader`. If it is, you are on the software
path and the number is meaningless (see "Verifying the adapter" below).

**Fixed config** (so numbers compare across machines and across runs):

| Setting       | Value                                                            |
| ------------- | ---------------------------------------------------------------- |
| Preset        | `Large (~2.7M params)` — the first preset whose own note says "switch to WebGPU if your browser supports it" |
| Backend       | WebGPU (then repeat: WASM)                                       |
| Other knobs   | leave at preset defaults (layers 6, d_model 192, ctx 128, batch 12, 600 steps) |
| Dataset       | Built-in `tiny-corpus.txt`                                       |
| Seed          | default (leave unchanged)                                        |

**Steps.**

1. Quit other GPU-heavy apps (Chrome tabs running WebGL, video calls, screen
   recorders). Plug the laptop in — battery-saver throttles the GPU.
2. Load the page, scroll to the run-config card. Pick the **Large** preset
   and set the backend to **WebGPU**. Leave the other knobs at the preset
   defaults.
3. Click **Start**. The first ~10 steps include shader compilation and buffer
   allocation — ignore the initial tokens/sec reading.
4. Once the step counter passes ~50, the live tokens/sec reading in the
   sticky stats bar at the top of the page has stabilised. Let it run for
   another ~150 steps (so you're reading steady-state numbers, not warm-up),
   then record the tokens/sec value. You can hit **Stop** at that point — no
   need to run the full 600 steps just for the bench number.
5. Switch the backend toggle to **WASM**, reload (so buffers are fresh), pick
   the same preset, click **Start** again, and repeat the measurement.
6. Repeat the whole pair once more from a fresh page load and average. (One
   run is enough to be useful; two confirms the number isn't noise.)

**Verifying the adapter.** In the same browser, open `chrome://gpu` (Chrome /
Edge) or the Safari WebGPU inspector and confirm `WebGPU: Hardware
accelerated`. If the line says `Software only, hardware acceleration
unavailable` or the adapter name contains `SwiftShader`, `llvmpipe`, or
`WARP`, you are on a software fallback — the WebGPU number is not a real-GPU
measurement and should not be posted as one.

**Output format.** Post results as a short block — paste it into the relevant
GitHub issue / discussion thread or a PR comment. Keep the field names exactly
as below so they are greppable:

```
device:      <vendor + model, e.g. "Apple M5 Pro (14-inch MacBook Pro, 2025)">
os/browser:  <e.g. "macOS 15.4, Chrome 131.0.6778.86">
adapter:     <WebGPU adapter name from chrome://gpu, e.g. "Apple M5 Pro">
preset:      Large (~2.7M params, ctx 128, batch 12)
steps read:  <step at which tokens/sec was recorded, e.g. "step 200">
webgpu:      <N> tokens/sec  (steady state, after step 50)
wasm-simd:   <N> tokens/sec  (steady state, after step 50)
ratio:       <webgpu / wasm-simd, e.g. "6.8×">
notes:       <anything unusual — thermal throttling, dGPU vs iGPU, etc.>
```

Example:

```
device:      Apple M5 Pro (14-inch MacBook Pro, 2025)
os/browser:  macOS 15.4, Chrome 131.0.6778.86
adapter:     Apple M5 Pro
preset:      Large (~2.7M params, ctx 128, batch 12)
steps read:  step 200
webgpu:      2,850 tokens/sec
wasm-simd:   420 tokens/sec
ratio:       6.8×
notes:       plugged in; no other GPU apps running
```

One real-hardware pair (WebGPU + WASM-SIMD) closes the evidence gap this doc
currently flags. Multiple pairs across vendors (Apple, NVIDIA discrete, Intel
integrated, AMD) would let the WebGPU section quote a range instead of a
single anecdotal number.

## Register + cache-blocked matmul — and why the microbench lied (2026-06-14)

`wasm/src/matmul.cpp` `matmul_forward` was rewritten from the naive ikn loop to a
register-blocked (MR×NR C-tile in registers, B reused across rows) + cache-blocked
(KC/NC panels resident in L2) micro-kernel. The k-accumulation order is preserved,
so it is **bit-identical** to the naive kernel — the native parity gate
(`wasm/build_native.sh`) passes with zero drift.

**The cautionary result.** An isolated, single-threaded native microbench
(`clang++ -O3`) showed **3–4×** on 256³–1024³ matmuls. But the clean end-to-end WASM
*training* A/B (`tests/bench_wasm.mjs`, idle CPU) was only **~1.03–1.06×**:

| config | naive ms/step | blocked ms/step | speedup |
|---|---|---|---|
| small 0.37M | 99.2 | 94.0 | 1.06× |
| medium 0.84M | 352.8 | 334.0 | 1.06× |
| large 2.74M | 1182.6 | 1152.7 | 1.03× |
| xl 6.42M | 1850.2 | 1800.0 | 1.03× |

Why the gap between microbench and reality:
1. The WASM build is **8-threaded** — the naive matmul is already split across cores,
   so it isn't the single-threaded bottleneck the microbench measured.
2. emcc **already autovectorizes** the naive kernel (`-msimd128`); the prior "1.6×
   SIMD" was already banked.
3. **Only `matmul_forward` was optimized.** A *training* step is dominated by
   `matmul_backward` (dA + dB ≈ 2× the forward FLOPs), still naive — so the training
   bench barely moves.

**Kept anyway**, because the browser's product path is *inference*: token-by-token
generation uses M=1, which takes the single-threaded path (`MIN_M_FOR_THREADS=64`),
where the 3–4× per-token matmul win applies in full. The change is bit-exact and
free of downside. Lesson worth keeping: *measure on the real workload (threaded,
real model sizes, forward+backward), not an isolated microbench.*

**Backward pass too (2026-06-14).** `backward_dA` (dA = dC·Bᵀ, a dot-product per
output) was register-blocked KR=4 k-rows per pass so `dc_row[n]` is loaded once and
reused across 4 dot-products (was re-streamed per k). Same n-order → bit-identical
(native gate passes). Since a *training* step is dominated by backward (dA + dB ≈ 2×
forward FLOPs), this roughly **doubled** the end-to-end training speedup:

| config | naive | fwd-blocked | **fwd+bwd-blocked** |
|---|---|---|---|
| small 0.37M | 99.2 | 94.0 | **90.3** |
| medium 0.84M | 352.8 | 334.0 | **317.6** |
| large 2.74M | 1182.6 | 1152.7 | **1057.4** |
| xl 6.42M | 1850.2 | 1800.0 | **1643.6** |

End-to-end WASM training is now **~1.10–1.13× over naive** (was ~1.05× forward-only).
`backward_dB` is already SAXPY-autovectorized; blocking it (rank-1 update over m) is
the remaining increment.
