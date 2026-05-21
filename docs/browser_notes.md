# Browser notes — WASM, Workers, OPFS, WebGPU

Phase 4–5. Move training/inference into the browser **after** the Python
reference is correct. Order: PyTorch reference → WASM CPU → WebGPU.

WebGPU is the browser GPU-compute API, but it is platform/browser-dependent and
HTTPS-only — the browser build needs a WASM fallback.

---

## 1. WASM backend (Phase 4, step 3)

Emscripten compiles C/C++ to WebAssembly. Baseline build:

```bash
emcc src/*.cpp -O3 -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 \
  -o dist/tinygpt.js
```

SIMD build (`-msimd128`):

```bash
emcc src/*.cpp -O3 -msimd128 -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 \
  -o dist/tinygpt.simd.js
```

Verify the SIMD build matches the scalar build before relying on it. Kernels
live in `../wasm/src/`.

---

## 2. Web Worker (Phase 4, step 4)

Training must not run on the main thread.

```
Main thread:  UI, file upload, charts, controls
Worker:       dataset, training loop, sampling, checkpoint coordination
WASM:         tensor ops, forward/backward, optimizer
```

Progress message:

```ts
type TrainingProgress = {
  step: number;
  trainLoss: number;
  valLoss?: number;
  tokensPerSecond: number;
  backend: "wasm" | "wasm-simd" | "webgpu";
};
```

---

## 3. Checkpointing (Phase 4, step 5)

Checkpoint layout (see `../checkpoints/README.md`):

```
checkpoint/
  model_config.json   training_config.json   dataset_manifest.json
  trainer_state.json  weights.f32            adam_m.f32  adam_v.f32
  loss_history.json
```

Persist to OPFS or IndexedDB. OPFS is origin-private storage, subject to browser
storage quotas; clearing site storage deletes it. Request durability:

```js
await navigator.storage.persist();
const estimate = await navigator.storage.estimate();
```

Storage can be best-effort unless persistence is granted; quota and eviction
behaviour vary by browser.

---

## 4. WebGPU acceleration (Phase 5, step 6)

Only after WASM correctness. A compute shader processes data in parallel across
workgroups and writes results to buffers.

Port kernels in this order:

```
1 matmul   2 linear backward   3 attention scores   4 softmax
5 attention value aggregation   6 layernorm   7 AdamW
```

Start with matmul alone — do not port the whole model at once. Acceptance:
WebGPU matmul equals WASM matmul within tolerance.

Things to learn: WGSL, GPU buffers, bind groups, compute pipelines, command
encoders, workgroups, device limits, buffer sharding.

---

## 5. Threading is later, not first

WASM threading via Emscripten pthreads depends on `SharedArrayBuffer` and
requires cross-origin isolation in deployed browsers. Required headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

**Start single-threaded.** Add threading only once single-threaded works.

---

## 6. Browser facts to internalize

- **WebGPU** is in modern browsers but still needs feature detection. Chrome/Edge
  desktop support began at version 113; Android came later; platform support
  varies. Always detect `navigator.gpu` and request an adapter before use.
- **OPFS** is useful for browser-local files/checkpoints but is subject to
  storage quotas.
- **WASM pthreads** require `SharedArrayBuffer` + cross-origin isolation — do not
  start there.

---

## 7. Browser LoRA flow

```
Load frozen base model
→ load or initialise adapter
→ user uploads text/examples
→ train only the adapter in the Worker
→ save adapter checkpoint
→ generate samples using base + adapter
```

Recommended backend order: PyTorch reference → WASM CPU inference → WASM CPU
LoRA training → WASM SIMD → WebGPU inference → WebGPU LoRA training.

---

## Deliverables

- Phase 4: browser app trains a tiny model in a Worker; UI stays responsive;
  checkpoint survives reload.
- Phase 5: one WebGPU kernel (matmul) is correct and faster than WASM.

## References

- WebGPU API: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- Emscripten / WebAssembly: https://emscripten.org/docs/compiling/WebAssembly.html
- Emscripten pthreads: https://emscripten.org/docs/porting/pthreads.html
- OPFS: https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system
- Storage quotas: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria
- WebGPU overview: https://developer.chrome.com/docs/web-platform/webgpu/overview
