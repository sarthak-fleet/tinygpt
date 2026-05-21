# browser/ — Phase 4

The browser app: main-thread UI + a training Web Worker + a tokenizer + storage.

## Architecture

```
Main thread (main.ts)      UI, file upload, charts, controls
        |  postMessage (dataset + config)  /  TrainingProgress
Worker (worker.ts)         dataset, training loop, sampling, checkpoints
        |  calls
WASM backend (../wasm/)    tensor ops, forward/backward, optimizer
```

Training must never run on the main thread — the UI has to stay responsive.

## Files

| File               | Role |
| ------------------ | ---- |
| `src/main.ts`         | UI controller; spawns the worker, renders progress |
| `src/worker.ts`       | Training loop off the main thread |
| `src/tokenizer.ts`    | Byte-level encode/decode (vocab 256) |
| `src/storage.ts`      | OPFS / IndexedDB checkpoint persistence |
| `src/charts.ts`       | Loss / throughput charts |
| `src/runtime_detect.ts` | Picks backend: webgpu → wasm-simd → wasm |

## Status

Documented stubs only. Implement after the Python reference (Phase 1–3) is
correct. See `../docs/browser_notes.md`.

Start single-threaded. Threaded WASM needs `SharedArrayBuffer` + cross-origin
isolation (`COOP: same-origin`, `COEP: require-corp`) — add that later, not first.
