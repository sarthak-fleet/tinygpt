/**
 * worker.ts — training Web Worker (Phase 4).
 *
 * STATUS: documented stub. No implementation yet.
 *
 * Runs the training loop off the main thread:
 *   get batch -> WASM/WebGPU forward -> backward -> optimizer step
 *             -> post TrainingProgress to the UI
 *
 * Posts this message to the main thread:
 *   type TrainingProgress = {
 *     step: number;
 *     trainLoss: number;
 *     valLoss?: number;
 *     tokensPerSecond: number;
 *     backend: "wasm" | "wasm-simd" | "webgpu";
 *   };
 *
 * Start single-threaded. WASM pthreads need SharedArrayBuffer + cross-origin
 * isolation (COOP/COEP headers) — add that only later. See docs/browser_notes.md.
 *
 * Guide: docs/browser_notes.md ("Web Worker", "Checkpointing")
 *
 * TODO(phase-4): own dataset + training loop + sampling + checkpoint coordination.
 * TODO(phase-4): call into the WASM backend (wasm/) for tensor ops.
 */
export {};
