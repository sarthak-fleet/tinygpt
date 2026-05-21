/**
 * main.ts — UI / main-thread controller (Phase 4).
 *
 * STATUS: documented stub. No implementation yet.
 *
 * The main thread owns ONLY: UI, file upload, charts, and controls.
 * Training NEVER runs on the main thread — it runs in worker.ts.
 *
 * Responsibilities:
 *   - capability panel: feature-detect WebGPU / WASM-SIMD (see runtime_detect.ts)
 *   - accept a text/dataset upload from the user
 *   - spawn the training Web Worker and post it the dataset + config
 *   - receive TrainingProgress messages and render loss charts (charts.ts)
 *   - start / pause / checkpoint / sample controls
 *
 * Guide: docs/browser_notes.md ("Web Worker", "Required tests")
 *
 * TODO(phase-4): wire upload -> worker; render progress; keep the UI responsive.
 */
export {};
