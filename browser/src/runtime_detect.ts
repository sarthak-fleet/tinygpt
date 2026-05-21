/**
 * runtime_detect.ts — browser capability detection (Phase 4).
 *
 * STATUS: documented stub. No implementation yet.
 *
 * Picks the best available compute backend and degrades gracefully:
 *   webgpu      -> wasm-simd -> wasm
 *
 * Detect:
 *   - WebGPU:     'gpu' in navigator  &&  await navigator.gpu.requestAdapter()
 *                 (also HTTPS-only; support varies by browser/platform)
 *   - WASM SIMD:  WebAssembly.validate of a known SIMD test module
 *   - SharedArrayBuffer / cross-origin isolation (needed only for threaded WASM)
 *
 * Guide: docs/browser_notes.md ("WebGPU acceleration", browser facts)
 *
 * TODO(phase-4): export detectBackend() -> "webgpu" | "wasm-simd" | "wasm".
 * TODO(phase-4): expose results to the UI capability panel.
 */
export {};
