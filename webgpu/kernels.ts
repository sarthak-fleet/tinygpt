/**
 * kernels.ts — WebGPU device setup + kernel dispatch (Phase 5).
 *
 * STATUS: documented stub. No implementation yet. Do Phase 4 (WASM) first.
 *
 * JS/TS glue around the WGSL shaders:
 *   - request a GPUAdapter + GPUDevice (feature-detect; HTTPS-only)
 *   - create GPUBuffers, bind groups, compute pipelines, command encoders
 *   - upload inputs, dispatch workgroups, read results back
 *
 * Port kernels in this order (only after each passes a CPU-parity test):
 *   1 matmul  2 linear backward  3 attention scores  4 softmax
 *   5 attention value aggregation  6 layernorm  7 adamw
 *
 * Guide: docs/browser_notes.md ("WebGPU acceleration")
 *
 * TODO(phase-5): initWebGPU(); runMatmul() dispatching matmul.wgsl.
 * TODO(phase-5): a parity harness comparing each kernel to the WASM backend.
 */
export {};
