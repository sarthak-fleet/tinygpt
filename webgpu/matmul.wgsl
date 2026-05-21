// matmul.wgsl — WebGPU compute shader for matrix multiply (Phase 5).
//
// STATUS: documented stub. No implementation yet. Do Phase 4 (WASM) first.
//
// The FIRST kernel to port to WebGPU. Do not port the whole model at once.
// A compute shader processes data in parallel across workgroups and writes
// results into a storage buffer.
//
// Plan:
//   - bind groups: input A, input B, output C, dimensions uniform
//   - one invocation computes one (or a tile of) output element(s)
//   - tile via workgroup-shared memory once the naive version is correct
//
// Acceptance: WebGPU matmul output equals the WASM matmul within tolerance
// (see tests/README.md "WebGPU matmul gives same output as WASM").
//
// Guide: docs/browser_notes.md ("WebGPU acceleration")
//
// TODO(phase-5): @compute @workgroup_size(...) entry point computing C = A @ B.
