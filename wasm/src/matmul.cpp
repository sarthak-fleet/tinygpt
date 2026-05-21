// matmul.cpp — matrix multiply, forward + backward (Phase 4).
//
// STATUS: documented stub. No implementation yet.
//
// The single most performance-critical kernel. Get it correct in scalar form
// first, then add a SIMD build (-msimd128). It is also the first kernel ported
// to WebGPU in Phase 5.
//
// Forward:   C = A @ B
// Backward:  dA = dC @ B^T      dB = A^T @ dC
//
// Used by: Linear layers, attention score/value products, the output head.
//
// Guide: docs/browser_notes.md ("WASM backend", "WebGPU acceleration")
//
// TODO(phase-4): scalar matmul + its backward.
// TODO(phase-4): SIMD variant verified equal to scalar within tolerance.
