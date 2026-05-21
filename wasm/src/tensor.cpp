// tensor.cpp — flat float32 tensor type + helpers (Phase 4).
//
// STATUS: documented stub. No implementation yet.
//
// Minimal building block for the WASM CPU backend: a contiguous float32 buffer
// plus a shape. No general autograd — each op (matmul, layernorm, attention,
// adamw) implements its own forward and backward by hand.
//
// Suggested surface:
//   struct Tensor { float* data; int* shape; int ndim; int size; };
//   alloc / free / zero / fill / shape helpers
//
// Build (see wasm/README.md):
//   emcc src/*.cpp -O3 -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 \
//     -o dist/tinygpt.js
//
// Guide: docs/browser_notes.md ("WASM backend")
//
// TODO(phase-4): define Tensor and allocation helpers; export an ES6 module.
