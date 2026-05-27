# Project status — 2026 update

A review-oriented snapshot of where TinyGPT stands. The detailed docs are linked
at the bottom; this page is the map.

TinyGPT is finished as a teaching project and continuing as a performance
project. The original ten milestones (PyTorch ref, training, LoRA, WASM
backend, browser app, WebGPU matmul, checkpointing, metrics dashboard,
write-up, public repo) are all complete and on `main`. The work past that
point is the perf round-trip — kernels, parity tests, the speedup curve,
and the lessons each failed lever taught.

## What's measured and shipped beyond the original milestones

| Area | What | State |
| --- | --- | --- |
| Perf | WASM SIMD (`-msimd128`) — measured 1.6× | shipped |
| Perf | Multi-threaded WASM (pthreads + SAB) — measured ~2× | shipped |
| Perf | WebGPU full stack (blocked4 + vec4 + subgroups + FA2 fwd+bwd) | shipped |
| Perf | End-to-end curve vs multi-thread WASM SIMD: Small 2.6×, Medium 6.8×, Large 9.3×, XL 12.1× | measured |
| Capacity | Memory64 module (`tinygpt64.{js,wasm}`) — 473M params in Node, browser blocked at d_model ≥ 256 (task #66) | partial |
| Data | Default corpus switched from inline 863-byte paragraph to TinyShakespeare (1.1 MB, `/shakespeare.txt`) | shipped |
| Data | Hugging Face dataset loading via public datasets-server API | shipped |
| Config | Default LR fixed: was `3e-3` (10× the Python ref), now `3e-4` — see [`lessons.md`](lessons.md) | shipped |
| UX | Banner reworked to make "load pretrained / train from scratch (~15 min)" explicit | shipped |
| UX | Pretrained Shakespeare demo model (Huge preset, 5000 steps) replaces `browser/public/demo.tinygpt` | shipped |
| Site | Astro migration — 5 static routes built into `dist/` | shipped |

## What's verified, and how

| Suite | Covers | Result |
| --- | --- | --- |
| `tests/test_phase1.py` | model, training, sampling | 8/8 |
| `tests/test_lora.py` | LoRA | 6/6 |
| `wasm/build_native.sh` | C++ kernels (finite-diff) + C++ model overfit | pass |
| `tests/smoke_wasm_node.mjs` | compiled WASM module trains | pass |
| `browser/npm run webgpu-test` | 24 WebGPU kernel parity checks + GPU overfit | pass |
| `tests/test_webgpu_train.mjs` | 50-step WASM vs WebGPU end-to-end parity | pass (drift 1.1–2.5%) |
| `tests/test_fa2_parity.mjs` + `test_fa2_backward_parity.mjs` | FA2 fwd + bwd vs naive ref | pass (≤ 1 ULP) |
| `tests/test_wasm64_xl_node.mjs` | reproduces the Memory64 ABI bug | reproduces (task #66) |
| `browser/npm run e2e` | full app in headless browser | pass |

Everything that can be checked by a machine, is — that was the method throughout.

## Open — worth your attention

- **The Memory64 ABI bug.** Same `.wasm` runs cleanly in Node when called
  directly, but the browser path hits a JS↔WASM bridge bug at d_model ≥ 256.
  `_malloc` returns Number; cwrap pointer args expect BigInt; the conversion
  throws. Tracked as task #66; the playground falls back to the 32-bit
  module for XL/Massive/Mega/Behemoth presets. Reproducer:
  `tests/test_wasm64_xl_node.mjs`. Full write-up: [`lessons.md`](lessons.md).
- **The LR-default bug, fixed.** Browser default LR was `3e-3` for months
  (10× the Python reference's `3e-4`). Loss plateaued at ~2.45 on real
  corpora and read as a modelling ceiling. Fixed in
  `browser/src/types.ts:35` and `browser/src/pages/index.astro:2621`.
- **Speedup is a curve, not a single number.** "9.7× end-to-end" is the
  Medium-preset point on the curve; Small is 2.6×, XL is 12.1×. Don't cite
  a flat ratio — see [`lessons.md`](lessons.md) for the framing.
- **PR history past #17 is in GitHub.** Too long to mirror here, but every
  shipped item above has a merged PR on `main`.

## Where the docs are

- [`learn.md`](learn.md) — start here to understand the repo
- [`notes.md`](notes.md) — what each component does and what each experiment showed
- [`performance.md`](performance.md) — the SIMD and WebGPU performance work
- [`lessons.md`](lessons.md) — the bugs and surprises worth more than the kernels
- [`model_guide.md`](model_guide.md), [`lora_guide.md`](lora_guide.md),
  [`browser_notes.md`](browser_notes.md), [`evaluation.md`](evaluation.md) —
  per-phase detail
- [`feature_ideas.md`](feature_ideas.md) — interactive-learning backlog
