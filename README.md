# TinyGPT

A GPT-2-shaped transformer, written from scratch and trained **in your browser
tab** — **2.6× → 12.1× faster** than the multi-threaded WebAssembly baseline
thanks to hand-written WebGPU kernels. The speedup is a curve, not a single
number: GPU work amortizes better as `d_model` grows. Parity-tested to within
2.5% loss drift across the curve.

Python reference, hand-written C++/WASM, hand-written WGSL — the same model at
three levels, with every gradient pinned down by a test.

**[Live playground →](https://tinygpt.sarthakagrawal.dev)**
· [Speedup chart](browser/speedup.html)
· [Devlog](browser/devlog.html)
· [Roadmap](browser/roadmap.html)

![TinyGPT playground](browser/public/og-image.png)

## Why this exists

It started as a teaching project — the goal was to build the whole modern LLM
stack at a size where nothing stays a black box. Every backward pass is derived
by hand, every kernel is parity-checked against a reference, no autograd engine
is involved on the C++/WebGPU side.

Somewhere along the way it became a performance project. The interesting work
stopped being "does the maths come out right" and started being "how fast can
this model train inside a browser tab, without lying about the numbers." Most
of what's in [`browser/devlog.html`](browser/devlog.html) is that second half.

## Key measured numbers

All on the same Apple M-series laptop, same model, same seed, same data.
Reproducible from the playground's bench button or `tests/test_webgpu_train.mjs`.

- **End-to-end speedup curve, WebGPU vs multi-thread WASM SIMD** — Small
  (d=96) 2.6×, Medium (d=128) 6.8×, Large (d=192) 9.3×, XL (d=256) 12.1×.
  The curve trends upward because the blocked-4×4 matmul kernel's win grows
  with matmul size — the bigger the model, the more the GPU pulls away.
  Mega/Behemoth aren't on this curve yet: a Memory64 ABI bug at the JS↔WASM
  bridge currently blocks an in-browser end-to-end run (see "Known issues").
  Loss drift across the curve: 1.1%–2.5% after 50 steps — float-reorder
  noise from different GPU accumulation order.
- **5.18× kernel speedup at 2048³ matmul** — the size that dominates the
  Mega/Behemoth presets. Naive WebGPU matmul: 47.24 ms. Workgroup-tiled:
  17.23 ms. Tiled + 4×4 register blocking: 9.12 ms.
- **473M-parameter model allocated in a tab** — `-sMEMORY64=1 -sWASM_BIGINT`
  lifts the 4 GB V8 heap ceiling. The same allocation hard-OOMs the 32-bit
  module; on the 64-bit one it allocates cleanly in 3.7 s and takes one
  training step in 82.2 s with `loss 5.78` (the correct initial loss for
  random init).
- **Flash Attention 2 forward + backward** in WGSL — workgroup-cooperative
  forward with online softmax in registers, backward that recomputes
  attention from a saved log-sum-exp instead of reading the cached matrix.
  The forward dropped its O(B·H·T²) attention writeback entirely; on
  Mega-class shapes (B=4, H=8, T=512) that's ~67 MB of global memory
  traffic per layer per step that now stays on-chip. End-to-end parity at
  2.5% drift vs. the WASM reference.

The full speed-evolution table — scalar → SIMD → threads → WebGPU naive →
WebGPU blocked — lives on the [roadmap](browser/roadmap.html). Each measured
bar is anchored to a number you can reproduce in the playground.

## Architecture in three sentences

`python_ref/` is the PyTorch reference — the clearest version, used as the
oracle when anything else disagrees. `wasm/` is the same maths in C++ with
every backward pass derived by hand, compiled to WebAssembly with Emscripten
(SIMD + pthreads, plus a Memory64 build that lifts the heap ceiling).
`webgpu/` is the whole training loop in WGSL — forward, backward, and AdamW
— every kernel finite-difference checked and parity-tested against the WASM
reference. All three read and write the same `.tinygpt` binary file format,
so a model trained in one path continues training in another.

## What's interesting under the hood

The long-form is in [`browser/devlog.html`](browser/devlog.html). Short version:

- **Memory64 in WebAssembly** lifts the per-tab heap cap from ~4 GB to tens
  of GB. Build script, runtime feature-detect, and a "Behemoth" preset that
  exercises it.
- **A 4×4-register blocked matmul kernel** in WGSL. Workgroup-shared tiling
  (Goto/VandeGeijn 16×16) plus a 4×4 output block per thread held in
  registers, so each shared-memory load gets reused 4× across the
  accumulator. The point where the kernel stops being bandwidth-bound and
  starts being compute-bound.
- **End-to-end parity testing as the only honest bar.** Standalone matmul
  benchmarks lie — they hide bugs that only show up in non-square production
  shapes. The `tests/test_webgpu_train.mjs` driver runs 50 training steps
  under WASM and 50 under WebGPU on the same seed, then asserts loss drift
  is below 5%. Every integration goes through that gate before it counts.

## Negative results — the most valuable lessons came from things that didn't work

This is the part of the project I'd most want a reviewer to look at, because
it's the part most blog posts skip.

- **f16-packed storage on top of tiled matmul** — standalone, packing weights
  as two f16 per u32 was 1.7× faster than naive WebGPU matmul. Stacked on
  top of the tiled kernel, the combined version ran *slower* than plain
  tiled at 2048³: 17.78 ms vs 16.90 ms. Once tiling has amortized the
  global-memory reads, the kernel is compute-bound on shared-memory ops and
  halving global bandwidth has nowhere left to help. **Lesson:** always
  bench an optimization against the *best* baseline, not the naive one.
- **8×8 register blocking** — the natural next step from 4×4, with 4× the
  arithmetic intensity per shared-memory load. Lost at every benchmarked
  size — 0.91× at 1024³, 0.88× at 2048³. Most likely cause: 64 floats per
  thread for the accumulator exceeds the per-thread register budget on
  Apple GPUs, forcing register spill and dropping workgroup occupancy.
  **Lesson:** more aggressive is not always faster.
- **vec4 global loads — broke once, then root-caused.** Wins by 1.37×
  standalone at 2048³, the best single-kernel speedup measured in the
  project. First integration attempt diverged loss to 88.67 vs WASM's 2.94
  — 30× off. Took the end-to-end parity test to catch it; the standalone
  square-shape bench passed cleanly. **Root cause:** the WGSL kernel
  declared `var<storage, read>` for the input buffers, but the shared
  bind-group layout in `ops.ts` declares them as `buffer: { type: "storage" }`
  (read-write). When WGSL access mode doesn't match the layout type,
  Chromium/Apple silently returns wrong data instead of erroring. Fixed by
  declaring all six bindings as `read_write` in `train_vec4.wgsl` — the
  kernel only reads from g0/g1 anyway, the decoration just has to match.
  Now passes parity at 1.6% drift. **Lesson:** standalone benchmarks miss
  bugs that only show up in real training, and "the validation passed" is
  not the same as "the data is right."

The first two are kept in the repo as documented negative results.
The vec4 fix is shipped.

## Lessons from this build

Three discoveries worth more than the kernels themselves. The long-form is in
[`docs/lessons.md`](docs/lessons.md).

- **The LR-default bug.** Browser default learning rate was `3e-3` for months;
  the Python reference uses `3e-4`. Ten times too hot. Loss plateaued at ~2.45
  on real corpora and looked like a modelling ceiling, not a config bug. Fixed
  in `browser/src/types.ts:35` and `browser/src/pages/index.astro:2621`.
  Lesson: parity-check the defaults the same way you parity-check the
  gradients.
- **The Memory64 ABI was untested.** `tests/bench_wasm.mjs` loads the 32-bit
  module, so the 64-bit pthread+Memory64 build had never been exercised in
  Node. The browser path was calling into a broken JS↔WASM bridge —
  `_malloc` returns Number but cwrap pointer args expect BigInt. Reproduced
  by `tests/test_wasm64_xl_node.mjs`. Tracked as task #66.
- **The speedup is a curve, not a number.** "9.7× end-to-end" was true for
  one preset on one day. The honest framing is the scaling curve above —
  2.6× → 12.1× as `d_model` climbs from 96 to 256. Don't quote a flat ratio
  as the project's identity number.

## Tech used

- [PyTorch](https://pytorch.org/) — the reference path
- [Emscripten](https://emscripten.org/) — C++ → WebAssembly (SIMD + pthreads + Memory64)
- [WebGPU](https://www.w3.org/TR/webgpu/) + [WGSL](https://www.w3.org/TR/WGSL/) — the GPU training loop
- [Vite](https://vitejs.dev/) + TypeScript — the playground UI
- [Cloudflare Pages](https://pages.cloudflare.com/) — hosting

## Try it

Open **[tinygpt.sarthakagrawal.dev](https://tinygpt.sarthakagrawal.dev)**.
Two paths: *Load pretrained model* serves a Shakespeare-trained checkpoint
and lets you generate immediately. *Train your own from scratch* runs in
~15 minutes on the larger presets and converges to readable pseudo-Shakespeare
on the bundled 1.1 MB TinyShakespeare corpus. The playground detects your
machine, suggests a model size, shows a live training-time estimate, and
saves checkpoints to OPFS so a run survives a refresh. WebGPU kicks in
automatically on Chrome/Edge 113+ and Safari 18+.

## What's next

- **Pre-trained model gallery** — Cloudflare R2-hosted, manifest-driven; let
  visitors load and continue-train from real checkpoints instead of just the
  one shipped demo.
- **Native macOS app** — MLX-Swift + SwiftUI, same `.tinygpt` file format both
  ways, lifts the model-size ceiling into the 7B–30B range on Apple Silicon.
  See [`docs/shared_vs_native.md`](docs/shared_vs_native.md) for the boundary.

Flash Attention 2 used to live in this list; it shipped — see
[`docs/fa2_forward_notes.md`](docs/fa2_forward_notes.md) and
[`docs/fa2_backward_notes.md`](docs/fa2_backward_notes.md). The forward
runs workgroup-cooperative tiling with online softmax across K blocks;
the backward recomputes attention on the fly from a saved log-sum-exp,
which let the forward drop the O(B·H·T²) attention writeback entirely.

## Repo layout

```
tinygpt/
  python_ref/   PyTorch reference: model, train, sample, LoRA, bench
  wasm/         C++ kernels + a full C++ model, compiled to WebAssembly
  webgpu/       WGSL kernels (forward, backward, AdamW) + JS glue
  browser/      The web app: UI, training Web Worker, tokenizer, storage
  configs/      Model / training / LoRA settings as JSON
  data/         Dataset builder + example corpora
  docs/         The learning guide and the per-phase specs
  tests/        Correctness tests — finite-diff, overfit, end-to-end parity
  native-mac/   (Planned) MLX-Swift macOS app
```

## Build it locally

```
# Python reference
python -m venv python_ref/.venv && source python_ref/.venv/bin/activate
pip install -r python_ref/requirements.txt
python tests/test_phase1.py
python python_ref/train.py --overfit

# Browser app
bash wasm/build_wasm.sh          # needs Emscripten SDK
cd browser && npm install && npm run dev
```

The C++ kernels can also be checked without Emscripten — `bash wasm/build_native.sh`
builds and tests them with a normal compiler. Full deploy notes:
[`docs/deploy.md`](docs/deploy.md).

## Docs

- [`docs/status.md`](docs/status.md) — where the project stands; a review map
- [`docs/learn.md`](docs/learn.md) — guided learning path through the repo
- [`docs/performance.md`](docs/performance.md) — the SIMD and WebGPU performance work
- [`docs/lessons.md`](docs/lessons.md) — the bugs and surprises worth more than the kernels
- [`docs/model_guide.md`](docs/model_guide.md) — the model, from scratch
- [`docs/lora_guide.md`](docs/lora_guide.md) — LoRA fine-tuning
- [`docs/online_softmax_in_attention.md`](docs/online_softmax_in_attention.md) — why and how, ties to the `attn_fused_sv` kernel
- [`docs/shared_vs_native.md`](docs/shared_vs_native.md) — browser vs. native boundary
- [`docs/feature_ideas.md`](docs/feature_ideas.md) — the future-ideas backlog

## License

MIT — see [`LICENSE`](LICENSE). Author: Sarthak Agrawal ([@sarthakagrawal927](https://github.com/sarthakagrawal927)).
