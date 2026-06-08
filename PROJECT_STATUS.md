# Project Status

Last updated: 2026-06-09

## Current Scope

TinyGPT is a from-scratch GPT-2-shaped transformer project with browser training/inference, Python/PyTorch references, C++/WASM, WGSL/WebGPU acceleration, and a native macOS research track for local model experimentation.

For Pace work, TinyGPT is now the development-time factory and eval lab: it produces planner LoRA artifacts, grammar/eval assets, dataset synthesis scripts, and porting helpers. Pace owns production runtime integration; shipped Pace should not depend on `tinygpt serve` or a localhost daemon.

## Done

- The original 10 milestone browser/research roadmap is complete and merged to main.
- Implemented work includes PyTorch baseline, training, LoRA, evaluation suite, browser WASM, WebGPU, checkpointing, metrics dashboard, write-up, and public repo readiness.
- The README documents shipped WebGPU, Memory64, FlashAttention-style work, performance lessons, and negative results.
- A native macOS app track exists for Hugging Face Llama architecture support and LoRA fine-tuning.
- Future work is documented around the single-machine roadmap rather than the completed browser milestone list.
- The Pace v9 serve path now precomputes tokenizer byte tables once at server boot for much faster grammar-constrained first-token latency. A trie-based grammar-mask experiment was implemented but left disabled after measurement showed it was slower than the legacy path.
- Pace v9/v10 grammar and dataset-helper assets are staged as factory inputs, with remaining train/eval/runtime work tracked as SaaS Maker tasks instead of uncompleted PRD files.

## Planned Next

1. Use SaaS Maker tasks for the active Pace/TinyGPT factory queue: DoRA serialization v2, Pace v9 body streaming, Pace v10 parameterized actions, factory-only scope docs, serve FSM latency follow-up, and WhisperKit qualification.
2. Use `docs/single_machine_roadmap.md` as the current roadmap for native Mac/local-model work.
3. Separate completed browser milestones from active native-app research tasks in docs and issue/task references.
4. Continue measuring backend changes with the existing eval/performance harness before claiming speed or quality wins.
5. Preserve trained checkpoints and generated gallery artifacts unless a cleanup is explicitly requested.

## Deferred / Parked

- The original Phases 1-10 browser roadmap is complete; do not reopen it as active work.
- Larger backend/evaluation ideas such as WebNN or alternate attention paths are deferred until they have a measured reason.
- Hosted model service or commercial API scope is parked.
- Pace runtime over TinyGPT HTTP/localhost is parked; keep `serve` as a development and evaluation tool.
