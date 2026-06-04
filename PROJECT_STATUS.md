# Project Status

Last updated: 2026-06-04

## Current Scope

TinyGPT is a from-scratch GPT-2-shaped transformer project with browser training/inference, Python/PyTorch references, C++/WASM, WGSL/WebGPU acceleration, and a native macOS research track for local model experimentation.

## Done

- The original 10 milestone browser/research roadmap is complete and merged to main.
- Implemented work includes PyTorch baseline, training, LoRA, evaluation suite, browser WASM, WebGPU, checkpointing, metrics dashboard, write-up, and public repo readiness.
- The README documents shipped WebGPU, Memory64, FlashAttention-style work, performance lessons, and negative results.
- A native macOS app track exists for Hugging Face Llama architecture support and LoRA fine-tuning.
- Future work is documented around the single-machine roadmap rather than the completed browser milestone list.

## Planned Next

1. Use `docs/single_machine_roadmap.md` as the current roadmap for native Mac/local-model work.
2. Separate completed browser milestones from active native-app research tasks in docs and issue/task references.
3. Continue measuring backend changes with the existing eval/performance harness before claiming speed or quality wins.
4. Preserve trained checkpoints and generated gallery artifacts unless a cleanup is explicitly requested.

## Deferred / Parked

- The original Phases 1-10 browser roadmap is complete; do not reopen it as active work.
- Larger backend/evaluation ideas such as WebNN or alternate attention paths are deferred until they have a measured reason.
- Hosted model service or commercial API scope is parked.
