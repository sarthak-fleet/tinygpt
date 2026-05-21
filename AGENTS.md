# agents.md — tinygpt

## Shared Fleet Standard

Also read and follow the shared fleet-level agent standard at `../AGENTS.md`.

## Purpose

A **learning project**, not a deployed product: build a browser-capable TinyGPT that
trains from scratch and adapts a small base model with LoRA. Priority is correctness
and understanding over output quality or shipping.

## Working rules specific to this repo

- **Respect the build order.** Python reference → WASM → WebGPU. Do not implement a
  browser/WebGPU path before the Python reference for that component is correct and
  tested. See `README.md` and `docs/learning_roadmap.md`.
- **Correctness gates.** Before scaling anything, the model must overfit a tiny
  (1–10 KB) repeated dataset. If it cannot, the bug is in model/backprop/data — fix
  that first. See `tests/README.md`.
- **Configs are the source of truth.** Exact specs live in `configs/*.json`. Code and
  docs should reference them rather than restating numbers.
- **Stubs.** Code files are currently documented stubs. When implementing one, follow
  the interface described in its header and the linked `docs/` section.

## Layout

See `README.md`. Specs in `configs/`, guide in `docs/`, tests in `tests/`.

## Not in scope for the fleet tooling

This project is a sandbox: no SaaS Maker product record, deployment, or analytics
wiring is expected unless explicitly requested.
