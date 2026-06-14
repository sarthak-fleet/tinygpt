# agents.md — tinygpt

## Shared Fleet Standard

Also read and follow the shared fleet-level agent standard at `../AGENTS.md`.

## Purpose

A **learning project**, not a deployed product: build a browser-capable TinyGPT that
trains from scratch and adapts a small base model with LoRA. Priority is correctness
and understanding over output quality or shipping.

## North-star (owner's goal — recorded 2026-06-14)

The owner is **not** trying to win at the large-scale / frontier paradigm — no
money, no compute, and that's fully accepted. The actual goal:

1. **Win on the Mac** — be best-in-class at Mac-local AI specifically.
2. **Learn the whole space like a sponge** — including the *single-machine ↔
   distributed boundary*: what a cluster/distributed system can do that one Mac
   can't, and the physics + economics of why.
3. **Build everything that's buildable on a Mac** — "if it can be built on this
   Mac, build it."
4. **Position for the future** — get the foundation, knowledge, and assets done
   now, so when money/opportunity to expand arrives we scale from a running start.

How this shapes the work:
- Value **completeness, depth, and learning as first-class outcomes**, not only
  commercial ROI. Coverage of the Mac-local surface IS the goal.
- Treat "failed" experiments as **learning wins** — e.g., A1 (a fine-tune not
  beating the base) mapped the fine-tuning frontier; the distillation result (a
  0.6B matching a 4B at 1/7th size on tool-calling) is a validated recipe.
- The eval/judgment "turnaround" is *one* valuable piece, not the sole focus —
  don't over-narrow to a single commercial niche.
- When scaled/distributed approaches come up (Prime Intellect, oMLX, teale),
  explaining them is **boundary-mapping**, not a detour.
- Tactically, ROI-scope per task still applies; strategically the north-star is
  comprehensive Mac mastery + future optionality.

> Owner preference: durable goal/context like this lives **here in AGENTS.md**
> (owner-readable, version-controlled), not in agent-private memory.

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

## Project sequencing (owner's call, do not re-litigate)

The order in which remaining major threads should land, as decided by the
owner. **Polish is treated as a moat, not a finish line** — when a project
catches a niche social moment, low-effort clones appear within days, and the
original survives only if it's polished enough that the clones look obviously
worse. So we spend longer here than the engineering-effort estimate suggests.

1.  **Polish** — editorial passes + UI sweep + every rough edge sanded. Owner
    explicit: clones will appear post-launch; this is what differentiates.
2.  **Docs / learning** — turn this from "perf demo" into "how to build a
    GPT from scratch, narrated."
3.  **Further perf** — small safe pushes (operator fusion, async dispatch,
    workgroup tuning). Capped at 1-2 days.
4.  **Astro migration** — chosen for the speed (static-first, partial
    hydration), Lighthouse scores, MDX content authoring, and the learning
    value for the owner. Not just cleanup.
5.  **Deploy + launch** (task #55) — blog post live, HN submission, X/Twitter
    thread, link from portfolio. After this, decisions become data-driven.
6.  **PostHog analytics** (task #56) — three events, no PII. Pairs with #5
    so launch-day data starts flowing immediately.
7.  **"Watch the model think" view** (task #57) — interactive forward-pass
    visualization. The teaching-visualization lever; screenshots travel.
8.  **Native macOS app** (`native-mac/`) — comes BEFORE FA2. Larger
    models become natural here, which makes FA2's payoff worth the
    multi-day effort. See `native-mac/ROADMAP.md`.
9.  **Flash Attention 2** (task #47) — once big models are real in the
    Mac app, FA2's long-context win pays off.
10. **Pre-trained model gallery** — last. Implicit promise of the gallery
    ("you can train these too") is only honest once the speed + Mac
    paths are both shipped.

Items 5 + 6 can land in parallel with later polish/docs work — the deploy
isn't gated on "everything perfect", and analytics is ~30 min once a project
key is in hand. Items 1-4 are the in-session loop; 5-10 are the launch +
beyond sequence. If asked to flip this order, ask the owner before acting.

## Safety rules for heavy GPU / compile loops (macOS host)

Some work on this repo — particularly **Flash Attention 2** (task #47), the
**native Mac app** (`native-mac/`), and any **benchmark sweeps over big-preset
configs** — can spawn workloads that stress the macOS graphics stack hard
enough to make WindowServer sluggish or unstable. This is **workload runaway
+ UI compositor stress, not hardware failure**, but it's still expensive in
user time.

Rules of engagement when working on this repo from an AI agent context:

- **Never run long benchmarks, training, or install/build loops without first
  asking the user.** "Long" here means: more than a few seconds of pinned
  CPU/GPU, more than a single training step on anything above the Small
  preset, or any sweep that repeats kernel dispatches in a tight loop.
- **Single-shot heavy work is OK** (e.g. one Behemoth allocation + one train
  step to verify Memory64 works) — but stop after the verification, don't loop.
- **Kill background processes you spawned** before ending a task. `npm run
  dev` workers, headed Playwright Chrome windows, and Emscripten compile jobs
  all count. Use `ps` / `kill -9` rather than leaving things to time out.
- **Workloads to flag explicitly before kicking off:** Flash Attention 2
  kernel development with iterated bench runs; MLX/Metal model runs from the
  native Mac app; any `pip install` of PyTorch/JAX/CUDA-adjacent packages;
  any parallel compile (`emcc -j`, `cmake --parallel`, `cargo build`).

If you suspect the host has degraded, ask the user to keep a guardrail
terminal open with:

```
top -o cpu
```

and if the screen starts lagging, identify and kill the runaway process from
a separate terminal (or via SSH from a phone if the GUI is locked up):

```
ps -arcwwwxo pid,pcpu,pmem,comm | head -30
kill -9 <pid>
```

This guidance came directly from the project owner after a heavy session.
File under "things that aren't obvious until they bite you."
