---
name: B34 batched eval-runtime + pluggable MLX backend (oMLX steals)
status: not-started
owner: unassigned
created: 2026-06-14
parent_plan: docs/PLAN.md §Market-landscape positioning (Tier B)
parent_strategy: docs/learn/competitive-landscape.md (serving is commoditizing → consume it; differentiate on eval)
related_prds: B32-eval-ci-gate.md (the gate this speeds up), B26-deferred-tools.md (eval-mode parity)
---

# PRD — Batched eval-runtime: steal oMLX's batching + prefix-KV cache for the eval harness

## Goal

Make tinyGPT's **eval harness** fast and scalable by stealing the two oMLX
techniques that fit eval's workload — **continuous batching** and a
**persistent shared-prefix KV cache** — and by making the harness drive a
**pluggable fast MLX backend** instead of the in-house single-stream serve.
The eval/judgment layer is the strategic asset post-A1; it's only as good as
how fast it runs. Pace consumes the same qualified backend.

## Why now

- **The turnaround makes eval the product.** B32 (eval-gate), the
  mac-assistant-judgment benchmark, BFCL/τ-bench/lm-eval — all gate on serve
  throughput. Faster eval = the differentiated layer actually ships.
- **Diagnosed bottleneck (2026-06-14):** tinyGPT's HF `serve` path runs ~**7
  tok/s** on a 4B (M5 Pro) and is **single-stream**, so a BFCL suite of
  hundreds of requests runs sequentially through a slow engine. The A1 eval
  (120 examples × 2 models) took ~tens of minutes for this reason.
- **oMLX proved the Mac techniques** (RAM→SSD tiered KV, continuous batching,
  any MLX HF model) and is a commoditizing utility — so *consume* the runtime,
  *steal* the eval-shaped wins.
- **Eval is the ideal workload for these steals:** every request in a suite
  shares the *identical* system prompt + tool schema → one prefix-KV compute,
  reused N times; and N independent requests → continuous batching.

## What we steal (and what we don't)

| oMLX feature | Steal for tinyGPT? | Why |
|---|---|---|
| Continuous / iteration-level batching | **Yes — top priority** | eval = many concurrent requests; biggest throughput lever |
| Persistent shared-prefix KV cache | **Yes** | eval requests share the system+tools prefix verbatim |
| Tiered KV cache RAM→SSD | **Adopt via backend** | lets eval drive big-context models; don't reimplement |
| OpenAI **+ Anthropic** dual API | **Yes (small)** | harness can drive any backend + matches Claude-Code clients |
| Native menu-bar app / general production serving | **No** | that's oMLX's lane; don't build runtime #5 |

## Scope — in

- **Pluggable backend for the eval harness.** `EvalHarnessSupport.startServe`
  gains a `--backend {tinygpt,omlx,mlx-server,openai-url}` selector. Default
  stays `tinygpt` (no behavior change); the new paths point the harness at a
  fast batching-capable backend (oMLX / `mlx_lm.server`) or an external
  OpenAI-compatible URL.
- **Concurrent request submission.** Refactor the eval drivers (BFCL/τ-bench/
  lm-eval shims) to fire requests with bounded concurrency (e.g. 16 in flight)
  so the backend's continuous batching engages, instead of one-at-a-time.
- **Shared-prefix exploitation.** Detect the suite's invariant prefix (system
  prompt + tool schema) and submit requests so a prefix-caching backend
  (oMLX/RadixAttention-style) gets cache hits; emit a one-line "prefix cached:
  N tokens" log.
- **Anthropic-format adapter** alongside the existing OpenAI one, so the
  harness can drive Anthropic-API backends.
- A `scripts/eval-throughput-smoke.sh` that runs a fixed N-request suite
  through `tinygpt` vs a batched backend and reports the speedup.

## Scope — out

- **Rebuilding tinyGPT's `serve` into a production server** (KV-SSD paging,
  multi-tenant). That's oMLX/LM Studio's commoditized lane — adopt, don't
  rebuild. The native from-scratch `.tinygpt` decode path (already fast, 293–767
  tok/s) is untouched.
- **Pace's in-process runtime** — separate track; Pace consumes the qualified
  backend (a one-line note in Pace's planner config, not this PRD).
- **Logit-level / training changes** — eval-only.

## Files to touch

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPT/EvalHarnessSupport.swift` | `--backend` selector + concurrent submission helper |
| `native-mac/Sources/TinyGPT/EvalBFCL.swift` / `EvalTauBench.swift` | fire requests with bounded concurrency |
| `native-mac/Sources/TinyGPT/*` (Anthropic shim) | new — Anthropic-format request adapter |
| `scripts/eval-throughput-smoke.sh` | new — throughput A/B (tinygpt vs batched backend) |
| `docs/learn/omlx-steals.md` | new — the steal rationale (house style) |
| `docs/PLAN.md` | B34 entry |

## Don't touch

- The native `.tinygpt` serve/decode path (it's the fast one).
- `Sources/TinyGPT/TinyGPT.swift` dispatch beyond one flag (maintainer merges).

## Acceptance criteria

- [ ] A fixed BFCL/τ-bench suite runs **≥3× faster** end-to-end via a batched
  backend vs the current sequential `tinygpt serve` path (measured by
  `eval-throughput-smoke.sh`).
- [ ] Shared system+tools prefix is computed once and reused (cache-hit logged);
  per-request TTFT drops accordingly.
- [ ] `--backend` defaults to `tinygpt` — existing eval invocations are
  byte-unchanged.
- [ ] Eval scores are within noise of the sequential baseline (batching/caching
  must not change outputs at T=0).
- [ ] Harness can drive an Anthropic-API backend, not just OpenAI.

## Reference patterns

- `docs/learn/advanced-llm-inference.md` §4–6 (PagedAttention, prefix/Radix
  caching, continuous batching) — the theory being stolen.
- oMLX (tiered KV RAM→SSD, continuous batching) — the Mac-proven implementation.
- `EvalHarnessSupport.startServe` (current serve-spawn) — the integration point.

## Open questions

- **Build vs adopt the backend.** Recommendation: **adopt** (`mlx_lm.server`
  has batching today; oMLX is more capable) and keep tinyGPT's value at the
  *harness* layer (concurrency + prefix exploitation + scoring). Building
  native batching into `serve` is a larger, lower-ROI rebuild — only if backend
  licensing/control becomes a blocker.
- Which backend to qualify first — `mlx_lm.server` (OSS, in-tree-ish) vs oMLX
  (more features, separate app). Qualify on the eval-throughput smoke.
- Concurrency ceiling that maximizes throughput without OOM on a 48 GB Mac.
