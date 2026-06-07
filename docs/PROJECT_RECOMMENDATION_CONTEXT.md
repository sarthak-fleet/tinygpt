# Project Recommendation Context

Generated: 2026-06-06T21:14:19.634Z

This file is a CodeVetter Repo Unpacked-inspired audit written for Starboard recommendations. It is intentionally local, evidence-oriented, and safe to commit: it records product context, feature areas, stack inventory, and recommendation guidance without secrets or environment values.

## Project Identity

- Slug: `tinygpt`
- Registry description: TinyGPT.
- Product grouping: `internal-first`
- Source path: `tinygpt`

## Product Context

TinyGPT.

TinyGPT is a from-scratch GPT-2-shaped transformer project with browser training/inference, Python/PyTorch references, C++/WASM, WGSL/WebGPU acceleration, and a native macOS research track for local model experimentation.

TinyGPT A GPT-2-shaped transformer, written from scratch and trained in your browser tab — 2.6× → 12.1× faster than the multi-threaded WebAssembly baseline thanks to hand-written WebGPU kernels. The speedup is a curve, not a single number: GPU work amortizes better as d model grows. Parity-tested to within 2.5% loss drift across the curve. Python reference, hand-written C++/WASM, hand-written WGSL — the same model at three levels, with every gradient pinned down by a test. Live playground → https://tinygpt.sarthakagrawal.dev · Speedup chart browser/speedup.html · Devlog browser/devlog.html · Roadmap browser/roadmap.html ! TinyGPT playground browser/public/og-image.png There is also a native 

## Feature Map

- **AI agents**: Agents, tool use, workflows, orchestration, RAG, evals, and model integration. Keywords: ai, agent, agents, llm, rag, embedding, eval, model.
- **UI workflows**: Dashboards, tables, forms, component systems, charts, and user workflows. Keywords: ui, ux, dashboard, table, component, react, next, tailwind.
- **Testing and quality**: Unit tests, browser tests, evals, CI quality gates, and regression checks. Keywords: test, testing, quality, vitest, playwright, ci, eval, benchmark.
- **Content and media**: Content production, video, reels, documents, markdown, and publishing workflows. Keywords: content, media, video, reel, markdown, document, publish, editor.
- **Browser and extensions**: Browser extensions, page capture, annotation, automation, and client-side integrations. Keywords: browser, extension, chrome, annotation, capture, webpage, reader.
- **Search and discovery**: Search, ranking, recommendations, feeds, semantic retrieval, and discovery UX. Keywords: search, discovery, recommend, ranking, semantic, feed, index, retrieval.
- **Analytics and intelligence**: Signal analysis, forecasting, monitoring, trends, metrics, and decision support. Keywords: analytics, intelligence, signal, forecast, monitoring, metric, trend, insight.

## Runtime Surfaces and Entrypoints

- `browser/src/pages/devlog.astro`
- `browser/src/pages/docs/[...slug].astro`
- `browser/src/pages/docs/index.astro`
- `browser/src/pages/eval-leaderboard.astro`
- `browser/src/pages/index.astro`
- `browser/src/pages/leaderboard.astro`
- `browser/src/pages/playground.astro`
- `browser/src/pages/roadmap.astro`
- `browser/src/pages/sae-timeline.astro`
- `browser/src/pages/speedup.astro`
- `browser/src/pages/training-dashboard.astro`
- `browser/src/pages/webgpu-test.astro`
- `native-mac/Sources/TinyGPT/Agent.swift`
- `native-mac/Sources/TinyGPT/AgentLoop.swift`
- `native-mac/Sources/TinyGPT/Bench.swift`
- `native-mac/Sources/TinyGPT/BestOfN.swift`
- `native-mac/Sources/TinyGPT/CausalTrace.swift`
- `native-mac/Sources/TinyGPT/CloudList.swift`
- `native-mac/Sources/TinyGPT/CloudPull.swift`
- `native-mac/Sources/TinyGPT/CloudPush.swift`
- `native-mac/Sources/TinyGPT/ColdStart.swift`
- `native-mac/Sources/TinyGPT/Compare.swift`
- `native-mac/Sources/TinyGPT/DPO.swift`
- `native-mac/Sources/TinyGPT/Debug.swift`
- `native-mac/Sources/TinyGPT/Dedupe.swift`
- `native-mac/Sources/TinyGPT/Distill.swift`
- `native-mac/Sources/TinyGPT/DownloadDataset.swift`
- `native-mac/Sources/TinyGPT/ES.swift`
- `native-mac/Sources/TinyGPT/Escalate.swift`
- `native-mac/Sources/TinyGPT/Eval.swift`
- `native-mac/Sources/TinyGPT/EvalBFCL.swift`
- `native-mac/Sources/TinyGPT/EvalCompare.swift`
- `native-mac/Sources/TinyGPT/EvalHarnessSupport.swift`
- `native-mac/Sources/TinyGPT/EvalHumanEval.swift`
- `native-mac/Sources/TinyGPT/EvalIndic.swift`
- `native-mac/Sources/TinyGPT/EvalMTEB.swift`
- `native-mac/Sources/TinyGPT/EvalTauBench.swift`
- `native-mac/Sources/TinyGPT/Extract.swift`
- `native-mac/Sources/TinyGPT/ExtractorData.swift`
- `native-mac/Sources/TinyGPT/FetchGitHub.swift`
- `native-mac/Sources/TinyGPT/Filter.swift`
- `native-mac/Sources/TinyGPT/Finetune.swift`

## Current Stack

- Languages: `Astro`, `Python`, `Rust`, `Swift`, `TypeScript`
- Frameworks/tools: `Astro`, `Cargo`, `Swift Package Manager`
- Config files:
- `browser/_legacy_html/vite.config.ts.bak`
- `browser/astro.config.mjs`
- `native-mac/Package.swift`
- `scripts/data-prep/pyproject.toml`
- `scripts/hf-downloader/Cargo.toml`
- `scripts/humaneval-sandbox/Cargo.toml`
- `scripts/parquet-decoder/Cargo.toml`
- `scripts/tokenizer-trainer/Cargo.toml`

## OSS Already In Use

Direct dependencies:
- `@floating-ui/dom`
- `posthog-js`

Development dependencies:
- `@astrojs/mdx`
- `@fontsource-variable/geist`
- `@fontsource-variable/geist-mono`
- `@types/node`
- `@webgpu/types`
- `astro`
- `lightningcss`
- `playwright`
- `sharp`
- `typescript`

Package scripts:
- `build`
- `dev`
- `e2e`
- `preview`
- `typecheck`
- `typecheck:tools`
- `webgpu-test`

## Testing and Quality Signals

- `tests/README.md`
- `tests/bench_wasm.mjs`
- `tests/smoke_wasm64_node.mjs`
- `tests/smoke_wasm_node.mjs`
- `tests/test_f16_packer.mjs`
- `tests/test_fa2_backward_parity.mjs`
- `tests/test_fa2_compile.mjs`
- `tests/test_fa2_parity.mjs`
- `tests/test_lora.py`
- `tests/test_phase1.py`
- `tests/test_wasm64_xl_node.mjs`
- `tests/test_wasm_kernels.cpp`
- `tests/test_wasm_model.cpp`
- `tests/train_demo.mjs`

## Recommendation Guidance

Good matches:
- Repos that strengthen ai agents without replacing already-installed libraries.
- Repos that strengthen ui workflows without replacing already-installed libraries.
- Repos that strengthen testing and quality without replacing already-installed libraries.
- Repos that strengthen content and media without replacing already-installed libraries.
- Repos that strengthen browser and extensions without replacing already-installed libraries.
- Repos that strengthen search and discovery without replacing already-installed libraries.
- Repos that strengthen analytics and intelligence without replacing already-installed libraries.
- Tools with concrete support for tinygpt, browser, native-mac, sources, webgpu, pages, model, src.
- Implementation repos, SDKs, CLIs, testing utilities, adapters, and focused libraries are higher value than generic awesome lists.

Avoid recommending:
- Do not recommend packages already listed under direct or development dependencies unless the task is migration research.
- Do not recommend broad framework replacements unless the project context explicitly calls for a rewrite.
- Downrank curated lists, archived repos, stale demos, and generic UI kits that do not map to the feature catalog.

## Evidence Read

Primary docs and handoff files:
- `AGENTS.md`
- `HANDOFF.md`
- `PROJECT_STATUS.md`
- `README.md`
- `docs/CITATIONS.md`
- `docs/MAP.md`
- `docs/PLAN.md`
- `docs/agent_runtime.md`
- `docs/async_tool_dispatch.md`
- `docs/audit_2026.md`
- `docs/backlog.md`
- `docs/benchmark_first_run.md`
- `docs/benchmark_harness_design.md`
- `docs/bpe_browser_scoring.md`
- `docs/browser_notes.md`
- `docs/capability_matrix.md`
- `docs/cold_start_results.md`
- `docs/constrained_generation.md`
- `docs/continue_provider.md`
- `docs/cpu_speedup_results.md`
- `docs/cpu_utilization_research.md`
- `docs/data_inventory.md`
- `docs/data_perf.md`
- `docs/dataset-inventory.md`
- `docs/decision_log.md`
- `docs/deploy.md`
- `docs/determinism.md`
- `docs/distillation.md`

Package manifests:
- `browser/package.json`

Inventory notes:
- Files scanned: 724
- This pass uses deterministic repo inventory plus local documentation/source-path evidence. It does not claim a full manual line-by-line review of every source file.

## Confidence

Confidence: **high**

Why:
- PROJECT_STATUS.md present
- README.md present
- 42 entrypoint/runtime files identified
- package dependencies inventoried
- 14 test/quality files identified

Refresh command:

```bash
cd /Users/sarthak/Desktop/fleet/starboard
pnpm fleet:audit-recommendation-context
pnpm fleet:extract-projects
```
