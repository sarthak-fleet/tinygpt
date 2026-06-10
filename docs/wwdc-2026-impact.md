# WWDC 2026 (June 8–9) — impact on tinygpt + Pace

Researched 2026-06-10 via two web sweeps (platform + product). Sources at
bottom; unverified items marked. macOS 27 is "Golden Gate", fall 2026.

## The three findings that change our roadmap

### 1. "Core AI" framework supersedes Core ML — and aims at our exact bottleneck

New inference framework (same runtime as Apple Intelligence): Swift API
(`AIModel`, `InferenceFunction`, `NDArray`), Python toolchain (`coreai-torch`
/ `coreai-opt` / `coreai-build`), **multi-function single-asset models**,
**in-place mutable state views** (no input/output state copies), AOT
compilation, embedded custom Metal kernels, int4/int8/FP4/FP8 + palettized
quantization, dedicated Instruments profiler.

**Why it matters:** M8's decode ceiling is ~1.6 ms × 28 dispatches/token.
A 28-function single Core AI asset + in-place state views is plausibly the
first-party fix for both the dispatch overhead AND our IOSurface ping-pong
hack. Also likely routes around the coremltools-9/macOS26 ANE-binding bug
(no coremltools fix shipped — still 9.0 from Nov 2025).

**Action (new M9 candidate):** prototype 2-block Qwen3 chain as a
2-function Core AI asset; measure per-token dispatch vs M8; numerics gate
applies. This replaces "retry M6 N≥2 on macOS 26" as the decode lever.

### 2. Foundation Models now runs CUSTOM models — Apple shipped our M8 idea

`LanguageModel` protocol: any model can back a `LanguageModelSession`.
Apple ships open-source **`CoreAILanguageModel` (ANE)** and
**`MLXLanguageModel` (GPU)** — the WWDC session literally demos custom
**Qwen3-0.6B/8B on ANE** through it. Plus: rebuilt on-device model (8k ctx,
image input, better tool calling), `@Generable` structured output, Spotlight
RAG tool, Evaluations framework, `fm` CLI.

**Why it matters:** part-threat (structured gen + tool calling + free model
commoditizes runtime surface), part-leverage (first-party ANE path for our
exact base model). Our remaining moat: grammar-constrained decoding fidelity
(FSM masks vs `@Generable`), the LoRA/DoRA adaptation loop, TTFW, and
any-app AX control.

**Action:** benchmark `CoreAILanguageModel(Qwen3-0.6B)` vs M8 chain vs
serve-int8 (tok/s, TTFW, constraint fidelity). Conform tinygpt's serve to
the `LanguageModel` protocol so Pace's planner is model-pluggable either way.

### 3. Siri-Gemini + macOS 27 defines Pace's launch window

Siri rebuilt on Gemini-derived models: screen awareness, multi-step
cross-app actions via App Intents (SiriKit deprecated). BUT: waitlisted
beta, **M3+/12 GB requirement, English-only, fall 2026, App-Intents-bound**
(only apps that opted in), and **Private Cloud Compute now runs on Google
Cloud/NVIDIA infrastructure**.

**Why it matters:** Apple just announced Pace's category and simultaneously
handed us the differentiation: *any app (AX, no developer opt-in), any
Apple Silicon Mac (M1+), zero cloud ("your voice never touches Google's
servers"), shipping now.* The window is **before fall 2026**.

## Secondary findings

- **MLX first-party embrace**: dedicated sessions, M5 neural-accelerator
  kernels (4× matmul vs M4), continuous batching in mlx-lm server,
  Thunderbolt-RDMA distributed inference (26.2). Validates our stack; pull
  M5 kernels + consider continuous batching for serve.
- **Dictation in macOS 27**: system-level auto-formatting, "hyper-accurate"
  local dictation (M3+/12 GB tier). Direct threat to the dictation
  specialist → re-scope Stage B toward custom vocab / app-aware formatting /
  voice-edit commands; treat dictation as feature, not wedge.
- **Visual Intelligence on macOS** (screenshot-based) + App Schemas: no
  general third-party computer-use API → the Pace Task Loop stays
  differentiated. Expose Pace skills as App Intents for Siri reach.
- **Speech/TTS**: nothing new at WWDC26; SpeechAnalyzer (WWDC25) still the
  comparison point. WhisperKit decision stands; re-bench on macOS 27 beta.
- **Extensions framework**: third-party AI providers (Gemini, Claude)
  selectable for Siri/Writing Tools. Monitor whether Pace can register as a
  provider. No new permission restrictions on AX/mic/screen found in beta-1
  coverage (re-check at beta 2).

## Roadmap deltas (concrete)

1. NEW research task: **Core AI multi-function chain prototype (M9)** — the
   decode-speed lever, supersedes the M6-retry idea.
2. NEW benchmark: **CoreAILanguageModel vs M8 vs serve-int8** on Qwen3-0.6B.
3. Landing page: add the PCC-on-Google-Cloud contrast line + "any app, any
   Apple Silicon, today" vs Siri's M3+/waitlist/fall.
4. Dictation Stage B re-scoped (custom vocab/voice-edit) — lower priority.
5. Launch clock: HN/public launch **before fall 2026**, ideally well before
   Golden Gate GA.
6. macOS 27 beta validation pass for Pace (AX intact, Intel dropped is fine).

## Sources

Platform: Apple WWDC26 sessions 324/325/326 (Core AI), 241/339/298
(Foundation Models), 232/233/328 (MLX), 297 (Visual Intelligence), 240/345
(App Intents), 347 (agentic security); coremltools GitHub releases.
Product: Apple Newsroom (next-gen Apple Intelligence), CNBC/TechCrunch/
Engadget WWDC26 coverage, MacRumors State of the Union, 9to5Mac (beta-1
waitlist), TechTimes (M3/12GB requirement), Neowin/digitimes (PCC on Google
Cloud), macOS 27 developer release notes. Press-sourced items (Siri-Gemini
details, "replaces Core ML" framing) are lower-confidence than Apple
session pages.
