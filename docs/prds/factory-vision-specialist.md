---
name: Vision-language specialist for Pace screen reading
status: m1-m3-shipped-m4-qwen3vl-scaffold-shipped-2026-06-07
owner: elf
created: 2026-06-08
priority: P0 — Pace's screen-reading currently uses Qwen3-VL-8B (~6 GB Q4). Replace with a 1-2B distilled specialist for ~3× RAM win + same/better screen-reading quality.
size: 3-4 weeks (real architecture work)
authorized-by: maintainer 2026-06-08 ("Well I want the best option and I want you to start now.")
---

# PRD — VLM specialist for Pace screen reading

## Status note — 2026-06-07

M1-M3 are shipped. The M4 architecture decision is now made in
`factory-vision-m4-architecture-decision.md`: Option A, full Qwen3-VL port
with UI-Venus-1.5-2B as base.

M4 scaffold v1 is shipped in `Qwen3VLScaffold.swift`: compile-safe contracts
for Qwen3-VL mRoPE metadata, image-token replacement validation, and
deepstack visual injection sites. This is intentionally not a HF weight loader
or a functional UI-Venus forward path.

This PRD remains active, not complete. M4 requires the named Qwen3-VL
features to be implemented and parity-gated against HF PyTorch:

- multimodal RoPE (`mrope_section`)
- `<image>` token embedding replacement
- deepstack visual feature injection at the configured LLM layers

Those are multi-day architecture tasks and should not be reported as done
until the per-feature parity tests exist.

## Goal

Distill Pace's current screen-reading VLM (Qwen3-VL-8B via LM Studio at
~6 GB Q4) into a **1-2B student** that:

1. Takes a screenshot + optional Apple Vision OCR hints
2. Emits Pace's structured element list:
   ```
   [N] role|x,y|label|text
   ```
3. Matches teacher quality on a held-out Pace screenshot eval set
4. Runs at ~1-2 GB RAM (~3× less than Qwen3-VL-8B)
5. Drops in via `tinygpt serve --vlm <path>` for Pace's `LocalVLMClient`

## Why P0

Pace's "watch" capability depends on this VLM. Without it, Pace can't
see the screen. The 30B planner specialist (Pace's other model) is
already in our distillation pipeline; the VLM is the parallel arc.

Combined wins when both ship:
- Planner: 0.6B on ANE — ~500 MB, ~600 tok/s, ~3W
- VLM: 1-2B distilled — ~1-2 GB Q4, fewer-second screen read
- **Total Pace footprint: ~2-3 GB** (vs current ~24 GB) — fits comfortably on any modern Mac

## Architecture

LLaVA-style:
```
Image (1024×1024 max)
  ↓ ViT encoder (e.g., CLIP-ViT-L-14 or Qwen3-VL's native vision tower)
  ↓ pooled image features [N_patches, hidden_dim]
  ↓ projection MLP [hidden_dim → llm_dim]
  ↓ prepended as image tokens to text input
LLM body (Qwen3-1B or similar text-only base)
  ↓ generates text response
```

Teacher: Qwen3-VL-8B (Sarthak has it in LM Studio).
Student: Qwen3-VL-2B (architecturally compatible vision+LLM) OR
custom (CLIP encoder + Qwen3-1B body via cross-modal projection
training).

## Scope — in (milestones, ELF reports at each)

### Milestone 1 — Vision encoder primitive
- Add `VisionEncoder.swift` to TinyGPTModel
- ViT forward pass (CLIP-ViT-L-14 reference; load weights from HF)
- Image preprocessing: resize, normalize, patchify (16×16 patches)
- Smoke: load CLIP-ViT-L-14 weights from HF, run forward on a 224×224 sample image, verify output shape

### Milestone 2 — Cross-modal projection
- Add `CrossModalProjection.swift` — a small MLP (768 → 2048 → llm_dim)
- For LLaVA-style: project pooled ViT features into LLM token embedding space
- Initialize randomly; will train via M5 (below)

### Milestone 3 — VLM forward pass

**LEVERAGE-FIRST gate before this milestone** (per `[[feedback_leverage_first]]`):
re-evaluate base-model choice. Three open-weights candidates to score on
(param count, screen-task fit, Mac-load cost, license):

1. **Qwen3-VL-2B** — current PRD pick. Generic VLM, no UI specialization.
2. **CogAgent (Zhipu, open weights)** — already trained for screen
   interaction. Bigger (~18B) but distillable.
3. **UI-TARS-1.5B (ByteDance, open weights, MIT)** — explicitly trained
   for desktop UI agents. Closest fit; smallest size.

Likely winner: **UI-TARS-1.5B as base**, fine-tune our cross-modal
projection + LoRA on the LLM body. Saves weeks of pretrain because the
"screens look like this" learning is already done.

- New `TinyGPTModelVLM.swift` wrapping vision-encoder + projection + LLM body
- Forward signature: `model(image: MLXArray, tokens: MLXArray) → logits`
- Image tokens prepended before text tokens (LLaVA convention)
- Smoke: instantiate with chosen base, forward returns logits

### Milestone 4 — VLM HF loader
- Add `HFVLMLoader.swift` parallel to `HFModelLoader.swift`
- Detect VLM architectures in config.json (`Qwen3VLForCausalLM`, `LlavaForConditionalGeneration`)
- Load vision tower + projection + LLM weights from sharded safetensors
- Smoke: load Qwen3-VL-2B weights from HF dir; produce TinyGPTModelVLM that loads cleanly

### Milestone 5 — VLM training pipeline
- Extend `Train.swift` or new `TrainVLM.swift` to accept image+text training pairs
- Loss: standard next-token CE on text output only (image is input, not output)
- LoRA support: wrap LLM body's Linears with LoRA (vision tower can be frozen)
- Smoke: train 10 steps on a tiny image+caption dataset, verify loss decreases

### Milestone 6 — VLM screen training data (LEVERAGE-FIRST)

**Principle**: warmstart on public datasets, fine-tune on tiny
high-quality Mac-specific set. Do NOT teacher-synthesize when public
pretrained data already exists for the same task. See
`[[feedback_leverage_first]]`.

**Three-stage data plan**:

**Stage A — warmstart on public UI-grounding data** (primary):
- **OS-Atlas** (~13M GUI grounding pairs, Apache 2.0) — biggest public
  desktop UI dataset. `huggingface.co/datasets/OS-Copilot/OS-Atlas-data`
- **SeeClick pretrain** (~1M screen→click pairs, Apache 2.0)
- **Rico** (~72K Android UI screens, auxiliary mobile-UI patterns)
- Combined: ~14M+ image+label+bbox pairs. Use for cross-modal projection
  adaptation. If we picked UI-TARS-1.5B at M3 we skip this — already done.

**Stage B — Mac-specific fine-tune via macOS Accessibility API**:
- New `scripts/vlm-ax-capture.py`:
  - Captures screenshot + walks AX tree on the same Mac
  - For each interactive AX element: emit `(screenshot, label, role, bbox)`
    training row. Deterministic ground truth, no teacher errors.
  - Sarthak runs in the background during normal Mac use → ~500-2000
    pairs across his actual app set in a few days
- Output: `~/.cache/tinygpt/datasets/vlm-ax-mac.jsonl`

**Stage C — teacher fill-in** (last resort, AX-blind cases only):
- Apps where AX is blind: Electron-rendered, canvas-rendered (Figma,
  some web), image-content-inside-elements
- ONLY for these, hit Qwen3-VL-8B (or UI-Venus-8B) teacher
- Target: <100 pairs — teacher noise is expensive; minimize

**Held-out eval**: ScreenSpot + ScreenSpot-Pro (~4K test pairs) — public,
objective. Don't train on these.

### Milestone 7 — VLM specialist SFT
- Train Qwen3-VL-2B (or smaller) on the screen-reading data
- Bake LoRA into base before deployment (similar to ANE arc)
- Output: `vlm-pace-v1.tinygpt` (or .safetensors)

### Milestone 8 — VLM serve integration
- Add `--vlm <path>` flag to `tinygpt serve`
- Multipart request body: image part + text part (mirrors OpenAI's image-in-message format)
- Output: standard OpenAI chat completion shape
- Pace's `LocalVLMClient` already expects OpenAI-compat shape

### Milestone 9 — Vision eval
- New `pace-eval-vlm.py`:
  - Held-out screenshot test set (~20 screens spanning different apps)
  - For each: feed screenshot → student VLM → parse element list → compare to teacher's reference list (or hand-curated gold)
  - Metric: element-detection precision + recall, label match accuracy
- Target: ≥90% of teacher's element-detection rate

### Milestone 10 — Pace integration handoff
- Document the Pace Info.plist key change to point at our serve VLM endpoint
- Smoke: Pace's `LocalVLMClient` makes an actual call → gets back a valid element list

## Scope — out (deferred)

- Multi-image inputs (single-screenshot only for v1)
- Video understanding (animation, scrolling) — falls out free in a later VLM
- Cross-language (English screens only for v1)
- Real-time streaming vision (capture-every-frame; v1 is on-demand)
- Multiple primary-focus screens (Pace can handle multi-screen; v1 specialist handles one at a time)

## Files involved

**New files (parallel-safe with ANE elf — different surface)**:
- `native-mac/Sources/TinyGPTModel/VisionEncoder.swift` (new)
- `native-mac/Sources/TinyGPTModel/CrossModalProjection.swift` (new)
- `native-mac/Sources/TinyGPTModel/TinyGPTModelVLM.swift` (new)
- `native-mac/Sources/TinyGPTModel/HFVLMLoader.swift` (new)
- `native-mac/Sources/TinyGPTModel/ImagePreprocess.swift` (new)
- `native-mac/Sources/TinyGPT/TrainVLM.swift` (new, optional — can extend Train.swift)
- `scripts/vlm-pace-prep.py` (new)
- `scripts/pace-eval-vlm.py` (new)

**Modified (light touch, please coordinate)**:
- `native-mac/Sources/TinyGPTModel/ModelConfig.swift` — add `visionConfig: VisionConfig?` field
- `native-mac/Sources/TinyGPTModel/HFModel.swift` — dispatch VLM vs text-only at load
- `native-mac/Sources/TinyGPTServe/Serve.swift` — `--vlm` flag (small)
- `native-mac/Sources/TinyGPT/TinyGPT.swift` — register new subcommands

**Don't touch**:
- ANE elf's work area (ToCoreML.swift, ANEInference.swift, BakeLora.swift, Merge.swift)
- Pace planner LoRA arc (pace-planner-v* directories)
- App UI (separate work)

## Resource discipline

- Multiple elves running. Don't fire heavy GPU jobs without checking.
- v6 SFT may still be running when you start — verify before any training smoke.
- VLM training will eventually need many epochs; PAUSE before kicking off >30 min runs.
- One large model loaded at a time. If validating with Qwen3-VL-8B teacher, kill any conflicting serve.

## Acceptance — full ship

1. `tinygpt serve <hf-vlm-dir> --vlm-mode auto --port 8765` boots and accepts image inputs
2. End-to-end smoke: send a Pace screenshot via curl → returns a valid `[N] role|x,y|label|text` element list
3. Eval ≥90% of teacher's element-detection rate on held-out screens
4. RAM: ≤2 GB peak (vs Qwen3-VL-8B's ~6 GB Q4)
5. Latency: ≤2× current Qwen3-VL-8B time on the same screen (target: <2s per screen read)
6. Pace daily-drives against our endpoint without rollback for 1 day

## Estimated effort

**3-4 weeks** for the elf, broken into milestones:
- Week 1: Vision encoder + projection + forward pass (M1-M3)
- Week 2: HF VLM loader + training pipeline (M4-M5)
- Week 3: Data prep + SFT training (M6-M7)
- Week 4: Serve integration + eval + Pace handoff (M8-M10)

## Why this is the right architecture (not Option 1)

Option 1 (Apple Vision + text head) was the fast path. Owner chose
Option 3 (full VLM) explicitly. Reasons that matter long-term:
- True visual understanding beyond OCR (icons, layouts, colors, spatial relationships)
- Generalizes to non-text UI (sliders, charts, images, hand-drawn input)
- Becomes the foundation for any "Pace can see" feature beyond text elements
- Avoids the Apple Vision API ceiling

## Milestone 1 ship note (2026-06-07)

**Status**: PASS. Vision encoder primitive lands; load + forward
verified against `openai/clip-vit-large-patch14`.

**What landed**:
- `VisionEncoder.swift`: `VisionConfig`, `CLIPAttention`, `CLIPMlp`,
  `CLIPEncoderLayer`, `CLIPEncoder`, `CLIPEmbeddings`,
  `CLIPVisionModel`, `CLIPVisionConfigParser`, `CLIPVisionLoader`.
  CLIP-style ViT with HF-native @ModuleInfo keys (including the
  intentional `pre_layrnorm` typo). Loader walks safetensors,
  permutes Conv2d patch-embed from PyTorch OIHW → MLX OHWI, accepts
  vision tensors under `vision_model.` / `vision_tower.` / `visual.`
  prefixes for forward-compat with non-CLIP repos. F32/F16/BF16
  dtypes handled.
- `ImagePreprocess.swift`: CGImage-backed resize (shortest-edge),
  center crop, RGB normalisation. Reads `preprocessor_config.json`
  if present (handles both old `size: int` and new
  `size: {shortest_edge}` shapes). Synthetic gradient helper for
  smoke tests with no image.
- `VLMSmoke.swift`: `tinygpt vlm-smoke <hf-dir> [image.png]`
  subcommand. Loads, forwards, asserts shape/finite/std. Registered
  in `TinyGPT.swift` (single new case).

**Files changed**:
- NEW `native-mac/Sources/TinyGPTModel/VisionEncoder.swift`
- NEW `native-mac/Sources/TinyGPTModel/ImagePreprocess.swift`
- NEW `native-mac/Sources/TinyGPT/VLMSmoke.swift`
- MODIFIED `native-mac/Sources/TinyGPT/TinyGPT.swift` (one new case)

**Smoke result**:
```
tinygpt vlm-smoke <openai/clip-vit-large-patch14 snapshot dir>
  → loads 24-layer ViT-L
  → forwards synthetic 224×224 → features [1, 257, 1024], finite
  → forwards real screenshot → features [1, 257, 1024], finite
  → PASS
```

**Parity check against HF PyTorch `CLIPVisionModel` (fp32)**:
```
mean diff:        0.00183     (target: dominated by fp32 noise)
median diff:      0.00113
95th-pctile diff: 0.00522
99th-pctile diff: 0.01234
max diff:         1.16  (single outlier on a value of magnitude ~30)
cosine sim:       0.999995
```
First parity attempt FAILED (cosine 0.78). Root cause: my forward was
applying `post_layernorm` to the FULL sequence; HF applies it only to
the CLS-pooled output. Fixed by changing `callAsFunction` to return
`pre_layrnorm + encoder` (no post-LN) and adding a separate
`pooled(_:)` method for the CLS path. Verified parity after fix.
Outlier diffs concentrate at high-magnitude positions (relative error
<5%) — consistent with MLXFast SDPA vs PyTorch SDPA fp32 accumulation
across 24 layers. Not a logic bug. The parity script lives at
`scripts/vlm/clip_parity.py` for re-runs.

## Milestone 2 ship note (2026-06-07)

**Status**: PASS. Cross-modal projection MLP primitive lands. Random
init forwards cleanly; weights are trained from scratch in M5 (no
load path here — that's the M4 VLM loader's job when reusing a
LLaVA-1.5 / Qwen3-VL checkpoint's pretrained projector).

**What landed**:
- `CrossModalProjection.swift`: `CrossModalProjectionConfig` +
  `CrossModalProjection` (LLaVA-1.5-style 2-layer MLP, exact GELU
  default, configurable visionHidden→llmHidden→llmHidden). `@ModuleInfo`
  keys (`linear_1`, `linear_2`) match LLaVA-1.5 safetensors naming so
  the HF VLM loader can splice in pretrained projector weights at M4
  without translation.
- Extended `tinygpt vlm-smoke` to chain the projection after the
  encoder forward: drops CLS, runs projection, asserts shape +
  finite + non-degenerate.

**Files changed**:
- NEW `native-mac/Sources/TinyGPTModel/CrossModalProjection.swift`
- MODIFIED `native-mac/Sources/TinyGPT/VLMSmoke.swift` (M2 stage in
  the smoke pipeline)

**Smoke result**:
```
CLIP vision → projection (vision_hidden=1024 → llm_hidden=2048)
  input  : [1, 256, 1024]  (patches only, CLS dropped)
  output : [1, 256, 2048]
  mean=-0.003, std=0.26, range [-4.66, 5.15], all finite
  PASS
```

**Known limitations / non-issues**:
- llmHidden=2048 in the smoke is a placeholder for Qwen3-1.7B (the
  current M4 candidate body). Once M4 picks a definite LLM, the
  smoke's hardcoded 2048 becomes parameterised. No structural impact:
  the projection is dim-agnostic by design.
- No normalisation in the projector — matches LLaVA-1.5. Newer
  LLaVA/Qwen-VL variants add a LN/RMSNorm before fc1; we can add a
  feature flag later if M4's chosen base needs it (Qwen3-VL ships
  its own projector style, see M4).
- Random init at M2 means output std is the rms-fan-in scale of
  MLX-Swift's default Linear init (~0.26 here). Real values will
  shift after M5 SFT. The std being non-zero is what we check.

**Known limitations / non-issues**:
- Synthetic-pixel std (~0.95) and real-image std (~0.96) match closely
  → features are not collapsed and the loader maps weights correctly.
  This is structural verification, not a numerics gate vs PyTorch — that
  parity check will land alongside M3 once we can pipe vision features
  into an LLM and compare logits.
- M1 only supports the CLIP-shaped vision tower (Conv2d patchify, 1
  CLS token, fixed `numPositions`). Qwen3-VL's native tower uses 2D RoPE
  + dynamic resolution and will likely need a sibling
  `Qwen3VLVisionEncoder` class at M4 — keeping `VisionConfig` as a
  separate struct (not extending HuggingFaceConfig) so we can grow the
  config surface without touching the text-LLM loader.
- The intentional HF typo `pre_layrnorm` (missing 'e') is preserved in
  our @ModuleInfo key so safetensors load cleanly. Verified against
  CLIP-ViT-L-14's actual checkpoint tensor names.
- LM Studio currently exposes `ui-venus-1.5-{2b,8b}` (not `qwen3-vl-8b`
  as the PRD prose claims). UI-Venus IS a Qwen3-VL-derived
  screen-reading model — treating it as the M6 teacher is consistent
  with the PRD's intent. Will surface this in M6 ship note.

## Milestone 3 ship note (2026-06-07)

**Status**: PASS. Full VLM forward composes end-to-end. Shape test
only (random LLM init, CLIP encoder, front-prepend) per advisor
guidance — real numerics validation happens at M4 against actual
checkpoint weights.

**What landed**:
- `TinyGPTModelVLM.swift`: `TinyGPTModelVLM.Config` (vision + llm +
  projectionAct) and the `TinyGPTModelVLM` Module composing
  `CLIPVisionModel + CrossModalProjection + TinyGPTModelHF`. Forward
  signature `(image NHWC, tokens [B, T]) → logits [B, N_patch+T, vocab]`.
  @ModuleInfo keys (`vision_tower`, `multi_modal_projector`,
  `language_model`) match LLaVA-1.5 safetensors naming so the M4
  loader can splice weights without translation. Preconditions catch
  misuse (no YOCO / MoE / MTP at this layer yet — explicitly
  out-of-scope until the simple path stabilises).
- VLM smoke stage runs after M1+M2 stages; constructs a tiny
  (2-layer, hidden=128) random LLM, forwards image+8 text tokens,
  asserts output shape `[1, 264, 1000]` and finite values.

**Files changed**:
- NEW `native-mac/Sources/TinyGPTModel/TinyGPTModelVLM.swift`
- MODIFIED `native-mac/Sources/TinyGPT/VLMSmoke.swift` (M3 stage in
  the smoke pipeline)

**Smoke result**:
```
vlm output shape: [1, 264, 1000]   (256 vision tokens + 8 text → logits)
logits mean=0.01, std=1.00, range [-3.68, 4.37], all finite
PASS
```

**M4 architectural reckoning (per the leverage-first gate the
maintainer added between M2 and M3)**:

Investigated the candidates against actual `config.json` evidence
(NOT recall):

| Candidate | Arch family | Notes |
|---|---|---|
| `inclusionAI/UI-Venus-1.5-2B` | `Qwen3VLForConditionalGeneration` | UI-Venus is the LM Studio teacher Sarthak is already running. Hidden=2048, 28L. Vision: 24L ViT, hidden=1024, patch=16, mRoPE, spatial_merge=2, `deepstack_visual_indexes=[5,11,17]` (features re-injected at multiple LLM depths). |
| `ByteDance-Seed/UI-TARS-2B-SFT` | `Qwen2VLForConditionalGeneration` | 1.5B params. Hidden=1536, 28L. Vision: Qwen2VL tower, patch=14, mRoPE. Tied embeddings. MIT license. |
| `ByteDance-Seed/UI-TARS-1.5-7B` | `Qwen2_5_VLForConditionalGeneration` | Too big (~5 GB Q4) — defeats the size budget. Skip. |
| `UI-TARS-1.5B` (PRD candidate) | n/a | Does not exist on HF — likely a typo for UI-TARS-1.5-7B (version 1.5, 7B) or UI-TARS-2B-SFT. |
| CogAgent | THUDM custom | 18B base, skip on size. |

**Verdict for M4**: Pick **UI-Venus-1.5-2B** as the student. Reasons:
1. Already screen-trained by inclusionAI (LM Studio confirms this is
   the production VLM Sarthak is running). Distilling from itself =
   max-leverage; the M6 teacher and M7 student share architecture,
   simplifying label parity and tokeniser sharing.
2. Qwen3-VL is the most-recent generation (better long-context, better
   tied-embedding savings vs Qwen2-VL).
3. UI-Venus-1.5-2B's vision tower hidden_size=1024 happens to match
   CLIP-ViT-L-14's hidden — our M1 encoder is the right shape for
   weight-copy experiments down the road if we want to swap towers.

**M4 design challenge (surface honestly now)**: Qwen3-VL is
ARCHITECTURALLY DIFFERENT from LLaVA-1.5 in three substantive ways
the PRD's "LLaVA-style" framing glosses over:

1. **mRoPE** — the LLM body uses multimodal RoPE with `mrope_section
   = [24, 20, 20]` (T, H, W splits across `head_dim=128`). Our
   existing `CausalSelfAttention` uses 1D RoPE. M4 will either need a
   new attention path or ships with a "force-1D-RoPE" mode that
   PARTIALLY loads the model — vision-stream position info will be
   wrong, hurting accuracy. Honest acknowledgement: this is the
   single hardest engineering item.

2. **Image-token replacement at embed stage** — the model expects
   text containing `image_token_id=151655` at the image's position,
   then vision tokens REPLACE that token's embedding. Front-prepend
   (M3) is structurally wrong for the trained model. The fix is
   straightforward at the forward call (find image-token positions,
   scatter vision tokens). Not yet implemented because random LLM
   doesn't care; M4+ requires correct token-id splicing.

3. **`deepstack_visual_indexes=[5,11,17]`** — UI-Venus-1.5-2B
   re-injects visual features at LLM layers 5, 11, 17 (not just at
   the front). This is part of why the architecture is accurate at
   fine-detail screen reading. Faithful M4 loading needs this;
   skipping it degrades quality. Pragmatic option: skip deepstack at
   M7 SFT (model is being fine-tuned anyway; missing skip-connections
   can be partially compensated by LoRA), document the gap.

These three items don't block M4 — they shape its scope. M4 will land
an honest "Qwen3-VL safetensors loader that handles the
embedding-replacement + mRoPE + deepstack contract correctly", and
the PRD's original "LLaVA convention" framing was approximate; reality
is "Qwen3-VL specifically, with three architectural quirks".

**Known limitations / non-issues (M3)**:
- Front-prepend is M3 only — M4 will switch to `<image>`-token
  embedding replacement, which is what Pace's `LocalVLMClient`
  semantically expects when posting OpenAI's `image_url` content.
- LLM body is random init in this smoke; real perplexity numbers land
  at M4 against UI-Venus-1.5-2B weights.
- Smoke uses CLIP encoder (M1) + random LLM. UI-Venus-1.5-2B's native
  Qwen3-VL vision tower is a different architecture; M4 will add
  `Qwen3VLVisionEncoder.swift` as a sibling and pick at load time
  based on `config.json`'s `model_type`.

**Cross-elf coordination note (2026-06-07)**: at the end of this
session, a clean `swift build -c release` is currently broken by a
pre-existing async/await error in `ANEInference.swift:339` (the ANE
elf's in-flight territory, called out as "Don't touch" by the PRD).
The VLM specialist binary at
`.build/arm64-apple-macosx/release/tinygpt` was built BEFORE that
file was edited and still includes the VLM smoke subcommand working
correctly — all three M1/M2/M3 smokes still pass against it. When
the ANE elf clears their error, an incremental rebuild will produce
a new binary with the same VLM code. None of the VLM files touch
ANEInference.swift; the dependency is purely module-level (they
live in the same TinyGPTModel target).
