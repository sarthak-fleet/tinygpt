# ANE research dossier for TinyGPT

Status: v1 research synthesis landed 2026-06-07  
Supports: `docs/prds/factory-ane-inference-pace.md`

This dossier is implementation guidance, not a benchmark result. Anything
marked repo-local comes from TinyGPT PRDs / smoke notes and should be verified
again before a long conversion or benchmark loop.

## Source map

- Apple ML Research, "Deploying Transformers on the Apple Neural Engine":
  <https://machinelearning.apple.com/research/neural-engine-transformers>
- Apple `ml-ane-transformers` repo:
  <https://github.com/apple/ml-ane-transformers>
- Apple ML Research, "Deploying Attention-Based Vision Transformers to Apple
  Neural Engine":
  <https://machinelearning.apple.com/research/vision-transformers>
- Core ML framework docs:
  <https://developer.apple.com/documentation/coreml>
- `MLState` docs:
  <https://developer.apple.com/documentation/coreml/mlstate>
- Core ML Tools stateful models guide:
  <https://apple.github.io/coremltools/docs-guides/source/stateful-models.html>
- Core ML Tools conversion formats:
  <https://apple.github.io/coremltools/docs-guides/source/target-conversion-formats.html>
- Core ML Tools ML Program guide:
  <https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html>
- Core ML Tools quantization docs:
  <https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html>,
  <https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html>
- Core ML Tools palettization docs:
  <https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html>,
  <https://apple.github.io/coremltools/docs-guides/source/opt-palettization-algos.html>
- Core ML Tools releases:
  <https://github.com/apple/coremltools/releases>
- Apple `ml-stable-diffusion` repo:
  <https://github.com/apple/ml-stable-diffusion>
- Apple FastVLM research page:
  <https://machinelearning.apple.com/research/fastvlm-efficient-vision-encoding>
- Apple MobileCLIP paper:
  <https://arxiv.org/abs/2311.17049>
- Apple M5 Pro / M5 Max newsroom:
  <https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/>
- Orion paper page:
  <https://arxiv.org/abs/2603.06728>

## 1. ANE compute model

The Apple Neural Engine is Apple's low-power neural accelerator. Apple's 2022
transformer writeup says the first ANE shipped in A11 / iPhone X and that the
A15 16-core ANE reached 15.8 FP16 TFLOPS, 26x the original A11 peak. The ANE
expanded from iPhone to iPad and then to Mac with M1.

Core ML is the public dispatch layer. It can place work on CPU, GPU, and ANE,
and it may choose a hybrid execution plan. That is useful for app developers,
but difficult for TinyGPT because we need to know whether decode actually uses
ANE and which graph pattern blocked compilation.

Apple's public ANE guidance is not "convert arbitrary PyTorch and hope." The
guidance is to write the model in an ANE-friendly layout before conversion.
The 2022 transformer post explicitly trades generic implementation flexibility
for a principled ANE implementation to reduce transfers and improve throughput.

Important implications:

- ANE likes 4D channels-first tensors.
- Transformer shapes should be arranged as `(B, C, 1, S)` where `B` is batch,
  `C` is hidden/channel width, and `S` is sequence.
- Linear layers should often become `Conv2d`.
- Avoid unnecessary reshape / transpose churn between attention blocks.
- Prefer FP16-first conversion and only add INT8/INT4 experiments after FP16
  compiles.
- A graph that "runs in Core ML" is not the same thing as a graph that compiles
  to ANE.

M5 Pro / M5 Max note: Apple's March 2026 newsroom describes a faster
power-efficient Neural Engine, higher unified-memory bandwidth, and GPU cores
with Neural Accelerators. Treat those GPU Neural Accelerators as separate from
the ANE until tools prove otherwise. For TinyGPT, the practical M5 win is
higher memory bandwidth plus more AI-capable GPU fallback; the ANE path still
needs an ANE-friendly Core ML graph.

## 2. Apple's published transformer patterns

### Layout

The key layout from `ml-ane-transformers` is `(B, C, 1, S)`. Standard training
code usually stores activations as `(B, S, C)`. ANE-friendly code moves
channels into the second dimension and treats sequence length as the final
spatial dimension.

Why it matters:

- Conv kernels can emulate dense projection efficiently.
- Channels-first avoids repeated transposes around projection / residual paths.
- Sequence length remains explicit and easier to slice for cache-like patterns.

For Qwen3, the M7 port should start by making a single decoder block use this
layout end-to-end. Do not start with the whole 28-layer model.

### Conv2d as Linear

Apple's reference rewrites linear projections as convolutional layers. For
language decoders this maps naturally to:

- token embedding projection paths
- Q, K, V projections
- output projection
- MLP up / gate / down projections

For Qwen3, the important test is whether grouped-query attention projections
can stay in this layout without introducing ANE-hostile shape ops.

### Normalization

Apple ships ANE-specific layer norm patterns such as `LayerNormANE`. Qwen3 uses
QK-Norm / RMS-style normalization rather than a vanilla BERT stack. The port
should not blindly copy LayerNorm. Instead:

1. Write a tiny Qwen RMSNorm / QK-Norm module.
2. Convert that module alone.
3. Inspect whether Core ML places it on ANE.
4. Replace unsupported reductions or sqrt/divide patterns before scaling.

### Activations

The Apple repo includes ANE-friendly GELU / SiLU approximations. Qwen3 uses
SwiGLU. M7 should port the MLP with the ANE SiLU approximation before chasing
the attention path, because MLPs are large and relatively static.

### File-level copy targets

Read the Apple repo in this order:

1. Package-level docs and examples to confirm install / conversion flow.
2. Transformer layer modules for layout and Conv2d projection patterns.
3. Activation / normalization helper modules.
4. Hugging Face DistilBERT example for conversion plumbing.
5. Any LLaMA or decoder-style example branches if present in the current repo.

Copy patterns, not class names. TinyGPT's Qwen path should remain shaped by
Qwen3 architecture and the local Swift/CoreML integration.

## 3. Core ML Tools capabilities and limits

### Model format

Core ML Tools 7+ produces `mlprogram` by default for iOS15 / macOS12 or newer.
Apple says new feature work targets ML Program; the older neural-network
format is maintenance mode. Stateful models also require ML Program.

For TinyGPT, default all serious ANE work to `.mlpackage` / ML Program.

### Stateful models

Core ML stateful prediction is available starting iOS18 / macOS15 for
`mlprogram`. Conversion uses `ct.StateType` in `ct.convert(...)`. Runtime uses
`MLState`; the client creates a state object and passes it into stateful
prediction calls. Apple's docs warn that predictions using the same state must
be serialized.

TinyGPT implication:

- One decode stream = one `MLState`.
- Do not share one state across concurrent requests.
- Serialize token steps per request.
- If serve concurrency is added, allocate state per request/session.

### Quantization

Core ML Tools supports:

- data-free linear weight quantization
- calibration-based activation quantization
- GPTQ-style weight quantization
- palettization / LUT compression

The docs distinguish Core ML model compression APIs from PyTorch compression
APIs. For ANE, the generated Core ML graph matters more than the training-time
quantizer name. Per-block / int4 combinations may reduce size but still fail
ANE placement.

Recommended order:

1. FP16 compile and ANE placement.
2. W8 weight quantization.
3. W8A8 with calibration.
4. INT4 / palettization only after a working FP16/W8 baseline.

### coremltools 9 caveat

The PRD says macOS 26 + coremltools 9. GitHub releases show 9.0 wheels, but
some Apple-hosted API pages still render as 8.1 documentation. Before locking a
script to a 9-only API, verify with:

```bash
python - <<'PY'
import coremltools as ct
print(ct.__version__)
PY
```

Do not run installs from an agent session without owner approval.

### Error handling

Apple does not publish a complete ANECCompile error-code manual. Treat code -14
as a symptom, not a diagnosis. Repo-local notes say TinyGPT saw -14 around
state slots / graph size. The diagnostic plan should bisect shape, state,
operator, and layer-count causes rather than assuming one root cause.

## 4. Qwen3-specific concerns

Repo-local target: Qwen3-0.6B-like decoder with:

- GQA, approximately 16 query heads / 8 KV heads
- head dimension 128
- QK-Norm
- SwiGLU MLP
- RoPE with large base
- tied embeddings
- 28 decoder layers

ANE friction points:

- GQA changes Q/K/V tensor shapes and repeats/broadcasts K/V across query
  groups.
- RoPE adds sin/cos elementwise work and shape handling that may prevent ANE
  placement if implemented dynamically.
- KV cache state can explode into many state buffers if every layer and tensor
  becomes a separate state slot.
- QK-Norm adds reductions in the attention path.
- Full 28-layer graphs may exceed compiler or planner limits even if one layer
  compiles.
- Decode is inherently step-by-step, so dispatch overhead matters.

Porting principle:

Do not begin with full Qwen. Begin with a single block and only two sequence
modes:

- prefill with fixed short sequence
- decode step with fixed one-token input and state

Once those compile, scale layer count geometrically: 1, 2, 4, 8, 14, 28.

## 5. Current blocker analysis

Repo-local facts to verify:

- A Core ML pipeline exists and can smoke on CPU/GPU.
- ANE runtime remains blocked.
- Earlier failure notes mention ANECCompile -14 and/or graph/state limits.
- State consolidation was attempted in a prior milestone.

Hypothesis tree:

1. Too many state slots. Test by compiling a stateless one-layer graph, then a
   stateful one-layer graph with consolidated KV, then N layers.
2. Unsupported op. Test each submodule alone: RMSNorm, RoPE, attention scores,
   softmax, MLP, residual.
3. Layout mismatch. Test standard `(B,S,C)` against ANE `(B,C,1,S)` for the
   same tiny block.
4. Graph size / layer count. Compile repeated identical blocks with fake
   weights and fixed shapes.
5. Dynamic shape issue. Freeze all shapes and compare to dynamic-shape export.
6. Quantization issue. Disable compression until FP16 compiles.

What has not been proven yet:

- Full Qwen3 can compile to ANE as one graph.
- Stateful KV cache is accepted by ANE for all 28 layers.
- Core ML can keep the entire decode step on ANE without CPU/GPU fallbacks.
- Quantization improves ANE throughput for this graph instead of only reducing
  package size.

## 6. Beyond-reference research arcs

### Hybrid ANE/GPU dispatch

Core ML can choose CPU/GPU/ANE, but public APIs do not expose a clean "put these
layers on ANE, those layers on GPU" planner. The practical hybrid route is
manual partitioning into multiple model packages:

- ANE MLP chunks
- GPU attention chunks
- CPU sampling / tokenizer

This increases host orchestration overhead. It is only worth testing after a
single block shows stable ANE wins.

### Layer-chunked conversion

Apple's Stable Diffusion repo is the precedent: text encoder, UNet, VAE, and
variants are converted as separate modules. For LLMs, chunking could be:

- embedding + first N layers
- middle chunks of N layers
- final norm + lm head

Chunking helps compiler limits and lets Core ML load smaller graphs. It hurts
latency if state copies cross model boundaries. Measure only with single-shot
decode tests first.

### ANE-native INT4 / INT8

Core ML Tools supports int8/uint8 and int4/uint4 weight compression APIs, plus
activation quantization. That does not guarantee native ANE INT4 execution for
every transformer op. Treat INT4 as a package-size and memory experiment until
compute-plan inspection proves ANE placement.

### MPSGraph bridge

Use MPSGraph when:

- Core ML compilation is blocked by an op that Metal can run well.
- You need dynamic control around an otherwise static Core ML graph.
- You want GPU fallback without fighting Core ML's planner.

Do not use MPSGraph as a substitute ANE path. It is a GPU route.

### Sparse / local attention

Apple's vision-transformer post emphasizes local attention for high-resolution
vision because global attention scales quadratically. That is directly useful
for VLM vision encoders and less directly useful for autoregressive text decode.
For Qwen text, local attention changes model behavior and is not a drop-in
optimization unless we train or distill for it.

### FastVLM and MobileCLIP

FastVLM's lesson is to reduce vision tokens before the LLM, not to make the
LLM itself magical. It uses an efficient hybrid vision encoder to reduce TTFT
and token count. For TinyGPT VLM work, this argues for a smaller ANE-friendly
vision encoder plus fewer visual tokens before spending time on large language
decoder ANE execution.

MobileCLIP's lesson is distillation/data efficiency for mobile image-text
models. It is a better pattern for the vision specialist than for Qwen text
decode.

### Direct ANE programming

Orion and similar 2026 research explores bypassing Core ML. This is promising
but outside the current repo scope. It uses private or reverse-engineered paths
and should not be a product dependency. Keep it as background reading only.

### Halved-depth distillation

If full 28-layer Qwen3 remains blocked, a 14-layer distilled specialist may be
more tractable than forcing the full model onto ANE. This fits TinyGPT's
specialist-model philosophy: narrow domain, smaller model, measurable eval.

## 7. Recommended next experiments

Run these in order, with owner approval before any long loops.

### M6 diagnostic sequence

1. Compile one FP16 Qwen MLP block in ANE layout.
2. Compile one FP16 RMSNorm / QK-Norm module.
3. Compile RoPE alone with fixed shapes and precomputed sin/cos.
4. Compile attention score + softmax alone with fixed short sequence.
5. Compile one full decoder block stateless.
6. Compile one full decoder block stateful with consolidated KV.
7. Scale layer count: 1, 2, 4, 8, 14, 28.
8. Only after FP16 works, test W8.

### M7 port specifics

Closest Apple reference: `ml-ane-transformers`, not Stable Diffusion. Copy:

- `(B,C,1,S)` layout
- Conv2d projections
- ANE-friendly activation approximations
- normalization implementation style

Do not copy:

- BERT assumptions
- encoder-only flow
- full-sequence-only evaluation

### M8 arcs ranked

1. Layer-chunked conversion: most practical, matches Apple Stable Diffusion
   modular packaging precedent.
2. Halved-depth distillation: high payoff if full graph remains blocked.
3. W8 activation/weight quantization: useful after FP16 placement works.
4. Hybrid ANE/GPU: plausible but orchestration-heavy.
5. FastVLM-style vision-token reduction: important for VLM, separate from text
   decoder ANE.
6. Direct ANE programming / Orion-style path: research-only, not product path.

## Maintainer checklist

- Keep ANE claims tied to compute-plan or runtime evidence.
- Never infer ANE execution from Core ML success alone.
- Keep every diagnostic model tiny and fixed-shape.
- Stop after one compile/run when testing a new hypothesis from an agent
  session.
- Record package size, compile result, runtime compute unit, first-token
  latency, decode-token latency, and memory.
- Prefer a smaller specialist that really uses ANE over a full model that
  silently falls back to GPU.
