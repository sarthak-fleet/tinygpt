# Architecture — Mac native-mac/

A top-down tour of every module + how they fit together. Written so
someone reading this for the first time (you, in a month, or anyone
else) can understand what's where and why, and reproduce the design
from scratch if needed.

## 1. The package layout

```
native-mac/
├── Package.swift                — SwiftPM manifest, deps, targets
├── Sources/
│   ├── TinyGPTIO/               — Pure Foundation, no MLX. File formats.
│   │   ├── Manifest.swift       — JSON header schema for .tinygpt
│   │   ├── TinyGPTFile.swift    — Binary reader/writer (fp32 + fp16)
│   │   ├── SafetensorsReader.swift — HuggingFace's weight format
│   │   └── HuggingFaceConfig.swift — HF config.json parser
│   ├── TinyGPTModel/            — MLX-Swift; model + training
│   │   ├── ModelConfig.swift    — One struct that describes any model
│   │   ├── TransformerBlock.swift — Attention + MLP + Block
│   │   ├── Norms.swift          — LayerNorm vs RMSNorm
│   │   ├── TinyGPTModel.swift   — The top-level Transformer
│   │   ├── WeightLoader.swift   — Load .tinygpt → model (with WASM transpose)
│   │   ├── HFWeightMapping.swift — HF param-name translation table
│   │   ├── Tokenizer.swift      — Byte-level + HF BPE wrapper
│   │   ├── KVCache.swift        — O(T²) → O(T) attention sampling
│   │   ├── Lora.swift           — Low-rank adapter Linear subclass
│   │   ├── LoraIO.swift         — .lora adapter file format
│   │   ├── LoraComposition.swift — Multi-adapter stacking
│   │   ├── Trainer.swift        — AdamW + compiled train step
│   │   └── ANEInference.swift   — Core ML / ANE inference path
│   ├── TinyGPT/                 — CLI executable
│   │   ├── TinyGPT.swift        — Entry point + subcommand dispatch
│   │   ├── Bench.swift          — Training throughput benchmark
│   │   ├── Train.swift          — Train from scratch + save
│   │   ├── Sample.swift         — Load + generate
│   │   ├── Eval.swift           — Score on held-out text
│   │   ├── Compare.swift        — Base vs base+LoRA delta
│   │   ├── Finetune.swift       — LoRA fine-tuning
│   │   ├── HFInspect.swift      — Inspect a HF model directory
│   │   └── Debug.swift          — Diagnostic helpers
│   └── TinyGPTApp/              — SwiftUI app
│       ├── TinyGPTApp.swift     — @main + AppDelegate (activate-on-launch)
│       ├── ContentView.swift    — Sidebar + tab bar + machine stats bar
│       ├── Theme.swift          — Colors matching the browser
│       ├── ModelController.swift — Owns the loaded model + sampling
│       ├── TrainController.swift — Owns a training run
│       ├── TrainView.swift      — Train tab UI
│       ├── LossChart.swift      — Canvas-rendered live loss curve
│       ├── GalleryDiscovery.swift — Find .bin/.tinygpt files
│       ├── CorpusDiscovery.swift — Find text files
│       └── MachineStats.swift   — Process RSS + GPU + RAM stats
└── Tests/                       — XCTest suites
```

## 2. The pipelines

### 2a. Load + sample (gallery / pretrained checkpoint)

```
.tinygpt file (fp16 storage)
   │
   │ TinyGPTFileReader.read()                    [TinyGPTIO]
   ▼
TinyGPTFile struct
   │
   │ TinyGPTWeightLoader.load(file, into: model) [TinyGPTModel]
   │   - Reads each tensor at its WASM-order shape
   │   - Transposes Linear weights to PyTorch shape
   ▼
TinyGPTModel (MLX-Swift Module tree)
   │
   │ model.forwardCached(idx, cache) per token   [KVCache]
   ▼
logits [B, T, vocab]
   │
   │ argMax / sample → next token id             [Sample.swift]
   ▼
emit byte, repeat
```

### 2b. Train from scratch

```
ByteCorpus (raw bytes loaded from .txt)
   │
   │ sampleBatch(B, T) → (x, y)                  [Trainer.swift]
   ▼
(MLXArray inputs, MLXArray targets)
   │
   │ Trainer.step()
   │   - valueAndGrad(loss(model, x, y))
   │   - optimizer.update(model, gradients)
   │   - eval(loss, model, optimizer)
   ▼
loss value
   │
   │ append to history, repeat                   [Train.swift]
   ▼
saveCheckpoint() → .tinygpt                      [Train.swift]
```

### 2c. LoRA fine-tune + compare

```
base.tinygpt
   │
   │ TinyGPTFileReader + WeightLoader.load
   ▼
TinyGPTModel (base weights, frozen)
   │
   │ LoraInjection.inject(model, config)         [Lora.swift]
   │   - Replace q_proj, v_proj Linears with LoraLinear
   │   - LoraLinear: y = base(x) + (x @ A) @ B * scale
   ▼
TinyGPTModel with LoraLinear sub-modules
   │
   │ LoraInjection.freezeBase(model)
   │   - Module.freeze(recursive: true)
   │   - then unfreeze loraA, loraB on each LoraLinear
   ▼
Trainable params shrink from 9.6M → 98K
   │
   │ Trainer.step() — same as full training,
   │ but gradients only flow into A, B
   ▼
LoraAdapterWriter.write() → .lora               [LoraIO.swift]
   - JSON header with rank/alpha/targets + base config snapshot
   - Raw A, B fp32 matrices

Then `tinygpt compare` runs eval on both base alone and base+adapter,
prints loss/BPB/perplexity table + a sample from each.
```

### 2d. ANE-routed inference (the perf path that's gated on Apple)

```
.tinygpt  ─python_ref/export_to_coreml.py─→  .mlpackage
   │                                              │
   │                                              │ ct.optimize.coreml.palettize_weights
   │                                              ▼
   │                                          .mlpackage (4-bit weights)
   │                                              │
   │                                              │ TinyGPTANE.load(mlpackageURL)  [ANEInference.swift]
   │                                              ▼
   │                                          MLModel with computeUnits=.all
   │                                              │
   │                                              │ predict(tokens)
   │                                              ▼
   │                                          ~365 pass/s, 2.6× vs CPU
   │
   │ Today the from-MLX-Swift path is FASTER than ANE for our model
   │ size (Metal vs ANE roughly parity for fp16 transformers).
   │ ANE wins genuinely when:
   │  - Apple ships int4-compute on ANE in coremltools
   │  - Model is large enough that ANE's parallelism dominates
   │    kernel-launch overhead
```

## 3. The .tinygpt file format

Two body layouts share one container:

```
[ 4 bytes  ] magic "TGPT"
[ 4 bytes  ] version u32 LE  (currently 2)
[ 4 bytes  ] header length u32 LE
[ N bytes  ] UTF-8 JSON header:
              { config, manifest, lossHistory?, sample?,
                weightDtype, includesOptimizerState, ... }
[ 4 bytes  ] step counter int32 LE
[ body     ] depends on header.weightDtype:

  trainingFP32 (default, train-resumable):
    per tensor in manifest order:
      [4*N bytes] weight float32 row-major
      [4*N bytes] AdamW m float32
      [4*N bytes] AdamW v float32

  inferenceFP16 (gallery distribution):
    one contiguous fp16 buffer of all weights, indexed by
    manifest entry's `floatOffset` field
```

The IMPORTANT gotcha that took an hour to find: WASM stores Linear
weights as `[in, out]` but the manifest declares PyTorch's `[out, in]`.
TinyGPTWeightLoader reads at WASM shape then transposes. Without this,
square weight matrices appear correct but compute wrong outputs;
non-square ones fail to load entirely.

## 4. The .lora file format

```
[ 4 bytes ] magic "TGLA"
[ 4 bytes ] version u32 LE (currently 1)
[ 4 bytes ] header length u32 LE
[ N bytes ] JSON header:
              { rank, alpha, targetSuffixes,
                baseLayers, baseDModel, baseCtx, baseHeads, baseDMlp,
                entries: [{name, loraAShape, loraBShape}, ...],
                savedAt?, finalLoss? }
[ body    ] raw fp32 [in, r] and [r, out] matrices per entry
```

Smaller than .tinygpt because adapter matrices are tiny. Rank-4 LoRA
on QV across 12 blocks = 12 × 2 × (256·4 + 4·256) = 49,152 fp32
floats = ~200 KB total.

## 5. ModelConfig — one struct, every architecture

`ModelConfig` carries everything needed to construct any supported
architecture:

```swift
ModelConfig(
    vocabSize: 256 or 32000-150000,    // byte-level vs BPE
    contextLength: 256 or 4096+,
    nLayers, nHeads,
    nKvHeads: nil (= nHeads, MHA) or less (GQA),
    dModel, dMlp,
    useRoPE: false (our default) or true (HF),
    ropeBase: 10000 (standard) or 500000 (Llama-3),
    useRMSNorm: false (LayerNorm) or true (HF),
    useSwiGLU: false (GELU MLP) or true (HF),
    attnBias: true (our default) or false (HF Llama family)
)
```

Two presets exist today: `.huge` (browser-comparable 9.6M) and
`.mega` (76M), `.behemoth` (404M), `.titan` (1.3B). When the HF
loader lands, a fourth factory will construct a config from a
`HuggingFaceConfig` automatically.

## 6. The HF-compat capabilities — what changed, why

To load any modern open-weight model, our model architecture needs
to support four things the from-scratch GPT skips:

1. **SwiGLU MLP** — gated feedforward `y = down(silu(up(x)) * gate(x))`.
   Three linears, ~1% better validation loss. Required for Llama,
   Mistral, Phi, Qwen, Gemma, LFM.

2. **RoPE** — rotate Q, K by position-proportional angle inside
   attention. No learned position embedding table. Generalises to
   longer contexts than training. `MLXFast.RoPE` is the fused kernel.

3. **GQA** — fewer K/V heads than Q heads; KV heads broadcast across
   their assigned Q heads inside attention. Shrinks KV cache 4-8×
   for sampling. MLX-Fast SDPA handles the broadcast natively.

4. **BPE tokenizer** — subword vocab of 32K-150K tokens. Massively
   more efficient than byte-level for natural language: same context
   window represents ~4× more text. We use `swift-transformers`
   for this — HuggingFace's official Swift port supports BPE,
   SentencePiece, and WordPiece via the same `AutoTokenizer` API.

## 7. Build + run

```sh
# CLI tools (must use Xcode toolchain for Metal compilation)
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcodebuild -scheme tinygpt -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .xcode-build build

# Run CLI subcommands
.xcode-build/Build/Products/Debug/tinygpt inspect path/to/model.tinygpt
.xcode-build/Build/Products/Debug/tinygpt sample path/to/model.tinygpt --prompt "ROMEO:"
.xcode-build/Build/Products/Debug/tinygpt finetune base --corpus my.txt --out my.lora
.xcode-build/Build/Products/Debug/tinygpt compare base --lora my.lora --corpus held-out.txt

# SwiftUI app
xcodebuild -scheme TinyGPTApp -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .xcode-build build
./scripts/make_app_bundle.sh
open .xcode-build/Build/Products/Debug/TinyGPT.app

# Tests (file-format suite passes via swift test; the MLX-using
# tests need Xcode for Metal libraries)
swift test
```

## 8. The roadmap

Shipped tonight:
- Train / save / load / sample / eval / compare / finetune CLI
- ANE conversion + 4-bit palettization (storage win)
- LoRA fine-tuning + adapter composition
- KV-cached sampling
- SwiftUI app with Sample + Train tabs, gallery sidebar, machine
  stats bar
- HuggingFace compat foundations: safetensors reader, config parser,
  param-name mapping, SwiGLU, RoPE, GQA, RMSNorm, BPE tokenizer
  (via swift-transformers)

Next session:
- Wire `tinygpt hf-load <dir>` end-to-end (config → ModelConfig →
  TinyGPTModel → load safetensors via name map → tokenizer
  attached)
- Reuse `tinygpt finetune` and `tinygpt compare` against the
  HF-loaded models so the same LoRA workflow applies
- SwiftUI Fine-tune tab that drives the LoRA pipeline visually
- Multi-modal / MoE / model ensembling — see `docs/parked_multi_model.md`
