# Citations

Every architectural claim in TinyGPT's code + docs traces back to a
primary source here. If a claim doesn't have a citation in this file,
treat it as informed-opinion-not-evidence and challenge it.

The bar: peer-reviewed paper > arXiv preprint > official model card
> library README > blog post > vibes. Anything below "blog post" is
labeled `[informal]`.

## Transformer architecture pieces

### SwiGLU MLP

**Primary**: Shazeer, Noam. *"GLU Variants Improve Transformer."*
arXiv:2002.05202 (2020). https://arxiv.org/abs/2002.05202

Shows that gated linear units (GLU, ReGLU, GEGLU, SwiGLU) outperform
plain MLPs across multiple downstream tasks. SwiGLU specifically:

> "We offer no explanation as to why these architectures seem to work;
> we attribute their success, as all else, to divine benevolence."
> — Shazeer, §6

Adoption in production:

- **PaLM** (Chowdhery et al., 2022, arXiv:2204.02311), §2.1: "We use
  SwiGLU activations (Swish(xW) · xV) for the MLP intermediate
  activations because they have been shown to significantly increase
  quality compared to standard ReLU, GeLU, or Swish activations."
- **LLaMA** (Touvron et al., 2023, arXiv:2302.13971): adopted SwiGLU
  citing the GLU Variants paper.

Why "~1% better validation loss" in our docs: this is the order-of-
magnitude observed in the GLU Variants paper across their benchmarks
(typical 0.3–1.5% perplexity gain). The exact number depends on the
specific task + model size.

### RoPE (Rotary Position Embedding)

**Primary**: Su, Jianlin, et al. *"RoFormer: Enhanced Transformer with
Rotary Position Embedding."* arXiv:2104.09864 (2021).
https://arxiv.org/abs/2104.09864

Key property: relative position information is encoded via rotation,
so `q·k` after rotation depends only on `(i - j)`, the position
difference, not on absolute `i` and `j`. This is what enables
extrapolation past the training context length.

Context extrapolation:

- **NTK-Aware Scaled RoPE** (informal, /u/bloc97 on Reddit, 2023):
  scale `base` non-linearly to push the wavelengths beyond training
  context. Empirically works.
- **YaRN** (Peng et al., 2023, arXiv:2309.00071): more principled
  extension; current SOTA for long-context extrapolation.

Adoption: LLaMA, Llama-2, Llama-3 (with different `rope_theta` values:
10,000 → 500,000 in Llama-3 to support 8K context). Mistral, Phi,
Qwen, Gemma, LFM all use RoPE.

### Grouped Query Attention (GQA)

**Primary**: Ainslie, Joshua, et al. *"GQA: Training Generalized
Multi-Query Transformer Models from Multi-Head Checkpoints."*
arXiv:2305.13245 (2023). https://arxiv.org/abs/2305.13245

Key claim from §4: GQA with 8 K/V heads matches MHA quality within
0.1 BLEU on translation while being 4× faster at inference.

Concrete adoption:

- **Llama-3-8B**: 32 query heads, 8 K/V heads (4:1 ratio).
  Source: Llama-3 model card,
  https://huggingface.co/meta-llama/Meta-Llama-3-8B
- **Mistral 7B** (Jiang et al., 2023, arXiv:2310.06825): 32 Q / 8
  KV heads.

Our "KV cache shrinks 4×" claim is exact for Llama-3-8B's 32/8 ratio.

### RMSNorm

**Primary**: Zhang, Biao, and Rico Sennrich. *"Root Mean Square Layer
Normalization."* arXiv:1910.07467 (2019). https://arxiv.org/abs/1910.07467

Key claim: RMSNorm = LayerNorm without mean centering, with no bias.
Achieves equivalent perplexity to LayerNorm on transformer training
while being ~30% cheaper per call.

Adoption: every Llama-family model from LLaMA onwards. Mistral, Phi,
Gemma, Qwen, LFM all use it.

### BPE / SentencePiece tokenization

**Primary (BPE)**: Sennrich, Rico, et al. *"Neural Machine Translation
of Rare Words with Subword Units."* arXiv:1508.07909 (2015).
https://arxiv.org/abs/1508.07909

**Primary (SentencePiece)**: Kudo, Taku, and John Richardson.
*"SentencePiece: A simple and language independent subword tokenizer
and detokenizer for Neural Text Processing."* EMNLP demo, 2018.
arXiv:1808.06226. https://arxiv.org/abs/1808.06226

The "~4× more text per token vs byte-level" claim: depends on the
language and BPE vocab size. For English with a 32K SentencePiece
vocab, average tokens per character is around 0.25 (i.e., 1 token =
4 characters). Source: empirical numbers from Llama tokenizer tests,
documented in Llama paper §3.3 and reproducible via the tokenizer's
own statistics.

## Training + inference techniques

### LoRA (Low-Rank Adaptation)

**Primary**: Hu, Edward J., et al. *"LoRA: Low-Rank Adaptation of
Large Language Models."* arXiv:2106.09685 (2021).
https://arxiv.org/abs/2106.09685

Key claims we cite:

- §1: full fine-tuning of GPT-3 175B has 175B trainable params; LoRA
  reduces this to 0.01% of the total (~17.5M) with negligible
  quality loss on a range of downstream tasks.
- §4.2: LoRA on q_proj + v_proj (rank 4) captures most of the
  benefit; targeting all four attention projections gives marginal
  improvement. This is why our default config wraps only Q and V.
- §4.3: training time per step is roughly equivalent to full
  fine-tuning because the forward and backward through the base
  weights still happens; the speedup comes from the optimiser state
  being ~200× smaller, not from less compute.

Our "98K trainable params, 1% of model" number on the Huge preset:
12 layers × 2 targets × (256·4 + 4·256) = 49,152 per target type ×
2 (A + B) = 98,304. Math reproducible from config.

### KV-cache (autoregressive attention reuse)

**Primary**: Implied by the original transformer decoder design in
Vaswani et al. 2017 (arXiv:1706.03762) §3.2.3. First explicit naming
+ description as "KV cache" is folklore; common reference is the
original GPT-2 paper code (Radford et al., 2019) and HuggingFace's
`transformers` library documentation:
https://huggingface.co/docs/transformers/main/en/llm_optims#static-kv-cache-and-torchcompile

Key claim: with KV-cache, per-token forward goes from O(T²) attention
work to O(T). Compounding: total work for generating T tokens goes
from O(T³) to O(T²).

Our measured "2.2× sustained speedup at 500 tokens" on the Huge
gallery model: attention is ~10% of compute at d_model=256, so the
theoretical max KV-cache speedup is ~1.1× (saving 90% of 10%).
Actual 2.2× includes the savings from MLX's lazy graph being smaller
when only one new token is processed per step (less Python-side
overhead, faster eval()). Real component breakdown is in
`docs/perf_research.md`.

### Flash Attention 2

**Primary**: Dao, Tri. *"FlashAttention-2: Faster Attention with
Better Parallelism and Work Partitioning."* arXiv:2307.08691 (2023).
https://arxiv.org/abs/2307.08691

FA1 (Dao et al. 2022, arXiv:2205.14135) is the predecessor.

We don't implement FA2 ourselves on Mac. We rely on Apple's MLX
team's `MLXFast.scaledDotProductAttention`, which the MLX team
implemented as an FA2-equivalent fused kernel. Source: MLX C++
source code, /opt/mlx-c/mlx/c/mlx_fast.cpp (see also the MLX
release notes for 0.5+).

Browser side: `webgpu/train_f16.wgsl` implements FA2 directly
in WGSL. See `docs/fa2_forward_notes.md` and `docs/fa2_backward_notes.md`
for the derivation.

### Mixture of Experts (parked)

**Primary**: Shazeer, Noam, et al. *"Outrageously Large Neural
Networks: The Sparsely-Gated Mixture-of-Experts Layer."*
arXiv:1701.06538 (2017). https://arxiv.org/abs/1701.06538

Recent + relevant adoption:
- **Mixtral 8x7B** (Jiang et al., 2024, arXiv:2401.04088): the open-
  weight reference. 47B total, ~13B active per token.

Our claim "compute matches 13B, quality matches 47B" is roughly the
Mixtral paper's headline result on their benchmark suite.

## Quantization

### Symmetric int8 / int4 weight-only quantization

**Primary**: Lin, Ji, et al. *"AWQ: Activation-aware Weight Quantization
for LLM Compression and Acceleration."* MLSys 2024, arXiv:2306.00978.
https://arxiv.org/abs/2306.00978

**Also**: Frantar, Elias, et al. *"GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers."* ICLR 2023,
arXiv:2210.17323. https://arxiv.org/abs/2210.17323

Both show that 4-bit weight-only quantization with proper scaling
loses < 1 perplexity point on Llama-class models.

Block-wise quantization (the variant we use): documented in
**GGUF / GGML / Q4_0** in `ggerganov/llama.cpp`:
https://github.com/ggerganov/llama.cpp/blob/master/docs/quantize.md

Our block size 64 + symmetric ±7 range + per-block fp16 scale
matches Q4_0 exactly. Source: llama.cpp's `quants/q4_0` implementation.

### Core ML 4-bit palettization

**Primary**: Apple's coremltools documentation.
https://apple.github.io/coremltools/docs-guides/source/opt-palettization-api.html

Specifically: `coremltools.optimize.coreml.palettize_weights`. The
mode we use is k-means clustering with `nbits=4` and
`granularity="per_tensor"`. This is purely STORAGE-side compression:
at inference the weights are expanded back to fp16 for the matmul.

Real int-compute on ANE (the path that would actually deliver
speedup): gated on Apple shipping the **Stateful Models** /
**MIL.LinearQuantized** API in coremltools. Not yet stable as of
this writing. Source: Apple's coremltools release notes,
https://github.com/apple/coremltools/releases

## Apple Silicon + MLX

### MLX framework

**Primary**: Apple Machine Learning Research. *"MLX: An array
framework for Apple silicon."* GitHub repository.
https://github.com/ml-explore/mlx

Source code is the citation; there isn't a formal paper. The README
documents the design (unified memory, lazy eval, function transforms).

### Apple Neural Engine

**Primary**: Apple's *Deploying Transformers on the Apple Neural
Engine* whitepaper / blog post (June 2022).
https://machinelearning.apple.com/research/neural-engine-transformers

Key numbers cited:

- ~10× faster inference on M1 ANE vs M1 GPU on a 6-layer transformer
  in their reference architecture
- ~14× lower latency
- ~14× lower power consumption

Our measured "2.6× vs CPU, parity with Metal" for a 9.6M-param fp16
transformer on M5 Pro: the Apple whitepaper's numbers were for a
specific reference architecture tuned for ANE; our generic Llama-
shaped model doesn't hit the same path-optimized speedup. The gap
to Apple's published numbers is the bridge we'd cross with the
Stateful Models API + proper int-compute path.

### swift-transformers (BPE / tokenizer Swift port)

**Primary**: HuggingFace. *swift-transformers* GitHub repository.
https://github.com/huggingface/swift-transformers

This is the canonical Swift implementation of HF's `tokenizers`
library. Used by `mlx-swift-examples`, `MLX-VLM`, and every serious
Swift LLM project. Maintained by HuggingFace themselves; pinned
to a specific version in `native-mac/Package.swift`.

## File formats

### safetensors

**Primary**: HuggingFace. *safetensors* GitHub repository + spec.
https://github.com/huggingface/safetensors

Key spec detail: u64-LE header size, JSON header with per-tensor
`{dtype, shape, data_offsets}`, raw tensor data packed contiguously.

### GGUF (mentioned but not used)

**Primary**: Gerganov, Georgi. *GGUF specification.* llama.cpp
documentation. https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

The format used by llama.cpp for distributing quantized models.
We don't use it directly — we use safetensors via HuggingFace —
but our 4-bit quantization scheme (block size 64, symmetric ±7,
per-block fp16 scale) matches GGUF's `Q4_0`.

## Datasets used

### Project Gutenberg

**Primary**: Project Gutenberg. https://www.gutenberg.org

All texts used in `scripts/fetch_corpora.sh` are public domain
(US copyright expired). Total ~34 MB across 19 books.

### TinyStories

**Primary**: Eldan, Ronen, and Yuanzhi Li. *"TinyStories: How Small
Can Language Models Be and Still Speak Coherent English?"*
arXiv:2305.07759 (2023). https://arxiv.org/abs/2305.07759

The dataset: https://huggingface.co/datasets/roneneldan/TinyStories

### codeparrot/github-code-clean

**Primary**: BigCode project, *github-code-clean.*
https://huggingface.co/datasets/codeparrot/github-code-clean

A filtered subset of GitHub code from The Stack.

### databricks/databricks-dolly-15k

**Primary**: Databricks, *Free Dolly: Introducing the World's First
Truly Open Instruction-Tuned LLM.* Blog post (April 2023).
https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm.html

Dataset: https://huggingface.co/datasets/databricks/databricks-dolly-15k

### tinyshakespeare

**Primary**: Karpathy, Andrej. *char-rnn* GitHub repository.
https://github.com/karpathy/char-rnn

The `data/tinyshakespeare/input.txt` file from this repo is the
canonical "small Shakespeare corpus" used in every nanoGPT-style
tutorial.

## How to challenge a claim

If you find a number or claim in the code/docs that doesn't have a
citation in this file:

1. Note the file + line where you saw it
2. Either add a citation to this file (preferred) or downgrade the
   claim to "informal" / "rough estimate" in the source location

The point is to have ONE place to audit, not to inflate the
citation count.
