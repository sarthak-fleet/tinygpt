# Roadmap — orthogonal categories

These are orthogonal to "what to build next" but matter for completeness.
Compact entries.

Status legend: 🟢 shipped · 🟡 partial · ⬜ not yet built · 🟣 parked.

## Optimizers (we currently use AdamW)

- **AdamW 🟢** — what we have. Standard. Memory: 2× params (m + v).
- **Lion ⬜** — sign-based optimizer; ~½ the memory of Adam. Sometimes
  matches Adam quality. Worth trying for big models where Adam's
  memory dominates.
- **Sophia ⬜** — second-order optimizer (uses Hessian estimates).
  Reported 2× faster convergence. More code complexity.
- **Muon ⬜** — 2024 optimizer. Orthogonalizes gradients via
  Newton-Schulz iterations. Big wins on small-scale benchmarks
  (Karpathy's nanoGPT speedrun adopted it). Worth a benchmark.
- **Adafactor ⬜** — sublinear-memory Adam variant. Trades some
  quality for much less memory.
- **BAdam ⬜** — block coordinate descent for full fine-tune at
  LoRA memory cost. Different mechanism from GaLore.
- **LISA ⬜** — Layer-wise Importance Sampled AdamW. Train only a
  random subset of layers per step. Memory savings + sometimes
  better quality.

## Training stability tricks

- **Gradient clipping ⬜** — clip gradient norm to a fixed value
  (~1.0 typical). Prevents loss spikes. Probably the cheapest +
  most universal stability lever. MLX-Swift has `MLX.clipNorm`;
  needs wiring into Trainer.
- **Z-loss / auxiliary loss ⬜** — add a small penalty term on
  logsumexp magnitudes. Stabilizes training at scale.
- **Embedding RMSNorm ⬜** — apply RMSNorm to the token embedding
  output. Helps with input distribution drift.
- **Layer-wise learning rate decay ⬜** — smaller LR for lower
  layers. Stabilizes fine-tuning.
- **Warmup curves beyond linear ⬜** — cosine warmup, exp warmup.
  Marginal vs linear.
- **DeepNorm ⬜** — residual scaling that improves stability of
  very deep models. Matters past ~50 layers.
- **Embedding tying 🟢** — already a config flag (`tieEmbeddings`).

## Data techniques

- **Curriculum learning ⬜** — order examples easy → hard. Modest
  gains; needs a difficulty metric.
- **DoReMi ⬜** — learns optimal data domain mixing ratios via a
  reference model. Useful when training on a mix (FineWeb + Wiki +
  code).
- **Data quality filtering ⬜** — perplexity-based filtering with
  a reference model. Drop the highest-PPL docs (likely noise).
- **Deduplication ⬜** — drop near-duplicate documents.
  FineWeb-edu is already deduped; matters for raw web scrapes.
- **Hard example mining ⬜** — oversample high-loss examples.
- **Importance sampling ⬜** — sample based on token-level
  importance scores.
- **Self-instruct ⬜** — use the model to generate its own training
  data; bootstrap from a small seed set.
- **Evol-instruct ⬜** — iteratively complicate prompts to grow
  instruction data quality.
- **Distillation-based synthesis ⬜** — generate data with a
  bigger model, train smaller on it. Pairs with knowledge
  distillation (Tier 1.1).
- **Document-level shuffling 🟢** — implicit in our random batch
  sampling.
- **Sample packing ⬜** — different from sequence packing (Tier
  1.2): combine examples from *different* sources into one batch
  to avoid intra-source correlation.

## Tokenization

- **Byte-level (vocab=256) 🟢** — what we have on the from-scratch
  path.
- **HF BPE / SentencePiece via swift-transformers 🟢** — shipped
  for from-scratch (Tier 1) and HF models.
- **BPE-dropout ⬜** — randomly merge less often during encoding;
  regularizer.
- **Train our own BPE on our corpus ⬜** — `tokenizers` Rust crate
  via Python wrapper. ~5% perplexity improvement at same step count
  vs using a foreign tokenizer.
- **Vocabulary trimming ⬜** — drop unused BPE tokens. Shrinks the
  embedding matrix.
- **tiktoken adoption ⬜** — OpenAI's tokenizer. Format isn't
  natively in swift-transformers but is reproducible.
- **Subword regularization ⬜** — present multiple valid
  tokenizations of the same text during training. Robustness
  technique.

## Interpretability tools (educational value)

See [`docs/interpretability.md`](../interpretability.md) for what's
shipped today.

- **Logit lens 🟢** — at every layer, project hidden state to vocab
  via the LM head, see what token the model would predict if
  forced to stop there. Reveals layer-wise prediction emergence.
- **Activation patching 🟢 (zero-patch variant)** — replace one
  example's hidden state with another's at a specific layer; see
  what changes downstream. The mechanistic-interpretability primitive.
- **Linear probes ⬜** — train a small linear classifier on hidden
  states for a specific property. Detects what each layer "knows."
- **Attention heat maps 🟢** — visualize attention weights per head
  per position. Browser playground ships this.
- **Top-k token-by-token 🟡** — show the top 5 alternatives at each
  generation position with their probabilities. Already partially
  in the browser playground; could be expanded.
- **Per-layer ablation 🟢** — zero out one layer at inference, see
  how much quality drops.
- **Sparse autoencoders for interpretability ⬜** — Anthropic-style
  feature decomposition. Substantial build.
- **Knowledge editing (ROME / MEMIT) ⬜** — surgical weight edits to
  modify specific facts.

## Inference optimizations (single-user, single-GPU)

- **KV cache 🟢** — shipped (both arch paths).
- **Flash Attention forward 🟢** — `MLXFast.scaledDotProductAttention`
  + WGSL FA2 in browser.
- **Flash Attention backward 🟢** — same.
- **Quantized inference (int4/int8) 🟢** — via `MLXNN.quantize`.
- **Speculative decoding** — Tier 1.8.
- **Prefix caching** — Tier 2.15.
- **Streaming attention sink** — Tier 2.14.
- **KV cache quantization** — Tier 2.11.
- **Multi-Token Prediction inference path** — Tier 2.12.
- **Token elimination ⬜** — drop low-probability past tokens
  from the KV cache. Trade slight quality for shorter effective
  cache.
- **Continuous batching ⬜** — multi-user only; not us.
- **PagedAttention ⬜** — multi-user only.
- **Tree decoding ⬜** — sample a tree, prune; better than top-k
  for some tasks.

## Browser-side performance

Already largely explored in [`docs/perf_quest.md`](../perf_quest.md).

- **f16 storage 🟢** — shipped.
- **Blocked 4×4 matmul 🟢** — shipped.
- **WebGPU subgroups ⬜** — better matmul via subgroup ops.
  Chrome only.
- **WebGPU cooperative matrix ⬜** — hardware matmul intrinsics.
  Chrome-experimental.
- **WebNN integration ⬜** — Chrome's neural-network API. Could
  delegate inference to native backend.
- **Memory64 🟢** — shipped (lifts 4 GB heap cap).
- **OPFS persistence 🟢** — shipped.

## Architecture variants

- **Standard transformer (RoPE + RMSNorm + SwiGLU + GQA) 🟢** —
  what we have.
- **MoE 🟢** — Tier 3.12. See [`docs/moe.md`](../moe.md).
- **Sliding window** — Tier 2.9.
- **Sparse attention (BigBird, Longformer) ⬜** — pattern-based
  sparse masks for very long context. Tier 4 unless we go past
  ctx=8192.
- **Linear attention (Performer, Linformer, Reformer) ⬜** —
  O(N) attention via kernel tricks. Quality usually worse than
  flash attention at moderate contexts.
- **State space models (Mamba, RWKV) ⬜** — Tier 4 (whole
  different family).
- **Hybrid attention/SSM (Jamba, Samba) ⬜** — interleave
  attention + SSM layers. Compromise architecture.
- **Multi-token prediction heads** — Tier 2.12. See [`docs/mtp.md`](../mtp.md).
- **Mixture of Depths** — Tier 3.14.
- **Differential attention** — Tier 3.13.
- **YOCO (cross-layer KV)** — Tier 3.8.
- **Pre-norm vs post-norm ⬜** — pre-norm is standard; post-norm
  is more stable at very large scale (used by GPT-2 original).
  Toggle in our `TransformerBlock`.

## Adapter / PEFT taxonomy summary

Compact table of the LoRA family for cross-reference. Mechanics in
[`docs/lora_guide.md`](../lora_guide.md).

| Name | Idea | Status | Tier |
|---|---|---|---|
| LoRA | Rank-r `A·B` delta | 🟢 | shipped |
| Multi-LoRA composition | Stack N adapters | 🟢 | shipped |
| DoRA | Magnitude + direction decomp | 🟢 | 2.3 |
| QLoRA | Int4 base + fp16 LoRA | 🟡 | 1.3 |
| VeRA | Frozen random A, B; scalars | ⬜ | 2.5 |
| GaLore | Low-rank gradient projection | ⬜ | 2.4 |
| LoftQ | Quantization-aware LoRA init | ⬜ | 2.6 |
| LoRA+ | Different LR for A vs B | 🟢 | 3.18 |
| LoRA-FA | Freeze A | ⬜ | 3.20 |
| RsLoRA | Different scaling at high r | ⬜ | 3.21 |
| ReLoRA | Periodic merge + restart | ⬜ | 3.16 |
| AdaLoRA | Adaptive per-layer rank | ⬜ | 3.17 |
| PISSA | SVD-based initialization | ⬜ | 3.19 |
| NEFTune | Noise on embeddings | 🟢 | 1.6 |
| Prefix tuning | Soft prompt tokens | ⬜ | 2.16 |
| Prompt tuning | Soft prompt; smaller variant | ⬜ | (subsumed) |
| IA³ | Element-wise scaling | ⬜ | 4 (skip) |
| BitFit | Train biases only | ⬜ | 4 (skip) |
| Adapter (Houlsby) | Bottleneck MLPs | ⬜ | 4 (skip) |

## Cross-cutting infrastructure (lifts ROI of everything)

- **Browser-side benchmark runner** — Tier 1.9.
- **Real submission upload flow** — drag-drop + auto-score; OPFS
  first, R2 later.
- **TinyGPT-as-library API** — surface the four primitives
  (`forward_backward`, `optim_step`, `sample`, `save_state`) as a
  public Python/TS API.
- **Persistent tokenized cache ⬜** — write `.tokens` files alongside
  `.txt` corpora. Saves the 30-min BPE-encode cost on every run.
- **Real CI ⬜** — GitHub Actions: build + test on every PR.
- **`tinygpt eval` benchmark harness ⬜** — `tinygpt eval --bench
  tinystories-ppl path/to/model.tinygpt` would replace the
  separate `score_gallery.mjs` scripts.
