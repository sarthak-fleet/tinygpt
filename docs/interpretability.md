# Interpretability tools — what is the model thinking?

Two interpretability surfaces ship in the browser playground:

1. **Attention heatmap** — for any prompt, show the per-head
   attention weights from the LAST transformer block. The "Watch the
   model think" panel that already exists.
2. **Logit lens** (Nostalgebraist 2020) — for any prompt, show what
   each layer "would predict" if its hidden state were projected
   straight through the final layernorm + LM head. A window into when
   specific knowledge first appears in the residual stream.

Both are WebGPU-only — they need access to intermediate tensors that
the WASM build doesn't currently expose.

---

## Attention heatmap

Already wired. After training or loading a gallery model, generate
once and the "Watch the model think" card appears. The heatmap shows,
for every token position in the prompt and every attention head in
the last block, which earlier tokens the head looked at.

Implementation: `gpu_model.ts:GpuModel.inspect`. One forward over the
prompt, save the last block's `[B, H, T, T]` attention matrix,
download to CPU, return.

## Logit lens

**New as of Phase 8.** A second button next to "Run benchmark" in the
Sample card: **Logit lens**. Click it, the worker runs forward over
the current prompt and returns one row per layer × one column per
input position; each cell is the top-1 byte the model "would output"
if it stopped at that depth.

What to look for:
- **Early layers** usually predict the most common byte (space or
  newline) regardless of context. The residual stream is still mostly
  positional + low-order ngrams.
- **Mid layers** start producing context-aware predictions (correct
  next byte for short n-grams).
- **Last layer** matches the actual next-byte prediction.
- **Where a prediction first becomes correct** is the depth at which
  the relevant feature crystallises. For ROMEO-style prompts that's
  often layer 4-6 on a 12-layer Huge model.

### Algorithm

For each layer L:

1. Run the standard forward through layers 0..L.
2. Apply the final layernorm (with its real `γ, β`) to the layer-L
   output.
3. Project through the tied LM head (`tokEmb.asLinear`).
4. Softmax over vocab → top-K per position.

The lens uses the FINAL layernorm's parameters even at intermediate
depths. That's the standard "logit lens" interpretation; alternative
"tuned lens" variants train layer-specific projection heads, which
this implementation does NOT ship.

### Cost

One extra layernorm + lm-head matmul per layer per inspection. For
the Huge config (12 layers, vocab=256, d=256), that's 12 extra ops
per lens run — a ~2× slowdown vs. a normal forward. Cheap enough to
run interactively for any prompt in the playground.

### Code map

- `webgpu/gpu_model.ts:GpuModel.logitLens` — the forward + per-layer
  head projection.
- `browser/src/worker.ts:doLens` — message handler that softmaxes
  and picks top-K per position.
- `browser/src/main.ts:renderLens` — the ASCII-table render of the
  result.
- `browser/src/types.ts:LensResult` — the result type definition.

## What's NOT yet shipped

- **Tuned lens.** The current lens reuses the final-layernorm
  parameters at every depth — a noisy approximation for mid layers.
  A tuned lens would train a small `Linear` per layer to better
  project that layer's residual stream into the LM head's space.
- **Per-layer ablation tool** (roadmap Phase 8, third item). The
  ability to zero out an attention head or MLP block and watch the
  prediction shift. Mechanism is clear (set the relevant tensor to
  zero between forwards) — the UI for it is the work.
- **Cross-layer attribution.** Tracing a specific output token's
  attribution back through layers (logit attribution / integrated
  gradients). Higher-leverage but substantially more complex.

---

# Appendix — Watch the model think (the playground UX)

(Merged from the former `docs/watch_the_model_think.md`.)

The TinyGPT playground has a small interpretability lever bolted onto the Sample card:
click any byte the model just generated and see the two things the model "actually
knew" at that position — the next-byte probability distribution it sampled from, and
the attention weights from the final transformer block.

Nothing about it is prettified. The bars and heatmaps come from the same forward pass
your sample came from, with one extra download from GPU memory.

## How the path works

When you click **Generate**, the worker calls `gpuModel.generate(…)` and an
autoregressive loop streams bytes back as text. Once the sample message lands in the
main thread, the UI immediately fires a second message — `inspect` — back into the
worker carrying the *full generated string* as the prompt. The worker calls a sibling
method, `GpuModel.inspect(promptIds, k=10)`, which is mostly a wrapper around the
existing forward path:

1. Encode the prompt as `Float32Array` byte ids, batch=1, length T.
2. Call `this.forward(ids, 1, T)` — the same routine used during training.
3. Download two GPU tensors:
   - `logits` shape `[T, V]` — used to softmax each row and pick the top-10 candidates
     per position.
   - `caches[L-1].attn` shape `[B=1, H, T, T]` — the last block's attention weights,
     already softmaxed and causal-masked by `attentionForward`.
4. For each position `t`, slice out `H` rows of length `T` and ship them back across
   the worker boundary using transferable buffers so we don't copy 64+ KB twice.

The forward pass is one GPU submission. On the Small preset (T=64, L=3, H=4) it's
about 4 ms on an M-series GPU after the buffer pool is warm. The download is the
expensive part — somewhere between 5 and 30 ms depending on driver.

This view is **WebGPU-only for the first cut**. The WASM build doesn't expose a
sibling `tg_inspect` symbol yet — adding one is straightforward (it's the same C++
forward, just with the attn buffer left undestroyed and a new export) but it's a
follow-up. When called against a WASM-only model in memory the UI shows a friendly
"switch backend to inspect" note instead of fake data.

We also chose **last-layer-only** attention rather than all layers. Reasoning: the
most narratively useful head behaviour (induction heads, "previous-token" heads,
positional copy patterns) tends to live in the final blocks of a small transformer.
Showing all `L * H` heads would tile fine but the marginal information per head drops
fast as you go earlier. Adding "all layers" is one line — keep the cache, slice per
layer, ship as `attention[l][t][h]`.

## What the visualization is honest about

The textbook claim is *"attention is interpretable — you can see what the model is
looking at."* This is partially true and mostly an aesthetic.

What attention actually tells you, at position `t`:

- **Which earlier position the value-vector mix came from.** That's a real signal:
  if head `h` puts 0.9 of its weight on position 12, the output of that head at
  position `t` is mostly the value-projected representation of position 12.
- **Nothing about the residual stream.** The model is a stack of additive updates.
  The attention output at any given block is one of many summands. So "head H2 looked
  at position 5" is true, but the final logit at position `t` is not just that —
  it's the sum of contributions from every block plus the embedding, plus the MLP at
  the same block, plus everything earlier.
- **Nothing about composition.** Heads in later layers operate on *outputs of earlier
  heads*, so a clean "this head copies the previous token" story only tells you what
  it does in isolation, not what its output then does to logits.

The top-K probability bars are more grounded. They are exactly the distribution your
sample was drawn from (or would have been drawn from with temperature 1.0 — the
inspect path uses the raw softmax, no top-k truncation). When you see one byte sitting
at 70% and the rest scrambling for crumbs, that's the model being confident. When you
see ten bytes all between 5% and 12%, that's the model genuinely unsure and the
sampler making most of the decision.

## What it's good for

- **Teaching what attention is** without lying about how much it explains.
- **Spotting "the model has no idea here"** moments — a flat top-10 with no candidate
  above 15% is a clearer signal than "the output looks bad."
- **Watching a positional copying head emerge during training** — train for ~30
  seconds on Tiny Shakespeare, then look at a sample. At least one head usually puts
  a strong diagonal weight on the previous token (the bigram baseline). That's the
  model rediscovering character-level n-grams, visible at the level of weights.

## Not yet

- **Per-layer view**, **per-position composition**, **logit-lens-style projections**,
  and an **attribution mode** ("which earlier byte most influenced this byte's
  logit?") are all natural follow-ups that don't need new kernels — just more
  downloads from the existing forward.
- **WASM parity.** Add a `tg_inspect_forward` export, return `attn[L*H*T*T]` and
  `logits[T*V]`, and the same UI works against the C++ build.

The whole feature is ~200 lines of TypeScript and one new GPU method. The lever you
want to make a transformer "feel" navigable is small once the forward pass is already
there.
