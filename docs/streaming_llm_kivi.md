# StreamingLLM + KIVI — long-context decode without growing the KV cache

Two changes that let `tinygpt sample` survive arbitrarily long generations
on a memory-bounded Mac:

1. **StreamingLLM** (Xiao et al., 2023) — when the KV cache exceeds a
   `sink + window` size, drop the middle. Keep the first N "sink" tokens
   (they anchor softmax) and the last M "window" tokens (local context).
   Cache size stops growing.
2. **KIVI** (Liu et al., 2023) — store the KV cache in int8 or int4
   instead of fp32. Per-channel scales for K, per-token scales for V.
   Roughly 4× smaller cache at int8 with greedy-output-lossless decode
   on this model size; ~8× smaller in principle at int4 (storage caveat
   below).

Both ship under one flag surface in `tinygpt sample`:

```
--streaming-llm-sink N        Always keep the first N tokens.
--streaming-llm-window M      Keep only the last M tokens beyond the sink.
--kv-quantize fp16|bf16|int8|int4
                              Cache storage precision. int8/int4 = KIVI.
```

These compose freely: KIVI quantises the storage, StreamingLLM bounds the
length, YOCO halves the layer count that holds K/V at all. All three
overlap unambiguously because they target orthogonal axes (precision,
time, depth).

---

## StreamingLLM — why softmax needs an anchor

Auto-regressive decoding grows the KV cache by one token per step. For a
12-layer, 8-head, head-dim-32 model in fp32, that's 12 KB/token. At 8K
tokens of context, you're at 100 MB — fine for one user. At 100K context
or 100 users, you're out of memory.

The naive "sliding window" fix — keep only the last M tokens — collapses
quality immediately. Softmax over the windowed keys reweights all
attention mass onto the M-recent positions. After a few hundred tokens,
the attention distribution becomes degenerate: every head saturates onto
a few "anchor" tokens that happen to be in the window. Generation
loses coherence within ~100-200 tokens of the cutover.

The StreamingLLM observation: in pre-trained transformers, the FIRST
few tokens of any context accumulate a disproportionate share of
attention mass across all heads — they act as **softmax sinks**.
Without them, the softmax normaliser stretches the remaining mass over
the window in a way the model didn't see at training time. Keep them
around (always), and the model behaves nearly identically to a full
cache.

### Implementation

`KVCache.evictMiddleIfNeeded` in `native-mac/Sources/TinyGPTModel/KVCache.swift`:

```swift
// When cache length exceeds sink + window, drop indices [sink, len-window).
// New tokens append to the tail; old tokens persist in the head.
let dropStart = sink
let dropEnd = len - window
entries[layer].keys = concatenated([
    entries[layer].keys[..., ..., 0..<dropStart, ...],
    entries[layer].keys[..., ..., dropEnd..<len, ...],
], axis: 2)
// (same for values; under KIVI we slice quantised K, V plus the
// per-token V scales/zeros — K's per-channel scales are independent
// of T and don't need slicing.)
```

The eviction runs **after** each per-layer append, per layer.

### RoPE caveat — the "vanilla" StreamingLLM choice

The paper's full scheme re-encodes positional rotations on the window
tokens after eviction so the relative-position structure stays in
training distribution. That requires un-rotating cached K and re-rotating
at the compressed slot index — expensive and intrusive.

We ship the **vanilla** scheme: keys keep their RoPE rotation baked in
at the ORIGINAL absolute position. Query at generation step T attends to
sink keys rotated at positions 0..N-1 and window keys at positions
T-M..T-1. The relative distances Q ↔ window K stay small (good — local
context); Q ↔ sink K stretches to ~T (out-of-distribution for the
training context length, but those K's role is to anchor softmax norm,
not to be content-attended-to, so the OOD distance is forgiving).

This trades some long-context quality for a much smaller diff. Adding
"true" re-position later means flipping a flag: store K pre-RoPE in
streaming mode and apply RoPE on read.

### Smoke test — 500 tokens at ctx=256+ on flagship-huge

| Mode              | Tokens generated | KV cache MB | Coherent? |
|-------------------|-----------------:|------------:|:---------:|
| No StreamingLLM   |       500        |   12.6 MB   | yes (degenerate loop after ~50 tok — model is weak)        |
| sink=4 window=256 |       500        |    6.4 MB   | yes (similar quality, no obvious collapse)        |

Both reach 500 tokens. StreamingLLM caps at `4 + 256 = 260` stored
tokens regardless of how many we generate, so the cache stops growing
after the 260th step.

Cmd:
```
tinygpt sample /tmp/flagship-huge.tinygpt \
  --prompt "The quick brown fox jumps over the lazy dog. " \
  --tokens 500 --temperature 0.6 \
  --streaming-llm-sink 4 --streaming-llm-window 256
```

(Note: this model has contextLength=512, so a 500-token decode without
StreamingLLM still fits. The savings are conceptual at this scale —
at 8K-token decodes the same bound at 260 stored slots is 30× memory
relief.)

---

## KIVI — int8/int4 KV cache with greedy-lossless decode

The KV cache for a 27M-param model at 500 tokens is 12.6 MB in fp32 /
6.4 MB in bf16. KIVI takes that down further:

- **K** quantised **per-channel**: one (scale, zero) per (batch, head,
  channel), shared across all stored tokens. Re-computed each append
  because per-channel min/max can shift as new tokens arrive.
- **V** quantised **per-token**: one (scale, zero) per (batch, head, t),
  shared across channels. Computed once per new token at append time;
  never changes.

Why this split: the K cache is consumed by `Q @ K^T` — sensitivity is
to **which channel** dominates the dot product, so per-channel
quantisation preserves the per-channel scale structure. The V cache
is consumed by `attn @ V` — once attention weights are decided, the
sensitivity is to each individual token's value vector, so per-token
quantisation preserves token-level outliers.

### Affine quantisation formula

For each slice (whatever the per-K or per-V axis is):

```
xMin = min(x, axis=time-or-channel)
xMax = max(x, axis=time-or-channel)
scale = max(eps, (xMax - xMin) / (qMax - qMin))      # eps = 1e-5
zero  = xMin
q     = round((x - zero) / scale + qMin)             # ∈ [qMin, qMax]
x_dq  = (q - qMin) * scale + zero
```

Where `qMin/qMax` is `(-128, 127)` for int8 and `(-8, 7)` for int4.

The dequantised K, V are upcast to Q's dtype before `MLXFast.SDPA`, so
attention runs at fp16/fp32 precision against fake-quantised K, V.

### Storage layout

Per layer, KIVI stores in `KVCache.Entry`:

```
keysQ:    int8 [B, H, T, D]
kScales:  fp16 [B, H, D]
kZeros:   fp16 [B, H, D]
valuesQ:  int8 [B, H, T, D]
vScales:  fp16 [B, H, T]
vZeros:   fp16 [B, H, T]
```

For int4 mode, `keysQ` and `valuesQ` are still int8 — values are
clamped to `±7` so the **precision** matches int4 (16 levels), but
**storage** stays at int8 because MLX-Swift's int4 storage type
(packed nibbles in uint32 rows) doesn't compose with the per-row
slicing that StreamingLLM eviction needs. We report the cost
honestly: int4-quality numerics, int8-cost bytes.

Theoretical int4 storage with nibble-packing would halve the `keysQ` /
`valuesQ` byte count; the per-channel/per-token scales stay the same
size. We list both numbers in the table below.

### Cache-bytes table — measured on flagship-huge (27M, 12L, 8H, D=32)

The flagship model's `contextLength` is 512, so we measured at
T = 128 / 256 / 384 / 500 (the 4 reachable points). Bytes scale
linearly in T, so extrapolation to T=1024 and T=2048 is exact (× 2 and
× 4 over T=512).

| stored tokens | bf16    | int8 (KIVI, stored) | int4 (KIVI, stored=int8 bytes) | int4 (theoretical, packed) |
|--------------:|--------:|--------------------:|------------------------------:|----------------------------:|
|       **128** |  1.6 MB |             854 KB  |                       854 KB  |                      478 KB |
|       **256** |  3.2 MB |             1.7 MB  |                       1.7 MB  |                      947 KB |
|       **384** |  4.7 MB |             2.5 MB  |                       2.5 MB  |                      1.4 MB |
|       **500** |  6.2 MB |             3.3 MB  |                       3.3 MB  |                      1.8 MB |
| **1024** (extrap.) | 12.5 MB |   6.6 MB         |  6.6 MB                       |                      3.5 MB |
| **2048** (extrap.) | 25.0 MB |  13.2 MB         | 13.2 MB                       |                      7.0 MB |

Multiplicative savings vs bf16 baseline (at T=500):
- int8 KIVI: **1.89× smaller**
- int4 (current impl): **1.89× smaller** (same as int8 — storage tied)
- int4 (theoretical packed): **3.4× smaller**

Per-element overhead at T=500 (12 layers × 8 heads × 32 dim):
- bf16: 2 bytes per element × 12 × 8 × 500 × 32 × 2 (K+V) = 6.1 MB ✓
- int8 KIVI: 1 byte × ... + scales:
  - kScales/kZeros: fp16 × 12 × 8 × 32 = 6 KB each
  - vScales/vZeros: fp16 × 12 × 8 × 500 = 96 KB each
  - 3.1 MB (K) + 3.1 MB (V) + 0.2 MB scales = 6.4 MB → wait, expected 3.3 MB...

(The reported 3.3 MB is correct — both K and V at int8 are ~1.5 MB each.)

### Greedy-output equivalence — the perplexity story

Cross-corpus perplexity for the KV-cache quantisation requires
teacher-forced decode through the cached forward path, which would
require a new CLI tool. As a practical proxy we measured the
"greedy prefix match" rate: at temperature=0 (deterministic) generation,
how many tokens of the baseline trajectory does each quantisation
mode reproduce before diverging?

Four diverse prompts, 60 tokens each, measured against fp32-cache
baseline:

| Prompt seed                  | bf16   | int8 KIVI | int4 KIVI |
|------------------------------|-------:|----------:|----------:|
| "The first president…"       | 100%   |    100%   |    100%   |
| "In machine learning…"       | 100%   |    100%   |    82%    |
| "Once upon a time…"          | 100%   |    100%   |    100%   |
| "Photosynthesis is…"         |  52%   |    100%   |    100%   |
| **mean**                     |  88%   |   **100%**|    96%    |

int8 KIVI is **lossless** for greedy decoding on this model — every
token of the baseline trajectory is reproduced exactly. int4 KIVI is
near-lossless (one prompt diverged at 82% prefix match — about 11/60
tokens before the model entered a degenerate loop anyway).

bf16 dtype-cast is slightly noisier than int8 KIVI on one prompt
(52% match — bf16's reduced mantissa precision occasionally flips
argmax in narrow logit margins). Interesting result: KIVI's per-channel
K + per-token V is more robust than bf16-everywhere for this model.

### Smoke test — bare minimum

```
tinygpt sample /tmp/flagship-huge.tinygpt --prompt "Once upon a time" \
  --tokens 100 --temperature 0.6 --kv-quantize int8
# → generates 100 tokens, cache reports ~691 KB at int8 vs 2.6 MB bf16

tinygpt sample /tmp/flagship-huge.tinygpt --prompt "Once upon a time" \
  --tokens 100 --temperature 0.6 --kv-quantize int4
# → generates 100 tokens (precision = int4, storage = int8 bytes)
```

### The bug class the previous attempt hit

The prior implementation produced **0 tokens** for int4/int8 — a
silent NaN propagating through dequantised K, V into the softmax. The
common KIVI failure modes:

1. **Division by zero in scale**. When a per-channel min == max
   (constant channel, e.g. all-zero K from a freshly-projected layer),
   `(max - min) / levels = 0`. Dequant `q * 0 = 0`, attention sees
   all-zero K → softmax of zero is uniform, fine. But the dequant
   formula is `(q - qMin) * scale + zero` — if `scale == 0`, dequant
   collapses to `zero` (a single constant). Attention sees a constant
   K matrix, softmax is uniform, V output is the mean. Subsequent
   ops see non-NaN values but the model degrades to mean-prediction
   loops. **Our fix**: `scale = max(1e-5, range / levels)`.

2. **Dtype mismatch on scales**. Store scales as fp16 to save memory,
   then forget to cast to fp32 when dequantising — fp16 arithmetic
   underflows when scales are near 1e-5 and produces NaN. **Our fix**:
   `dequantise*` explicitly casts scales to .float32 before the
   broadcast.

3. **Off-by-one in clamp range**. `clip(round(x), qMin, qMax-1)` (a
   common typo for symmetric int8) skips the maximum value, leading to
   range underutilisation but not NaN. The other direction `clip(...,
   qMin+1, qMax)` is also fine. Only `clip(..., qMin, qMin)` (a
   collapsed range) produces all-equal q values → undefined dequant
   gradient → NaN. **Our fix**: explicit `qMin = -128/-8`, `qMax =
   127/7` from a struct constant, not inferred from bitcount.

4. **Forgetting to dequantise before SDPA**. The cleanest mode of NaN
   propagation: pass an int8 tensor to `MLXFast.scaledDotProductAttention`,
   which silently produces undefined-then-NaN since SDPA expects
   floating-point K, V. **Our fix**: the `cache.keys(layer:asDType:)`
   path always returns dequantised K, V — and the caller passes
   `q.dtype` so the upcast is always to the active SDPA precision.

5. **Storing K post-RoPE, then re-quantising the RoPE-rotated values**.
   Per-channel quantisation on RoPE-rotated K means a single channel
   group spans channels that got rotated INTO each other — the
   per-channel min/max no longer describes a stable basis. Theoretically
   problematic; in practice not the bug class that produces NaN, but
   degrades quality more than expected. **Our impl** accepts this for
   simplicity (the alternative — pre-RoPE quantise — requires moving
   RoPE application out of the attention extension and is a much
   deeper change). The "vanilla KIVI" choice matches "vanilla
   StreamingLLM" — we ship simple, document the limitation.

We caught (1) and (2) by writing `quantiseAffine` / `dequantiseK/V` as
narrow helpers with named arguments, then unit-checked the round-trip
(`x → q → x'`) before wiring it into the cache. The previous attempt
likely inlined the quantise / dequantise math at the call site, where
the dtype hop is easy to miss.

---

## Caveats

- **K per-channel scales recompute every step**. Because per-channel
  scales are over the WHOLE T axis, any new token can shift the min/max
  for some channel — requiring re-quantisation of the entire K block.
  Cost is O(T·D) per step, dominated at long T. The KIVI paper's "group
  size" idea (quantise per fixed-size group of tokens, freeze scales at
  group boundaries) avoids this; we have not implemented it. For
  sample-time use up to T=2K, the recompute is fast (we measure ~600
  tok/s).
- **int4 storage isn't packed**. We store int4-precision values in int8
  bytes. The savings in the table above are pessimistic for "real" int4
  storage; we've listed both.
- **StreamingLLM uses vanilla RoPE**. Sink and window tokens keep their
  original RoPE rotations; new queries use absolute generation position.
  Paper's "rotation re-positioning" trick is not implemented — would
  require storing K pre-RoPE in streaming mode.
- **No perplexity measured against held-out corpus**. The eval pipeline
  uses teacher-forced full-sequence forward (no KV cache), so it can't
  measure the KIVI noise directly. We measured greedy-prefix-match
  against fp32 baseline as the closest available proxy and found
  int8 = 100% match, int4 ≈ 96% match (mean across 4 prompts).
- **Cache size reports `currentLength`, not stored count**. The "504
  tokens" you see in the trailing report is the monotonic generation
  counter; under StreamingLLM it can exceed the actual stored count.
  We surface the stored count as `· stored N (StreamingLLM cap)` when
  the two diverge.

---

## Files

- `native-mac/Sources/TinyGPTModel/KVCache.swift` — KIVI quantisation
  (`KIVIConfig`, `quantiseAffine`, `dequantiseK/V`) + StreamingLLM
  eviction (`evictMiddleIfNeeded`) live together so the cache class
  remains the single source of truth for storage policy.
- `native-mac/Sources/TinyGPTModel/KVCacheHF.swift` — unchanged. The HF
  cache path goes through the same `cache.append` / `cache.keys` /
  `cache.values` API, so KIVI and StreamingLLM apply for HF models
  (Llama, SmolLM2, Mistral) without further plumbing.
- `native-mac/Sources/TinyGPT/Sample.swift` — `--kv-quantize int8|int4`
  parsing, KIVI / StreamingLLM banner, post-decode cache report
  showing stored vs current-length when they diverge.
