# Online softmax in attention — why and how

Audience: an ML-curious engineer who can read C++/WGSL but hasn't derived
the trick before. This doc walks through what "online softmax" means, why
it matters for attention specifically, and where the idea shows up in the
TinyGPT codebase. It builds toward the existing `attn_fused_sv` kernel in
[`webgpu/train.wgsl`](../webgpu/train.wgsl) and gestures at where Flash
Attention 2 takes the same idea further.

## The textbook softmax: stable, but two passes

Softmax over a vector `x` of length `T` is:

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

Computed naively, `exp(x_i)` overflows fast — fp32 `exp(89)` is already
infinity. The fix is the standard trick: subtract the max before
exponentiating. Algebraically nothing changes (the `exp(-max)` factor
cancels top and bottom), but every `exp` now sees a non-positive argument
and stays in range.

The "safe softmax" you'll see in every textbook is two passes:

```
# Pass 1: find the max
m = -inf
for i in 0..T:
    m = max(m, x[i])

# Pass 2: normalised exponentials and their sum
s = 0
for i in 0..T:
    e[i] = exp(x[i] - m)
    s   += e[i]

# Pass 3: normalise
for i in 0..T:
    p[i] = e[i] / s
```

Three passes, actually, if you separate out the divide. This is exactly
what the C++ reference does in [`wasm/src/attention.cpp`](../wasm/src/attention.cpp):
compute scores, find `maxv`, exponentiate, sum, divide.

The two-pass structure is fine when the input vector fits in a register
file or in shared memory. It becomes painful when:

1. `T` is very large (long-context attention), so the `T`-sized scores
   array doesn't fit anywhere fast, or
2. you only get to stream the input once because reading it again means
   another round trip to global memory.

Both of those are true for transformer attention at long context, and
that's where online softmax earns its keep.

## The online formulation: one pass, with running state

The idea is to compute `m`, `s`, and the running output incrementally —
revising both as you see new elements — instead of doing three separate
passes. You carry two scalars and update them with each new score.

Say you've seen scores `x_1, ..., x_k` and maintain:

- `m_k = max(x_1, ..., x_k)` — the running max
- `s_k = sum_i exp(x_i - m_k)` — the running normalisation, **rescaled to
  the current max**

When a new score `x_{k+1}` arrives, two things can happen:

**Case A — `x_{k+1} ≤ m_k`.** The max doesn't move. Just add the new
exponentiated term to the sum:

```
m_{k+1} = m_k
s_{k+1} = s_k + exp(x_{k+1} - m_k)
```

**Case B — `x_{k+1} > m_k`.** The max moves to `x_{k+1}`. Every previously
accumulated `exp(x_i - m_k)` term was computed against the old max, so
each one needs to be rescaled by `exp(m_k - m_{k+1})` (which is < 1, since
the max only grew):

```
m_{k+1} = x_{k+1}
s_{k+1} = s_k * exp(m_k - m_{k+1}) + 1     // the last term is exp(x_{k+1} - m_{k+1}) = 1
```

Both cases collapse to one update if you write it as:

```
m_new = max(m_old, x_new)
s_new = s_old * exp(m_old - m_new) + exp(x_new - m_new)
```

This is the **online softmax** — see Milakov & Gimelshein (NVIDIA, 2018),
which is also the trick FlashAttention reuses.

The same logic extends to whatever you want to *do* with the softmax. In
attention you want a weighted sum of value vectors:

```
ctx = sum_i softmax(x)_i * v_i
```

Maintain a running unnormalised context `O_k = sum_{i≤k} exp(x_i - m_k) * v_i`,
rescale it when the max moves (same `exp(m_k - m_{k+1})` factor), and
divide by `s` at the end. Now you've computed the softmax-weighted output
in a single streaming pass over `(x_i, v_i)` pairs.

## Why this matters for attention specifically

In the per-head, per-query-position attention computation, the inner loop
over the key/value sequence is exactly the streaming setup the online
formulation wants. For each query position `t1`, you're computing:

- scores `s_{t2} = (q_{t1} · k_{t2}) / sqrt(hd)` for `t2 = 0..t1` (causal mask)
- weights `a_{t2} = softmax(s)_{t2}`
- context `ctx_{t1} = sum_{t2} a_{t2} · v_{t2}`

The naive way: do it in three separate kernel passes. First write `s`
into a `[B,H,T,T]` buffer. Then read that buffer, apply softmax, write
attention weights `a` back. Then read `a` and `v`, write context `ctx`.
That's two full round trips through the `[B,H,T,T]` attention tensor in
global memory, which gets very expensive at long context — that buffer
grows as O(T²).

TinyGPT's [`attn_fused_sv`](../webgpu/train.wgsl) kernel fuses the second
and third pass into one. It's still two-pass *within* a query position
(it has to materialise the scores into shared memory to find the max), but
the second pass produces both the softmax weights *and* the context
accumulation in one loop:

```wgsl
// Pass 1: scores + max.
for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
  // ... q · k inner product ...
  sc[t2] = s;
  if (s > maxv) { maxv = s; }
}
// Pass 2a: sum.
var sum = 0.0;
for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
  sc[t2] = exp(sc[t2] - maxv);
  sum = sum + sc[t2];
}
let inv = 1.0 / sum;

// Pass 2b: ctx = sum_{t2} softmax(t2) * v[t2]
for (var t2 = 0u; t2 <= t1; t2 = t2 + 1u) {
  let a = sc[t2] * inv;
  // ... accumulate a * v[t2, :hd] into ctx ...
}
```

We still write the attention weights into the `[B,H,T,T]` buffer because
the backward kernels (`attn_dscores`, `attn_dvalue`) read them — backward
recomputation is the next step, not yet shipped. But the forward no longer
has to read the attention buffer back into a separate kernel to compute
`ctx`. That's the measurable forward-pass win from fusion.

**Why two-pass and not fully online here?** Because we already have the
whole `[T]` scores array sitting in shared memory (`var sc: array<f32, 1024>`)
once Pass 1 finishes. The advantage of going fully online — updating `m`
and `s` element-by-element — is that you never need to materialise the
full scores vector. That advantage shows up when `T` is large enough that
the scores array would spill out of fast memory, which is the regime
Flash Attention 2 targets with workgroup-level tiling. At Small/Mega
preset sizes (`T ≤ 256`), the scores array fits, and the two-pass form is
fine.

## Where this is heading: FA2

Flash Attention 2 takes the online softmax idea and applies it at a
**tile** level, not an element level. Conceptually:

1. Process the K/V sequence in tiles of, say, 64 positions at a time.
2. For each tile, compute scores, find a tile-local max, exponentiate.
3. Maintain workgroup-level running `(m, s, O)` state across tiles —
   when a new tile arrives, rescale the previous accumulator by
   `exp(m_old - m_new)` if the global max changed, then add the new
   tile's contribution.
4. Never materialise the full `[B,H,T,T]` attention matrix. On backward,
   recompute scores from cached `(m, s)` instead of reading them back.

The two payoffs: peak memory drops from O(T²) to O(T) per head, and the
arithmetic intensity climbs because you keep the small running state hot
in registers and stream K/V through tiles. That's the lever TinyGPT
hasn't pulled yet — `attn_fused_sv` is one step toward it, kernel-fusion
of softmax + value, but it still pays the O(T²) memory and the per-query
loop is still serial.

## What to take away

- "Online softmax" = update the max and sum incrementally so you don't
  need a separate max pass. It's algebraically identical to the textbook
  formulation; the win is structural, not numerical.
- The single fused kernel in TinyGPT (`attn_fused_sv`) fuses softmax and
  value-projection but still keeps the full scores vector in shared
  memory. That's a one-tile-per-query special case of the more general
  online algorithm.
- Flash Attention 2 generalises the same trick to tile-level streaming
  over K/V, which is what makes long-context attention possible without
  an O(T²) memory footprint.

If you want to verify the trick yourself: take the textbook two-pass
softmax, plug in a small example by hand (say `x = [1, 5, 3]`), then
re-derive it incrementally with the running `m, s` updates. The numbers
should match to floating-point noise. After that, the WGSL kernel is
just bookkeeping.
