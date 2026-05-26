# Flash Attention 2 — backward pass notes

Implementation log for the second half of task #47. Forward landed last
session (`webgpu/attention_fa2.wgsl` + `tests/test_fa2_parity.mjs` +
`docs/fa2_forward_notes.md`); this is the runway for backward.

This delivery: **algorithm-level parity verified in Node**
(`tests/test_fa2_backward_parity.mjs`). The WGSL kernel + ops.ts wiring
are left for a follow-up session because of a real binding-budget
problem documented below.

## The math (FA2 backward with recompute)

Standard attention backward starts from `dO = ∂loss/∂ctx` and produces
`dQ, dK, dV`. Naive version uses the cached attention matrix `P`. The
FA2 trick: with the forward's saved `L = m_final + log(l_final)`
(log-sum-exp per Q row), we reconstruct `P = exp(S − L)` from `q/k`
without ever reading the full `[B, H, T, T]` attn matrix.

Per `(batch, head, q_row)`:

```
D[q_row] = ⟨dO[q_row], O[q_row]⟩          // scalar; O = ctx from forward
for t2 in causal range (t2 ≤ q_row):
    S    = q[q_row] · k[t2] · scale
    P    = exp(S − L[q_row])               // recompute
    dP   = ⟨dO[q_row], V[t2]⟩
    dS   = P · (dP − D[q_row])
    dQ[q_row] += dS · k[t2] · scale        // own thread
    dK[t2]    += dS · q[q_row] · scale     // others contribute too
    dV[t2]    += P  · dO[q_row]            // others contribute too
```

`dQ` is owned by `q_row`'s thread — no contention. `dK` and `dV` are
gathered: every `q_row ≥ t2` contributes to the same `k[t2]` and `v[t2]`.
That contention shapes the kernel design (see below).

## Parity verification — what's already done

`tests/test_fa2_backward_parity.mjs` mirrors the planned WGSL kernel
in plain JS. Six shapes including the masked-boundary case (T=20),
the Behemoth-shaped head dim (hd=64), and the regime-crossover case
(T=256). All 18 checks (6 shapes × dQ/dK/dV) pass within 1 ULP of
the naive cached-attn reference:

```
$ node tests/test_fa2_backward_parity.mjs
ok   fa2 dQ vs naive [B=1 T=16 C=32 H=4 hd=8]    maxAbs=2.98e-8
ok   fa2 dK vs naive [B=1 T=16 C=32 H=4 hd=8]    maxAbs=2.98e-8
ok   fa2 dV vs naive [B=1 T=16 C=32 H=4 hd=8]    maxAbs=2.38e-7
...
ok   fa2 dV vs naive [B=1 T=256 C=64 H=2 hd=32]  maxAbs=2.98e-7
ALL PASS
```

`dV` shows higher absolute error because it sums `T` contributions
per output element; 2.98e-7 at T=256 is ~T·ε for f32 — clean.

## Why the WGSL kernel isn't shipped this turn

The natural binding layout for the backward kernel is **7 storage
buffers**:

```
g0 = q       (read)
g1 = k       (read)
g2 = v       (read)
g3 = L       (read)     ← new in forward, saved per (b, h, t1)
g4 = dO      (read)
g5 = ctx     (read)     ← needed to compute D[t1] = ⟨dO, O⟩
g? = dQ/dK/dV  (write)
```

The shared bind layout in `webgpu/ops.ts` exposes exactly six storage
buffers + one uniform. Three workable resolutions, none of which
deserve to be rushed:

1. **Pre-pass kernel** computes `D` once and stores it in a small
   `[B, H, T]` buffer. The main backward kernel then drops `ctx` and
   reads `D` instead — fits in six bindings cleanly. Two kernels total.
2. **Buffer aliasing trick** — overwrite `g4 = dO` with `dq` in place.
   Each thread reads its full `dO[t1, :]` row into private registers
   before the K-block loop, then writes `dq[t1, :]` to the same slot.
   No race because the buffer is per-thread (per-Q-row). One kernel,
   but the aliasing rule (`ops.ts` already comments that WebGPU forbids
   aliasing the same buffer in two writable bindings of one bind group)
   means we need a slightly different bind-group construction path.
3. **Two-kernel split** by output role: `fa2_backward_dq` (one workgroup
   per Q tile, dQ-only — no contention) and `fa2_backward_dkv`
   (one workgroup per K tile, walking all Q rows ≥ k_row, dK + dV).
   This is the FA2 paper's actual implementation choice. Avoids float
   atomics entirely; uses the natural data-flow split (each kernel's
   threads each own one row of their output buffer). Likely the right
   long-term shape.

Picking between (1) and (3) is a 20-minute design decision; doing it
right is another half-day of careful WGSL plus the parity gate
(`tests/test_webgpu_train.mjs` end-to-end, 5% drift bar). Today's
delivery pins the math so whichever path the next session picks, the
algorithm itself is verified.

## What the next session needs to do

1. **Decide on (1) vs (3) above.** My recommendation: (3), because it
   matches the published FA2 backward and parallelises naturally.
2. **Modify forward** to save `L = m + log(l)` into a new `[B, H, T]`
   buffer (small) so backward can read it. Today the kernel saves
   `m_final` and `l_final` to `var<workgroup>` arrays — we just expose
   them to a host-visible storage buffer.
3. **Write the WGSL kernel(s)** following the plan in
   `test_fa2_backward_parity.mjs`. The Node mirror's loops are the
   exact WGSL structure modulo cooperative loading of K/V tiles into
   shared memory.
4. **Live WGSL compile check** via Playwright + a real WebGPU adapter,
   same pattern as `tests/test_fa2_compile.mjs` did for the forward.
5. **Integration in `ops.ts`**: add an `attentionBackwardFA2` method
   that calls the new kernel(s); route `attentionBackward` to it
   conditionally on whether `attentionForward` used the FA2 path
   (today both are conditional on `hd ≤ 64`).
6. **End-to-end gate**: `tests/test_webgpu_train.mjs` — drift < 5%.
7. **Drop the second `attn` pass from the forward kernel** once
   backward is on the recompute path. Saves the O(B·H·T²) global
   memory write that the FA1-style backward currently needs. This is
   the other half of the FA2 memory win.
8. **Bench**: one Mega-preset (ctx=512) step with FA1 forward + FA1
   backward vs FA2 forward + FA2 backward. Document in
   `browser/devlog.html`. **Single shot**, per AGENTS.md "Safety
   rules" — no loops.

## Caveats specific to the backward path

- **The L = -∞ row.** For Q rows where every K position is masked
  (shouldn't happen with causal attention as long as `q_row < T`, but
  defensive code wins): `L = -∞` means `exp(S − L) = exp(+∞)` blows up
  to NaN. The forward already writes zero `ctx` for those lanes; the
  backward needs the same guard (skip rows where `l == 0`).
- **D = ⟨dO, O⟩ as scalar.** Cheap, but needs `O` (= forward ctx)
  available. If we go with split (3), `fa2_backward_dq` can compute
  D from O+dO on the fly; `fa2_backward_dkv` doesn't need D directly
  (it needs `dS` which factors through D, so it might need a small
  D buffer either pre-computed or duplicated in shared memory).
- **No dropout, no relative position bias.** Same scope as the forward.

## Where this lives in the codebase

```
docs/fa2_forward_notes.md          — forward writeup (last session)
docs/fa2_backward_notes.md         — this file
tests/test_fa2_parity.mjs          — forward parity (last session)
tests/test_fa2_backward_parity.mjs — backward parity (this session)
webgpu/attention_fa2.wgsl          — forward kernel (last session)
webgpu/attention_fa2_backward.wgsl — backward kernel (next session)
webgpu/ops.ts                      — integration site (next session)
```
