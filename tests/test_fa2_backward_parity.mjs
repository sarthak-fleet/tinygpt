// test_fa2_backward_parity.mjs — algorithm-level parity test for Flash
// Attention 2 BACKWARD (causal). Same approach as test_fa2_parity.mjs:
// mirror the planned WGSL kernel in plain JS, and check the outputs
// against a naive backward reference.
//
// The key claim: with the forward's saved L = m + log(l) (log-sum-exp
// per Q row), the backward can reconstruct P = exp(S − L) from q/k
// without ever reading the cached attention matrix. This is the FA2
// memory win — forward stops materialising [B, H, T, T] entirely.
//
// Run:  node tests/test_fa2_backward_parity.mjs

// --- Naive forward + L computation -------------------------------------------
//
// We need a forward that produces (a) attn, (b) ctx, AND (c) L, so the
// backward can be checked both ways: against the cached-attn backward
// and the recompute-from-L backward. Same algorithm as the existing
// naiveCausalAttention; this version also returns L per (b, h, t1).
function naiveCausalAttentionWithL(q, k, v, B, T, C, H) {
  const hd = C / H;
  const scale = 1 / Math.sqrt(hd);
  const attn = new Float32Array(B * H * T * T);
  const ctx = new Float32Array(B * T * C);
  const L = new Float32Array(B * H * T);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const off = h * hd;
      for (let t1 = 0; t1 < T; t1++) {
        const sc = new Float64Array(t1 + 1);
        let mx = -Infinity;
        for (let t2 = 0; t2 <= t1; t2++) {
          let s = 0;
          for (let d = 0; d < hd; d++) {
            s += q[(b * T + t1) * C + off + d] * k[(b * T + t2) * C + off + d];
          }
          s *= scale;
          sc[t2] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let t2 = 0; t2 <= t1; t2++) {
          sc[t2] = Math.exp(sc[t2] - mx);
          sum += sc[t2];
        }
        const arow = ((b * H + h) * T + t1) * T;
        for (let t2 = 0; t2 <= t1; t2++) attn[arow + t2] = sc[t2] / sum;
        for (let d = 0; d < hd; d++) {
          let acc = 0;
          for (let t2 = 0; t2 <= t1; t2++) {
            acc += attn[arow + t2] * v[(b * T + t2) * C + off + d];
          }
          ctx[(b * T + t1) * C + off + d] = acc;
        }
        // L[t1] = log-sum-exp = m_final + log(sum_of_unnormalised_exps).
        L[(b * H + h) * T + t1] = mx + Math.log(sum);
      }
    }
  }
  return { attn, ctx, L };
}

// --- Naive backward (reads cached attn). -----------------------------------
function naiveBackward(q, k, v, attn, dctx, ctx, B, T, C, H) {
  const hd = C / H;
  const scale = 1 / Math.sqrt(hd);
  const dq = new Float32Array(q.length);
  const dk = new Float32Array(k.length);
  const dv = new Float32Array(v.length);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const off = h * hd;
      // D[t1] = sum_d dO[t1,d] · O[t1,d]
      const D = new Float64Array(T);
      for (let t1 = 0; t1 < T; t1++) {
        let s = 0;
        for (let d = 0; d < hd; d++) {
          const i = (b * T + t1) * C + off + d;
          s += dctx[i] * ctx[i];
        }
        D[t1] = s;
      }
      // dS[t1, t2] then accumulate dQ, dK; dV uses P directly.
      for (let t1 = 0; t1 < T; t1++) {
        const arow = ((b * H + h) * T + t1) * T;
        for (let t2 = 0; t2 <= t1; t2++) {
          const P = attn[arow + t2];
          // dP[t1, t2] = sum_d dO[t1,d] · V[t2,d]
          let dP = 0;
          for (let d = 0; d < hd; d++) {
            dP += dctx[(b * T + t1) * C + off + d] *
                  v[(b * T + t2) * C + off + d];
          }
          const dS = P * (dP - D[t1]);
          // dQ[t1] += dS * K[t2] * scale
          for (let d = 0; d < hd; d++) {
            dq[(b * T + t1) * C + off + d] +=
              dS * k[(b * T + t2) * C + off + d] * scale;
          }
          // dK[t2] += dS * Q[t1] * scale
          for (let d = 0; d < hd; d++) {
            dk[(b * T + t2) * C + off + d] +=
              dS * q[(b * T + t1) * C + off + d] * scale;
          }
          // dV[t2] += P * dO[t1]
          for (let d = 0; d < hd; d++) {
            dv[(b * T + t2) * C + off + d] +=
              P * dctx[(b * T + t1) * C + off + d];
          }
        }
      }
    }
  }
  return { dq, dk, dv };
}

// --- FA2 backward with recompute (reads L instead of attn). ----------------
//
// Mirrors the planned WGSL kernel. For each (b, h, Q-row) we recompute
// the scores against q[t1] and every k[t2], then derive P = exp(S − L[t1])
// for the causal columns. The cached attn matrix is never touched.
function fa2BackwardRecompute(q, k, v, L, dctx, ctx, B, T, C, H) {
  const hd = C / H;
  const scale = 1 / Math.sqrt(hd);
  const dq = new Float32Array(q.length);
  const dk = new Float32Array(k.length);
  const dv = new Float32Array(v.length);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const off = h * hd;
      const D = new Float64Array(T);
      for (let t1 = 0; t1 < T; t1++) {
        let s = 0;
        for (let d = 0; d < hd; d++) {
          const i = (b * T + t1) * C + off + d;
          s += dctx[i] * ctx[i];
        }
        D[t1] = s;
      }
      for (let t1 = 0; t1 < T; t1++) {
        const Lt1 = L[(b * H + h) * T + t1];
        for (let t2 = 0; t2 <= t1; t2++) {
          // Recompute score: S = q·k * scale, with the same precision as forward.
          let S = 0;
          for (let d = 0; d < hd; d++) {
            S += q[(b * T + t1) * C + off + d] *
                 k[(b * T + t2) * C + off + d];
          }
          S *= scale;
          const P = Math.exp(S - Lt1);
          let dP = 0;
          for (let d = 0; d < hd; d++) {
            dP += dctx[(b * T + t1) * C + off + d] *
                  v[(b * T + t2) * C + off + d];
          }
          const dS = P * (dP - D[t1]);
          for (let d = 0; d < hd; d++) {
            dq[(b * T + t1) * C + off + d] +=
              dS * k[(b * T + t2) * C + off + d] * scale;
          }
          for (let d = 0; d < hd; d++) {
            dk[(b * T + t2) * C + off + d] +=
              dS * q[(b * T + t1) * C + off + d] * scale;
          }
          for (let d = 0; d < hd; d++) {
            dv[(b * T + t2) * C + off + d] +=
              P * dctx[(b * T + t1) * C + off + d];
          }
        }
      }
    }
  }
  return { dq, dk, dv };
}

// --- Test driver ------------------------------------------------------------
function rand(n, seed) {
  // Deterministic PRNG (mulberry32) so the test is reproducible.
  let s = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    out[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  }
  return out;
}

function maxAbs(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    if (e > m) m = e;
  }
  return m;
}

let fail = 0;
function check(name, ok, detail) {
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(48)} ${detail}`);
  if (!ok) fail++;
}

// Shapes mirror test_fa2_parity.mjs.
const shapes = [
  { B: 1, T: 16, C: 32, H: 4 },
  { B: 1, T: 8,  C: 32, H: 4 },
  { B: 1, T: 20, C: 32, H: 2 },
  { B: 2, T: 48, C: 24, H: 3 },
  { B: 1, T: 32, C: 64, H: 1 },
  { B: 1, T: 256, C: 64, H: 2 },
];

for (const s of shapes) {
  const { B, T, C, H } = s;
  const q = rand(B * T * C, 1);
  const k = rand(B * T * C, 2);
  const v = rand(B * T * C, 3);
  const dctx = rand(B * T * C, 4);
  const { attn, ctx, L } = naiveCausalAttentionWithL(q, k, v, B, T, C, H);

  const ref = naiveBackward(q, k, v, attn, dctx, ctx, B, T, C, H);
  const fa2 = fa2BackwardRecompute(q, k, v, L, dctx, ctx, B, T, C, H);

  const tag = `[B=${B} T=${T} C=${C} H=${H} hd=${C / H}]`;
  // Tolerance scales like K*ε ≈ T * 1e-7 because we sum T contributions
  // per output element. Allow 1e-4 absolute — well above f32 ULP noise.
  const tol = 1e-4;
  check(`fa2 dQ vs naive ${tag}`, maxAbs(ref.dq, fa2.dq) < tol,
    `maxAbs=${maxAbs(ref.dq, fa2.dq).toExponential(2)}`);
  check(`fa2 dK vs naive ${tag}`, maxAbs(ref.dk, fa2.dk) < tol,
    `maxAbs=${maxAbs(ref.dk, fa2.dk).toExponential(2)}`);
  check(`fa2 dV vs naive ${tag}`, maxAbs(ref.dv, fa2.dv) < tol,
    `maxAbs=${maxAbs(ref.dv, fa2.dv).toExponential(2)}`);
}

console.log(fail === 0 ? "\nALL PASS" : `\n${fail} FAILED`);
process.exit(fail === 0 ? 0 : 1);
