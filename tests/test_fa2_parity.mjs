// test_fa2_parity.mjs — algorithm-level parity test for Flash Attention 2
// forward (causal). Mirrors the WGSL kernel in `webgpu/attention_fa2.wgsl`
// in plain JS, and checks its output against the naive softmax(QKᵀ/√d) V
// reference. If the algorithm is correct here, the shader (which implements
// the same algorithm with the same tile sizes) is also correct — modulo
// WebGPU bugs that would surface in the in-browser kernel parity harness
// (browser/src/webgpu-test.ts, run via `npm run webgpu-test`).
//
// Why this style instead of a live WGSL dispatch:
//   - Node doesn't have WebGPU natively; the existing project runs every
//     WGSL parity check through a headed Chromium + Playwright (see
//     tests/test_webgpu_train.mjs and browser/webgpu_test.mjs).
//   - The integration step that wires this kernel into ops.ts will live in
//     a follow-up session; that's where the live WGSL parity assertion
//     belongs (one extra `check("fa2 forward ctx", ...)` in webgpu-test.ts).
//   - This test pins the algorithm itself, which is the actual unit of
//     work being delivered here. See docs/fa2_forward_notes.md.
//
// Run:  node tests/test_fa2_parity.mjs

// --- Naive reference: exactly what `attn_fused_sv` computes. -----------------
function naiveCausalAttention(q, k, v, B, T, C, H) {
  const hd = C / H;
  const scale = 1 / Math.sqrt(hd);
  const attn = new Float32Array(B * H * T * T);
  const ctx = new Float32Array(B * T * C);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const off = h * hd;
      for (let t1 = 0; t1 < T; t1++) {
        // scores
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
        for (let t2 = 0; t2 <= t1; t2++) { sc[t2] = Math.exp(sc[t2] - mx); sum += sc[t2]; }
        const arow = ((b * H + h) * T + t1) * T;
        for (let t2 = 0; t2 <= t1; t2++) attn[arow + t2] = sc[t2] / sum;
        for (let d = 0; d < hd; d++) {
          let acc = 0;
          for (let t2 = 0; t2 <= t1; t2++) {
            acc += attn[arow + t2] * v[(b * T + t2) * C + off + d];
          }
          ctx[(b * T + t1) * C + off + d] = acc;
        }
      }
    }
  }
  return { attn, ctx };
}

// --- FA2-shaped reference: mirrors the WGSL kernel exactly. ----------------
// One workgroup per (b, h, q_tile). Inside that workgroup, BR threads each
// own one Q row; the workgroup walks K/V in BC-row blocks, doing the online
// softmax merge after each block. The post-pass second walk rewrites attn.
function fa2CausalAttention(q, k, v, B, T, C, H) {
  const hd = C / H;
  const scale = 1 / Math.sqrt(hd);
  const BR = 16, BC = 16;
  const ctx = new Float32Array(B * T * C);
  const attn = new Float32Array(B * H * T * T);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const off = h * hd;
      const nQTiles = Math.ceil(T / BR);
      for (let qt = 0; qt < nQTiles; qt++) {
        const q_start = qt * BR;
        if (q_start >= T) continue;
        const q_end = Math.min(q_start + BR - 1, T - 1);
        const nKBlocks = Math.ceil(T / BC);

        // Per-row state, simulating per-thread registers in the WGSL kernel.
        const m_i = new Array(BR).fill(-1e30);
        const l_i = new Array(BR).fill(0);
        const O_i = Array.from({ length: BR }, () => new Float64Array(hd));

        for (let kj = 0; kj < nKBlocks; kj++) {
          const k_start = kj * BC;
          if (k_start > q_end) break;     // causal: whole block past every Q row

          for (let lane = 0; lane < BR; lane++) {
            const q_row = q_start + lane;
            if (q_row >= T) continue;     // out-of-range Q lane

            // Scores for this block against q_row.
            const S = new Float64Array(BC);
            let m_block = -1e30;
            for (let jj = 0; jj < BC; jj++) {
              const t2 = k_start + jj;
              let s = -1e30;
              if (t2 < T && t2 <= q_row) {
                s = 0;
                for (let d = 0; d < hd; d++) {
                  s += q[(b * T + q_row) * C + off + d] * k[(b * T + t2) * C + off + d];
                }
                s *= scale;
              }
              S[jj] = s;
              if (s > m_block) m_block = s;
            }

            const m_new = Math.max(m_i[lane], m_block);
            if (m_new > -1e29) {
              const alpha = Math.exp(m_i[lane] - m_new);
              let l_new = alpha * l_i[lane];
              for (let d = 0; d < hd; d++) O_i[lane][d] *= alpha;
              for (let jj = 0; jj < BC; jj++) {
                if (S[jj] > -1e29) {
                  const pj = Math.exp(S[jj] - m_new);
                  l_new += pj;
                  const t2 = k_start + jj;
                  for (let d = 0; d < hd; d++) {
                    O_i[lane][d] += pj * v[(b * T + t2) * C + off + d];
                  }
                }
              }
              m_i[lane] = m_new;
              l_i[lane] = l_new;
            }
          }
        }

        // Write ctx + attn (second pass for attn, identical to what the shader does).
        for (let lane = 0; lane < BR; lane++) {
          const q_row = q_start + lane;
          if (q_row >= T) continue;
          const inv = l_i[lane] > 0 ? 1 / l_i[lane] : 0;
          const cb = (b * T + q_row) * C + off;
          for (let d = 0; d < hd; d++) ctx[cb + d] = O_i[lane][d] * inv;

          const arow = ((b * H + h) * T + q_row) * T;
          // Zero the row, then walk K blocks and recompute scores → attn.
          // (Identical pattern to the WGSL kernel's second pass.)
          for (let t2 = 0; t2 < T; t2++) attn[arow + t2] = 0;
          for (let kj = 0; kj < nKBlocks; kj++) {
            const k_start = kj * BC;
            if (k_start > q_row) break;
            for (let jj = 0; jj < BC; jj++) {
              const t2 = k_start + jj;
              if (t2 < T && t2 <= q_row) {
                let s = 0;
                for (let d = 0; d < hd; d++) {
                  s += q[(b * T + q_row) * C + off + d] * k[(b * T + t2) * C + off + d];
                }
                s *= scale;
                attn[arow + t2] = Math.exp(s - m_i[lane]) * inv;
              }
            }
          }
        }
      }
    }
  }
  return { attn, ctx };
}

// --- Driver -----------------------------------------------------------------
function rand(n, scale = 1) {
  // Deterministic seed via mulberry32 — the test should be reproducible.
  let s = 0xdeadbeef ^ n;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    a[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
    a[i] *= scale;
  }
  return a;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

let failed = 0;
function check(name, ok, detail) {
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(40)} ${detail}`);
  if (!ok) failed++;
}

const cases = [
  // small, T == multiple of 16 — exercises the cooperative-tile happy path
  { B: 1, T: 16, C: 32, H: 4 },
  // T < BR — single partial Q tile, all causal
  { B: 1, T: 8,  C: 32, H: 4 },
  // T not a multiple of 16 — exercises the boundary K block mask
  { B: 1, T: 20, C: 32, H: 2 },
  // larger T (small-preset shape): exercises multi-block walk
  { B: 2, T: 48, C: 24, H: 3 },
  // Behemoth-shaped (hd = 64 = MAX_HD), small B/T to keep the test cheap
  { B: 1, T: 32, C: 64, H: 1 },
  // ctx=256 — the regime where FA2 starts to actually pay off
  { B: 1, T: 256, C: 64, H: 2 },
];

// Tolerance: 1e-4 absolute. FA2 reorders the softmax sum across blocks vs
// the naive single-pass softmax, so the result differs in the last few ulps
// of f32. 1e-4 absorbs that without admitting real bugs.
const TOL = 1e-4;

for (const { B, T, C, H } of cases) {
  const N = B * T * C;
  const q = rand(N), k = rand(N), v = rand(N);
  const ref = naiveCausalAttention(q, k, v, B, T, C, H);
  const got = fa2CausalAttention(q, k, v, B, T, C, H);
  const dCtx  = maxAbsDiff(ref.ctx,  got.ctx);
  const dAttn = maxAbsDiff(ref.attn, got.attn);
  const label = `B=${B} T=${T} C=${C} H=${H} hd=${C / H}`;
  check(`fa2 ctx   [${label}]`,  dCtx  < TOL, `maxAbsDiff=${dCtx.toExponential(2)}`);
  check(`fa2 attn  [${label}]`,  dAttn < TOL, `maxAbsDiff=${dAttn.toExponential(2)}`);
}

if (failed === 0) {
  console.log("\nALL PASS");
  process.exit(0);
} else {
  console.log(`\n${failed} test(s) FAILED`);
  process.exit(1);
}
