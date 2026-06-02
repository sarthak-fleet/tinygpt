// train_f16_compute.wgsl — matmul variant with f16 SHARED-MEMORY tiles and
// f16-MULTIPLY + f32-ACCUMULATE inner loop. Requires the `shader-f16`
// adapter feature (Chrome 121+ stable). Compiled separately because
// `enable f16;` validates only on devices that advertise the extension.
//
// What this kernel adds on top of `matmul_blocked_f16` (the f16-storage
// path in train_f16.wgsl):
//   - shared tiles stored as f16 (halves workgroup-shared bandwidth)
//   - inner multiply runs in f16 (one Metal-half-MAC instead of FP32 MAC)
//   - accumulator stays f32 so the K-direction sum keeps full precision
//
// Numerical expectation: the per-step rounding error of one f16 × f16
// product is bounded by eps_f16 ≈ 5e-4 of the operand magnitude. Summed
// over K random-sign terms in f32 the error grows like √K × 5e-4 —
// the same √K factor as the storage-only path, because the accumulator
// is still f32. The numerics gate in ops.ts validates this empirically;
// in practice the f16-compute gate is roughly equivalent in tolerance
// to the f16-storage one (we're not adding any new lossy accumulation).
//
// Speedup expectation on Apple M-series: ~1.2× on top of the f16-storage
// path. The big win is halved shared-memory bandwidth + faster f16 MACs;
// gains compound less than 2× because the compute is already
// register-blocked at 4×4 and shared-memory bandwidth isn't the only
// limit.
//
// Layout requirements: same as matmul_blocked_f16 — K and N even.

enable f16;

struct P {
  a: u32, b: u32, c: u32, d: u32,
  fa: f32, fb: f32, fc: f32, fd: f32,
};

@group(0) @binding(0) var<storage, read_write> g0: array<f32>;      // A [M,K] f32
@group(0) @binding(1) var<storage, read_write> g1: array<u32>;      // B [K,N] packed f16
@group(0) @binding(2) var<storage, read_write> g2: array<f32>;      // C [M,N] f32
@group(0) @binding(3) var<storage, read_write> g3: array<f32>;
@group(0) @binding(4) var<storage, read_write> g4: array<f32>;
@group(0) @binding(5) var<storage, read_write> g5: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

// Shared tiles in f16 — halved bandwidth + matches the inner multiply
// precision. K-tile depth = 16, output tile = 64×64 (16×16 workgroup,
// 4×4 register block per thread), same as matmul_blocked_f16.
var<workgroup> mc_tileA: array<array<f16, 16>, 64>;
var<workgroup> mc_tileB: array<array<f16, 64>, 16>;

@compute @workgroup_size(16, 16)
fn matmul_blocked_f16_compute(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;
  let halfN = N / 2u;
  let blockRow = wid.x * 64u;
  let blockCol = wid.y * 64u;
  let lrow = lid.x; let lcol = lid.y;
  let tid = lrow * 16u + lcol;

  // f32 accumulators — keep the K-direction sum in f32 so √K × eps_f16
  // is the only drift source. With K=256 that's ~8e-3 RMS, well under
  // the 0.5% mean-rel gate.
  var acc: array<array<f32, 4>, 4>;
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  let nTiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kBase = t * 16u;

    // Load A — convert f32 → f16 on load into shared memory.
    {
      let row = tid / 16u;
      let col = tid % 16u;
      let kCol = kBase + col;
      for (var rb: u32 = 0u; rb < 4u; rb = rb + 1u) {
        let tileR = row + rb * 16u;
        let aRow = blockRow + tileR;
        var v: f32 = 0.0;
        if (aRow < M && kCol < K) {
          v = g0[aRow * K + kCol];
        }
        mc_tileA[tileR][col] = f16(v);
      }
    }

    // Load B — unpack packed-f16 storage directly into the f16 shared
    // tile (no f32 detour). `unpack2x16float` returns vec2<f32>; cast the
    // selected component back to f16 for storage.
    {
      let row = tid / 16u;
      let col = tid % 16u;
      let bRow = kBase + row;
      for (var cb: u32 = 0u; cb < 4u; cb = cb + 1u) {
        let tileC = col + cb * 16u;
        let bCol = blockCol + tileC;
        var v: f16 = 0.0h;
        if (bRow < K && bCol < N) {
          let bWordCol = bCol / 2u;
          let bIsHigh = (bCol & 1u) == 1u;
          let pair = unpack2x16float(g1[bRow * halfN + bWordCol]);
          v = f16(select(pair.x, pair.y, bIsHigh));
        }
        mc_tileB[row][tileC] = v;
      }
    }
    workgroupBarrier();

    // Inner loop: f16 multiplies, f32 accumulates. The cast from
    // `a * b` (f16) to f32 is the standard "mixed-precision MAC" pattern.
    let myA0 = lrow * 4u;
    let myB0 = lcol * 4u;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let a0 = mc_tileA[myA0 + 0u][k];
      let a1 = mc_tileA[myA0 + 1u][k];
      let a2 = mc_tileA[myA0 + 2u][k];
      let a3 = mc_tileA[myA0 + 3u][k];
      let b0 = mc_tileB[k][myB0 + 0u];
      let b1 = mc_tileB[k][myB0 + 1u];
      let b2 = mc_tileB[k][myB0 + 2u];
      let b3 = mc_tileB[k][myB0 + 3u];
      acc[0][0] = acc[0][0] + f32(a0 * b0); acc[0][1] = acc[0][1] + f32(a0 * b1);
      acc[0][2] = acc[0][2] + f32(a0 * b2); acc[0][3] = acc[0][3] + f32(a0 * b3);
      acc[1][0] = acc[1][0] + f32(a1 * b0); acc[1][1] = acc[1][1] + f32(a1 * b1);
      acc[1][2] = acc[1][2] + f32(a1 * b2); acc[1][3] = acc[1][3] + f32(a1 * b3);
      acc[2][0] = acc[2][0] + f32(a2 * b0); acc[2][1] = acc[2][1] + f32(a2 * b1);
      acc[2][2] = acc[2][2] + f32(a2 * b2); acc[2][3] = acc[2][3] + f32(a2 * b3);
      acc[3][0] = acc[3][0] + f32(a3 * b0); acc[3][1] = acc[3][1] + f32(a3 * b1);
      acc[3][2] = acc[3][2] + f32(a3 * b2); acc[3][3] = acc[3][3] + f32(a3 * b3);
    }
    workgroupBarrier();
  }

  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let outRow = blockRow + lrow * 4u + i;
    if (outRow >= M) { continue; }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let outCol = blockCol + lcol * 4u + j;
      if (outCol < N) {
        g2[outRow * N + outCol] = acc[i][j];
      }
    }
  }
}
