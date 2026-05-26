// matmul_blocked.wgsl — tiled + thread-blocked matmul.
//
// Stacks two well-known wins:
//   1. Workgroup-shared tiling (same as matmul_tiled.wgsl) — load a 64×16
//      block of A and a 16×64 block of B into shared memory, then do all the
//      multiply-accumulates for the 64×64 output tile from shared.
//   2. Thread blocking — each of the 16×16 threads in the workgroup computes
//      a 4×4 block of output, held in registers. Outer-product structure:
//      every shared-memory load is reused 4× across the thread's accumulator,
//      so the inner-loop arithmetic intensity goes from 1 mul-add per global
//      read to ~16. This is where matmul actually becomes compute-bound.
//
// Layout:
//   Workgroup = 16×16 threads (= 256)
//   Each thread computes a 4×4 output block, so workgroup output = 64×64
//   K is walked in tiles of 16; per K-tile, the workgroup loads
//   A[64×16] and B[16×64] cooperatively (4 elements per thread).
//
// Register budget per thread: 4×4 = 16 f32s for the accumulator, plus a few
// for the loop. Within Apple GPU per-thread register limits.

const BM: u32 = 64u;  // output tile M
const BN: u32 = 64u;  // output tile N
const BK: u32 = 16u;  // K tile
const TM: u32 = 4u;   // per-thread M block
const TN: u32 = 4u;   // per-thread N block

struct Dims {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<array<f32, 16>, 64>; // [BM][BK]
var<workgroup> tileB: array<array<f32, 64>, 16>; // [BK][BN]

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = dims.M;
  let K = dims.K;
  let N = dims.N;

  let blockRow = wid.x * BM;  // first row of this workgroup's output tile
  let blockCol = wid.y * BN;
  let lrow = lid.x;            // 0..15
  let lcol = lid.y;            // 0..15
  let tid = lrow * 16u + lcol; // 0..255 — for cooperative loads

  // Per-thread 4×4 register accumulator (initialised to zero).
  var acc: array<array<f32, 4>, 4>;
  for (var i: u32 = 0u; i < TM; i = i + 1u) {
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  let nTiles = (K + BK - 1u) / BK;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kBase = t * BK;

    // Cooperative load of A: 64 rows × 16 cols = 1024 elements / 256 threads
    // = 4 elements per thread. Layout: each thread loads positions
    //   (lrow*4 + i, lcol) for i in 0..4.
    for (var i: u32 = 0u; i < TM; i = i + 1u) {
      let aRow = blockRow + lrow * TM + i;
      let aCol = kBase + lcol;
      var v: f32 = 0.0;
      if (aRow < M && aCol < K) { v = A[aRow * K + aCol]; }
      tileA[lrow * TM + i][lcol] = v;
    }

    // Cooperative load of B: 16 rows × 64 cols. 4 elements per thread:
    //   tileB[lrow][lcol*4 + j] for j in 0..4
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      let bRow = kBase + lrow;
      let bCol = blockCol + lcol * TN + j;
      var v: f32 = 0.0;
      if (bRow < K && bCol < N) { v = B[bRow * N + bCol]; }
      tileB[lrow][lcol * TN + j] = v;
    }

    workgroupBarrier();

    // Inner K-tile loop. The outer-product structure is the trick: load
    // 4 A-values and 4 B-values into registers, then do 16 fused mul-adds
    // that all reuse those 8 loaded values. Arithmetic intensity per
    // shared-mem load: 4 multiply-accumulates instead of 1.
    let myA_row0 = lrow * TM;
    let myB_col0 = lcol * TN;
    for (var k: u32 = 0u; k < BK; k = k + 1u) {
      // Load the 4 A-values this thread needs from shared into registers.
      var a0 = tileA[myA_row0 + 0u][k];
      var a1 = tileA[myA_row0 + 1u][k];
      var a2 = tileA[myA_row0 + 2u][k];
      var a3 = tileA[myA_row0 + 3u][k];
      // Load the 4 B-values into registers.
      var b0 = tileB[k][myB_col0 + 0u];
      var b1 = tileB[k][myB_col0 + 1u];
      var b2 = tileB[k][myB_col0 + 2u];
      var b3 = tileB[k][myB_col0 + 3u];
      // Outer product into the 4×4 accumulator.
      acc[0][0] = acc[0][0] + a0 * b0;
      acc[0][1] = acc[0][1] + a0 * b1;
      acc[0][2] = acc[0][2] + a0 * b2;
      acc[0][3] = acc[0][3] + a0 * b3;
      acc[1][0] = acc[1][0] + a1 * b0;
      acc[1][1] = acc[1][1] + a1 * b1;
      acc[1][2] = acc[1][2] + a1 * b2;
      acc[1][3] = acc[1][3] + a1 * b3;
      acc[2][0] = acc[2][0] + a2 * b0;
      acc[2][1] = acc[2][1] + a2 * b1;
      acc[2][2] = acc[2][2] + a2 * b2;
      acc[2][3] = acc[2][3] + a2 * b3;
      acc[3][0] = acc[3][0] + a3 * b0;
      acc[3][1] = acc[3][1] + a3 * b1;
      acc[3][2] = acc[3][2] + a3 * b2;
      acc[3][3] = acc[3][3] + a3 * b3;
    }

    workgroupBarrier();
  }

  // Write the 4×4 block of output. Guard the ragged edge.
  for (var i: u32 = 0u; i < TM; i = i + 1u) {
    let outRow = blockRow + lrow * TM + i;
    if (outRow >= M) { continue; }
    let outBase = outRow * N + blockCol + lcol * TN;
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      let outCol = blockCol + lcol * TN + j;
      if (outCol < N) {
        C[outBase + j] = acc[i][j];
      }
    }
  }
}
