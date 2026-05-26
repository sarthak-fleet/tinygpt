// matmul_blocked8.wgsl — same tiled+blocked pattern as matmul_blocked.wgsl,
// scaled up to 8×8 per-thread register blocks.
//
//   workgroup     = 16×16 threads (256 total, same as before)
//   per-thread    = 8×8 output block (was 4×4)
//   per-workgroup = 128×128 output tile (was 64×64)
//   K-tile        = 16 (unchanged)
//
// Each shared-memory load now amortizes across 64 multiply-accumulates
// (vs 16 in the 4×4 version). Per-thread register pressure climbs from
// ~16 floats to ~64 floats for the accumulator — still well under Apple
// GPU per-thread register budgets.
//
// Workgroup-shared memory: A (128×16) + B (16×128) = 8 KB + 8 KB = 16 KB.
// Apple per-workgroup limit is 32 KB, so this fits.

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 16u;
const TM: u32 = 8u;
const TN: u32 = 8u;

struct Dims { M: u32, K: u32, N: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<array<f32, 16>, 128>;
var<workgroup> tileB: array<array<f32, 128>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = dims.M;
  let K = dims.K;
  let N = dims.N;

  let blockRow = wid.x * BM;
  let blockCol = wid.y * BN;
  let lrow = lid.x;
  let lcol = lid.y;

  var acc: array<array<f32, 8>, 8>;
  for (var i: u32 = 0u; i < TM; i = i + 1u) {
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  let nTiles = (K + BK - 1u) / BK;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kBase = t * BK;

    // Cooperative load A: 128 rows × 16 cols = 2048 elements / 256 threads
    // = 8 elements per thread.
    for (var i: u32 = 0u; i < TM; i = i + 1u) {
      let aRow = blockRow + lrow * TM + i;
      let aCol = kBase + lcol;
      var v: f32 = 0.0;
      if (aRow < M && aCol < K) { v = A[aRow * K + aCol]; }
      tileA[lrow * TM + i][lcol] = v;
    }

    // Cooperative load B: 16 rows × 128 cols. 8 elements per thread.
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      let bRow = kBase + lrow;
      let bCol = blockCol + lcol * TN + j;
      var v: f32 = 0.0;
      if (bRow < K && bCol < N) { v = B[bRow * N + bCol]; }
      tileB[lrow][lcol * TN + j] = v;
    }

    workgroupBarrier();

    // Inner K-tile loop. Outer-product structure into 8×8 register block.
    let myA0 = lrow * TM;
    let myB0 = lcol * TN;
    for (var k: u32 = 0u; k < BK; k = k + 1u) {
      // Load 8 A-values into registers.
      let a0 = tileA[myA0 + 0u][k];
      let a1 = tileA[myA0 + 1u][k];
      let a2 = tileA[myA0 + 2u][k];
      let a3 = tileA[myA0 + 3u][k];
      let a4 = tileA[myA0 + 4u][k];
      let a5 = tileA[myA0 + 5u][k];
      let a6 = tileA[myA0 + 6u][k];
      let a7 = tileA[myA0 + 7u][k];
      // Load 8 B-values into registers.
      let b0 = tileB[k][myB0 + 0u];
      let b1 = tileB[k][myB0 + 1u];
      let b2 = tileB[k][myB0 + 2u];
      let b3 = tileB[k][myB0 + 3u];
      let b4 = tileB[k][myB0 + 4u];
      let b5 = tileB[k][myB0 + 5u];
      let b6 = tileB[k][myB0 + 6u];
      let b7 = tileB[k][myB0 + 7u];
      // 8×8 = 64 fused multiply-adds.
      acc[0][0] = acc[0][0] + a0 * b0; acc[0][1] = acc[0][1] + a0 * b1;
      acc[0][2] = acc[0][2] + a0 * b2; acc[0][3] = acc[0][3] + a0 * b3;
      acc[0][4] = acc[0][4] + a0 * b4; acc[0][5] = acc[0][5] + a0 * b5;
      acc[0][6] = acc[0][6] + a0 * b6; acc[0][7] = acc[0][7] + a0 * b7;
      acc[1][0] = acc[1][0] + a1 * b0; acc[1][1] = acc[1][1] + a1 * b1;
      acc[1][2] = acc[1][2] + a1 * b2; acc[1][3] = acc[1][3] + a1 * b3;
      acc[1][4] = acc[1][4] + a1 * b4; acc[1][5] = acc[1][5] + a1 * b5;
      acc[1][6] = acc[1][6] + a1 * b6; acc[1][7] = acc[1][7] + a1 * b7;
      acc[2][0] = acc[2][0] + a2 * b0; acc[2][1] = acc[2][1] + a2 * b1;
      acc[2][2] = acc[2][2] + a2 * b2; acc[2][3] = acc[2][3] + a2 * b3;
      acc[2][4] = acc[2][4] + a2 * b4; acc[2][5] = acc[2][5] + a2 * b5;
      acc[2][6] = acc[2][6] + a2 * b6; acc[2][7] = acc[2][7] + a2 * b7;
      acc[3][0] = acc[3][0] + a3 * b0; acc[3][1] = acc[3][1] + a3 * b1;
      acc[3][2] = acc[3][2] + a3 * b2; acc[3][3] = acc[3][3] + a3 * b3;
      acc[3][4] = acc[3][4] + a3 * b4; acc[3][5] = acc[3][5] + a3 * b5;
      acc[3][6] = acc[3][6] + a3 * b6; acc[3][7] = acc[3][7] + a3 * b7;
      acc[4][0] = acc[4][0] + a4 * b0; acc[4][1] = acc[4][1] + a4 * b1;
      acc[4][2] = acc[4][2] + a4 * b2; acc[4][3] = acc[4][3] + a4 * b3;
      acc[4][4] = acc[4][4] + a4 * b4; acc[4][5] = acc[4][5] + a4 * b5;
      acc[4][6] = acc[4][6] + a4 * b6; acc[4][7] = acc[4][7] + a4 * b7;
      acc[5][0] = acc[5][0] + a5 * b0; acc[5][1] = acc[5][1] + a5 * b1;
      acc[5][2] = acc[5][2] + a5 * b2; acc[5][3] = acc[5][3] + a5 * b3;
      acc[5][4] = acc[5][4] + a5 * b4; acc[5][5] = acc[5][5] + a5 * b5;
      acc[5][6] = acc[5][6] + a5 * b6; acc[5][7] = acc[5][7] + a5 * b7;
      acc[6][0] = acc[6][0] + a6 * b0; acc[6][1] = acc[6][1] + a6 * b1;
      acc[6][2] = acc[6][2] + a6 * b2; acc[6][3] = acc[6][3] + a6 * b3;
      acc[6][4] = acc[6][4] + a6 * b4; acc[6][5] = acc[6][5] + a6 * b5;
      acc[6][6] = acc[6][6] + a6 * b6; acc[6][7] = acc[6][7] + a6 * b7;
      acc[7][0] = acc[7][0] + a7 * b0; acc[7][1] = acc[7][1] + a7 * b1;
      acc[7][2] = acc[7][2] + a7 * b2; acc[7][3] = acc[7][3] + a7 * b3;
      acc[7][4] = acc[7][4] + a7 * b4; acc[7][5] = acc[7][5] + a7 * b5;
      acc[7][6] = acc[7][6] + a7 * b6; acc[7][7] = acc[7][7] + a7 * b7;
    }

    workgroupBarrier();
  }

  // Write the 8×8 output block. Guard the ragged edge.
  for (var i: u32 = 0u; i < TM; i = i + 1u) {
    let outRow = blockRow + lrow * TM + i;
    if (outRow >= M) { continue; }
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      let outCol = blockCol + lcol * TN + j;
      if (outCol < N) {
        C[outRow * N + outCol] = acc[i][j];
      }
    }
  }
}
