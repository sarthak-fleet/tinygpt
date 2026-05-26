// matmul_blocked_vec4.wgsl — same blocked4 algorithm as matmul_blocked.wgsl
// but with vec4<f32> global memory loads. Standard CUDA / Metal optimization:
// issue 128-bit memory transactions instead of 32-bit, fewer load instructions
// per K-tile, better memory-bandwidth utilization.
//
// Apple's Metal compiler often coalesces adjacent f32 loads from adjacent
// threads into wider transactions, so the win can be smaller than on NVIDIA
// where vec4 is explicit. Benchmark will tell us.
//
// Layout requirements:
//   K and N must be multiples of 4 (vec4 alignment along the contiguous axis).
//   All preset matmul shapes already satisfy this (d_model values 96, 128,
//   192, 256, 384, 1280 all div-by-4).
//
// Bind layout / dispatch geometry unchanged from matmul_blocked.wgsl:
//   workgroup_size(16, 16); dispatch ceil(M/64) × ceil(N/64) workgroups.

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const BK4: u32 = 4u;  // BK / 4 — number of vec4s per K-tile row

struct Dims { M: u32, K: u32, N: u32, _pad: u32 };

// A and B viewed as vec4 arrays — each row contributes K/4 vec4s.
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<array<f32, 16>, 64>;
var<workgroup> tileB: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = dims.M;
  let K = dims.K;
  let N = dims.N;
  let K4 = K / 4u;  // vec4s per row of A
  let N4 = N / 4u;  // vec4s per row of B

  let blockRow = wid.x * BM;
  let blockCol = wid.y * BN;
  let lrow = lid.x;
  let lcol = lid.y;

  var acc: array<array<f32, 4>, 4>;
  for (var i: u32 = 0u; i < TM; i = i + 1u) {
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  let nTiles = (K + BK - 1u) / BK;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kBase = t * BK;
    let kBase4 = t * BK4;  // vec4 offset into A's row / B's row dimension

    // Load A: 64 rows × 16 cols = 64 rows × 4 vec4s. 256 threads / 4 vec4s
    // per row = 4 rows per thread group of 4 lcol — but simpler pattern:
    // each thread loads ONE vec4 = 4 contiguous f32 values into tileA.
    // lcol ∈ 0..15, so lcol/4 picks one of 4 vec4 positions, lcol%4 unused.
    //
    // Pattern: For i in 0..4, each thread loads tileA row (lrow*4 + i),
    // and vec4 column (lcol/4). Only threads with lcol%4 == 0 do the load;
    // the others wait. That wastes 3/4 of threads on load — not great.
    //
    // Better: all 256 threads cooperate. 64 rows × 4 vec4s = 256 vec4s.
    // Each thread loads exactly one vec4 (one i, one row, one col-vec4).
    // Thread (lrow, lcol) loads tileA[lcol/4 + (lrow*4) * 4 / 4][?] ... no,
    // simpler: linearise tid = lrow*16 + lcol. tid in 0..255 maps to:
    //   row = tid / 4   (0..63)
    //   col4 = tid % 4  (0..3)  → f32 columns 4*col4 .. 4*col4+3
    let tid = lrow * 16u + lcol;
    {
      let row = tid / 4u;            // 0..63
      let col4 = tid % 4u;            // 0..3 — vec4 offset within the 16-tile
      let aRow = blockRow + row;
      if (aRow < M && kBase + col4 * 4u < K) {
        let v = A[aRow * K4 + kBase4 + col4];
        tileA[row][col4 * 4u + 0u] = v.x;
        tileA[row][col4 * 4u + 1u] = v.y;
        tileA[row][col4 * 4u + 2u] = v.z;
        tileA[row][col4 * 4u + 3u] = v.w;
      } else {
        tileA[row][col4 * 4u + 0u] = 0.0;
        tileA[row][col4 * 4u + 1u] = 0.0;
        tileA[row][col4 * 4u + 2u] = 0.0;
        tileA[row][col4 * 4u + 3u] = 0.0;
      }
    }

    // Load B: 16 rows × 64 cols = 16 rows × 16 vec4s = 256 vec4s.
    // Thread tid maps to:  row = tid / 16, col4 = tid % 16.
    {
      let row = tid / 16u;            // 0..15
      let col4 = tid % 16u;            // 0..15 — vec4 column within the tile
      let bRow = kBase + row;
      let bCol4 = blockCol / 4u + col4;
      if (bRow < K && bCol4 < N4) {
        let v = B[bRow * N4 + bCol4];
        tileB[row][col4 * 4u + 0u] = v.x;
        tileB[row][col4 * 4u + 1u] = v.y;
        tileB[row][col4 * 4u + 2u] = v.z;
        tileB[row][col4 * 4u + 3u] = v.w;
      } else {
        tileB[row][col4 * 4u + 0u] = 0.0;
        tileB[row][col4 * 4u + 1u] = 0.0;
        tileB[row][col4 * 4u + 2u] = 0.0;
        tileB[row][col4 * 4u + 3u] = 0.0;
      }
    }

    workgroupBarrier();

    // Inner K-tile loop — identical to matmul_blocked.wgsl from this point.
    let myA0 = lrow * TM;
    let myB0 = lcol * TN;
    for (var k: u32 = 0u; k < BK; k = k + 1u) {
      let a0 = tileA[myA0 + 0u][k];
      let a1 = tileA[myA0 + 1u][k];
      let a2 = tileA[myA0 + 2u][k];
      let a3 = tileA[myA0 + 3u][k];
      let b0 = tileB[k][myB0 + 0u];
      let b1 = tileB[k][myB0 + 1u];
      let b2 = tileB[k][myB0 + 2u];
      let b3 = tileB[k][myB0 + 3u];
      acc[0][0] = acc[0][0] + a0 * b0; acc[0][1] = acc[0][1] + a0 * b1;
      acc[0][2] = acc[0][2] + a0 * b2; acc[0][3] = acc[0][3] + a0 * b3;
      acc[1][0] = acc[1][0] + a1 * b0; acc[1][1] = acc[1][1] + a1 * b1;
      acc[1][2] = acc[1][2] + a1 * b2; acc[1][3] = acc[1][3] + a1 * b3;
      acc[2][0] = acc[2][0] + a2 * b0; acc[2][1] = acc[2][1] + a2 * b1;
      acc[2][2] = acc[2][2] + a2 * b2; acc[2][3] = acc[2][3] + a2 * b3;
      acc[3][0] = acc[3][0] + a3 * b0; acc[3][1] = acc[3][1] + a3 * b1;
      acc[3][2] = acc[3][2] + a3 * b2; acc[3][3] = acc[3][3] + a3 * b3;
    }

    workgroupBarrier();
  }

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
