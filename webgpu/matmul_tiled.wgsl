// matmul_tiled.wgsl — classic 16×16 tiled matmul using workgroup-shared
// memory. The naive matmul reads each A and B value `size` times from
// global memory; this version stages them through fast workgroup memory so
// each load is amortized across 16 multiplications. On big matrices that
// turns matmul from bandwidth-bound to compute-bound, which is where the
// GPU starts looking like a GPU.
//
// Algorithm (textbook Goto/VandeGeijn flavour):
//   1. Each workgroup is 16×16; it computes one 16×16 tile of C.
//   2. K is walked in tiles of 16.
//   3. For each K-tile, every thread cooperatively loads one element of A's
//      16×16 block and one element of B's 16×16 block into shared memory.
//   4. Workgroup barrier; then each thread does 16 multiply-accumulates from
//      the shared tiles into its private accumulator.
//   5. Move to next K-tile.
//
// Buffers + dims identical to matmul.wgsl so the host can swap kernels
// without changing bind-group layout.

const TILE: u32 = 16u;

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

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.x;
  let col = gid.y;
  let lrow = lid.x;
  let lcol = lid.y;

  let M = dims.M;
  let K = dims.K;
  let N = dims.N;

  var acc: f32 = 0.0;

  // Number of K-tiles to walk; ceil(K / 16).
  let nTiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    // Each thread loads ONE element of A and ONE element of B.
    let aCol = t * TILE + lcol;
    let bRow = t * TILE + lrow;

    if (row < M && aCol < K) {
      tileA[lrow][lcol] = A[row * K + aCol];
    } else {
      tileA[lrow][lcol] = 0.0;
    }
    if (bRow < K && col < N) {
      tileB[lrow][lcol] = B[bRow * N + col];
    } else {
      tileB[lrow][lcol] = 0.0;
    }

    workgroupBarrier();

    // Inner product across this K-tile, reading only from shared memory.
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + tileA[lrow][k] * tileB[k][lcol];
    }

    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}
