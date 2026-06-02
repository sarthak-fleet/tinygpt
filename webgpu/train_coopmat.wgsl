// train_coopmat.wgsl — matmul via WGSL `chromium_experimental_subgroup_matrix`.
//
// Maps to hardware MMA units: NVIDIA tensor cores, AMD MFMA, Apple AMX /
// simdgroup matrices. One subgroup-cooperative call replaces an 8×8 tile of
// scalar multiply-adds, with the matrix tiles distributed across subgroup
// registers.
//
// The extension's API surface is documented in the WebGPU CTS at
// https://github.com/gpuweb/cts and the Chromium tint implementation. The
// form used here is the current (mid-2026) shape:
//
//   subgroup_matrix_left<f16, M, K>   — left operand
//   subgroup_matrix_right<f16, K, N>  — right operand
//   subgroup_matrix_result<f32, M, N> — accumulator
//   subgroupMatrixLoad<T>(ptr, col_major, stride)
//   subgroupMatrixStore(ptr, matrix, col_major, stride)
//   subgroupMatrixMultiplyAccumulate(left, right, result) -> result
//
// Tile size: 8×8×8 — the smallest size supported on every documented vendor
// (Apple simdgroup_matrix is fixed at 8×8 for half-precision; NVIDIA mma at
// 16×16 accepts 8×8 via padding). 8×8 is the safe minimum.
//
// Numerical expectation: same as the shader-f16 compute path — f16
// multiplies, f32 accumulator, √K × eps_f16 K-direction drift. The gate in
// ops.ts uses the same magnitude-aware tolerance as verifyShaderF16Compute.
//
// Defensive design: if the API form here doesn't match what the running
// Chrome ships, the module fails to compile and the gate stays inactive —
// matmul() falls back to f16-compute / f16-storage / vec4 unchanged. Zero
// regression risk to existing paths.

enable chromium_experimental_subgroup_matrix;

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

// Per-workgroup staging buffers. Each workgroup owns one 8×8 output tile.
// 32 threads cooperate to stage the A and B input tiles + scatter the
// accumulator back to global memory. K_tile = 8 matches the inner matrix
// dimension so each K-step does one MultiplyAccumulate.
var<workgroup> cm_tileA: array<f16, 64>;  // 8 × 8 (row-major)
var<workgroup> cm_tileB: array<f16, 64>;  // 8 × 8 (row-major)
var<workgroup> cm_tileC: array<f32, 64>;  // 8 × 8 (row-major) — for store

@compute @workgroup_size(32, 1, 1)
fn matmul_blocked_coopmat(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = p.a; let K = p.b; let N = p.c;
  let halfN = N / 2u;
  let blockRow = wid.x * 8u;
  let blockCol = wid.y * 8u;
  let tid = lid.x;  // 0..31

  // f32 accumulator matrix; default-init is zero.
  var acc: subgroup_matrix_result<f32, 8, 8>;

  let nTiles = (K + 7u) / 8u;
  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kBase = t * 8u;

    // 32 threads write 2 entries each into the 8×8 = 64-entry tiles.
    for (var pair: u32 = 0u; pair < 2u; pair = pair + 1u) {
      let idx = tid * 2u + pair;
      let r = idx / 8u;
      let c = idx % 8u;
      // A: f32 → f16.
      let aRow = blockRow + r;
      let aCol = kBase + c;
      var aVal: f16 = 0.0h;
      if (aRow < M && aCol < K) {
        aVal = f16(g0[aRow * K + aCol]);
      }
      cm_tileA[r * 8u + c] = aVal;
      // B: unpacked-f16 along N axis.
      let bRow = kBase + r;
      let bCol = blockCol + c;
      var bVal: f16 = 0.0h;
      if (bRow < K && bCol < N) {
        let bWordCol = bCol / 2u;
        let bIsHigh = (bCol & 1u) == 1u;
        let bPair = unpack2x16float(g1[bRow * halfN + bWordCol]);
        bVal = f16(select(bPair.x, bPair.y, bIsHigh));
      }
      cm_tileB[r * 8u + c] = bVal;
    }
    workgroupBarrier();

    // Load the 8×8 tiles into subgroup-matrix registers, then MMA.
    let aMat = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(
      &cm_tileA, false, 8u);
    let bMat = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(
      &cm_tileB, false, 8u);
    acc = subgroupMatrixMultiplyAccumulate(aMat, bMat, acc);
    workgroupBarrier();
  }

  // Scatter the 8×8 accumulator to the workgroup-shared scratch buffer,
  // then 32 threads each write 2 entries to g2.
  subgroupMatrixStore(&cm_tileC, acc, false, 8u);
  workgroupBarrier();
  for (var pair: u32 = 0u; pair < 2u; pair = pair + 1u) {
    let idx = tid * 2u + pair;
    let r = idx / 8u;
    let c = idx % 8u;
    let outRow = blockRow + r;
    let outCol = blockCol + c;
    if (outRow < M && outCol < N) {
      g2[outRow * N + outCol] = cm_tileC[r * 8u + c];
    }
  }
}
