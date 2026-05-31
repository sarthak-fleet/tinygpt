// train_sg.wgsl — subgroup-using variants of the reduction-heavy kernels.
// Compiled separately from train.wgsl because `enable subgroups;` is a
// module-level directive and the base train.wgsl needs to work on devices
// that lack the feature. ops.ts dispatches into this module only when
// the device advertises the `subgroups` feature.
//
// Convention: one WORKGROUP per row / column (vs. one thread in train.wgsl).
// Each workgroup uses all 64 threads to cooperatively reduce that slice.
// On big d_model (Mega / Behemoth) this turns serial 1280-element scans
// into 20-element-per-thread scans + a single subgroupAdd — the regime
// where this lever actually pays off.
//
// Entry points (each registered in ops.ts's SG_ENTRIES list):
//   layernorm_forward_sg — row-cooperative LN (mean + variance + apply).
//   cross_entropy_sg     — row-cooperative softmax + CE + dlogits.
//   bias_grad_sg         — column-cooperative reduction over N rows.

enable subgroups;

struct P {
  a: u32, b: u32, c: u32, d: u32,
  fa: f32, fb: f32, fc: f32, fd: f32,
};

@group(0) @binding(0) var<storage, read_write> g0: array<f32>;
@group(0) @binding(1) var<storage, read_write> g1: array<f32>;
@group(0) @binding(2) var<storage, read_write> g2: array<f32>;
@group(0) @binding(3) var<storage, read_write> g3: array<f32>;
@group(0) @binding(4) var<storage, read_write> g4: array<f32>;
@group(0) @binding(5) var<storage, read_write> g5: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

// Within the workgroup, subgroups produce one partial sum each. We fold
// those down to a single value via shared memory. Max subgroups per
// workgroup of 64 is 64 (sg_size = 1) — set the array big enough for the
// worst case.
const WG: u32 = 64u;
var<workgroup> sg_partial_sum: array<f32, 64>;
var<workgroup> sg_partial_max: array<f32, 64>;
var<workgroup> wg_mu: f32;
var<workgroup> wg_rs: f32;
var<workgroup> wg_sum: f32;
var<workgroup> wg_max: f32;

// LayerNorm forward — one workgroup per row, all 64 threads collaborate.
// g0=x[N,D] g1=gamma[D] g2=beta[D] g3=y[N,D] g4=mean[N] g5=rstd[N]
// p.a=N p.b=D p.fa=eps
@compute @workgroup_size(WG)
fn layernorm_forward_sg(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let row = wid.x;
  let tid = lid.x;
  let D = p.b;
  if (row >= p.a) { return; }
  let base = row * D;
  let invD = 1.0 / f32(D);
  let nSg = (WG + sgSize - 1u) / sgSize;
  let sgId = tid / sgSize;

  // Pass 1 — sum.
  var localSum: f32 = 0.0;
  var i = tid;
  loop {
    if (i >= D) { break; }
    localSum = localSum + g0[base + i];
    i = i + WG;
  }
  let sgSum = subgroupAdd(localSum);
  if (sid == 0u) { sg_partial_sum[sgId] = sgSum; }
  workgroupBarrier();
  if (tid == 0u) {
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < nSg; k = k + 1u) { s = s + sg_partial_sum[k]; }
    wg_mu = s * invD;
  }
  workgroupBarrier();
  let mu = wg_mu;

  // Pass 2 — variance.
  var localVar: f32 = 0.0;
  i = tid;
  loop {
    if (i >= D) { break; }
    let diff = g0[base + i] - mu;
    localVar = localVar + diff * diff;
    i = i + WG;
  }
  let sgVar = subgroupAdd(localVar);
  if (sid == 0u) { sg_partial_sum[sgId] = sgVar; }
  workgroupBarrier();
  if (tid == 0u) {
    var v: f32 = 0.0;
    for (var k: u32 = 0u; k < nSg; k = k + 1u) { v = v + sg_partial_sum[k]; }
    wg_rs = 1.0 / sqrt(v * invD + p.fa);
    g4[row] = mu;
    g5[row] = wg_rs;
  }
  workgroupBarrier();
  let rs = wg_rs;

  // Pass 3 — apply.
  i = tid;
  loop {
    if (i >= D) { break; }
    g3[base + i] = g1[i] * ((g0[base + i] - mu) * rs) + g2[i];
    i = i + WG;
  }
}

// Cross-entropy with subgroup reductions over the vocab (V=256 for the
// byte-level model, so each thread handles 4 elements with WG=64). Same
// shape: per-row softmax + loss + gradient.
// g0=logits[N,V] g1=targets[N] g2=dlogits[N,V] g3=loss[N]   p.a=N p.b=V
@compute @workgroup_size(WG)
fn cross_entropy_sg(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let n = wid.x;
  let tid = lid.x;
  let V = p.b;
  if (n >= p.a) { return; }
  let base = n * V;
  let nSg = (WG + sgSize - 1u) / sgSize;
  let sgId = tid / sgSize;

  // Pass 1 — max.
  var localMax: f32 = -3.4e38;
  var i = tid;
  loop {
    if (i >= V) { break; }
    let x = g0[base + i];
    if (x > localMax) { localMax = x; }
    i = i + WG;
  }
  let sgMax = subgroupMax(localMax);
  if (sid == 0u) { sg_partial_max[sgId] = sgMax; }
  workgroupBarrier();
  if (tid == 0u) {
    var m: f32 = sg_partial_max[0];
    for (var k: u32 = 1u; k < nSg; k = k + 1u) {
      let v = sg_partial_max[k];
      if (v > m) { m = v; }
    }
    wg_max = m;
  }
  workgroupBarrier();
  let mx = wg_max;

  // Pass 2 — exp + sum.
  var localSum: f32 = 0.0;
  i = tid;
  loop {
    if (i >= V) { break; }
    localSum = localSum + exp(g0[base + i] - mx);
    i = i + WG;
  }
  let sgSum2 = subgroupAdd(localSum);
  if (sid == 0u) { sg_partial_sum[sgId] = sgSum2; }
  workgroupBarrier();
  if (tid == 0u) {
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < nSg; k = k + 1u) { s = s + sg_partial_sum[k]; }
    wg_sum = s;
    let tgt = u32(g1[n]);
    g3[n] = -((g0[base + tgt] - mx) - log(s));
  }
  workgroupBarrier();
  let sum = wg_sum;

  // Pass 3 — gradient.
  let invN = 1.0 / f32(p.a);
  let tgt = u32(g1[n]);
  i = tid;
  loop {
    if (i >= V) { break; }
    let prob = exp(g0[base + i] - mx) / sum;
    var onehot: f32 = 0.0;
    if (i == tgt) { onehot = 1.0; }
    g2[base + i] = (prob - onehot) * invN;
    i = i + WG;
  }
}

// Matmul forward — subgroup-cooperative K reduction.
//
// One workgroup outputs a ROW_TILE × COL_TILE block of C. All 64 threads in
// the workgroup cooperate on each output's K-reduction: each thread walks K
// with stride WG (= 64), accumulating partial dot-products for every output
// in the tile simultaneously. After the K loop, a subgroupAdd folds each
// output's per-lane partial within the subgroup, and a cross-subgroup
// gather (through shared memory) finalises the sum.
//
// Numerics: identical algebra to the f32 vec4 path (sum of products in
// fp32, no truncation), just different reduction tree. Order-of-summation
// differs because the partial sums collapse via subgroupAdd's
// implementation-defined tree rather than left-fold; this is the same
// reordering the existing layernorm_forward_sg / cross_entropy_sg gates
// already tolerate. The numerics gate at the end of create() catches
// real bugs (anything beyond float-reassociation noise).
//
// Why this shape:
//   - Each WG amortises 64 outputs across the 256-1024-element K reduction
//     typical of the Huge preset, where K=256 (attn projs / MLP fc_out) or
//     K=1024 (MLP fc_in). Each thread does K/64 = 4-16 MACs per output,
//     well above the per-lane work threshold where subgroupAdd dominates
//     the cost of loading B from global.
//   - ROW_TILE × COL_TILE = 64 = WG. Each thread holds exactly one f32
//     accumulator per output (a [ROW_TILE][COL_TILE] private array, all
//     threads identical) — no register spill on the Apple ISA budget.
//   - g0=A[M,K]  g1=B[K,N]  g2=C[M,N]   p.a=M  p.b=K  p.c=N
//   - Dispatch: workgroups = ceil(M/ROW_TILE) × ceil(N/COL_TILE).
//
// Speedup vs matmul_blocked_vec4: depends entirely on whether the device's
// subgroupAdd implementation is faster than the workgroup-shared-mem tile
// load. Apple Metal subgroup ops compile to native simdgroup intrinsics —
// the reduction is a single hardware instruction per stage, vs N
// instructions of shared-mem traffic in the blocked variant. Expected
// 1.3-2× on attn projections (K=256) at Huge; less at K=1024 because the
// blocked kernel's L1-resident tiles already amortise B's bandwidth there.
const MM_ROW_TILE: u32 = 4u;
const MM_COL_TILE: u32 = 16u;
// Per-output cross-subgroup partials: max 2 subgroups per WG (sgSize ≥ 32),
// indexed [output][subgroup]. 64 outputs × 2 subgroups × 4 bytes = 512 B
// of shared memory — comfortably under Apple's 16 KB threadgroup budget.
var<workgroup> mm_sg_partial: array<array<f32, 2>, 64>;

@compute @workgroup_size(WG)
fn matmul_sg(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let M = p.a; let K = p.b; let N = p.c;
  let rowBase = wid.x * MM_ROW_TILE;
  let colBase = wid.y * MM_COL_TILE;
  let tid = lid.x;
  let sgId = tid / sgSize;
  let nSg = (WG + sgSize - 1u) / sgSize;

  // Per-thread partial accumulators — one per output in the tile.
  var acc: array<array<f32, MM_COL_TILE>, MM_ROW_TILE>;
  for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
    for (var j: u32 = 0u; j < MM_COL_TILE; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  // Walk K with stride WG, accumulating MM_ROW_TILE × MM_COL_TILE products
  // simultaneously per K step. Each thread takes ceil(K/WG) elements.
  var k: u32 = tid;
  loop {
    if (k >= K) { break; }
    // Pre-load this thread's slice of A and B (one column of B, ROW_TILE
    // entries of A). Bounds check on rows / cols inside the inner loop.
    var aCol: array<f32, MM_ROW_TILE>;
    for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
      let row = rowBase + i;
      if (row < M) {
        aCol[i] = g0[row * K + k];
      } else {
        aCol[i] = 0.0;
      }
    }
    for (var j: u32 = 0u; j < MM_COL_TILE; j = j + 1u) {
      let col = colBase + j;
      var b: f32 = 0.0;
      if (col < N) { b = g1[k * N + col]; }
      for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
        acc[i][j] = acc[i][j] + aCol[i] * b;
      }
    }
    k = k + WG;
  }

  // Per-output subgroup reduction. mm_sg_partial[outIdx][sgId] receives
  // one lane's contribution from each subgroup.
  for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
    for (var j: u32 = 0u; j < MM_COL_TILE; j = j + 1u) {
      let sgSum = subgroupAdd(acc[i][j]);
      if (sid == 0u) {
        let outIdx = i * MM_COL_TILE + j;
        mm_sg_partial[outIdx][sgId] = sgSum;
      }
    }
  }
  workgroupBarrier();

  // Cross-subgroup gather + write. 64 threads → one per output.
  let outIdx = tid;
  if (outIdx < MM_ROW_TILE * MM_COL_TILE) {
    let i = outIdx / MM_COL_TILE;
    let j = outIdx % MM_COL_TILE;
    let row = rowBase + i;
    let col = colBase + j;
    if (row < M && col < N) {
      var s: f32 = 0.0;
      for (var k2: u32 = 0u; k2 < nSg; k2 = k2 + 1u) {
        s = s + mm_sg_partial[outIdx][k2];
      }
      g2[row * N + col] = s;
    }
  }
}

// matmul_abt forward — C = A @ Bᵀ with B in [N, K] layout. Same K-reduction
// strategy as matmul_sg, but B's row/col indexing flips so each lane reads
// B[col, k] not B[k, col].
//
// g0=A[M,K]  g1=B[N,K]  g2=C[M,N]   p.a=M  p.b=K  p.c=N
@compute @workgroup_size(WG)
fn matmul_abt_sg(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let M = p.a; let K = p.b; let N = p.c;
  let rowBase = wid.x * MM_ROW_TILE;
  let colBase = wid.y * MM_COL_TILE;
  let tid = lid.x;
  let sgId = tid / sgSize;
  let nSg = (WG + sgSize - 1u) / sgSize;

  var acc: array<array<f32, MM_COL_TILE>, MM_ROW_TILE>;
  for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
    for (var j: u32 = 0u; j < MM_COL_TILE; j = j + 1u) {
      acc[i][j] = 0.0;
    }
  }

  var k: u32 = tid;
  loop {
    if (k >= K) { break; }
    var aCol: array<f32, MM_ROW_TILE>;
    for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
      let row = rowBase + i;
      if (row < M) {
        aCol[i] = g0[row * K + k];
      } else {
        aCol[i] = 0.0;
      }
    }
    for (var j: u32 = 0u; j < MM_COL_TILE; j = j + 1u) {
      let col = colBase + j;
      var b: f32 = 0.0;
      // B is [N, K] in abt convention — flip row/col.
      if (col < N) { b = g1[col * K + k]; }
      for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
        acc[i][j] = acc[i][j] + aCol[i] * b;
      }
    }
    k = k + WG;
  }

  for (var i: u32 = 0u; i < MM_ROW_TILE; i = i + 1u) {
    for (var j: u32 = 0u; j < MM_COL_TILE; j = j + 1u) {
      let sgSum = subgroupAdd(acc[i][j]);
      if (sid == 0u) {
        let outIdx = i * MM_COL_TILE + j;
        mm_sg_partial[outIdx][sgId] = sgSum;
      }
    }
  }
  workgroupBarrier();

  let outIdx = tid;
  if (outIdx < MM_ROW_TILE * MM_COL_TILE) {
    let i = outIdx / MM_COL_TILE;
    let j = outIdx % MM_COL_TILE;
    let row = rowBase + i;
    let col = colBase + j;
    if (row < M && col < N) {
      var s: f32 = 0.0;
      for (var k2: u32 = 0u; k2 < nSg; k2 = k2 + 1u) {
        s = s + mm_sg_partial[outIdx][k2];
      }
      g2[row * N + col] = s;
    }
  }
}

// Bias-gradient (column reduction): db[d] = Σ_n dy[n, d].
//
// Subgroup variant: one workgroup per output column, all 64 threads
// cooperatively sum the N rows for that column. Each thread strides
// through the column at stride WG, accumulates locally, then a single
// subgroupAdd folds the partial sums per subgroup and a cross-subgroup
// reduction in shared memory produces the final value.
//
// Base train.wgsl runs this as one thread per column, serially scanning
// N rows — wins as soon as N gets big enough for parallel reduction
// throughput to dominate the per-step dispatch overhead. On Huge
// (N = B·T = 4·256 = 1024) the SG path runs ~10-15× faster on a single
// dispatch when the device supports subgroups.
//
// g0 = dy[N, D]; g1 = db[D]; p.a = N; p.b = D
@compute @workgroup_size(WG)
fn bias_grad_sg(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) sgSize: u32,
) {
  let dcol = wid.x;
  let tid = lid.x;
  let N = p.a;
  let D = p.b;
  if (dcol >= D) { return; }
  let nSg = (WG + sgSize - 1u) / sgSize;
  let sgId = tid / sgSize;

  var localSum: f32 = 0.0;
  var n = tid;
  loop {
    if (n >= N) { break; }
    localSum = localSum + g0[n * D + dcol];
    n = n + WG;
  }
  let sgSum = subgroupAdd(localSum);
  if (sid == 0u) { sg_partial_sum[sgId] = sgSum; }
  workgroupBarrier();
  if (tid == 0u) {
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < nSg; k = k + 1u) { s = s + sg_partial_sum[k]; }
    g1[dcol] = s;
  }
}
