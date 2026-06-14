// matmul.cpp — matrix multiply, forward + backward (Phase 4 + threading pass).
//
// The single most performance-critical kernel. The inner loop is the original
// "iko" order — emcc autovectorises this cleanly with -msimd128. Threading
// splits the M dimension across CPU cores via std::thread.
//
//   Forward:   C = A @ B            A:[M,K]  B:[K,N]  C:[M,N]
//   Backward:  dA = dC @ B^T        dB = A^T @ dC
//
// Used by: Linear layers, the attention projections, the output head.
//
// For matmul_forward and the dA path, every output row is independent —
// each thread gets a contiguous slice of M and writes only to its own region.
// dB accumulates over M so we use per-thread scratch and a final reduction.
//
// Threading kicks in only when M is large enough that thread-creation
// overhead is amortised (M >= 64). Below that, single-threaded.
//
// Guide: docs/browser_notes.md ("WASM backend", "WebGPU acceleration")

#include "kernels.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// Cap pthread count to match the WASM build's PTHREAD_POOL_SIZE.
static constexpr int MAX_THREADS = 8;
// Threshold below which threading overhead exceeds the parallel win.
static constexpr int MIN_M_FOR_THREADS = 64;

static int chosen_threads(int M) {
  if (M < MIN_M_FOR_THREADS) return 1;
  const int hw = static_cast<int>(std::thread::hardware_concurrency());
  const int cap = hw > 0 ? std::min(hw, MAX_THREADS) : MAX_THREADS;
  // Each thread should chew on at least ~16 rows.
  return std::min(cap, std::max(1, M / 16));
}

// ---------------------------------------------------------------------------
// C = A @ B    A:[M,K]  B:[K,N]  C:[M,N]
//
// Register- + cache-blocked. An MR×NR tile of C is held in registers and a B
// value is reused across MR rows, lifting the kernel from memory-bandwidth-bound
// toward compute-bound (the autovectoriser then fills NR with SIMD lanes). KC/NC
// keep the active B panel [KC×NC] resident in L2 across the M sweep. For any
// fixed (m,n) the k-accumulation still runs k = 0..K-1 in order, so the result
// is BIT-IDENTICAL to the naive ikn loop — the parity gate sees zero drift.
// (~3-4× over the naive kernel natively; see docs/performance.md.)
static constexpr int MR = 4;     // C-tile rows held in registers
static constexpr int NR = 16;    // C-tile cols (4 SIMD lanes × 4)
static constexpr int KC = 256;   // K-panel (B panel height)
static constexpr int NC = 256;   // N-panel (B panel width) — [KC×NC] ≈ 256 KB fits L2

static void matmul_forward_serial(const float* A, const float* B, float* C,
                                   int m_lo, int m_hi, int K, int N) {
  // Micro-kernels accumulate into C across k-panels, so zero the slice first.
  for (int m = m_lo; m < m_hi; ++m) {
    float* c_row = C + static_cast<long>(m) * N;
    for (int n = 0; n < N; ++n) c_row[n] = 0.0f;
  }
  const int m_full = m_lo + ((m_hi - m_lo) / MR) * MR;  // last MR-aligned row
  for (int n0 = 0; n0 < N; n0 += NC) {
    const int nmax = (n0 + NC < N) ? n0 + NC : N;
    for (int k0 = 0; k0 < K; k0 += KC) {
      const int kmax = (k0 + KC < K) ? k0 + KC : K;
      // Full MR×NR register tiles.
      for (int m = m_lo; m < m_full; m += MR) {
        int n = n0;
        for (; n + NR <= nmax; n += NR) {
          float acc[MR][NR];
          for (int i = 0; i < MR; ++i) {
            const float* c = C + static_cast<long>(m + i) * N + n;
            for (int j = 0; j < NR; ++j) acc[i][j] = c[j];
          }
          for (int k = k0; k < kmax; ++k) {
            const float* b = B + static_cast<long>(k) * N + n;
            const float a0 = A[static_cast<long>(m + 0) * K + k];
            const float a1 = A[static_cast<long>(m + 1) * K + k];
            const float a2 = A[static_cast<long>(m + 2) * K + k];
            const float a3 = A[static_cast<long>(m + 3) * K + k];
            for (int j = 0; j < NR; ++j) {
              const float bv = b[j];
              acc[0][j] += a0 * bv; acc[1][j] += a1 * bv;
              acc[2][j] += a2 * bv; acc[3][j] += a3 * bv;
            }
          }
          for (int i = 0; i < MR; ++i) {
            float* c = C + static_cast<long>(m + i) * N + n;
            for (int j = 0; j < NR; ++j) c[j] = acc[i][j];
          }
        }
        // N remainder (< NR) for this MR row group.
        for (; n < nmax; ++n)
          for (int i = 0; i < MR; ++i) {
            float s = C[static_cast<long>(m + i) * N + n];
            for (int k = k0; k < kmax; ++k) s += A[static_cast<long>(m + i) * K + k] * B[static_cast<long>(k) * N + n];
            C[static_cast<long>(m + i) * N + n] = s;
          }
      }
      // M remainder rows (< MR).
      for (int m = m_full; m < m_hi; ++m)
        for (int n = n0; n < nmax; ++n) {
          float s = C[static_cast<long>(m) * N + n];
          for (int k = k0; k < kmax; ++k) s += A[static_cast<long>(m) * K + k] * B[static_cast<long>(k) * N + n];
          C[static_cast<long>(m) * N + n] = s;
        }
    }
  }
}

WASM_EXPORT void matmul_forward(const float* A, const float* B, float* C,
                                int M, int K, int N) {
  const int nthreads = chosen_threads(M);
  if (nthreads == 1) {
    matmul_forward_serial(A, B, C, 0, M, K, N);
    return;
  }
  const int chunk = (M + nthreads - 1) / nthreads;
  std::vector<std::thread> workers;
  workers.reserve(nthreads - 1);
  for (int t = 1; t < nthreads; ++t) {
    int m_lo = t * chunk;
    int m_hi = std::min(m_lo + chunk, M);
    if (m_lo >= M) break;
    workers.emplace_back(matmul_forward_serial, A, B, C, m_lo, m_hi, K, N);
  }
  matmul_forward_serial(A, B, C, 0, std::min(chunk, M), K, N);
  for (auto& w : workers) w.join();
}

// ---------------------------------------------------------------------------
// dA = dC @ B^T     dB = A^T @ dC.   Both outputs overwritten.
//
// dA: each row dA[m,:] is independent — splittable like forward.
// dB: each cell dB[k,n] accumulates over m. Per-thread scratch + reduction.
// ---------------------------------------------------------------------------
// dA[m,k] = sum_n dC[m,n]·B[k,n]. Register-block KR k-rows per pass so dc_row[n]
// is loaded once and reused across KR dot-products (was re-streamed per k). Same
// n-order accumulation → bit-identical to the naive reduction.
static constexpr int KR = 4;
static void backward_dA_serial(const float* B, const float* dC, float* dA,
                                int m_lo, int m_hi, int K, int N) {
  const int k_full = (K / KR) * KR;
  for (int m = m_lo; m < m_hi; ++m) {
    float* da_row = dA + static_cast<long>(m) * K;
    const float* dc_row = dC + static_cast<long>(m) * N;
    int k = 0;
    for (; k < k_full; k += KR) {
      const float* b0 = B + static_cast<long>(k + 0) * N;
      const float* b1 = B + static_cast<long>(k + 1) * N;
      const float* b2 = B + static_cast<long>(k + 2) * N;
      const float* b3 = B + static_cast<long>(k + 3) * N;
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      for (int n = 0; n < N; ++n) {
        const float dn = dc_row[n];
        a0 += dn * b0[n]; a1 += dn * b1[n]; a2 += dn * b2[n]; a3 += dn * b3[n];
      }
      da_row[k + 0] = a0; da_row[k + 1] = a1; da_row[k + 2] = a2; da_row[k + 3] = a3;
    }
    for (; k < K; ++k) {  // remainder k
      const float* b_row = B + static_cast<long>(k) * N;
      float acc = 0.0f;
      for (int n = 0; n < N; ++n) acc += dc_row[n] * b_row[n];
      da_row[k] = acc;
    }
  }
}

static void backward_dB_partial(const float* A, const float* dC, float* dB_local,
                                 int m_lo, int m_hi, int K, int N) {
  for (long i = 0; i < static_cast<long>(K) * N; ++i) dB_local[i] = 0.0f;
  for (int m = m_lo; m < m_hi; ++m) {
    const float* a_row = A + static_cast<long>(m) * K;
    const float* dc_row = dC + static_cast<long>(m) * N;
    for (int k = 0; k < K; ++k) {
      const float a = a_row[k];
      float* db_row = dB_local + static_cast<long>(k) * N;
      for (int n = 0; n < N; ++n) db_row[n] += a * dc_row[n];
    }
  }
}

WASM_EXPORT void matmul_backward(const float* A, const float* B, const float* dC,
                                 float* dA, float* dB, int M, int K, int N) {
  const int nthreads = chosen_threads(M);
  if (nthreads == 1) {
    backward_dA_serial(B, dC, dA, 0, M, K, N);
    backward_dB_partial(A, dC, dB, 0, M, K, N);
    return;
  }
  const int chunk = (M + nthreads - 1) / nthreads;

  // --- dA: independent rows, no reduction needed --------------------------
  {
    std::vector<std::thread> workers;
    workers.reserve(nthreads - 1);
    for (int t = 1; t < nthreads; ++t) {
      int m_lo = t * chunk;
      int m_hi = std::min(m_lo + chunk, M);
      if (m_lo >= M) break;
      workers.emplace_back(backward_dA_serial, B, dC, dA, m_lo, m_hi, K, N);
    }
    backward_dA_serial(B, dC, dA, 0, std::min(chunk, M), K, N);
    for (auto& w : workers) w.join();
  }

  // --- dB: accumulate per-thread, reduce at the end -----------------------
  const long kn = static_cast<long>(K) * N;
  std::vector<std::vector<float>> partials(nthreads);
  for (auto& p : partials) p.resize(kn);

  std::vector<std::thread> workers;
  workers.reserve(nthreads - 1);
  for (int t = 1; t < nthreads; ++t) {
    int m_lo = t * chunk;
    int m_hi = std::min(m_lo + chunk, M);
    if (m_lo >= M) break;
    workers.emplace_back(backward_dB_partial, A, dC, partials[t].data(),
                         m_lo, m_hi, K, N);
  }
  backward_dB_partial(A, dC, partials[0].data(), 0, std::min(chunk, M), K, N);
  for (auto& w : workers) w.join();

  // Reduction: dB = sum_t partials[t]. O(KN), << the matmul itself.
  std::memcpy(dB, partials[0].data(), kn * sizeof(float));
  for (int t = 1; t < nthreads; ++t) {
    const float* src = partials[t].data();
    for (long i = 0; i < kn; ++i) dB[i] += src[i];
  }
}
