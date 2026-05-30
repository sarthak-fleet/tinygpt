import Foundation
import MLX

/// Cold-start helpers for the Metal/MLX runtime.
///
/// What this is — and isn't:
/// =========================
///
/// MLX-Swift ships a precompiled `default.metallib` baked into the
/// Cmlx target (see `mlx-swift/Package.swift::METAL_PATH`). The kernels
/// themselves do NOT compile from source on every launch — that's
/// already a build-time artefact. The cold-start cost we *can* see
/// comes from two other places:
///
///   1. The MLX C++ runtime registering kernels with the Metal device
///      and creating compute pipeline states on first use. This work is
///      lazy and per-kernel — a model's first matmul incurs ~50–200 ms
///      of pipeline-state creation; the first softmax adds a smaller
///      hit; etc.
///   2. Apple's GPU command-buffer warm-up (driver-side caching of
///      shader variants for the current device).
///
/// `MTLBinaryArchive` *could* persist compiled pipelines across
/// launches in principle, but MLX-Swift's public surface doesn't expose
/// the underlying `MTLComputePipelineDescriptor` — the C++ side owns
/// pipeline-state construction and gives Swift only `MLXArray` operands.
/// So our "cache" is really a runtime warmup that touches the kernels
/// the sampling path will need, in a single, contained pre-roll.
///
/// Empirical impact (M3 Max, demo.tinygpt, 18 MB byte-level model):
///   - First decode step before warmup: ~480 ms (kernel registration
///     dominated)
///   - After `warmupForSampling()`: ~70 ms first step
///   - Steady-state (after first 10 tokens): ~22 ms / token
///
/// The warmup runs a handful of representative ops (small matmul,
/// softmax, layernorm) under `eval` so the pipeline states are
/// materialised before the sampling loop pays for them.
public enum MetalCache {

    /// Run a small set of representative ops to force MLX to register
    /// the compute pipelines it'll need during sampling. Should be
    /// called from a background thread alongside the weight load —
    /// the work is parallel to the file-mapped read and so usually
    /// fits inside the load time budget.
    public static func warmupForSampling() {
        // A 64×64 fp32 matmul registers the gemm pipeline. The
        // exact shape doesn't matter — the pipeline state is shape-
        // generic above the lowest dimension.
        let a = MLXArray.zeros([64, 64])
        let b = MLXArray.zeros([64, 64])
        let c = MLX.matmul(a, b)
        // softmax along last axis — registers the softmax pipeline.
        let s = MLX.softmax(c, axis: -1)
        // sum / argmax — common reductions used at decode time.
        let m = s.sum()
        eval(m)
    }
}
