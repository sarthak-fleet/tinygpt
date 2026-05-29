import Foundation
import TinyGPTModel

/// Pre-flight memory estimate for `tinygpt train`.
///
/// The Mega-bf16 attempt died at step 1 with no error message — almost
/// certainly OOM during the first backward, after we'd already paid the
/// 30-minute tokenize cost. This estimator runs BEFORE tokenisation, so
/// the user sees the projected footprint and can abort cheaply.
///
/// The breakdown:
///   - weights:    P × dtype_bytes
///   - gradients:  P × dtype_bytes (one shadow copy)
///   - AdamW:      P × 8 (m + v, both fp32 — the optimiser state never
///                 follows the training dtype)
///   - activations:  the forward-pass and backward intermediates. We
///                 model the dominant terms only — the per-layer
///                 attention matrix (B × H × T² × dtype) is the OOM
///                 driver at long context, and the per-layer
///                 (in/out activations + MLP hidden) is the
///                 second-order term.
///
/// All numbers are upper bounds within a factor of ~1.3× — good enough
/// to refuse a 24 GB projection on a 16 GB host while letting a 10 GB
/// projection through with a heads-up.
enum OOMGuard {

    struct Estimate {
        let params: Int
        let dtypeBytes: Int
        let weights: Int
        let gradients: Int
        let optimizer: Int
        let activations: Int
        var total: Int { weights + gradients + optimizer + activations }
    }

    /// Compute the projected footprint for the given config + batch.
    /// `effectiveBatch` is the actual per-step batch (B × accum is fine
    /// because activations live and die within a single micro-batch).
    static func estimate(cfg: ModelConfig, params: Int, batch: Int) -> Estimate {
        let dtypeBytes = dtypeSize(cfg.dtype)
        let weights = params * dtypeBytes
        let gradients = params * dtypeBytes
        // Adam state is fp32-only regardless of weight dtype: bf16/fp16
        // weights are kept in their training dtype, but m/v are 4-byte
        // floats so the optimiser update keeps numerical headroom.
        let optimizer = params * 8

        let B = batch
        let T = cfg.contextLength
        let L = cfg.nLayers
        let C = cfg.dModel
        let M = cfg.dMlp
        let H = cfg.nHeads

        // Attention matrix [B, H, T, T] per layer is the cost driver at
        // long ctx. Counted twice (forward + saved-for-backward).
        let attnMatrix = 2 * B * H * T * T * L * dtypeBytes
        // Per-token-per-layer activations: the 6 residual-stream-shaped
        // tensors (ln1 in/out, q/k/v, attn out, ln2 in/out, residual)
        // plus MLP hidden of width M.
        let perTokC = 6 * C
        let perTokM = M
        let actsCT = B * T * L * (perTokC + perTokM) * dtypeBytes

        let activations = attnMatrix + actsCT
        return Estimate(
            params: params, dtypeBytes: dtypeBytes,
            weights: weights, gradients: gradients,
            optimizer: optimizer, activations: activations
        )
    }

    /// Print the breakdown + a coloured warning if the estimate exceeds
    /// 60% of physical RAM. Threshold chosen because macOS keeps a sizable
    /// page-cache slice — at 60% the trainer is already squeezing the
    /// system; at >80% the wired-memory pressure starts forcing the
    /// dramatic compress-and-swap loop that killed Mega-bf16.
    static func reportAndWarn(_ e: Estimate) {
        let ram = physicalRAMBytes()
        print("""

        Memory estimate (pre-flight)
        ---------------------------
        weights     \(format(e.weights))    \(e.dtypeBytes) B / param × \(formatLargeInt(e.params)) params
        gradients   \(format(e.gradients))
        AdamW m+v   \(format(e.optimizer))   fp32 (8 B / param)
        activations \(format(e.activations))   forward + backward intermediates
                    --------
        total       \(format(e.total))   of \(ram > 0 ? format(ram) : "?") physical RAM
        """)
        if ram > 0 {
            let ratio = Double(e.total) / Double(ram)
            if ratio > 0.6 {
                let band = ratio > 0.8 ? "⚠ DANGER" : "⚠ heads-up"
                fputs("""

                \(band): projected \(String(format: "%.0f%%", ratio * 100)) of physical RAM.
                  macOS keeps page-cache + window-server overhead; >60% often crashes
                  silently on first backward. Mitigations: --dtype bfloat16, lower
                  --ctx or --batch, raise --accum.

                """, stderr)
            }
        }
    }

    // MARK: - helpers

    private static func dtypeSize(_ dt: String) -> Int {
        switch dt.lowercased() {
        case "float16", "fp16", "half": return 2
        case "bfloat16", "bf16":         return 2
        default:                         return 4    // fp32 or anything we don't recognise
        }
    }

    private static func physicalRAMBytes() -> Int {
        var size: UInt64 = 0
        var len = MemoryLayout<UInt64>.size
        let ok = sysctlbyname("hw.memsize", &size, &len, nil, 0) == 0
        return ok ? Int(size) : 0
    }

    private static func format(_ n: Int) -> String {
        let gb = Double(n) / 1_073_741_824
        if gb >= 1 { return String(format: "%6.2f GB", gb) }
        let mb = Double(n) / 1_048_576
        return String(format: "%6.0f MB", mb)
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}
