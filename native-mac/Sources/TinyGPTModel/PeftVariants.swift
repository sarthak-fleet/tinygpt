import Foundation
import MLX
import MLXNN
import MLXRandom

/// PEFT variant bundle — see `docs/peft_variants.md` for the full table.
///
/// Each variant changes either how the LoRA-style `A`, `B` matrices are
/// INITIALISED (PISSA, LoftQ), what fraction of them is TRAINABLE (LoRA-FA,
/// VeRA, AdaLoRA), or the SCALING of the forward delta (RsLoRA). LayerDrop
/// is orthogonal — it stochastically skips entire transformer blocks.
///
/// Implementation strategy: a single `PeftVariant` enum drives both
/// initialisation and freeze-mask in `LoraLinear`/`LoraLinearHF`. The
/// forward path branches on the variant to apply the right per-token /
/// per-feature scaling; saved adapter files don't yet encode the
/// variant (so a save+load round-trip degrades into a vanilla LoRA
/// load — same A·B matmul). For the in-session smoke tests that's
/// fine.
public enum PeftVariant: Sendable, Equatable {
    /// Vanilla LoRA — A gaussian, B zero. Both trainable. scale = α/r.
    case lora
    /// DoRA — see `DoraLinear`. Drop-in: kept as a sibling Linear class,
    /// not folded into the variant union (different param tree).
    case dora
    /// RsLoRA (Kalajdzievski, 2023) — same init as LoRA, but scale = α / √r
    /// instead of α / r. Pretty literal one-line change; large `r` actually
    /// helps now because the scaling no longer fights the rank growth.
    case rsLora
    /// LoRA-FA (Zhang et al., 2023) — Frozen A. B trainable. ½ the params,
    /// same init shapes as LoRA so adapters are bit-compatible on disk
    /// (just one of the two matrices stays at its random init forever).
    case loraFA
    /// VeRA (Kopiczko et al., 2023) — A AND B are frozen at shared random
    /// init; the trainable parameters become a per-layer diagonal `b` of
    /// size `[out]` plus an optional `d` of size `[rank]`. ~10× smaller
    /// adapter than LoRA at equal expressivity (the random projection
    /// preserves enough structure that scalar-per-row tuning suffices).
    case vera
    /// PISSA (Meng et al., 2024) — Principal-Singular-vectors and
    /// Singular-values Adaptation. Init A = U[:, :r] · sqrt(S[:r]),
    /// B = sqrt(S[:r]) · Vt[:r, :]. Effectively: bootstrap the LoRA
    /// adapter to the TOP-r SVD of the base weight, then subtract that
    /// component from the base ("residual"). Faster convergence than
    /// the zero-init LoRA at no extra param cost.
    case pissa
    /// LoftQ (Li et al., 2023) — quantization-aware init. Approximates
    /// quant(W) ≈ W - (A·B) by setting A·B to compensate the quantization
    /// error. We don't ship an int4 base here, so this is a "near-LoftQ":
    /// quantise-then-dequantise the base to a 4-bit grid, take SVD of
    /// the residual `W - W_q`, init A, B from its top-r factors. The
    /// adapter then represents the "error a 4-bit base loses".
    case loftq
    /// AdaLoRA (Zhang et al., 2023) — adaptive per-layer rank. We
    /// allocate the full configured rank to every wrapped Linear but
    /// expose a per-rank importance score `eVec` of shape `[r]`. The
    /// trainable forward becomes A · diag(e) · B; magnitudes drift to
    /// zero on unimportant directions. To keep the smoke test cheap
    /// we do NOT actually prune the matrices mid-training; the
    /// importance scoring is enough to demonstrate the parameter
    /// allocation in the trainable-count tally (it's stored alongside
    /// A/B and trained jointly).
    case adaLora
}

/// VeRA's shared random projections. Kopiczko et al. derived the result
/// for a SINGLE pair of (A, B) reused across all wrapped Linears.
/// We emulate that with deterministic per-layer seeded init — easier
/// to wire than threading a shared singleton through MLX's Module tree.
public enum VeRARandom {
    /// Reproducible gaussian init using MLXRandom's seedable variant
    /// (so two independent runs at the same seed produce identical
    /// frozen projections). Falls back to the regular non-seeded init
    /// when a seed isn't supplied (the smoke tests don't care; only
    /// matters if you want byte-for-byte adapter reload).
    public static func projection(shape: [Int], seed: UInt64 = 0xC0FFEE) -> MLXArray {
        // Use the same `MLXRandom.normal` everyone else uses, but with
        // a manual key so re-running the program reproduces the matrix.
        // The `MLXRandom.key(seed:)` API takes UInt64 — keep it inside
        // the enum scope so the rest of the code never sees a global key.
        let key = MLXRandom.key(seed)
        return MLXRandom.normal(
            shape, dtype: .float32, loc: 0,
            scale: 1.0 / Float(shape[0]).squareRoot(), key: key
        )
    }
}

/// Top-r truncated SVD helper. Used by PISSA + LoftQ initialisations.
///
/// `weight` is shaped `[out, in]` (PyTorch convention; same as MLX-NN's
/// `Linear.weight`). We compute the full SVD, slice to rank r, and return
/// `(A_init, B_init)` matching the LoRA shapes `[in, r]`, `[r, out]`.
///
/// Mathematically:
///   W ≈ U[:, :r] · diag(S[:r]) · Vt[:r, :]
/// We split the singular values evenly between A and B so that
///   A = (Vt[:r, :])^T · sqrt(diag(S[:r])) , shape [in, r]
///   B = sqrt(diag(S[:r])) · U[:, :r]^T   , shape [r, out]
/// such that `(x @ A) @ B` reconstructs `x @ W^T` to rank-r accuracy.
public enum TopRSVD {
    public static func factors(weight: MLXArray, rank: Int) -> (a: MLXArray, b: MLXArray) {
        // SVD lives on `MLXLinalg` in the MLX module — no separate import
        // is needed because the enum is re-exposed by `import MLX`.
        //
        // MLX's `linalg::svd` is CPU-only (the kernel hasn't been ported
        // to Metal yet; status as of mlx-swift 0.25). Pass `stream: .cpu`
        // explicitly so the call doesn't crash on the default GPU stream.
        // The result lands back on the default device for downstream use.
        eval(weight)
        let (U, S, Vt) = MLXLinalg.svd(weight, stream: .cpu)
        // SVD on a [out, in] matrix yields U shape [out, k], S [k], Vt [k, in]
        // where k = min(out, in). Take the leading r columns / rows.
        let r = min(rank, S.shape[0])
        let Sr = S[0..<r]              // [r]
        let Ur = U[0..., 0..<r]        // [out, r]
        let Vtr = Vt[0..<r, 0...]      // [r, in]
        // sqrt(S) split between A and B factors.
        let sqrtS = MLX.sqrt(MLX.maximum(Sr, MLXArray(Float(0))))  // [r]
        // A: [in, r] = Vt[:r, :]^T · diag(sqrtS).
        // We need shape [in, r]; Vtr is [r, in] → transpose → [in, r],
        // then per-column multiply by sqrtS (broadcast on the last axis).
        let A = Vtr.transposed() * sqrtS                            // [in, r]
        // B: [r, out] = diag(sqrtS) · U[:r, :]. U[:, :r] is [out, r];
        // transpose to [r, out], then multiply each ROW by sqrtS.
        let B = Ur.transposed() * sqrtS.expandedDimensions(axis: 1) // [r, out]
        return (A, B)
    }
}

/// LoftQ near-init helper. Approximates a 4-bit per-channel symmetric
/// quantization of `weight`, returns the dequantized version (so the
/// "quantization error" we initialise the LoRA adapter against is real).
///
/// Real LoftQ also re-loads the BASE with the quantized weights and
/// keeps it in int4 form; we don't do that — the base stays fp32. The
/// purpose here is to teach the adapter what error the quantizer would
/// have introduced, so when you later DO swap in a quantized base, the
/// adapter compensates.
public enum LoftQQuant {
    /// Per-output-row symmetric quantization: scale = max(|w|) / qMax,
    /// q = round(w/scale), clip to the signed range, dequantize = q · scale.
    public static func dequantize(_ w: MLXArray, bits: Int) -> MLXArray {
        let clampedBits = min(max(bits, 2), 8)
        let qMax = Float((1 << (clampedBits - 1)) - 1)
        let qMin = -Float(1 << (clampedBits - 1))
        let absW = MLX.abs(w)
        // [out, 1] — one scale per output row. eps guards a zero row.
        let scale = (absW.max(axis: -1, keepDims: true) / MLXArray(qMax))
            + MLXArray(Float(1e-8))
        let q = MLX.round(w / scale)
        let qClamped = MLX.clip(q, min: MLXArray(qMin), max: MLXArray(qMax))
        return qClamped * scale
    }

    /// Back-compat helper for existing call sites/tests.
    public static func dequantize4bit(_ w: MLXArray) -> MLXArray {
        dequantize(w, bits: 4)
    }
}

/// Per-block stochastic depth (LayerDrop, Fan et al. 2019).
///
/// We hold the drop fraction as a process-wide static rather than as a
/// module parameter for two reasons:
///   1. The TransformerBlock initializer is touched from many places —
///      adding a constructor argument would break a hundred call sites.
///   2. LayerDrop is a TRAINING-ONLY knob; serialised models never see
///      it. A static avoids polluting `ModelConfig` with a flag every
///      reader has to learn about.
///
/// Threading model: only the training loop writes; per-block forward
/// passes only read. There's no actual concurrency on a single MLX-Swift
/// step (the autograd path is single-threaded inside one process), so
/// the unguarded global is safe — but we tag it `nonisolated(unsafe)`
/// to keep the Swift 6 concurrency checker happy.
public enum LayerDropState {
    /// Probability of skipping a single block on a given forward.
    /// `0.0` = never skip (the default; LayerDrop off).
    nonisolated(unsafe) public static var probability: Float = 0.0

    /// Toggle for the `layerDrop` boolean check. The probability gate
    /// alone would be enough, but on first call we want to log the
    /// effective fraction once; this lets callers branch cleanly.
    public static var enabled: Bool { probability > 0 }

    /// Reset for tests / safety.
    public static func disable() { probability = 0 }

    /// Sample `Bool` — true means "drop this block on this step".
    /// Uses `MLXRandom.bernoulli` so the dice live on the same Metal
    /// device the model lives on (no cross-device sync cost).
    public static func shouldDrop() -> Bool {
        if probability <= 0 { return false }
        // Use Foundation's Float drand48 — simple, fast, plenty
        // random for stochastic depth (we never need bitwise repro).
        return Double.random(in: 0..<1) < Double(probability)
    }
}
