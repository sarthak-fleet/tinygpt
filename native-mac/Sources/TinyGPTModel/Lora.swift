import Foundation
import MLX
import MLXNN
import MLXRandom

/// LoRA (Low-Rank Adaptation): a Linear subclass that adds a low-rank
/// delta on top of the frozen base weight.
///
///     y = base_linear(x) + (x @ A) @ B * (alpha / r)
///
///   - `weight` and `bias` (inherited from Linear): FROZEN at fine-tune time
///   - `loraA: [in, r]` trainable, gaussian-init std=0.02
///   - `loraB: [r, out]` trainable, ZERO-init
///   - `rank r`: typically 4-16
///   - `alpha`: scaling, typically 2× r
///
/// B starts at zero → initial output exactly equals base output. Training
/// is purely additive — never destructive.
///
/// Param math: Huge has ~9.6M base params; LoRA-on-QV at r=4 across 12
/// blocks adds 12 × 2 × (256·4 + 4·256) = 49 152 trainable params (~200×
/// fewer). Training fits in minutes; adapter files are 100KB-1MB instead
/// of tens of MB, so multiple "voices" (legal text, code style, lyrics
/// register) share the same base cheaply.
public final class LoraLinear: Linear {
    public let loraA: MLXArray  // [in, r]
    public let loraB: MLXArray  // [r, out]
    /// VeRA / AdaLoRA per-rank trainable scalars `[r]`. Used as a
    /// diagonal between A and B in the forward delta. For variants
    /// that don't need it (vanilla LoRA, RsLoRA, LoRA-FA, PISSA, LoftQ)
    /// this stays at all-ones and contributes a no-op — kept as a
    /// parameter so the MLX module tree shape is constant across
    /// variants (simpler trainable-param walker).
    public let loraD: MLXArray  // [r]
    public let rank: Int
    public let alpha: Float
    /// Which PEFT variant this Linear is wired for. Drives the forward
    /// scale, the trainable-mask, and (for VeRA) which keys get
    /// unfrozen by `LoraInjection.freezeBase`.
    public let variant: PeftVariant
    public let quantizationBits: Int
    /// Effective forward scale. RsLoRA uses α/√r (Kalajdzievski 2023);
    /// every other variant keeps the LoRA-paper default α/r.
    public var scale: Float {
        switch variant {
        case .rsLora: return alpha / Foundation.sqrt(Float(rank))
        default:      return alpha / Float(rank)
        }
    }

    /// Wrap an existing Linear with LoRA adapters. The base weight + bias
    /// are reused verbatim (no copy); the adapter matrices are fresh.
    public init(wrapping base: Linear, rank: Int = 4, alpha: Float = 8.0,
                variant: PeftVariant = .lora, quantizationBits: Int = 4) {
        precondition(rank > 0, "LoRA rank must be > 0")
        self.rank = rank
        self.alpha = alpha
        self.variant = variant
        self.quantizationBits = min(max(quantizationBits, 2), 8)
        let outFeatures = base.weight.shape[0]
        let inFeatures = base.weight.shape[1]
        // Effective forward scale. Mirrors `var scale` — duplicated here
        // because the init body needs the value BEFORE super.init runs.
        let scale: Float = (variant == .rsLora)
            ? alpha / Foundation.sqrt(Float(rank))
            : alpha / Float(rank)
        // Initialise A, B per the variant. Vanilla LoRA: A ~ N(0, .02), B = 0.
        // PISSA / LoftQ: SVD-based. VeRA: frozen random projections (we use a
        // seed-derived gaussian; A is shape [in, r] with std ≈ 1/√in).
        // LoRA-FA / RsLoRA: same init as vanilla LoRA.
        var effectiveWeight = base.weight
        switch variant {
        case .lora, .rsLora, .loraFA, .adaLora:
            self.loraA = MLXRandom.normal([inFeatures, rank], scale: 0.02)
            self.loraB = MLXArray.zeros([rank, outFeatures])
        case .vera:
            // VeRA — both matrices are FROZEN random projections. Seed differs
            // per Linear shape so unrelated Linears get independent projections;
            // identical-shape Linears across layers share the same projection
            // (matching the paper's "shared random A,B" prescription as closely
            // as MLX module-tree semantics allow). The per-rank trainable
            // scalar `d` starts at zero so the forward at step 0 equals the
            // base forward — gradient descent grows the projection
            // contributions monotonically from a frozen anchor.
            let seedA: UInt64 = 0xA5A5A5A5 &+ UInt64(inFeatures) &+ UInt64(rank << 8)
            let seedB: UInt64 = 0x5A5A5A5A &+ UInt64(outFeatures) &+ UInt64(rank << 8)
            self.loraA = VeRARandom.projection(shape: [inFeatures, rank], seed: seedA)
            self.loraB = VeRARandom.projection(shape: [rank, outFeatures], seed: seedB)
        case .pissa:
            // PISSA: bake the inverse-scale into the SVD factors so the
            // forward at step 0 reconstructs `top_r(W)` exactly (not
            // `scale · top_r(W)`); then subtract that reconstruction from
            // the base weight. Net effect: forward at init equals base
            // forward, AND the top-r component is now ALREADY in the
            // adapter — gradient descent immediately starts refining the
            // most important directions instead of discovering them.
            let (aRaw, bRaw) = TopRSVD.factors(weight: base.weight, rank: rank)
            let invSqrtScale = MLXArray(Float(1) / Foundation.sqrt(scale))
            let a = aRaw * invSqrtScale
            let b = bRaw * invSqrtScale
            self.loraA = a
            self.loraB = b
            // The residual weight: W - scale · (A @ B).T = W - top_r(W).
            // Computed once at init; the result is the new frozen base.
            let deltaW = matmul(a, b) * MLXArray(scale)                // [in, out]
            effectiveWeight = base.weight - deltaW.transposed()         // [out, in]
            eval(effectiveWeight)
        case .loftq:
            // LoftQ: A·B should compensate the quantization error (W - W_q).
            // We compute W_q by per-row symmetric int4 quantize-then-
            // dequantize, take the residual, and seed the adapter from its
            // top-r SVD. With our fp32 base + simulated quant this is a
            // "compensation adapter" demo; swapping in a real int4 base
            // is the production setup. Same inverse-scale baking as PISSA
            // so the forward at init equals the QUANTIZED base + the
            // (now-compensating) adapter — exact reconstruction of W.
            eval(base.weight)
            let wQ = LoftQQuant.dequantize(base.weight, bits: self.quantizationBits)
            let residual = base.weight - wQ
            let (aRaw, bRaw) = TopRSVD.factors(weight: residual, rank: rank)
            let invSqrtScale = MLXArray(Float(1) / Foundation.sqrt(scale))
            let a = aRaw * invSqrtScale
            let b = bRaw * invSqrtScale
            self.loraA = a
            self.loraB = b
            // Pin the base to the quantized version. Combined with the
            // top-r residual we just put in A,B, forward equals the
            // ORIGINAL fp32 base at step 0 (to rank-r precision).
            effectiveWeight = wQ
            eval(effectiveWeight)
        case .dora:
            // Variant value `dora` is reserved for the DoraLinear class;
            // a LoraLinear with variant=.dora is a programming error.
            // Fall back to vanilla init so we never crash on a misconfig.
            self.loraA = MLXRandom.normal([inFeatures, rank], scale: 0.02)
            self.loraB = MLXArray.zeros([rank, outFeatures])
        }
        // Per-rank diagonal. For most variants this stays at all-ones and
        // contributes a no-op multiplication in the forward. VeRA flips
        // it to ZEROS so the random A·B projection contributes nothing
        // at step 0 (the trainable `d` then grows from zero — same
        // "start at base forward" guarantee LoRA's zero-init B gives).
        // AdaLoRA: ones so the per-rank importance starts neutral; the
        // optimiser pushes scores toward 0 for un-useful directions.
        if variant == .vera {
            self.loraD = MLXArray.zeros([rank])
        } else {
            self.loraD = MLXArray.ones([rank])
        }
        super.init(weight: effectiveWeight, bias: base.bias)
    }

    /// Build with explicit A, B (used by the adapter-loader to restore
    /// a saved fine-tune). Always loads as plain LoRA — the on-disk
    /// adapter format doesn't yet carry the variant tag.
    public init(wrapping base: Linear, loraA: MLXArray, loraB: MLXArray,
                rank: Int, alpha: Float) {
        self.rank = rank
        self.alpha = alpha
        self.variant = .lora
        self.quantizationBits = 4
        self.loraA = loraA
        self.loraB = loraB
        self.loraD = MLXArray.ones([rank])
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // base(x): inherited Linear forward — x @ weight.T + bias
        // delta:   x @ loraA · diag(d) @ loraB * scale
        // VeRA / AdaLoRA make `d` trainable; other variants leave it = 1.
        let baseOut = super.callAsFunction(x)
        let xA = matmul(x, loraA)                // [B*T, r]
        let scaled = xA * loraD                   // broadcast over the last dim [r]
        let delta = matmul(scaled, loraB) * MLXArray(scale)
        return baseOut + delta
    }
}

/// DoRA (Weight-Decomposed Low-Rank Adaptation; Liu et al., 2024).
///
/// Decomposes a weight update into a MAGNITUDE component (per-output-
/// row scalar) and a DIRECTION component (low-rank LoRA delta). The
/// forward computes:
///
///     V       = W_base + (A @ B).T · scale       // [out, in]
///     V_dir   = V / ‖V‖_row                       // unit-norm rows
///     W_new   = diag(m) @ V_dir                   // per-row scale
///     y       = x @ W_newᵀ + b
///
/// Where `m` is a per-output-row magnitude initialised to the original
/// row-norms — so the DoRA forward exactly equals the base linear at
/// init, and training adjusts both magnitude (m) and direction (BA).
///
/// Outperforms LoRA at the same rank on most benchmarks (paper: 0.5-2
/// PPL improvement at the same parameter count). Memory cost over
/// LoRA: one extra `[out]` vector per wrapped Linear — negligible.
///
/// Implementation notes:
/// - Save/load adapter format isn't yet extended for the `m` vector,
///   so DoRA is currently in-session only. Saving a DoRA-trained model
///   serialises only the merged `(base + LoRA delta) · m / ‖·‖` view.
///   (Follow-up: add `m` to `LoraAdapter`.)
/// - We RECOMPUTE V and its norm every forward. The recomputation is
///   `O(d_out · d_in)` extra work per linear per step — small relative
///   to the matmul that dominates.
public final class DoraLinear: Linear {
    public let loraA: MLXArray   // [in, r]
    public let loraB: MLXArray   // [r, out]
    public let m: MLXArray       // [out] — per-output-row magnitude (trainable)
    public let rank: Int
    public let alpha: Float
    public var scale: Float { alpha / Float(rank) }

    public init(wrapping base: Linear, rank: Int = 4, alpha: Float = 8.0) {
        precondition(rank > 0, "DoRA rank must be > 0")
        self.rank = rank
        self.alpha = alpha
        let outF = base.weight.shape[0]
        let inF  = base.weight.shape[1]
        self.loraA = MLXRandom.normal([inF, rank], scale: 0.02)
        self.loraB = MLXArray.zeros([rank, outF])
        // Initialise m to the row norms of the base weight, so the DoRA
        // forward at step 0 exactly equals the wrapped Linear forward.
        let rowSquared = (base.weight * base.weight).sum(axis: 1)           // [out]
        self.m = MLX.sqrt(rowSquared + MLXArray(Float(1e-12)))
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Combined weight V = W + ΔW. ΔW = (A @ B)ᵀ scaled.
        // (loraA: [in, r], loraB: [r, out]) so loraA @ loraB is [in, out],
        // which is the TRANSPOSE of how PyTorch stores W [out, in].
        let deltaAB = matmul(loraA, loraB) * MLXArray(scale)               // [in, out]
        let V = weight + deltaAB.transposed()                              // [out, in]
        // Per-row L2 norm. keepDims so we can broadcast-divide V's rows.
        let rowNorm = MLX.sqrt((V * V).sum(axis: 1, keepDims: true)
                                 + MLXArray(Float(1e-9)))                  // [out, 1]
        // W_new = (V / rowNorm) · m_row, broadcast over input dim.
        let mCol = m.expandedDimensions(axis: -1)                          // [out, 1]
        let Wnew = (V / rowNorm) * mCol                                    // [out, in]
        // y = x @ Wnewᵀ + b. (Inherited Linear forward does x @ weight.T +
        // bias; we replicate that pattern manually since `weight` here is
        // the frozen base, not the magnitude-scaled active matrix.)
        var y = matmul(x, Wnew.transposed())
        if let b = bias { y = y + b }
        return y
    }
}

/// LoRA wiring config. The default ("QV at rank 4, alpha 8") is the
/// recipe from the original LoRA paper that captures most of the
/// fine-tuning benefit on small adapter sizes.
public struct LoraConfig: Sendable {
    public var rank: Int
    public var alpha: Float
    /// Suffixes of fully-qualified parameter names to wrap (matched at
    /// the Linear leaf level — "q_proj", "fc_in", etc.).
    public var targetSuffixes: [String]
    /// `true` swaps `LoraLinear` for `DoraLinear` at the wrapping step.
    /// DoRA decomposes the weight update into a learnable magnitude +
    /// a LoRA-style direction (Liu et al., 2024). Same rank/alpha
    /// semantics; same targets list.
    public var useDora: Bool
    /// PEFT variant. `.lora` is the historical default. See `PeftVariant`
    /// for the full list. `useDora == true` overrides `variant` and
    /// routes wrapping through `DoraLinear`.
    public var variant: PeftVariant
    /// AdaLoRA's target average rank — the per-Linear rank still equals
    /// `rank`, but at end-of-training we soft-prune via the per-rank
    /// importance scores so the EFFECTIVE rank averages this target.
    /// Negative or zero means "no pruning, train full rank" (the default).
    public var adaLoraTargetRank: Int
    /// Quantization grid used by LoftQ / QLoRA-style initialisation.
    /// The current runtime stores the quantized base as dequantized
    /// floating-point weights; this controls the grid, not packed storage.
    public var quantizationBits: Int

    public init(rank: Int = 4, alpha: Float = 8.0,
                targetSuffixes: [String] = ["q_proj", "v_proj"],
                useDora: Bool = false,
                variant: PeftVariant = .lora,
                adaLoraTargetRank: Int = 0,
                quantizationBits: Int = 4) {
        self.rank = rank
        self.alpha = alpha
        self.targetSuffixes = targetSuffixes
        self.useDora = useDora
        // `useDora == true` is honoured first (it predates the variant
        // enum); callers passing both get DoRA. Otherwise the variant
        // drives everything downstream.
        self.variant = useDora ? .dora : variant
        self.adaLoraTargetRank = adaLoraTargetRank
        self.quantizationBits = min(max(quantizationBits, 2), 8)
    }

    /// Conservative: just QV. Smaller adapter, faster training.
    public static let qv = LoraConfig(rank: 4, alpha: 8.0,
                                       targetSuffixes: ["q_proj", "v_proj"])
    /// More expressive: all attention projections.
    public static let attention = LoraConfig(rank: 8, alpha: 16.0,
                                              targetSuffixes: ["q_proj", "k_proj", "v_proj", "o_proj"])
    /// Maximum: every Linear in the network.
    public static let full = LoraConfig(rank: 8, alpha: 16.0,
                                         targetSuffixes: ["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "fc_in", "fc_out"])
}

/// LoRA+ (Hayou et al., 2024): the B matrix needs a higher learning
/// rate than A to actually use the adapter capacity. The paper
/// recommends λ_B = 16 (i.e. B's effective LR is 16× A's).
///
/// Implemented as a post-grad-fn scaling: walk the gradient tree and
/// multiply any leaf reachable through a `loraB` dict key by the
/// ratio. With a single AdamW, that's equivalent to running B's
/// updates at `lr * ratio` while A stays at `lr`.
///
/// Always called AFTER grad-clipping — the clip sees the un-scaled
/// gradients (matching how per-param-LR LoRA+ implementations clip).
public func scaleLoraBGradients(_ grads: ModuleParameters, ratio: Float) -> ModuleParameters {
    let m = MLXArray(ratio)
    var result = NestedDictionary<String, MLXArray>()
    for (k, v) in grads {
        result[k] = scaleLoraBItem(v, dictKey: k, multiplier: m)
    }
    return result
}

private func scaleLoraBItem(_ item: NestedItem<String, MLXArray>,
                             dictKey: String, multiplier: MLXArray)
    -> NestedItem<String, MLXArray>
{
    switch item {
    case .none: return .none
    case .value(let g): return dictKey == "loraB" ? .value(g * multiplier) : .value(g)
    case .array(let elems):
        // Propagate the parent's dictKey so the "loraB" check fires at the
        // dict level. Arrays in MLX parameter trees are layer lists — the
        // loraB key never appears as an array index.
        return .array(elems.map { scaleLoraBItem($0, dictKey: dictKey, multiplier: multiplier) })
    case .dictionary(let dict):
        var newDict: [String: NestedItem<String, MLXArray>] = [:]
        for (k, v) in dict {
            newDict[k] = scaleLoraBItem(v, dictKey: k, multiplier: multiplier)
        }
        return .dictionary(newDict)
    }
}

/// Build the right adapter wrapper for the chosen variant. DoRA still
/// lives in its own `DoraLinear` class (different param tree — a `m`
/// vector instead of A/B/d); every other variant uses `LoraLinear`
/// with the variant tag driving the init + forward + freeze logic.
public func makeAdapterLinear(wrapping base: Linear, config: LoraConfig) -> Module {
    switch config.variant {
    case .dora:
        return DoraLinear(wrapping: base, rank: config.rank, alpha: config.alpha)
    default:
        return LoraLinear(wrapping: base, rank: config.rank,
                          alpha: config.alpha, variant: config.variant,
                          quantizationBits: config.quantizationBits)
    }
}

/// Unfreeze just the trainable parts of a `LoraLinear` per its variant.
/// Centralised so the from-scratch and HF injection paths can't drift.
public func unfreezeLoraLinear(_ lora: LoraLinear) {
    switch lora.variant {
    case .lora, .rsLora, .pissa, .loftq:
        // Standard LoRA-style training: A and B both trainable.
        lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
    case .loraFA:
        // Frozen-A: only B updates. Halves the trainable params per
        // wrapped Linear; quality matches LoRA on most benchmarks.
        lora.unfreeze(recursive: false, keys: ["loraB"])
    case .vera:
        // VeRA: A and B are frozen random projections; only the per-
        // rank diagonal `d` is trainable (and the per-rank `d` plus an
        // implicit per-output scalar in B's row dim, but we treat them
        // as one — keeping a clean param count comparison).
        lora.unfreeze(recursive: false, keys: ["loraD"])
    case .adaLora:
        // AdaLoRA: A, B, AND the rank-importance scalars all train.
        // The importance score `d` decides which rank components
        // survive pruning at the end of training.
        lora.unfreeze(recursive: false, keys: ["loraA", "loraB", "loraD"])
    case .dora:
        // Unreachable — DoRA wrapping goes through DoraLinear.
        break
    }
}

/// Trainable-element count for one wrapped Linear, per variant.
public func trainableElementCount(of lora: LoraLinear) -> Int {
    switch lora.variant {
    case .lora, .rsLora, .pissa, .loftq:
        return lora.loraA.shape.reduce(1, *) + lora.loraB.shape.reduce(1, *)
    case .loraFA:
        return lora.loraB.shape.reduce(1, *)
    case .vera, .adaLora:
        // VeRA trains only loraD; AdaLoRA trains A + B + d. The unified
        // function returns the trainable count, NOT the parameter count.
        if lora.variant == .vera { return lora.loraD.shape.reduce(1, *) }
        return lora.loraA.shape.reduce(1, *) + lora.loraB.shape.reduce(1, *)
             + lora.loraD.shape.reduce(1, *)
    case .dora: return 0
    }
}

/// Inject LoRA adapters into a TinyGPTModel via `Module.update(modules:)`.
/// The @ModuleInfo storage for q/k/v/o/fc_in/fc_out is private, so we
/// build a ModuleChildren nested dict naming the targets and ask the
/// framework to swap them in.
public enum LoraInjection {
    @discardableResult
    public static func inject(_ model: TinyGPTModel, config: LoraConfig) -> TinyGPTModel {
        let suffixes = Set(config.targetSuffixes)

        // Build the replacement tree:
        //   blocks: [
        //     0: { attn: { q_proj: LoraLinear, v_proj: LoraLinear }, mlp: { ... } },
        //     1: { ... },
        //     ...
        //   ]
        var blocksList: [NestedItem<String, Module>] = []
        for block in model.blocks {
            var attnEntries: [String: NestedItem<String, Module>] = [:]
            var mlpEntries: [String: NestedItem<String, Module>] = [:]
            if suffixes.contains("q_proj") {
                attnEntries["q_proj"] = .value(makeAdapterLinear(wrapping: block.attn.qProj, config: config))
            }
            if suffixes.contains("k_proj") {
                attnEntries["k_proj"] = .value(makeAdapterLinear(wrapping: block.attn.kProj, config: config))
            }
            if suffixes.contains("v_proj") {
                attnEntries["v_proj"] = .value(makeAdapterLinear(wrapping: block.attn.vProj, config: config))
            }
            if suffixes.contains("o_proj") {
                attnEntries["o_proj"] = .value(makeAdapterLinear(wrapping: block.attn.oProj, config: config))
            }
            // MoE blocks have `block.mlp == nil` — they expose router +
            // experts instead of fc_in/fc_out, and aren't LoRA-targetable
            // in this first cut. The attn projections above are still wrapped.
            if let dense = block.mlp {
                if suffixes.contains("fc_in") {
                    mlpEntries["fc_in"] = .value(makeAdapterLinear(wrapping: dense.fcIn, config: config))
                }
                if suffixes.contains("fc_out") {
                    mlpEntries["fc_out"] = .value(makeAdapterLinear(wrapping: dense.fcOut, config: config))
                }
            }
            var blockChildren: [String: NestedItem<String, Module>] = [:]
            if !attnEntries.isEmpty { blockChildren["attn"] = .dictionary(attnEntries) }
            if !mlpEntries.isEmpty { blockChildren["mlp"] = .dictionary(mlpEntries) }
            blocksList.append(.dictionary(blockChildren))
        }
        var root = NestedDictionary<String, Module>()
        root["blocks"] = .array(blocksList)
        model.update(modules: root)
        return model
    }

    /// Count the trainable parameters once LoRA (or DoRA) is injected.
    /// Honours the variant per-Linear so VeRA / LoRA-FA return their
    /// (smaller) count even though the underlying `LoraLinear` still
    /// holds the full A / B matrices for the forward.
    public static func trainableParamCount(in model: TinyGPTModel) -> Int {
        var n = 0
        for block in model.blocks {
            var leaves: [Linear] = [block.attn.qProj, block.attn.kProj,
                                     block.attn.vProj, block.attn.oProj]
            if let dense = block.mlp { leaves.append(contentsOf: [dense.fcIn, dense.fcOut]) }
            for layer in leaves {
                if let lora = layer as? LoraLinear {
                    n += trainableElementCount(of: lora)
                } else if let dora = layer as? DoraLinear {
                    n += dora.loraA.shape.reduce(1, *) + dora.loraB.shape.reduce(1, *)
                        + dora.m.shape.reduce(1, *)
                }
            }
        }
        return n
    }

    /// Freeze base weights; only the variant-specific subset (LoRA's A+B,
    /// LoRA-FA's B, VeRA's d, AdaLoRA's A+B+d, DoRA's A+B+m) gets gradients.
    public static func freezeBase(_ model: TinyGPTModel) {
        model.freeze(recursive: true)
        for block in model.blocks {
            var leaves: [Linear] = [block.attn.qProj, block.attn.kProj,
                                     block.attn.vProj, block.attn.oProj]
            if let dense = block.mlp { leaves.append(contentsOf: [dense.fcIn, dense.fcOut]) }
            for layer in leaves {
                if let lora = layer as? LoraLinear {
                    unfreezeLoraLinear(lora)
                } else if let dora = layer as? DoraLinear {
                    dora.unfreeze(recursive: false, keys: ["loraA", "loraB", "m"])
                }
            }
        }
    }
}
