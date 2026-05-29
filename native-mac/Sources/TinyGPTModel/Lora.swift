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
    public let rank: Int
    public let alpha: Float
    public var scale: Float { alpha / Float(rank) }

    /// Wrap an existing Linear with LoRA adapters. The base weight + bias
    /// are reused verbatim (no copy); the adapter matrices are fresh.
    public init(wrapping base: Linear, rank: Int = 4, alpha: Float = 8.0) {
        precondition(rank > 0, "LoRA rank must be > 0")
        self.rank = rank
        self.alpha = alpha
        let outFeatures = base.weight.shape[0]
        let inFeatures = base.weight.shape[1]
        self.loraA = MLXRandom.normal([inFeatures, rank], scale: 0.02)
        self.loraB = MLXArray.zeros([rank, outFeatures])
        super.init(weight: base.weight, bias: base.bias)
    }

    /// Build with explicit A, B (used by the adapter-loader to restore
    /// a saved fine-tune).
    public init(wrapping base: Linear, loraA: MLXArray, loraB: MLXArray,
                rank: Int, alpha: Float) {
        self.rank = rank
        self.alpha = alpha
        self.loraA = loraA
        self.loraB = loraB
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // base(x): inherited Linear forward — x @ weight.T + bias
        // delta:   x @ loraA @ loraB * scale
        let baseOut = super.callAsFunction(x)
        let delta = matmul(matmul(x, loraA), loraB) * MLXArray(scale)
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

    public init(rank: Int = 4, alpha: Float = 8.0,
                targetSuffixes: [String] = ["q_proj", "v_proj"],
                useDora: Bool = false) {
        self.rank = rank
        self.alpha = alpha
        self.targetSuffixes = targetSuffixes
        self.useDora = useDora
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

/// Build the right adapter wrapper (LoraLinear or DoraLinear) per
/// `LoraConfig.useDora`. Returns a `Module` because the two share
/// no protocol; downstream code only needs to know it's a Module
/// for the `update(modules:)` swap-in.
public func makeAdapterLinear(wrapping base: Linear, config: LoraConfig) -> Module {
    if config.useDora {
        return DoraLinear(wrapping: base, rank: config.rank, alpha: config.alpha)
    }
    return LoraLinear(wrapping: base, rank: config.rank, alpha: config.alpha)
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
    public static func trainableParamCount(in model: TinyGPTModel) -> Int {
        var n = 0
        for block in model.blocks {
            var leaves: [Linear] = [block.attn.qProj, block.attn.kProj,
                                     block.attn.vProj, block.attn.oProj]
            if let dense = block.mlp { leaves.append(contentsOf: [dense.fcIn, dense.fcOut]) }
            for layer in leaves {
                if let lora = layer as? LoraLinear {
                    n += lora.loraA.shape.reduce(1, *) + lora.loraB.shape.reduce(1, *)
                } else if let dora = layer as? DoraLinear {
                    n += dora.loraA.shape.reduce(1, *) + dora.loraB.shape.reduce(1, *)
                        + dora.m.shape.reduce(1, *)
                }
            }
        }
        return n
    }

    /// Freeze base weights; only LoRA's A, B (and DoRA's m) get gradients.
    public static func freezeBase(_ model: TinyGPTModel) {
        model.freeze(recursive: true)
        for block in model.blocks {
            var leaves: [Linear] = [block.attn.qProj, block.attn.kProj,
                                     block.attn.vProj, block.attn.oProj]
            if let dense = block.mlp { leaves.append(contentsOf: [dense.fcIn, dense.fcOut]) }
            for layer in leaves {
                if let lora = layer as? LoraLinear {
                    lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
                } else if let dora = layer as? DoraLinear {
                    dora.unfreeze(recursive: false, keys: ["loraA", "loraB", "m"])
                }
            }
        }
    }
}
