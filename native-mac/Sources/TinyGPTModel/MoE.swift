import Foundation
import MLX
import MLXNN

/// Mixture-of-Experts MLP block (Switch Transformer / Mixtral style).
///
/// A learned router picks the top-`topK` experts for each token; the
/// final output is the per-token-weighted sum of those experts' outputs.
/// Each expert is the same shape as the dense baseline's MLP (`MLP`),
/// so a dense d_mlp=1024 model and a `nExperts=8` MoE model with the
/// same d_mlp expose 8× the parameter capacity per layer while
/// activating only `topK/nExperts` of those params per token.
///
/// **First-cut implementation: dense compute.** Every expert runs on
/// every token; the router weight zeros out non-selected experts. This
/// is mathematically equivalent to true sparse dispatch but doesn't
/// realise the FLOP saving — that needs a scatter-gather kernel (next
/// optimisation pass). The architecture, training dynamics, and
/// parameter-capacity story are all correct as-is.
///
/// Reference: Fedus et al., 2021 (Switch Transformer); Jiang et al.,
/// 2024 (Mixtral-of-Experts). Load-balance loss form follows Switch
/// Transformer's `α · N · Σ_e (f_e · P_e)` recipe.
public final class MoEMLP: Module {
    @ModuleInfo(key: "router") public var router: Linear
    @ModuleInfo(key: "experts") public var experts: [MLP]
    public let nExperts: Int
    public let topK: Int

    /// Side-channel for the auxiliary load-balance loss. The training
    /// step pulls this AFTER the model forward and adds it (scaled by
    /// `loadBalanceWeight`) to the main NLL. Boxed so the closure that
    /// computes the loss can read it after the forward writes it,
    /// without threading return tuples through every Module signature.
    public final class AuxLossBox: @unchecked Sendable {
        public var value: MLXArray = MLXArray(Float(0))
    }
    public let auxLoss = AuxLossBox()

    public init(_ cfg: ModelConfig) {
        precondition(cfg.nExperts > 1, "MoEMLP needs nExperts > 1")
        self.nExperts = cfg.nExperts
        self.topK = cfg.moeTopK
        // Router is a single bias-free Linear from d_model → n_experts.
        // No bias keeps the routing decisions purely activation-driven,
        // which is the Switch Transformer recipe.
        self._router.wrappedValue = Linear(cfg.dModel, cfg.nExperts, bias: false)
        self._experts.wrappedValue = (0..<cfg.nExperts).map { _ in MLP(cfg) }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let logits = router(x)                                      // [B, T, E]
        let probs = MLX.softmax(logits, axis: -1)                   // [B, T, E]

        // Top-1 fast path (Switch Transformer): the common case is K=1
        // and it lets us skip the argSort. argMax → one-hot multiplied
        // by the full probs is mathematically identical to "renormalised
        // top-1 = the prob itself, since renorm by 1 == 1, then *prob".
        let chosenIdx: MLXArray             // [B, T] int32
        let chosenWeight: MLXArray          // [B, T, E] — nonzero only at chosen expert
        if topK == 1 {
            chosenIdx = argMax(logits, axis: -1)
            let oneHot = MoEMLP.oneHotLastAxis(chosenIdx, classes: nExperts, dtype: probs.dtype)
            // Use the router prob at the chosen position. This keeps the
            // weight differentiable (chosenIdx itself isn't, but the prob
            // tensor IS), so gradients flow through to the router.
            chosenWeight = oneHot * probs                            // [B, T, E]
        } else {
            // Top-K via descending argSort + slice. Build a soft mask
            // that's `prob` at top-K positions and 0 elsewhere, then
            // renormalise so the kept probs sum to 1 per token.
            let asc = argSort(probs, axis: -1)                       // [B, T, E] ascending
            // The top-K indices live at the END of the sorted array
            // (highest probs). We grab them and build a one-hot per K.
            // E - topK ≤ idx < E.
            var topKMask = MLXArray.zeros([B, T, nExperts]).asType(probs.dtype)
            for offset in 0..<topK {
                let idxK = asc[0..., 0..., nExperts - 1 - offset]    // [B, T]
                let oneHotK = MoEMLP.oneHotLastAxis(idxK, classes: nExperts, dtype: probs.dtype)
                topKMask = topKMask + oneHotK
            }
            chosenIdx = argMax(logits, axis: -1)
            let masked: MLXArray = probs * topKMask                  // [B, T, E]
            let denom: MLXArray = masked.sum(axis: -1, keepDims: true) + MLXArray(Float(1e-9))
            chosenWeight = masked / denom
        }

        // Run every expert. Each emits [B, T, C]; weight by its column
        // of `chosenWeight` and sum. With dense compute, the per-token
        // FLOPs are nExperts × MLP — the router's contribution is
        // shape-correct, the saving lands when we replace this loop
        // with a scatter-gather kernel.
        var out = MLXArray.zeros(x.shape).asType(x.dtype)
        for e in 0..<nExperts {
            let weight = chosenWeight[0..., 0..., e ..< (e + 1)]     // [B, T, 1]
            out = out + experts[e](x) * weight
        }

        // Load-balance loss (Switch Transformer):
        //   L_aux = N · Σ_e ( f_e · P_e )
        // where f_e = fraction of tokens routed to expert e
        //       P_e = mean router prob assigned to expert e (across tokens)
        // Zero when f and P are uniform → no penalty; maximal when one
        // expert dominates both → big penalty pushing the router back.
        let routedMask = MoEMLP.oneHotLastAxis(chosenIdx, classes: nExperts, dtype: probs.dtype)
        let fractionE: MLXArray = routedMask.mean(axes: [0, 1])      // [E]
        let probsMeanE: MLXArray = probs.mean(axes: [0, 1])          // [E]
        let prod: MLXArray = fractionE * probsMeanE
        let aux: MLXArray = MLXArray(Float(nExperts)) * prod.sum()
        auxLoss.value = aux

        return out
    }

    /// One-hot encode an integer `[*]` tensor along a NEW trailing axis.
    /// MLX-Swift doesn't ship a `oneHot` op as of writing, so we build it
    /// with a broadcasted equality: `idx[..., None] == arange(classes)`.
    /// Returns a tensor of shape `idx.shape + [classes]` in `dtype`.
    private static func oneHotLastAxis(_ idx: MLXArray, classes: Int, dtype: DType) -> MLXArray {
        let arange = MLXArray(0..<Int32(classes))                    // [classes] int32
        let idxAs = idx.asType(.int32).expandedDimensions(axis: -1)  // [..., 1]
        return (idxAs .== arange).asType(dtype)
    }
}

/// Sum the auxiliary load-balance losses across every MoE block in the
/// model. Returns a SCALAR MLXArray (zero if no MoE in the model).
///
/// The training step calls this AFTER the main forward + loss, then
/// folds the result into the gradient target as
/// `loss = mainLoss + cfg.loadBalanceWeight * sumAuxLosses()`.
public func sumMoEAuxLosses(_ blocks: [TransformerBlock]) -> MLXArray {
    var acc = MLXArray(Float(0))
    for b in blocks {
        if let moe = b.mlpUnit as? MoEMLP {
            acc = acc + moe.auxLoss.value
        }
    }
    return acc
}

public func sumMoEAuxLossesHF(_ blocks: [TransformerBlockHF]) -> MLXArray {
    var acc = MLXArray(Float(0))
    for b in blocks {
        if let moe = b.mlpUnit as? MoEMLP {
            acc = acc + moe.auxLoss.value
        }
    }
    return acc
}
