import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Compile-friendly AdamW where the learning rate is an ``MLXArray`` scalar
/// rather than a Swift ``Float``. That single change is what lets us put
/// the whole train step — including cosine/warmup LR — inside a single
/// `compile(...)` trace and re-use it across every step, instead of
/// burning a fresh trace each time LR changes (or, today, gating compile
/// off entirely when LR scheduling is on).
///
/// Implementation mirrors MLX-Swift's ``MLXOptimizers/AdamW``:
///
///     m_t = β1·m + (1 − β1)·g
///     v_t = β2·v + (1 − β2)·g²
///     θ   = θ · (1 − lr · wd)            // decoupled weight decay
///     θ  ←  θ − lr · m_t / (sqrt(v_t) + ε)
///
/// Differences from MLX-Swift's AdamW:
///   * `learningRate` is an ``MLXArray`` (scalar, fp32). When set via the
///     ``LearningRateMutable`` protocol the scalar is updated **in place**
///     via `_updateInternal`, which keeps any captured compiled graph
///     pointing at the same array object — same trace, fresh value.
///   * No bias correction (matches MLX-Swift's choice — comment in their
///     `AdamW.applySingle`).
///   * State (m, v) is per-leaf, allocated lazily on first sight.
///
/// This optimiser is wired up by ``Trainer.init(... useCompiledLR: true)``
/// and ``TrainerHF`` when the caller wants the compiled-with-schedule
/// path; it's invisible elsewhere.
public final class CompiledAdamW: Optimizer, LearningRateMutable {
    /// LR as a scalar MLXArray. Reused across every step — we never
    /// re-bind this property after init, only update its contents.
    public let lrArray: MLXArray
    public var beta1: Float
    public var beta2: Float
    public var eps: Float
    public var weightDecay: Float

    /// Per-parameter (m, v). Same shape strategy as the project's other
    /// custom optimisers (see Sophia / Muon in Optimizers.swift).
    private var stateStorage = NestedDictionary<String, PairState>()

    public init(
        learningRate: Float,
        betas: (Float, Float) = (0.9, 0.95),
        eps: Float = 1e-8,
        weightDecay: Float = 0.1
    ) {
        // Build the LR as a fresh array so we own a stable identity for
        // the lifetime of this optimiser. The compile trace captures
        // **this** MLXArray; subsequent setLR(...) calls mutate it
        // in-place via `_updateInternal`.
        self.lrArray = MLXArray(learningRate)
        self.beta1 = betas.0
        self.beta2 = betas.1
        self.eps = eps
        self.weightDecay = weightDecay
    }

    /// ``LearningRateMutable`` conformance. The setter writes through to
    /// the stable ``lrArray`` so any compiled closure keeps working.
    public var learningRate: Float {
        get { lrArray.item(Float.self) }
        set { lrArray._updateInternal(MLXArray(newValue)) }
    }

    /// Two contributions to `innerState`: the per-parameter (m, v) AND
    /// the LR scalar. Listing the LR here lets `compile(inputs: [opt])`
    /// pick it up as part of the optimiser's state, which is what makes
    /// LR changes between steps "free" (no re-trace).
    public func innerState() -> [MLXArray] {
        [lrArray] + stateStorage.flattenedValues().flatMap { $0.innerState() }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let modelParams = model.parameters()
        let (p, s) = gradients.mapValues(modelParams, stateStorage) {
            (grad, param, state) -> (MLXArray, PairState?) in
            let pState = state ?? PairState(zeros: param!)
            let (newParam, newState) = applySingle(
                gradient: grad, parameter: param!, state: pState
            )
            return (newParam, newState)
        }
        self.stateStorage = s
        model.update(parameters: p)
    }

    private func applySingle(
        gradient: MLXArray, parameter: MLXArray, state: PairState
    ) -> (MLXArray, PairState) {
        var m = state.a
        var v = state.b
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient * gradient)

        // Decoupled weight-decay: shrink params *before* subtracting
        // the Adam update. Same shape as MLX-Swift's AdamW.
        var p = parameter
        if weightDecay > 0 {
            p = p * (1 - lrArray * MLXArray(weightDecay))
        }
        let update = m / (MLX.sqrt(v) + MLXArray(eps))
        return (p - lrArray * update, PairState(m, v))
    }
}
