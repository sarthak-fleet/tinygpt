import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// AdamW trainer for TinyGPTModelHF. Parallel to Trainer (which is
/// specialised on TinyGPTModel). Same step semantics, same compile
/// option — just bound to the HF model class because Swift generics
/// don't compose nicely with the MLX value-and-grad signature today.
public final class TrainerHF {
    public let model: TinyGPTModelHF
    /// Generic optimiser handle — see `Trainer.optimizer`.
    public let optimizer: any Optimizer & LearningRateMutable
    public let optimizerKind: OptimizerKind
    public private(set) var stepCount: Int = 0
    public let gradClipNorm: Float?
    /// See `Trainer.galore` for the rationale. Forces uncompiled path.
    public let galore: GaLoreManager?
    /// See `Trainer.lrLayerDecay`.
    public let lrLayerDecay: Float
    private let trainStepFn: (MLXArray, MLXArray) -> MLXArray
    /// Compiled gradient-accumulation step — non-nil when caller passes
    /// `accumMicroBatches: N`. See ``Trainer.accumStepFn`` for the
    /// rationale.
    private let accumStepFn: (([MLXArray]) -> [MLXArray])?
    public let compiledAccumN: Int?
    private let gradFn: (TinyGPTModelHF, MLXArray, MLXArray) -> (MLXArray, ModuleParameters)

    public init(model: TinyGPTModelHF,
                learningRate: Float = 3e-4,
                weightDecay: Float = 0.0,
                betas: (Float, Float) = (0.9, 0.95),
                eps: Float = 1e-8,
                compileStep: Bool = true,
                gradClipNorm: Float? = nil,
                optimizer optimizerKind: OptimizerKind = .adamw,
                galore: GaLoreManager? = nil,
                lrLayerDecay: Float = 1.0,
                /// See ``Trainer.init(useCompiledLR:)``. AdamW only.
                useCompiledLR: Bool = false,
                /// See ``Trainer.init(accumMicroBatches:)``.
                accumMicroBatches: Int? = nil) {
        self.model = model
        self.gradClipNorm = gradClipNorm
        self.optimizerKind = optimizerKind
        self.galore = galore
        self.lrLayerDecay = lrLayerDecay
        if useCompiledLR {
            precondition(optimizerKind == .adamw,
                         "useCompiledLR currently supports only --optimizer adamw (got \(optimizerKind.rawValue))")
            self.optimizer = CompiledAdamW(
                learningRate: learningRate,
                betas: betas,
                eps: eps,
                weightDecay: weightDecay
            )
        } else {
            self.optimizer = makeOptimizer(
                kind: optimizerKind,
                learningRate: learningRate,
                weightDecay: weightDecay,
                betas: betas,
                eps: eps
            )
        }
        let m = model
        let opt = self.optimizer
        let clip = gradClipNorm
        let lossFn = { (mod: TinyGPTModelHF, x: MLXArray, y: MLXArray) -> MLXArray in
            mod.loss(x, y)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        self.gradFn = gradFn
        let layerDecay = lrLayerDecay
        let nLayers = model.config.nLayers
        let galoreMgr = galore
        // Same constraint as Trainer: GaLore mutates projector state out-of-graph.
        let canCompile = compileStep && galoreMgr == nil
        if canCompile {
            self.trainStepFn = compile(
                inputs: [m, opt], outputs: [m, opt]
            ) { x, y in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                opt.update(model: m, gradients: processed)
                return loss
            }
        } else {
            self.trainStepFn = { x, y in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                if let g = galoreMgr {
                    processed = g.processGradients(processed)
                }
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                opt.update(model: m, gradients: processed)
                return loss
            }
        }
        // Compiled gradient-accumulation step. See ``Trainer`` for the
        // mechanism — TrainerHF is bound to the HF model class but the
        // shape is identical otherwise.
        if let N = accumMicroBatches, N > 1, canCompile {
            self.compiledAccumN = N
            let nF = Float(N)
            self.accumStepFn = compile(
                inputs: [m, opt], outputs: [m, opt]
            ) { (xs: [MLXArray]) -> [MLXArray] in
                precondition(xs.count == 2 * N,
                             "compiled accum step expects exactly 2N arrays (got \(xs.count) for N=\(N))")
                var accumGrads: ModuleParameters? = nil
                var lossSum = MLXArray(Float(0))
                for i in 0..<N {
                    let x = xs[2 * i]
                    let y = xs[2 * i + 1]
                    let (loss, grads) = gradFn(m, x, y)
                    lossSum = lossSum + loss
                    if let acc = accumGrads {
                        accumGrads = acc.mapValues(grads) { a, b in a + (b ?? a) }
                    } else {
                        accumGrads = grads
                    }
                }
                let scale = MLXArray(1.0 / nF)
                var avg = accumGrads!.mapValues { (g: MLXArray) -> MLXArray in g * scale }
                if let cn = clip { avg = clipGradNorm(avg, maxNorm: cn) }
                if layerDecay < 0.9999 {
                    avg = scaleLayerwiseLR(avg, decay: layerDecay, nLayers: nLayers)
                }
                opt.update(model: m, gradients: avg)
                return [lossSum / MLXArray(nF)]
            }
        } else {
            self.accumStepFn = nil
            self.compiledAccumN = nil
        }
    }

    public func step(inputs: MLXArray, targets: MLXArray) -> Float {
        let loss = trainStepFn(inputs, targets)
        eval(loss, model, optimizer)
        stepCount += 1
        return loss.item(Float.self)
    }

    /// Gradient-accumulated step. Mirrors ``Trainer.accumulatedStep`` —
    /// when `accumStepFn` is non-nil and the caller passes exactly N
    /// micro-batches, the compiled-fold path runs; otherwise it falls
    /// back to a Swift `for` loop over uncompiled `gradFn` calls. The HF
    /// side is otherwise structurally identical to the byte-level Trainer.
    public func accumulatedStep(microBatches: [(MLXArray, MLXArray)]) -> Float {
        precondition(!microBatches.isEmpty, "accumulatedStep needs ≥1 micro-batch")
        if let fn = accumStepFn, let N = compiledAccumN, microBatches.count == N {
            var flat: [MLXArray] = []
            flat.reserveCapacity(2 * N)
            for (x, y) in microBatches { flat.append(x); flat.append(y) }
            let outs = fn(flat)
            let loss = outs[0]
            eval(loss, model, optimizer)
            stepCount += 1
            return loss.item(Float.self)
        }
        var accumGrads: ModuleParameters? = nil
        var lossSum: Float = 0
        let n = microBatches.count
        for (x, y) in microBatches {
            let (loss, grads) = gradFn(model, x, y)
            eval(loss)
            lossSum += loss.item(Float.self)
            if let accum = accumGrads {
                accumGrads = accum.mapValues(grads) { a, b in a + (b ?? a) }
            } else {
                accumGrads = grads
            }
        }
        let scale = MLXArray(1.0 / Float(n))
        var avg = accumGrads!.mapValues { (g: MLXArray) -> MLXArray in g * scale }
        if let cn = gradClipNorm {
            avg = clipGradNorm(avg, maxNorm: cn)
        }
        if let g = galore {
            avg = g.processGradients(avg)
        }
        if lrLayerDecay < 0.9999 {
            avg = scaleLayerwiseLR(avg, decay: lrLayerDecay, nLayers: model.config.nLayers)
        }
        optimizer.update(model: model, gradients: avg)
        eval(model, optimizer)
        stepCount += 1
        return lossSum / Float(n)
    }
}
