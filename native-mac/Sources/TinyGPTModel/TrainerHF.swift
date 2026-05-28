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
    public let optimizer: AdamW
    public private(set) var stepCount: Int = 0
    private let trainStepFn: (MLXArray, MLXArray) -> MLXArray

    public init(model: TinyGPTModelHF,
                learningRate: Float = 3e-4,
                weightDecay: Float = 0.0,
                betas: (Float, Float) = (0.9, 0.95),
                eps: Float = 1e-8,
                compileStep: Bool = false) {
        self.model = model
        self.optimizer = AdamW(learningRate: learningRate, betas: betas,
                                eps: eps, weightDecay: weightDecay)
        let m = model
        let opt = self.optimizer
        let lossFn = { (mod: TinyGPTModelHF, x: MLXArray, y: MLXArray) -> MLXArray in
            let logits = mod(x)
            let v = logits.shape.last!
            return crossEntropy(logits: logits.reshaped([-1, v]),
                                targets: y.reshaped([-1]),
                                reduction: .mean)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        if compileStep {
            self.trainStepFn = compile(
                inputs: [m, opt], outputs: [m, opt]
            ) { x, y in
                let (loss, grads) = gradFn(m, x, y)
                opt.update(model: m, gradients: grads)
                return loss
            }
        } else {
            self.trainStepFn = { x, y in
                let (loss, grads) = gradFn(m, x, y)
                opt.update(model: m, gradients: grads)
                return loss
            }
        }
    }

    public func step(inputs: MLXArray, targets: MLXArray) -> Float {
        let loss = trainStepFn(inputs, targets)
        eval(loss, model, optimizer)
        stepCount += 1
        return loss.item(Float.self)
    }
}
