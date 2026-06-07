import Foundation
import MLX
import TinyGPTModel

/// Helper bridge — runs a single full-prompt forward through the MLX-Swift
/// HF model and returns the last-position logits as a `[Float]`.
///
/// Split into its own file so AneValidate.swift can stay CoreML-focused
/// (CoreML is conditionally imported; this side always pulls in MLX).
func mlxForwardLast(model: TinyGPTModelHF, ids: [Int32]) -> [Float] {
    let arr = MLXArray(ids, [1, ids.count])
    let logits = model(arr)             // [1, T, vocab]
    let last = logits[0..., logits.shape[1] - 1, 0...]  // [1, vocab]
    eval(last)
    return last.asArray(Float.self)
}
