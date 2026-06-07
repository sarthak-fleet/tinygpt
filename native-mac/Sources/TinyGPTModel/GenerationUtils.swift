import Foundation
import MLX

public enum GenerationUtils {
    /// Simple greedy generation for short eval prompts. Re-feeds the current
    /// context window each step; callers that need high-throughput sampling
    /// should use the KV-cache path in the sample command instead.
    public static func greedyGenerate(
        model: AnyModel,
        cfg: ModelConfig,
        tokenizer: HFTokenizer?,
        prompt: String,
        maxNewTokens: Int
    ) -> String {
        let promptIds: [Int32]
        if let tok = tokenizer {
            promptIds = ((try? tok.encode(prompt)) ?? []).map { Int32($0) }
        } else {
            promptIds = [UInt8](prompt.utf8).map { Int32($0) }
        }

        let ctx = cfg.contextLength
        let seed = promptIds.count > ctx
            ? Array(promptIds[(promptIds.count - ctx)..<promptIds.count])
            : promptIds
        guard !seed.isEmpty else { return "" }

        var idx = MLXArray(seed, [1, seed.count])
        var generated: [Int32] = []
        for _ in 0..<maxNewTokens {
            let t = idx.shape.last!
            let lo = max(0, t - ctx)
            let cond = idx[0..., lo..<t]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let next = MLX.argMax(last, axis: -1).reshaped([1, 1])
            eval(next)
            let id = Int32(next.item(Int32.self))
            generated.append(id)
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }

        if let tok = tokenizer {
            return tok.decode(generated.map { Int($0) })
        }
        var out = ""
        for id in generated {
            if let scalar = UnicodeScalar(Int(id)), id >= 9 {
                out.append(Character(scalar))
            }
        }
        return out
    }
}
