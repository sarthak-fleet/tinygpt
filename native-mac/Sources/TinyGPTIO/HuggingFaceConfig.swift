import Foundation

/// Parser for HuggingFace `config.json` files. Most modern transformer
/// repos (Llama, Mistral, Phi, LFM, Qwen, Gemma) share a common subset of
/// fields, with a few model-family-specific extras.
///
/// We decode the common fields strictly and pass everything else through
/// as `extras` so model-family-specific loaders can read what they need.
public struct HuggingFaceConfig {
    /// Architectures listed in the config — e.g. ["LlamaForCausalLM"],
    /// ["PhiForCausalLM"], ["LFM2ForCausalLM"]. Use this to dispatch the
    /// right model-family adapter.
    public let architectures: [String]

    // Vocabulary and embedding dims
    public let vocabSize: Int
    public let hiddenSize: Int             // a.k.a. d_model
    public let intermediateSize: Int       // a.k.a. d_mlp / d_ff
    public let numHiddenLayers: Int        // a.k.a. n_layers
    public let numAttentionHeads: Int      // a.k.a. n_heads (query heads)
    public let numKeyValueHeads: Int       // for GQA; defaults to numAttentionHeads
    /// Per-head channel count. When the config sets `head_dim` explicitly
    /// (Qwen3, Llama-3.1+) the value can be DIFFERENT from
    /// `hidden_size / num_attention_heads` — Qwen3-0.6B for example has
    /// hidden_size=1024, num_attention_heads=16, head_dim=128. Falls back
    /// to `hidden_size / num_attention_heads` when absent (Phi-3, Llama-2,
    /// Mistral 7B v0.1).
    public let headDim: Int
    public let maxPositionEmbeddings: Int  // a.k.a. context_length

    // Norm + activation
    public let rmsNormEps: Float
    public let hiddenAct: String           // "silu", "gelu", "gelu_pytorch_tanh", etc.

    // RoPE
    public let ropeTheta: Float            // base frequency, typically 10_000 or 500_000
    public let ropeScaling: [String: Any]? // {type, factor} for LongRoPE / NTK etc.

    // Tied embeddings
    public let tieWordEmbeddings: Bool

    /// Whatever else the config carries — model-family-specific (e.g.
    /// LFM's `block_dim`, Phi's `partial_rotary_factor`).
    public let extras: [String: Any]

    public static func read(_ url: URL) throws -> HuggingFaceConfig {
        let data = try Data(contentsOf: url)
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            ?? [:]
        return try fromDict(raw)
    }

    public static func fromDict(_ d: [String: Any]) throws -> HuggingFaceConfig {
        func req<T>(_ k: String, _ type: T.Type) throws -> T {
            guard let v = d[k] as? T else {
                throw HFConfigError.missingOrWrongType(field: k, expected: "\(T.self)")
            }
            return v
        }
        let arch = (d["architectures"] as? [String]) ?? []
        let numHeads = try req("num_attention_heads", Int.self)
        let nkvh = (d["num_key_value_heads"] as? Int) ?? numHeads
        let hidden = try req("hidden_size", Int.self)
        // Explicit `head_dim` overrides the derived `hidden_size / num_heads`.
        // Qwen3-0.6B: hidden_size=1024, num_heads=16, head_dim=128 — Q out
        // is num_heads*head_dim=2048, NOT hidden_size. Phi-3 / Llama-2
        // omit the field and the fallback gives the canonical value.
        let hd = (d["head_dim"] as? Int) ?? (hidden / numHeads)
        let normEps = (d["rms_norm_eps"] as? Double).map(Float.init)
            ?? (d["layer_norm_eps"] as? Double).map(Float.init)
            ?? 1e-5
        let ropeBase = (d["rope_theta"] as? Double).map(Float.init) ?? 10_000.0
        let act = (d["hidden_act"] as? String) ?? "silu"
        let tied = (d["tie_word_embeddings"] as? Bool) ?? false

        // Whitelist of known keys — anything else passes through as extras
        let known: Set<String> = [
            "architectures", "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim",
            "max_position_embeddings", "rms_norm_eps", "layer_norm_eps",
            "hidden_act", "rope_theta", "rope_scaling", "tie_word_embeddings",
            "model_type", "torch_dtype", "transformers_version",
        ]
        let extras = d.filter { !known.contains($0.key) }

        return HuggingFaceConfig(
            architectures: arch,
            vocabSize: try req("vocab_size", Int.self),
            hiddenSize: hidden,
            intermediateSize: try req("intermediate_size", Int.self),
            numHiddenLayers: try req("num_hidden_layers", Int.self),
            numAttentionHeads: numHeads,
            numKeyValueHeads: nkvh,
            headDim: hd,
            maxPositionEmbeddings: try req("max_position_embeddings", Int.self),
            rmsNormEps: normEps,
            hiddenAct: act,
            ropeTheta: ropeBase,
            ropeScaling: d["rope_scaling"] as? [String: Any],
            tieWordEmbeddings: tied,
            extras: extras
        )
    }

    /// Used by the TinyGPTModel adapter to decide if it can run this
    /// architecture without further engineering. Returns nil if all
    /// known constraints pass; otherwise a human-readable reason.
    public func unsupportedReason() -> String? {
        if hiddenAct != "silu" && hiddenAct != "gelu" && hiddenAct != "gelu_new"
            && hiddenAct != "gelu_pytorch_tanh" {
            return "hidden activation '\(hiddenAct)' isn't in our shortlist (silu / gelu)"
        }
        // GQA (numAttentionHeads != numKeyValueHeads) is wired into the
        // attention path via cfg.nKvHeads + MLX-Fast SDPA's native head-
        // broadcast. No blocker here.
        if ropeScaling != nil {
            return "RoPE scaling configuration present — needs the long-context variant"
        }
        if numAttentionHeads % numKeyValueHeads != 0 {
            return "num_attention_heads \(numAttentionHeads) isn't divisible by num_key_value_heads \(numKeyValueHeads)"
        }
        return nil
    }
}

public enum HFConfigError: Error, CustomStringConvertible {
    case missingOrWrongType(field: String, expected: String)
    public var description: String {
        switch self {
        case .missingOrWrongType(let f, let t):
            return "config.json: field '\(f)' missing or not \(t)"
        }
    }
}
