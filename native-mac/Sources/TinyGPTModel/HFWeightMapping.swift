import Foundation
import TinyGPTIO

/// Translation table from HuggingFace parameter names to TinyGPTModel's.
/// Most modern transformer architectures use one of a few naming
/// conventions; we handle the dominant ones (Llama / Mistral / Phi-3,
/// LFM2, Qwen2, Gemma2) and reject the rest with a clear error.
///
/// The HF naming pattern most repos use:
///
///     model.embed_tokens.weight                            → token_embedding.weight
///     model.layers.0.self_attn.q_proj.weight               → blocks.0.attn.q_proj.weight
///     model.layers.0.self_attn.k_proj.weight               → blocks.0.attn.k_proj.weight
///     model.layers.0.self_attn.v_proj.weight               → blocks.0.attn.v_proj.weight
///     model.layers.0.self_attn.o_proj.weight               → blocks.0.attn.o_proj.weight
///     model.layers.0.mlp.gate_proj.weight                  → (SwiGLU; see note)
///     model.layers.0.mlp.up_proj.weight                    → (SwiGLU; see note)
///     model.layers.0.mlp.down_proj.weight                  → blocks.0.mlp.fc_out.weight
///     model.layers.0.input_layernorm.weight                → blocks.0.ln1.weight
///     model.layers.0.post_attention_layernorm.weight       → blocks.0.ln2.weight
///     model.norm.weight                                    → ln_final.weight
///     lm_head.weight                                       → (untied LM head; we tie by default)
///
/// SwiGLU note: Llama-family models use gated MLP (gate × up → silu → down),
/// not a plain feedforward (fc_in → gelu → fc_out). TinyGPTModel today is
/// the plain version. Loading a SwiGLU model means we'd need to either
///   (a) add a SwiGLU block variant to TinyGPTModel and pick at load time, or
///   (b) approximate by merging gate+up into fc_in (loses some quality).
/// (a) is the right answer; (b) is the demo-quality shortcut. This file
/// flags both Llama-family SwiGLU configs and exits with a clear message
/// until (a) lands.
public enum HFWeightMapping {
    /// Map a single HF param name to our TinyGPTModel name, or return nil
    /// if this name doesn't correspond to anything we own.
    public static func map(_ hfName: String) -> String? {
        if hfName == "model.embed_tokens.weight" { return "token_embedding.weight" }
        if hfName == "model.norm.weight" || hfName == "model.norm.bias" {
            return hfName.replacingOccurrences(of: "model.norm", with: "ln_final")
        }
        if hfName == "lm_head.weight" {
            // Untied LM head — we tie by default; loader needs to choose
            // whether to import this slot (untie) or skip it.
            return "lm_head.weight"
        }
        // Per-layer pattern: model.layers.N.{...}
        guard hfName.hasPrefix("model.layers.") else { return nil }
        let rest = String(hfName.dropFirst("model.layers.".count))
        guard let dotIdx = rest.firstIndex(of: "."),
              let layerN = Int(rest[..<dotIdx]) else { return nil }
        let suffix = String(rest[rest.index(after: dotIdx)...])
        switch suffix {
        case "self_attn.q_proj.weight": return "blocks.\(layerN).attn.q_proj.weight"
        case "self_attn.q_proj.bias":   return "blocks.\(layerN).attn.q_proj.bias"
        case "self_attn.k_proj.weight": return "blocks.\(layerN).attn.k_proj.weight"
        case "self_attn.k_proj.bias":   return "blocks.\(layerN).attn.k_proj.bias"
        case "self_attn.v_proj.weight": return "blocks.\(layerN).attn.v_proj.weight"
        case "self_attn.v_proj.bias":   return "blocks.\(layerN).attn.v_proj.bias"
        case "self_attn.o_proj.weight": return "blocks.\(layerN).attn.o_proj.weight"
        case "self_attn.o_proj.bias":   return "blocks.\(layerN).attn.o_proj.bias"
        case "mlp.down_proj.weight":    return "blocks.\(layerN).mlp.fc_out.weight"
        case "mlp.down_proj.bias":      return "blocks.\(layerN).mlp.fc_out.bias"
        case "mlp.up_proj.weight":      return "blocks.\(layerN).mlp.fc_in.weight"
        case "mlp.up_proj.bias":        return "blocks.\(layerN).mlp.fc_in.bias"
        case "mlp.gate_proj.weight":    return "blocks.\(layerN).mlp.gate_proj.weight"  // SwiGLU
        case "input_layernorm.weight":  return "blocks.\(layerN).ln1.weight"
        case "input_layernorm.bias":    return "blocks.\(layerN).ln1.bias"
        case "post_attention_layernorm.weight": return "blocks.\(layerN).ln2.weight"
        case "post_attention_layernorm.bias":   return "blocks.\(layerN).ln2.bias"
        default:
            // Family-specific bits not yet wired (rotary inv_freq buffer,
            // QK norm, etc.) — let the loader decide whether to skip or fail.
            return nil
        }
    }

    /// What's needed to load this config that TinyGPTModel doesn't have yet.
    /// Returns an empty array if we can load. Returns one entry per
    /// missing capability so the user sees the full punch list.
    public static func missingCapabilities(for cfg: HuggingFaceConfig) -> [String] {
        var missing: [String] = []

        // Activation: TinyGPTModel uses GELU. Most HF models use SiLU+SwiGLU.
        if cfg.hiddenAct == "silu" {
            missing.append("SwiGLU MLP (model uses silu + gate_proj/up_proj/down_proj; our MLP is plain fc_in → GELU → fc_out)")
        } else if !["gelu", "gelu_new", "gelu_pytorch_tanh"].contains(cfg.hiddenAct) {
            missing.append("activation '\(cfg.hiddenAct)' not yet wired")
        }

        // Norm: TinyGPTModel uses LayerNorm. Most HF use RMSNorm.
        // Heuristic: the presence of rms_norm_eps in the config (vs
        // layer_norm_eps) signals RMSNorm. We don't have direct access to
        // the raw key here, so we check the architectures list as a proxy.
        let rmsArchs = ["LlamaForCausalLM", "MistralForCausalLM",
                         "Qwen2ForCausalLM", "Phi3ForCausalLM",
                         "Gemma2ForCausalLM", "LFM2ForCausalLM"]
        if cfg.architectures.contains(where: { rmsArchs.contains($0) }) {
            missing.append("RMSNorm (our model uses LayerNorm — adding RMSNorm is a 30-line shim around MLXFast.rmsNorm)")
        }

        // Positional encoding: we use learned positional embeddings; HF
        // models use RoPE (rotary). Different code path entirely.
        missing.append("RoPE positional encoding (we use learned embeddings; HF models apply rotary to q,k inside attention)")

        // GQA: when num_heads > num_kv_heads
        if cfg.numAttentionHeads != cfg.numKeyValueHeads {
            missing.append("Grouped Query Attention: heads=\(cfg.numAttentionHeads), kv_heads=\(cfg.numKeyValueHeads) — need to broadcast KV during attention")
        }

        // Vocab: we hard-code 256 (byte-level). HF models use 32K-150K BPE.
        if cfg.vocabSize > 256 {
            missing.append("BPE tokenizer for vocab_size=\(cfg.vocabSize) (our tokenizer is byte-level; ports needed for SentencePiece / tiktoken style)")
        }

        return missing
    }
}
