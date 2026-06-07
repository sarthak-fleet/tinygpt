import Foundation
import MLX
import MLXNN

/// LLaVA-style cross-modal projection. Maps the ViT patch-feature
/// space (`visionHidden`) into the LLM embedding space (`llmHidden`)
/// so vision tokens can be spliced into the text token stream.
///
/// Architecture (LLaVA-1.5 reference):
///
///     vision_features [B, N_patches, vision_hidden]
///       ↓ Linear(vision_hidden → llm_hidden)
///       ↓ GELU (exact, not approximate)
///       ↓ Linear(llm_hidden → llm_hidden)
///       → vision_tokens [B, N_patches, llm_hidden]
///
/// Why no normalisation here: LLaVA-1.5 deliberately omits it because
/// the LLM body's own ln-pre-attention handles the scale problem.
/// LLaVA-1.6 / Qwen2-VL add a normalisation step inside the projector
/// when the LLM body is sensitive (e.g., when the LLM uses RMSNorm
/// and the projection scale matters for first-layer attention). We
/// can add it later via a feature flag.
///
/// Why no weight loading here: the projection is trained from scratch
/// in M5 alongside the LoRA on the LLM body. The PRD calls this out
/// explicitly: "initialize randomly; will train via M5". For Qwen3-VL
/// / LLaVA-1.5 checkpoint reuse (M4) the projection weights ship as
/// part of the parent VLM safetensors and the loader splices them in.
///
/// Param layout (matches LLaVA-1.5's `multi_modal_projector.*` naming
/// when this projection sits inside a TinyGPTModelVLM at M3, so the HF
/// VLM loader (M4) can splice weights directly without re-mapping):
///
///     linear_1.{weight, bias}     → fc1 (vision_hidden → llm_hidden)
///     linear_2.{weight, bias}     → fc2 (llm_hidden → llm_hidden)
///
/// Some VLM repos (Qwen2-VL) name them `mlp.0` / `mlp.2` instead. The
/// HF VLM loader (M4) will normalise those names before applying. The
/// in-Swift @ModuleInfo keys stay LLaVA-1.5 — that's the most common.
public struct CrossModalProjectionConfig: Sendable, Equatable {
    public var visionHidden: Int
    public var llmHidden: Int
    /// Hidden activation. Default is exact GELU (LLaVA-1.5 spec). Set
    /// to "gelu_approximate" for the tanh-approximation variant.
    public var hiddenAct: String

    public init(visionHidden: Int, llmHidden: Int, hiddenAct: String = "gelu") {
        self.visionHidden = visionHidden
        self.llmHidden = llmHidden
        self.hiddenAct = hiddenAct
    }
}

public final class CrossModalProjection: Module {
    @ModuleInfo(key: "linear_1") public var fc1: Linear
    @ModuleInfo(key: "linear_2") public var fc2: Linear

    public let config: CrossModalProjectionConfig

    public init(_ cfg: CrossModalProjectionConfig) {
        self.config = cfg
        self._fc1.wrappedValue = Linear(cfg.visionHidden, cfg.llmHidden, bias: true)
        self._fc2.wrappedValue = Linear(cfg.llmHidden, cfg.llmHidden, bias: true)
        super.init()
    }

    /// `x`: `[B, N_patches, vision_hidden]` → `[B, N_patches, llm_hidden]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        switch config.hiddenAct {
        case "gelu", "gelu_new":
            h = MLXNN.gelu(h)
        case "gelu_approximate", "gelu_pytorch_tanh":
            h = MLXNN.geluApproximate(h)
        case "quick_gelu":
            h = h * MLX.sigmoid(MLXArray(1.702) * h)
        case "silu", "swish":
            h = MLXNN.silu(h)
        default:
            h = MLXNN.gelu(h)
        }
        return fc2(h)
    }
}
