import Foundation
import MLX
import MLXNN

/// LLaVA-style VLM wrapper. Composes a CLIP-shape vision encoder + a
/// cross-modal projection + an existing HF-style text LLM into a single
/// forward that takes `(image, text_tokens)` and returns logits over
/// the vocab.
///
/// Forward (M3 — front-prepend; LLaVA-1.5 convention):
///
///     image NHWC [B, H, W, 3]
///       ↓ visionEncoder  → features [B, 1+N_patch, vision_hidden]
///       ↓ drop CLS, take patches [B, N_patch, vision_hidden]
///       ↓ projection      → vision tokens [B, N_patch, llm_hidden]
///     text_tokens [B, T_text]
///       ↓ tokenEmbedding  → text embeddings [B, T_text, llm_hidden]
///     concat(vision_tokens, text_embeddings) [B, N_patch+T_text, llm_hidden]
///       ↓ LLM blocks (forward through the SAME blocks as the text-only model)
///       ↓ lnFinal
///       ↓ lmHead (or tied token embedding)
///       → logits [B, N_patch+T_text, vocab]
///
/// What this model does NOT do yet (each is a separate milestone):
/// - LLaVA-1.6 / Qwen-VL `<image>` token replacement (M8 — needed
///   for OpenAI-compat `image_url` chat shape; M3 uses front-prepend
///   because we only need to verify the forward runs end-to-end here).
/// - Variable image resolution + dynamic patch counts (Qwen2.5-VL /
///   Qwen3-VL — M4-onwards if we adopt that vision tower).
/// - mRoPE / 2D-RoPE (multimodal RoPE for spatial positions) — the
///   underlying `TinyGPTModelHF` block uses 1D RoPE; vision-token
///   positions are treated as flat sequence indices [0..N_patch).
///   When we move to Qwen3-VL as the student we'll need mRoPE.
///
/// - YOCO / MoE / MTP / multi-token-prediction — disabled by
///   construction. `TinyGPTModelVLM` only supports a vanilla
///   forward path through the LLM body. If you build the underlying
///   `TinyGPTModelHF` with `useYOCO=true`, the VLM forward branch
///   panics (we'd be wiring vision tokens into an anchor-K/V
///   capture that wasn't designed for them). Future work — for now
///   the assertion at init catches misuse.
///
/// Param-name layout (consumed by HFVLMLoader at M4):
///     vision_tower.*                 → visionEncoder.*
///     multi_modal_projector.*        → projection.*
///     language_model.*               → llm.*
public final class TinyGPTModelVLM: Module {
    /// Combined config: a `VisionConfig` + a `ModelConfig` for the LLM
    /// body. The projection config is derived (vision_hidden →
    /// llm_dModel) at init.
    public struct Config: Sendable {
        public var vision: VisionConfig
        public var llm: ModelConfig
        /// Projection hidden act. LLaVA-1.5 uses exact GELU; Qwen-VL
        /// projectors use SiLU. Default to GELU.
        public var projectionAct: String

        public init(vision: VisionConfig, llm: ModelConfig, projectionAct: String = "gelu") {
            self.vision = vision
            self.llm = llm
            self.projectionAct = projectionAct
        }
    }

    public let config: Config

    @ModuleInfo(key: "vision_tower")          public var visionEncoder: CLIPVisionModel
    @ModuleInfo(key: "multi_modal_projector") public var projection: CrossModalProjection
    @ModuleInfo(key: "language_model")        public var llm: TinyGPTModelHF

    public init(_ cfg: Config) {
        // VLM forward assumes vanilla LLM (no YOCO / MoE / MTP). Catch
        // misconfig early so we get a clear error instead of a strange
        // numerics fail downstream.
        precondition(!cfg.llm.useYOCO,
            "TinyGPTModelVLM does not yet support YOCO LLM bodies")
        precondition(cfg.llm.nExperts <= 1,
            "TinyGPTModelVLM does not yet support MoE LLM bodies")
        precondition(cfg.llm.mtpHorizons <= 1,
            "TinyGPTModelVLM does not yet support MTP LLM bodies")
        self.config = cfg
        self._visionEncoder.wrappedValue = CLIPVisionModel(cfg.vision)
        self._projection.wrappedValue = CrossModalProjection(
            CrossModalProjectionConfig(
                visionHidden: cfg.vision.hiddenSize,
                llmHidden: cfg.llm.dModel,
                hiddenAct: cfg.projectionAct
            )
        )
        self._llm.wrappedValue = TinyGPTModelHF(cfg.llm)
        super.init()
    }

    /// Embed text tokens, prepend vision tokens, run the LLM blocks +
    /// head. Returns logits over the LLM vocabulary at every output
    /// position.
    ///
    /// `image`: NHWC `[B, H, W, 3]`.
    /// `tokens`: int32 token ids `[B, T_text]`.
    /// Returns: `[B, N_patch + T_text, vocab]`.
    public func callAsFunction(_ image: MLXArray, _ tokens: MLXArray) -> MLXArray {
        // 1) Vision tower + projection. Drop CLS, project patches.
        let visionFeatures = visionEncoder(image)            // [B, 1+N_patch, vis_h]
        let patchOnly = visionFeatures[0..., 1..., 0...]      // [B, N_patch, vis_h]
        let visionTokens = projection(patchOnly)              // [B, N_patch, llm_h]

        // 2) Text token embedding via the LLM's embedding lookup. We
        // reach into the public `tokenEmbedding` slot of TinyGPTModelHF
        // rather than recomputing it — keeps weight tying intact for
        // tied-embedding models (Qwen3, Llama-3-instruct).
        var textEmb = llm.tokenEmbedding(tokens)              // [B, T_text, llm_h]
        if let en = llm.embedNorm {
            textEmb = en(textEmb)
        }

        // 3) Concat vision tokens in FRONT of text embeddings
        // (LLaVA-1.5 convention; explicit doc-comment above explains
        // why we don't do `<image>`-replacement at M3).
        var x = MLX.concatenated([visionTokens, textEmb], axis: 1)

        // 4) LLM blocks. We forward through the same blocks the
        // text-only model uses — vision tokens look like ordinary
        // sequence positions to attention. M4+ may need to inject
        // 2D position info (mRoPE) for the vision-token range
        // specifically; for M3 the test only cares about shape.
        for block in llm.blocks {
            x = block(x)
        }
        x = llm.lnFinal(x)

        // 5) LM head — uses tied embeddings when lmHead is nil.
        if let head = llm.lmHead {
            return head(x)
        }
        return llm.tokenEmbedding.asLinear(x)
    }
}
