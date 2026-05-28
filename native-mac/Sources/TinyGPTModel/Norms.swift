import Foundation
import MLX
import MLXNN
import MLXFast

/// Common normalization layers used by transformer architectures.
/// TinyGPTModel today is built around LayerNorm (mean + variance). HF
/// modern models (Llama 2+, Mistral, Phi-3, Gemma, LFM, Qwen) all use
/// RMSNorm (no mean subtraction, no bias) — cheaper and empirically
/// equivalent on transformer training.
///
/// RMSNorm formula:
///     y = (x / sqrt(mean(x^2) + eps)) * weight
///
/// vs LayerNorm:
///     y = ((x - mean(x)) / sqrt(var(x) + eps)) * weight + bias
///
/// RMSNorm dropped the mean-centering and the bias, in exchange for
/// 30% fewer FLOPs and one fewer trainable scalar per dimension.

/// Drop-in replacement for LayerNorm that uses RMSNorm semantics.
/// Used by HF-imported models (Llama-family architectures) — see
/// `HFWeightMapping.missingCapabilities`. Dispatches to MLXFast.rmsNorm
/// when possible (fused kernel, faster on Metal).
public final class RMSNorm: Module, UnaryLayer {
    public let weight: MLXArray
    public let eps: Float
    public let dimensions: Int

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.dimensions = dimensions
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
        super.init()
    }

    /// Init from a loaded weight tensor (HF safetensors path).
    public init(weight: MLXArray, eps: Float = 1e-5) {
        self.dimensions = weight.shape.last ?? 0
        self.eps = eps
        self.weight = weight
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLXFast.rmsNorm fuses the rsqrt + scale; ~2× faster than the
        // expanded form on Metal. We pass `weight` (1D), the kernel
        // broadcasts across the leading axes.
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Enum-tagged norm choice so TinyGPTModel can pick at config time
/// without two parallel block classes. The block's @ModuleInfo slot
/// holds a UnaryLayer (the parent type both LayerNorm and RMSNorm
/// conform to via MLXNN's protocol hierarchy).
public enum NormKind {
    case layerNorm
    case rmsNorm

    /// Build a fresh norm of the given kind.
    public func build(dimensions: Int, eps: Float = 1e-5) -> Module & UnaryLayer {
        switch self {
        case .layerNorm: return LayerNorm(dimensions: dimensions, eps: eps)
        case .rmsNorm:   return RMSNorm(dimensions: dimensions, eps: eps)
        }
    }
}
