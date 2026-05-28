import Foundation
import MLX
import MLXFast
import MLXNN

/// KV cache for autoregressive sampling.
///
/// Without a cache, every generated token re-runs attention over the full
/// context: O(T²) work per token × T tokens = O(T³) total. With a cache,
/// each step we only compute Q for the new position and use the stored
/// K and V from past positions. Per-token work drops to O(T), making
/// total generation O(T²) — and the practical speedup is 10-50× depending
/// on context length and model depth.
///
/// Per-layer state: stacked tensors of shape `[B, H, T_so_far, D]` for
/// keys and values. Grown by `T_new` (usually 1) each step.
public final class KVCache {
    public struct Entry {
        public var keys: MLXArray   // [B, H, T, D]
        public var values: MLXArray // [B, H, T, D]
    }

    public var entries: [Entry]
    public let nLayers: Int
    public var currentLength: Int = 0

    public init(nLayers: Int) {
        self.nLayers = nLayers
        self.entries = []
        self.entries.reserveCapacity(nLayers)
    }

    public func append(layer: Int, keys: MLXArray, values: MLXArray) {
        if entries.count <= layer {
            // First step — initialise.
            while entries.count <= layer {
                entries.append(Entry(keys: keys, values: values))
            }
        } else {
            // Subsequent steps — concatenate along the time axis (axis=2).
            entries[layer].keys = concatenated([entries[layer].keys, keys], axis: 2)
            entries[layer].values = concatenated([entries[layer].values, values], axis: 2)
        }
    }

    public func keys(layer: Int) -> MLXArray? { entries.indices.contains(layer) ? entries[layer].keys : nil }
    public func values(layer: Int) -> MLXArray? { entries.indices.contains(layer) ? entries[layer].values : nil }
}

/// KV-cached attention forward. Used by `TinyGPTModel.forwardWithCache`.
/// Returns the new keys + values that the caller should append to the
/// cache so the next call can re-use them.
extension CausalSelfAttention {
    public func forwardCached(_ x: MLXArray, cache: KVCache, layer: Int) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        // Project Q/K/V from x; reshape to [B, T, H, D] → transpose to [B, H, T, D]
        let q = qProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let kNew = kProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let vNew = vProj(x).reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)

        // Concatenate with past K, V (if any). Then save back.
        let kFull: MLXArray
        let vFull: MLXArray
        if let kPast = cache.keys(layer: layer), let vPast = cache.values(layer: layer) {
            kFull = concatenated([kPast, kNew], axis: 2)
            vFull = concatenated([vPast, vNew], axis: 2)
            // Replace cache entry rather than append (we already concatenated).
            cache.entries[layer].keys = kFull
            cache.entries[layer].values = vFull
        } else {
            kFull = kNew
            vFull = vNew
            cache.append(layer: layer, keys: kNew, values: vNew)
        }

        // Attention. For the prefill case (T > 1), we need causal masking
        // among the new tokens. For the per-token decode case (T == 1),
        // the new token attends to all past + itself with no masking
        // (single position is trivially valid). MLX-Fast's `.causal` mask
        // works correctly because it masks j > i within the query range
        // (which is 1 row when T_q == 1, so the mask is empty).
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kFull, values: vFull,
            scale: scale,
            mask: T == kFull.shape[2] ? .causal : .none
        )
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])
        return oProj(merged)
    }
}

extension TransformerBlock {
    public func forwardCached(_ x: MLXArray, cache: KVCache, layer: Int) -> MLXArray {
        var x = x
        x = x + attn.forwardCached(ln1(x), cache: cache, layer: layer)
        x = x + mlp(ln2(x))
        return x
    }
}

extension TinyGPTModel {
    /// KV-cached forward pass. On the first call (when `cache` is empty),
    /// processes the full prompt and populates the cache. On subsequent
    /// calls, processes only the new token(s) — typically `idx` is
    /// `[B, 1]` for streaming generation.
    ///
    /// Returns logits of shape `[B, T_new, vocab_size]`.
    public func forwardCached(_ idx: MLXArray, cache: KVCache) -> MLXArray {
        let T = idx.shape[1]
        // Position offset: how many tokens are already in the cache.
        // The new tokens' positions are [cache.currentLength, ..., cache.currentLength + T - 1].
        let basePos = cache.currentLength
        precondition(basePos + T <= config.contextLength,
                     "KV cache + new tokens (\(basePos + T)) exceeds context \(config.contextLength)")
        let positions = MLXArray((0..<T).map { Int32($0 + basePos) })
        let posEmb = positionEmbedding(positions).expandedDimensions(axis: 0) // [1, T, C]
        var x = tokenEmbedding(idx) + posEmb
        for (i, block) in blocks.enumerated() {
            x = block.forwardCached(x, cache: cache, layer: i)
        }
        cache.currentLength = basePos + T
        x = lnFinal(x)
        return tokenEmbedding.asLinear(x)
    }
}
