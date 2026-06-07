import Foundation
import MLX
import MLXNN
import MLXFast
import MLXRandom
import TinyGPTIO

/// CLIP-style ViT vision encoder. Used as the vision tower in the
/// LLaVA-style VLM (see `TinyGPTModelVLM.swift` — added in M3).
///
/// Architecture (CLIP-ViT-L-14 reference):
///
///     image NHWC [B, H, W, 3]
///       ↓ Conv2d patch_embed: kernel=14, stride=14, out_channels=1024
///       ↓ reshape → [B, H/14*W/14, 1024]
///       ↓ prepend class_embedding token → [B, 1+N, 1024]
///       ↓ + position_embedding[0..1+N]
///       ↓ pre_layrnorm                                ← HF typo preserved
///       ↓ 24 × { LN1 → MHA → +x ; LN2 → MLP(quick_gelu) → +x }
///       ↓ post_layernorm
///       → features [B, 1+N, 1024]
///
/// The class token at position 0 is the global image representation; the
/// other tokens are patch features. LLaVA-style cross-modal projection
/// (M2) typically takes the patch tokens (skipping the CLS) and projects
/// them into LLM embedding space.
///
/// Param-name layout (matches HF safetensors under the `vision_model.`
/// prefix, which the loader strips):
///
///     embeddings.class_embedding              → classEmbedding ([1024])
///     embeddings.patch_embedding.weight       → patchEmbed.weight ([1024, 3, 14, 14] PyTorch -> permuted to [1024, 14, 14, 3] MLX NHWC)
///     embeddings.position_embedding.weight    → positionEmbedding.weight ([257, 1024])
///     pre_layrnorm.{weight,bias}              → preLayerNorm   ← HF typo
///     encoder.layers.N.layer_norm1.*          → blocks[N].ln1
///     encoder.layers.N.self_attn.{q,k,v}_proj → blocks[N].attn.{q,k,v}Proj
///     encoder.layers.N.self_attn.out_proj     → blocks[N].attn.outProj
///     encoder.layers.N.layer_norm2.*          → blocks[N].ln2
///     encoder.layers.N.mlp.fc1.*              → blocks[N].mlp.fc1
///     encoder.layers.N.mlp.fc2.*              → blocks[N].mlp.fc2
///     post_layernorm.{weight,bias}            → postLayerNorm
///
/// Image preprocessing (resize + normalize + center crop) lives in
/// `ImagePreprocess.swift` so the encoder stays a pure forward primitive.
public struct VisionConfig: Sendable, Equatable {
    public var hiddenSize: Int           // 1024 for L-14
    public var numHiddenLayers: Int       // 24
    public var numAttentionHeads: Int    // 16
    public var intermediateSize: Int      // 4096
    public var imageSize: Int             // 224
    public var patchSize: Int             // 14
    public var numChannels: Int           // 3
    public var layerNormEps: Float        // 1e-5
    /// Hidden activation. "quick_gelu" for CLIP (x * sigmoid(1.702*x));
    /// "gelu" for newer ViTs. Anything else: defaults to gelu.
    public var hiddenAct: String

    public init(
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        intermediateSize: Int = 4096,
        imageSize: Int = 224,
        patchSize: Int = 14,
        numChannels: Int = 3,
        layerNormEps: Float = 1e-5,
        hiddenAct: String = "quick_gelu"
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.layerNormEps = layerNormEps
        self.hiddenAct = hiddenAct
    }

    /// Number of patch tokens (no CLS).
    public var numPatches: Int { (imageSize / patchSize) * (imageSize / patchSize) }
    /// Sequence length the encoder produces (CLS + patches).
    public var numPositions: Int { numPatches + 1 }
}

/// CLIP multi-head self-attention. Plain MHA — no RoPE, no GQA, no mask
/// (the patch sequence is fully bidirectional). q/k/v/out_proj are
/// separately-parameterised Linears WITH bias (CLIP-style), matching
/// HF's `self_attn.{q,k,v,out}_proj` naming exactly.
public final class CLIPAttention: Module {
    @ModuleInfo(key: "q_proj")  public var qProj: Linear
    @ModuleInfo(key: "k_proj")  public var kProj: Linear
    @ModuleInfo(key: "v_proj")  public var vProj: Linear
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    public let numHeads: Int
    public let headDim: Int

    public init(_ cfg: VisionConfig) {
        precondition(cfg.hiddenSize % cfg.numAttentionHeads == 0,
                     "hidden_size must be divisible by num_attention_heads")
        self.numHeads = cfg.numAttentionHeads
        self.headDim = cfg.hiddenSize / cfg.numAttentionHeads
        self._qProj.wrappedValue = Linear(cfg.hiddenSize, cfg.hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(cfg.hiddenSize, cfg.hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(cfg.hiddenSize, cfg.hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(cfg.hiddenSize, cfg.hiddenSize, bias: true)
        super.init()
    }

    /// x: [B, T, hidden] → [B, T, hidden]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let q = qProj(x).reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
        let scale = 1.0 / sqrt(Float(headDim))
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([B, T, numHeads * headDim])
        return outProj(merged)
    }
}

/// CLIP MLP: fc1 → activation → fc2. Both Linears include biases.
/// Activation is `quick_gelu` for CLIP (x * sigmoid(1.702*x)) or
/// standard `gelu` for newer encoders.
public final class CLIPMlp: Module {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "fc2") public var fc2: Linear
    public let activationName: String

    public init(_ cfg: VisionConfig) {
        self._fc1.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: true)
        self.activationName = cfg.hiddenAct
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        switch activationName {
        case "quick_gelu":
            // Original CLIP "quick_gelu": x * sigmoid(1.702 * x)
            h = h * MLX.sigmoid(MLXArray(1.702) * h)
        case "gelu", "gelu_new":
            h = MLXNN.geluApproximate(h)
        case "gelu_pytorch_tanh":
            h = MLXNN.geluApproximate(h)
        default:
            // Fall back to standard GELU. Newer ViTs (Qwen3-VL etc.)
            // typically set "gelu"; older CLIP uses "quick_gelu".
            h = MLXNN.gelu(h)
        }
        return fc2(h)
    }
}

/// One CLIP encoder layer: pre-norm, attention residual, pre-norm, MLP
/// residual. Matches HF's `CLIPEncoderLayer`.
public final class CLIPEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") public var ln1: LayerNorm
    @ModuleInfo(key: "self_attn")   public var attn: CLIPAttention
    @ModuleInfo(key: "layer_norm2") public var ln2: LayerNorm
    @ModuleInfo(key: "mlp")          public var mlp: CLIPMlp

    public init(_ cfg: VisionConfig) {
        self._ln1.wrappedValue = LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps)
        self._attn.wrappedValue = CLIPAttention(cfg)
        self._ln2.wrappedValue = LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps)
        self._mlp.wrappedValue = CLIPMlp(cfg)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x + attn(ln1(x))
        y = y + mlp(ln2(y))
        return y
    }
}

/// CLIP patch + class-token + position embeddings. Output is the
/// pre-norm-input sequence `[B, 1+N_patches, hidden]`.
///
/// Patch embedding: HF stores the conv weight in PyTorch's OIHW layout
/// `[out_C, in_C, kH, kW]`. MLX-Swift's Conv2d expects NHWC tensors
/// AND OHWI weights `[out_C, kH, kW, in_C]`. The loader (added later)
/// permutes the weight at load time so we can keep using MLXNN's
/// Conv2d here.
public final class CLIPEmbeddings: Module {
    @ParameterInfo(key: "class_embedding") public var classEmbedding: MLXArray
    @ModuleInfo(key: "patch_embedding")    public var patchEmbed: Conv2d
    @ModuleInfo(key: "position_embedding") public var positionEmbedding: Embedding

    public let numPositions: Int

    public init(_ cfg: VisionConfig) {
        self.numPositions = cfg.numPositions
        self._classEmbedding.wrappedValue = MLXRandom.normal([cfg.hiddenSize])
        self._patchEmbed.wrappedValue = Conv2d(
            inputChannels: cfg.numChannels,
            outputChannels: cfg.hiddenSize,
            kernelSize: IntOrPair(cfg.patchSize),
            stride: IntOrPair(cfg.patchSize),
            padding: IntOrPair(0),
            bias: false
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: cfg.numPositions, dimensions: cfg.hiddenSize)
        super.init()
    }

    /// `pixels` is NHWC `[B, H, W, 3]`. Returns `[B, 1+N_patches, hidden]`.
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let B = pixels.shape[0]
        // Conv2d output: NHWC `[B, H/patch, W/patch, hidden]`.
        let patched = patchEmbed(pixels)
        let pH = patched.shape[1]
        let pW = patched.shape[2]
        let hidden = patched.shape[3]
        // Flatten patches → `[B, N_patches, hidden]`.
        var seq = patched.reshaped([B, pH * pW, hidden])
        // Prepend the class token. Broadcast classEmbedding `[hidden]`
        // → `[B, 1, hidden]`.
        let cls = MLX.broadcast(
            classEmbedding.reshaped([1, 1, hidden]),
            to: [B, 1, hidden])
        seq = MLX.concatenated([cls, seq], axis: 1)
        // Positional embedding lookup over [0..numPositions).
        let posIds = MLXArray(0 ..< Int32(numPositions))
        let posEmb = positionEmbedding(posIds)        // [numPositions, hidden]
        seq = seq + posEmb                            // broadcast over batch
        return seq
    }
}

/// The full CLIP vision tower. Forward signature:
///
///     pixels NHWC [B, H, W, 3] → features [B, 1+N, hidden]
///
/// Use case in this codebase: feed pixels in, take the patch tokens
/// (typically `features[:, 1:, :]`) into the cross-modal projection
/// to splice into the LLM token stream (LLaVA convention).
public final class CLIPVisionModel: Module {
    public let config: VisionConfig

    @ModuleInfo(key: "embeddings")    public var embeddings: CLIPEmbeddings
    /// `pre_layrnorm` is the HF spelling — missing 'e' is INTENTIONAL.
    /// CLIP's checkpoints ship with the typo and we have to match it.
    @ModuleInfo(key: "pre_layrnorm")  public var preLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder")       public var encoder: CLIPEncoder
    @ModuleInfo(key: "post_layernorm") public var postLayerNorm: LayerNorm

    public init(_ cfg: VisionConfig) {
        self.config = cfg
        self._embeddings.wrappedValue = CLIPEmbeddings(cfg)
        self._preLayerNorm.wrappedValue = LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps)
        self._encoder.wrappedValue = CLIPEncoder(cfg)
        self._postLayerNorm.wrappedValue = LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps)
        super.init()
    }

    /// Forward returns the FULL `last_hidden_state` sequence — i.e.
    /// `encoder_output` (post pre_layrnorm + 24 layers), with NO
    /// post_layernorm applied. This matches HF's
    /// `CLIPVisionModel.forward → BaseModelOutputWithPooling.last_hidden_state`:
    /// `post_layernorm` is applied ONLY to the CLS-pooled output (see
    /// `pooled(_:)` below). For LLaVA-style cross-modal projection (M2)
    /// we feed `last_hidden_state[:, 1:, :]` (patch tokens, no
    /// post_layernorm) into the projection MLP — that's the LLaVA-1.5
    /// recipe.
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var x = embeddings(pixels)
        x = preLayerNorm(x)
        x = encoder(x)
        return x
    }

    /// HF-equivalent pooled output: `post_layernorm(last_hidden_state[:, 0])`.
    /// CLIP's "image embedding" downstream of the contrastive head uses
    /// this. M1's parity check exercises both this path and the raw
    /// sequence path.
    public func pooled(_ pixels: MLXArray) -> MLXArray {
        let h = self(pixels)
        let cls = h[0..., 0, 0...]  // [B, hidden]
        return postLayerNorm(cls)
    }
}

/// CLIP encoder — sequential stack of `CLIPEncoderLayer`. Wrapped as a
/// distinct module so the HF parameter path `encoder.layers.N` works
/// without an extra translation step.
public final class CLIPEncoder: Module {
    @ModuleInfo(key: "layers") public var layers: [CLIPEncoderLayer]

    public init(_ cfg: VisionConfig) {
        self._layers.wrappedValue = (0..<cfg.numHiddenLayers).map { _ in CLIPEncoderLayer(cfg) }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in layers { y = layer(y) }
        return y
    }
}

/// Parser for CLIP-style `vision_config` blocks. The full CLIP repo
/// (e.g., `openai/clip-vit-large-patch14`) has a nested `vision_config`
/// inside `config.json`. Standalone-vision repos may put these at the
/// top level. This helper accepts either shape.
public enum CLIPVisionConfigParser {
    public static func parse(_ visionDict: [String: Any]) -> VisionConfig {
        return VisionConfig(
            hiddenSize: (visionDict["hidden_size"] as? Int) ?? 1024,
            numHiddenLayers: (visionDict["num_hidden_layers"] as? Int) ?? 24,
            numAttentionHeads: (visionDict["num_attention_heads"] as? Int) ?? 16,
            intermediateSize: (visionDict["intermediate_size"] as? Int) ?? 4096,
            imageSize: (visionDict["image_size"] as? Int) ?? 224,
            patchSize: (visionDict["patch_size"] as? Int) ?? 14,
            numChannels: (visionDict["num_channels"] as? Int) ?? 3,
            layerNormEps: (visionDict["layer_norm_eps"] as? Double).map(Float.init) ?? 1e-5,
            hiddenAct: (visionDict["hidden_act"] as? String) ?? "quick_gelu"
        )
    }
}

/// Load a CLIP vision encoder from a HuggingFace directory containing
/// `config.json` + `model.safetensors`. Only the `vision_model.*`
/// tensors are read; text/projection tensors are skipped. The loader
/// permutes Conv2d weights from PyTorch OIHW to MLX OHWI layout.
///
/// Used by the M1 smoke test and by `HFVLMLoader` (M4) when the parent
/// VLM ships a CLIP-style vision tower.
public enum CLIPVisionLoader {
    public enum LoadError: Error, CustomStringConvertible {
        case missingConfig(URL)
        case noVisionConfig
        case noSafetensors(URL)
        case missingTensor(String)
        case shapeMismatch(name: String, expected: [Int], got: [Int])

        public var description: String {
            switch self {
            case .missingConfig(let u): return "no config.json in \(u.path)"
            case .noVisionConfig: return "config.json has no vision_config block and no top-level vision fields"
            case .noSafetensors(let u): return "no .safetensors in \(u.path)"
            case .missingTensor(let n): return "expected vision tensor '\(n)' not present"
            case .shapeMismatch(let n, let exp, let got): return "\(n) shape mismatch: model wants \(exp), file has \(got)"
            }
        }
    }

    public struct LoadResult {
        public let model: CLIPVisionModel
        public let config: VisionConfig
    }

    /// Load a CLIP vision model. `dir` is the snapshot directory under
    /// `~/.cache/huggingface/hub/.../snapshots/<hash>/`.
    public static func load(from dir: URL) throws -> LoadResult {
        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw LoadError.missingConfig(dir)
        }
        let raw = try Data(contentsOf: configURL)
        let parsed = (try JSONSerialization.jsonObject(with: raw) as? [String: Any]) ?? [:]
        // Prefer nested vision_config (CLIP repos) over top-level fields.
        let visionDict: [String: Any]
        if let nested = parsed["vision_config"] as? [String: Any] {
            visionDict = nested
        } else if parsed["hidden_size"] != nil || parsed["patch_size"] != nil {
            visionDict = parsed
        } else {
            throw LoadError.noVisionConfig
        }
        let cfg = CLIPVisionConfigParser.parse(visionDict)
        let model = CLIPVisionModel(cfg)

        // Walk all .safetensors shards in the dir.
        let allFiles = (try? FileManager.default.contentsOfDirectory(at: dir,
                          includingPropertiesForKeys: nil)) ?? []
        let shards = allFiles
            .filter { $0.pathExtension == "safetensors" }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        guard !shards.isEmpty else { throw LoadError.noSafetensors(dir) }

        // Collect all vision-model tensors. The CLIP repo prefixes
        // every vision tensor with "vision_model."; standalone repos
        // skip the prefix. We try both.
        struct TensorSource {
            let file: SafetensorsReader.File
            let info: SafetensorsReader.TensorInfo
            let hfName: String
        }
        var sources: [String: TensorSource] = [:]
        for shardURL in shards {
            let file = try SafetensorsReader.read(shardURL)
            for (name, info) in file.tensors {
                // Match either "vision_model.<rest>" or just "<rest>"
                // (for standalone vision repos).
                let rest: String? = {
                    if name.hasPrefix("vision_model.") {
                        return String(name.dropFirst("vision_model.".count))
                    } else if name.hasPrefix("vision_tower.") {
                        return String(name.dropFirst("vision_tower.".count))
                    } else if name.hasPrefix("visual.") {
                        return String(name.dropFirst("visual.".count))
                    }
                    return nil
                }()
                guard let r = rest else { continue }
                sources[r] = TensorSource(file: file, info: info, hfName: name)
            }
        }

        // Construct param updates. Most tensors map 1:1 to our
        // @ModuleInfo paths (which mirror HF naming). The Conv2d
        // patch embedding needs an OIHW → OHWI permutation:
        //   PyTorch: [out, in, kH, kW]
        //   MLX:     [out, kH, kW, in]
        // `position_ids` is a non-trainable buffer HF emits; we skip it.
        var updates: [String: MLXArray] = [:]
        for (key, src) in sources {
            if key == "embeddings.position_ids" {
                continue  // non-trainable index buffer
            }
            let bytes = src.file.tensorData(src.hfName)!
            var array = makeMLXArray(bytes: bytes, dtype: src.info.dtype, shape: src.info.shape)
            if key == "embeddings.patch_embedding.weight" && array.ndim == 4 {
                // OIHW [out, in, kH, kW] → OHWI [out, kH, kW, in]
                array = array.transposed(0, 2, 3, 1)
            }
            // The class_embedding ships as a 1-D vector in HF; our
            // ParameterInfo slot also stores it 1-D. No reshape.
            updates[key] = array
        }

        let nested = buildNested(updates, model: model)
        try model.update(parameters: nested, verify: [])
        return LoadResult(model: model, config: cfg)
    }

    /// Convert raw safetensors bytes into an MLXArray. Mirrors the
    /// helper inside `HFModelLoader` — we keep a private copy so the
    /// vision loader stays self-contained (no cross-imports).
    private static func makeMLXArray(bytes: Data, dtype: String, shape: [Int]) -> MLXArray {
        let n = max(1, shape.reduce(1, *))
        switch dtype {
        case "F32":
            let f32 = bytes.withUnsafeBytes { ptr -> [Float] in
                Array(UnsafeBufferPointer<Float>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: n))
            }
            return MLXArray(f32, shape)
        case "F16":
            let f16 = bytes.withUnsafeBytes { ptr -> [UInt16] in
                Array(UnsafeBufferPointer<UInt16>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: n))
            }
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n { out[i] = Float(Float16(bitPattern: f16[i])) }
            return MLXArray(out, shape)
        case "BF16":
            let bf16 = bytes.withUnsafeBytes { ptr -> [UInt16] in
                Array(UnsafeBufferPointer<UInt16>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: n))
            }
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n {
                let bits = UInt32(bf16[i]) << 16
                out[i] = Float(bitPattern: bits)
            }
            return MLXArray(out, shape)
        case "I64":
            let i64 = bytes.withUnsafeBytes { ptr -> [Int64] in
                Array(UnsafeBufferPointer<Int64>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Int64.self),
                    count: n))
            }
            // Position-id buffers ship as I64; cast to Int32 for MLX.
            let asInt32 = i64.map { Int32(clamping: $0) }
            return MLXArray(asInt32, shape)
        default:
            fatalError("unsupported CLIP-vision safetensors dtype: \(dtype)")
        }
    }

    /// Walk model.parameters() and rebuild a nested-dict update tree
    /// keyed by dotted HF-name paths. Mirrors HFModelLoader.buildNested.
    private static func buildNested(_ flat: [String: MLXArray],
                                     model: CLIPVisionModel) -> ModuleParameters {
        var result = NestedDictionary<String, MLXArray>()
        let existing = model.parameters()
        for (key, item) in existing {
            result[key] = rewriteItem(item, path: [key], flat: flat)
        }
        return result
    }

    private static func rewriteItem(_ item: NestedItem<String, MLXArray>,
                                     path: [String],
                                     flat: [String: MLXArray]) -> NestedItem<String, MLXArray> {
        switch item {
        case .none: return .none
        case .value:
            let key = path.joined(separator: ".")
            if let v = flat[key] {
                return .value(v)
            }
            return item
        case .array(let elements):
            return .array(elements.enumerated().map { (i, e) in
                rewriteItem(e, path: path + [String(i)], flat: flat)
            })
        case .dictionary(let dict):
            var newDict: [String: NestedItem<String, MLXArray>] = [:]
            for (k, v) in dict {
                newDict[k] = rewriteItem(v, path: path + [k], flat: flat)
            }
            return .dictionary(newDict)
        }
    }
}
