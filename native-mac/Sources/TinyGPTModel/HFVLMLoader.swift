import Foundation
import TinyGPTIO

/// HFVLMLoader — load a Qwen3-VL HF directory (e.g. UI-Venus-1.5-2B
/// dequant'd via `scripts/ane/dequant_mlx_generic.py`) into our
/// `TinyGPTModelVLM` + companion structs.
///
/// SKELETON — milestone M4.1 work-in-progress. The data types it
/// produces match what `Qwen3VLScaffold.swift` declares; the actual
/// MLX weight materialization for vision tower + deepstack mergers
/// + LLM body is TODO and gates M4.2-M4.6.
///
/// Math reference for what the forward pass does: see
/// `docs/learn/qwen3-vl-mrope-deepstack.md`.
///
/// Important correction from the HF reference: `deepstack_visual_indexes`
/// in vision_config are VISION-TOWER tap indices, NOT LLM injection
/// indices. Injection happens at LLM layers `[0, 1, 2]` for a model
/// with 3 deepstack tappers. The scaffold's `Qwen3VLDeepstackPlan`
/// currently treats them as LLM indices and must be revised when M4.4
/// lands.
public enum HFVLMLoader {

    public struct Qwen3VLConfig {
        public let textNumLayers: Int       // 28 for UI-Venus-1.5-2B
        public let textHiddenSize: Int      // 2048
        public let textHeadDim: Int         // 128
        public let textNumQHeads: Int       // 16
        public let textNumKVHeads: Int      // 8
        public let textIntermediateSize: Int // 6144
        public let textVocabSize: Int       // 151_936
        public let textRopeTheta: Double    // 5_000_000
        public let textRmsNormEps: Double   // 1e-6
        public let textMRoPESection: [Int]  // [24, 20, 20]
        public let textTieEmbeddings: Bool  // false for UI-Venus

        public let visionDepth: Int         // 24
        public let visionHidden: Int        // 1024
        public let visionPatchSize: Int     // 16
        public let visionSpatialMergeSize: Int  // 2
        public let visionOutHiddenSize: Int // 2048 (= text hidden)
        public let visionDeepstackTapIndexes: [Int]  // [5, 11, 17]
        public let visionFusedQKV: Bool     // true

        public let imageTokenID: Int        // 151_655
        public let visionStartTokenID: Int  // 151_652
        public let visionEndTokenID: Int    // 151_653
        public let videoTokenID: Int        // 151_656

        public static func parse(jsonPath: URL) throws -> Qwen3VLConfig {
            let data = try Data(contentsOf: jsonPath)
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw LoadError.malformedConfig("root not a dict")
            }
            guard let textCfg = root["text_config"] as? [String: Any] else {
                throw LoadError.malformedConfig("missing text_config")
            }
            guard let visionCfg = root["vision_config"] as? [String: Any] else {
                throw LoadError.malformedConfig("missing vision_config")
            }
            // mrope_section lives inside rope_scaling
            var mropeSection: [Int] = [24, 20, 20]
            if let ropeScaling = textCfg["rope_scaling"] as? [String: Any],
               let raw = ropeScaling["mrope_section"] as? [Int] {
                mropeSection = raw
            }
            guard let imageTokenID = root["image_token_id"] as? Int else {
                throw LoadError.malformedConfig("missing image_token_id")
            }

            func intField(_ dict: [String: Any], _ k: String, _ fallback: Int? = nil) throws -> Int {
                if let v = dict[k] as? Int { return v }
                if let f = fallback { return f }
                throw LoadError.malformedConfig("missing field \(k)")
            }
            func doubleField(_ dict: [String: Any], _ k: String, _ fallback: Double) -> Double {
                if let v = dict[k] as? Double { return v }
                if let v = dict[k] as? Int { return Double(v) }
                return fallback
            }
            func boolField(_ dict: [String: Any], _ k: String, _ fallback: Bool) -> Bool {
                (dict[k] as? Bool) ?? fallback
            }

            return try Qwen3VLConfig(
                textNumLayers: try intField(textCfg, "num_hidden_layers"),
                textHiddenSize: try intField(textCfg, "hidden_size"),
                textHeadDim: try intField(textCfg, "head_dim", 128),
                textNumQHeads: try intField(textCfg, "num_attention_heads"),
                textNumKVHeads: try intField(textCfg, "num_key_value_heads"),
                textIntermediateSize: try intField(textCfg, "intermediate_size"),
                textVocabSize: try intField(textCfg, "vocab_size"),
                textRopeTheta: doubleField(textCfg, "rope_theta", 10_000.0),
                textRmsNormEps: doubleField(textCfg, "rms_norm_eps", 1e-6),
                textMRoPESection: mropeSection,
                textTieEmbeddings: boolField(textCfg, "tie_word_embeddings", false),
                visionDepth: try intField(visionCfg, "depth"),
                visionHidden: try intField(visionCfg, "hidden_size"),
                visionPatchSize: try intField(visionCfg, "patch_size", 16),
                visionSpatialMergeSize: try intField(visionCfg, "spatial_merge_size", 2),
                visionOutHiddenSize: try intField(visionCfg, "out_hidden_size"),
                visionDeepstackTapIndexes: visionCfg["deepstack_visual_indexes"] as? [Int] ?? [5, 11, 17],
                visionFusedQKV: true,  // empirically true for UI-Venus; revisit if other Qwen3-VL variants ship
                imageTokenID: imageTokenID,
                visionStartTokenID: root["vision_start_token_id"] as? Int ?? 151_652,
                visionEndTokenID: root["vision_end_token_id"] as? Int ?? 151_653,
                videoTokenID: root["video_token_id"] as? Int ?? 151_656
            )
        }

        private init(textNumLayers: Int, textHiddenSize: Int, textHeadDim: Int,
                     textNumQHeads: Int, textNumKVHeads: Int, textIntermediateSize: Int,
                     textVocabSize: Int, textRopeTheta: Double, textRmsNormEps: Double,
                     textMRoPESection: [Int], textTieEmbeddings: Bool,
                     visionDepth: Int, visionHidden: Int, visionPatchSize: Int,
                     visionSpatialMergeSize: Int, visionOutHiddenSize: Int,
                     visionDeepstackTapIndexes: [Int], visionFusedQKV: Bool,
                     imageTokenID: Int, visionStartTokenID: Int,
                     visionEndTokenID: Int, videoTokenID: Int) throws {
            self.textNumLayers = textNumLayers
            self.textHiddenSize = textHiddenSize
            self.textHeadDim = textHeadDim
            self.textNumQHeads = textNumQHeads
            self.textNumKVHeads = textNumKVHeads
            self.textIntermediateSize = textIntermediateSize
            self.textVocabSize = textVocabSize
            self.textRopeTheta = textRopeTheta
            self.textRmsNormEps = textRmsNormEps
            self.textMRoPESection = textMRoPESection
            self.textTieEmbeddings = textTieEmbeddings
            self.visionDepth = visionDepth
            self.visionHidden = visionHidden
            self.visionPatchSize = visionPatchSize
            self.visionSpatialMergeSize = visionSpatialMergeSize
            self.visionOutHiddenSize = visionOutHiddenSize
            self.visionDeepstackTapIndexes = visionDeepstackTapIndexes
            self.visionFusedQKV = visionFusedQKV
            self.imageTokenID = imageTokenID
            self.visionStartTokenID = visionStartTokenID
            self.visionEndTokenID = visionEndTokenID
            self.videoTokenID = videoTokenID
        }
    }

    public enum LoadError: Error, CustomStringConvertible {
        case missingFile(String)
        case malformedConfig(String)
        case missingTensor(String)
        case unsupportedDtype(String)
        case notImplemented(String)
        public var description: String {
            switch self {
            case .missingFile(let s): return "missing file: \(s)"
            case .malformedConfig(let s): return "malformed config: \(s)"
            case .missingTensor(let s): return "missing tensor: \(s)"
            case .unsupportedDtype(let s): return "unsupported dtype: \(s)"
            case .notImplemented(let s): return "not implemented: \(s)"
            }
        }
    }

    /// Stage 1 — inspect-only load. Reads config.json and every safetensors
    /// shard's manifest WITHOUT materializing weights. Reports tensor
    /// counts + sizes per major branch (vision_tower / language_model).
    ///
    /// This is enough for the M4.1 acceptance test: `tinygpt qwen3vl-load
    /// <dir>` prints "loaded N LLM layers + M vision blocks + K deepstack
    /// mergers + main merger; total bytes X".
    public static func inspect(hfDir: URL) throws -> InspectionReport {
        let cfgURL = hfDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: cfgURL.path) else {
            throw LoadError.missingFile("config.json at \(cfgURL.path)")
        }
        let cfg = try Qwen3VLConfig.parse(jsonPath: cfgURL)

        let shards = try FileManager.default.contentsOfDirectory(
            at: hfDir, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }
         .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else {
            throw LoadError.missingFile("no *.safetensors in \(hfDir.path)")
        }

        var llmLayerSet: Set<Int> = []
        var visionBlockSet: Set<Int> = []
        var deepstackMergerSet: Set<Int> = []
        var hasMainMerger = false
        var totalBytes: Int = 0
        var nTensors = 0

        for shard in shards {
            let f = try SafetensorsReader.read(shard)
            for (name, info) in f.tensors {
                nTensors += 1
                totalBytes += info.byteCount
                if let m = name.range(of: #"language_model\.model\.layers\.(\d+)\."#,
                                       options: .regularExpression) {
                    let numStr = name[m].dropFirst(28).prefix(while: { $0.isNumber })
                    if let n = Int(numStr) { llmLayerSet.insert(n) }
                } else if let m = name.range(of: #"vision_tower\.blocks\.(\d+)\."#,
                                              options: .regularExpression) {
                    let numStr = name[m].dropFirst(20).prefix(while: { $0.isNumber })
                    if let n = Int(numStr) { visionBlockSet.insert(n) }
                } else if let m = name.range(of: #"vision_tower\.deepstack_merger_list\.(\d+)\."#,
                                              options: .regularExpression) {
                    let numStr = name[m].dropFirst(36).prefix(while: { $0.isNumber })
                    if let n = Int(numStr) { deepstackMergerSet.insert(n) }
                } else if name.hasPrefix("vision_tower.merger.") {
                    hasMainMerger = true
                }
            }
        }

        return InspectionReport(
            config: cfg,
            nShards: shards.count,
            nTensors: nTensors,
            totalBytes: totalBytes,
            llmLayers: llmLayerSet.count,
            visionBlocks: visionBlockSet.count,
            deepstackMergers: deepstackMergerSet.count,
            hasMainMerger: hasMainMerger
        )
    }

    public struct InspectionReport {
        public let config: Qwen3VLConfig
        public let nShards: Int
        public let nTensors: Int
        public let totalBytes: Int
        public let llmLayers: Int
        public let visionBlocks: Int
        public let deepstackMergers: Int
        public let hasMainMerger: Bool

        public var summary: String {
            let mb = Double(totalBytes) / 1_048_576
            return """
            HFVLMLoader inspection
              shards:            \(nShards)
              tensors:           \(nTensors)
              total bytes:       \(String(format: "%.0f", mb)) MB
              LLM layers:        \(llmLayers) (config says \(config.textNumLayers))
              vision blocks:     \(visionBlocks) (config says \(config.visionDepth))
              deepstack mergers: \(deepstackMergers) (config taps \(config.visionDeepstackTapIndexes.count))
              main merger:       \(hasMainMerger ? "present" : "MISSING")
              image_token_id:    \(config.imageTokenID)
              vision_start_id:   \(config.visionStartTokenID)
              mrope_section:     \(config.textMRoPESection)
            """
        }
    }

    // ---------------------------------------------------------------
    // M4.1 — weights into TinyGPTModelVLM. NOT YET IMPLEMENTED.
    // ---------------------------------------------------------------
    //
    // The next layer of M4.1 work, after `inspect` passes its smoke
    // test, is to materialize MLX arrays for:
    //   - LLM body (qNorm/kNorm + QK proj + V proj + O proj + RMSNorms
    //     + MLP gates + final norm + lm_head, all weights non-tied for
    //     UI-Venus)
    //   - Vision tower (24 ViT blocks with FUSED QKV — single weight
    //     instead of three separate projections)
    //   - 3 deepstack_merger_list MLPs (one per tap layer)
    //   - 1 main merger MLP
    //   - tokenizer + chat template if not already loaded elsewhere
    //
    // The forward integration (mRoPE in attention, image-token
    // replacement at embed, deepstack residual at LLM layers [0, 1, 2])
    // belongs in M4.2/M4.3/M4.4. See docs/learn/qwen3-vl-mrope-deepstack.md
    // for the exact math.
    //
    // For now: `load()` throws `.notImplemented`.

    @available(*, unavailable, message: "M4.1 weight materialization not yet shipped; use `inspect()` instead")
    public static func load(hfDir: URL) async throws -> Never {
        throw LoadError.notImplemented("HFVLMLoader.load — M4.1 weight materialization is next")
    }
}
