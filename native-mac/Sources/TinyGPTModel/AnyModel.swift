import Foundation
import MLX
import MLXNN
import TinyGPTIO

/// A unified wrapper so the CLI commands (sample / finetune / compare /
/// eval) can operate on EITHER a from-scratch TinyGPTModel or an
/// HF-loaded TinyGPTModelHF without branching at every call site.
///
/// The wrapper:
///   - exposes the common interface (callAsFunction, loss, numParameters)
///   - forwards to the underlying concrete model
///   - knows which LoRA injection variant to use
public enum AnyModel {
    case fromScratch(TinyGPTModel)
    case huggingFace(TinyGPTModelHF)

    public var config: ModelConfig {
        switch self {
        case .fromScratch(let m): return m.config
        case .huggingFace(let m): return m.config
        }
    }

    public func callAsFunction(_ idx: MLXArray) -> MLXArray {
        switch self {
        case .fromScratch(let m): return m(idx)
        case .huggingFace(let m): return m(idx)
        }
    }

    public func loss(_ idx: MLXArray, _ targets: MLXArray) -> MLXArray {
        switch self {
        case .fromScratch(let m): return m.loss(idx, targets)
        case .huggingFace(let m):
            // HF model doesn't have a built-in loss helper; compute inline.
            let logits = m(idx)
            let v = logits.shape.last!
            return crossEntropy(
                logits: logits.reshaped([-1, v]),
                targets: targets.reshaped([-1]),
                reduction: .mean
            )
        }
    }

    public func numParameters() -> Int {
        switch self {
        case .fromScratch(let m): return m.numParameters()
        case .huggingFace(let m): return m.numParameters()
        }
    }

    public func parameters() -> ModuleParameters {
        switch self {
        case .fromScratch(let m): return m.parameters()
        case .huggingFace(let m): return m.parameters()
        }
    }

    /// Inject LoRA on the right variant; returns trainable param count.
    @discardableResult
    public func injectLora(config: LoraConfig) -> Int {
        switch self {
        case .fromScratch(let m):
            LoraInjection.inject(m, config: config)
            LoraInjection.freezeBase(m)
            return LoraInjection.trainableParamCount(in: m)
        case .huggingFace(let m):
            LoraInjectionHF.inject(m, config: config)
            LoraInjectionHF.freezeBase(m)
            return LoraInjectionHF.trainableParamCount(in: m)
        }
    }

    /// Apply a saved LoRA adapter to whichever variant we are.
    public func applyLora(_ adapter: LoraAdapter) throws {
        switch self {
        case .fromScratch(let m):
            try LoraAdapterReader.apply(adapter, to: m)
        case .huggingFace(let m):
            try LoraAdapterHFReader.apply(adapter, to: m)
        }
    }

    /// Save a LoRA adapter to disk. The model must have been injected
    /// + trained; this serialises just the A/B matrices.
    public func saveLora(baseConfig: ModelConfig, loraConfig: LoraConfig,
                          finalLoss: Float?, to url: URL) throws {
        switch self {
        case .fromScratch(let m):
            try LoraAdapterWriter.write(model: m, baseConfig: baseConfig,
                                          loraConfig: loraConfig,
                                          finalLoss: finalLoss, to: url)
        case .huggingFace(let m):
            try LoraAdapterHFWriter.write(model: m, baseConfig: baseConfig,
                                            loraConfig: loraConfig,
                                            finalLoss: finalLoss, to: url)
        }
    }

    /// Underlying Module — used by `freeze`/`unfreeze`/optimiser plumbing.
    public var module: Module {
        switch self {
        case .fromScratch(let m): return m
        case .huggingFace(let m): return m
        }
    }
}

/// Detect whether a path is a from-scratch `.tinygpt` checkpoint or an
/// HuggingFace model directory. Returns the loaded model + its config.
public enum ModelLoader {
    public struct LoadResult {
        public let model: AnyModel
        public let config: ModelConfig
        /// HF-model variants come with their own tokenizer on disk.
        /// Set to the model directory's URL so callers can load the
        /// tokenizer separately. nil for from-scratch byte-level models.
        public let hfTokenizerDir: URL?
    }

    public static func load(_ path: String) throws -> LoadResult {
        let url = URL(fileURLWithPath: path)
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)

        if isDirectory.boolValue {
            // HF model directory — expects config.json inside.
            let configURL = url.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw NSError(domain: "TinyGPT", code: 10,
                              userInfo: [NSLocalizedDescriptionKey:
                                "directory \(path) has no config.json — not an HF model dir"])
            }
            let hfResult = try HFModelLoader.load(from: url)
            return LoadResult(model: .huggingFace(hfResult.model),
                              config: hfResult.config, hfTokenizerDir: url)
        }

        // .tinygpt file path
        let file = try TinyGPTFileReader.read(url)
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )
        let m = TinyGPTModel(cfg)
        try TinyGPTWeightLoader.load(file, into: m)
        return LoadResult(model: .fromScratch(m), config: cfg, hfTokenizerDir: nil)
    }
}
