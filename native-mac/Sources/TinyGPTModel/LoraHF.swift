import Foundation
import MLX
import MLXNN
import TinyGPTIO

/// LoRA injection for TinyGPTModelHF. Parallel to LoraInjection but
/// uses the HF-native @ModuleInfo keys ("layers", "self_attn", etc.)
/// when building the Module.update NestedDictionary. Same on-disk
/// adapter format as the from-scratch path so the same .lora files
/// load against either model class.
public enum LoraInjectionHF {

    @discardableResult
    public static func inject(_ model: TinyGPTModelHF, config: LoraConfig) -> TinyGPTModelHF {
        let suffixes = Set(config.targetSuffixes)
        var layersList: [NestedItem<String, Module>] = []
        for block in model.blocks {
            var attnEntries: [String: NestedItem<String, Module>] = [:]
            var mlpEntries: [String: NestedItem<String, Module>] = [:]
            // Build the adapter via the shared factory so DoRA / VeRA /
            // PISSA / LoftQ / etc. all flow through one path.
            if suffixes.contains("q_proj") {
                attnEntries["q_proj"] = .value(makeAdapterLinear(wrapping: block.attn.qProj, config: config))
            }
            if suffixes.contains("k_proj") {
                attnEntries["k_proj"] = .value(makeAdapterLinear(wrapping: block.attn.kProj, config: config))
            }
            if suffixes.contains("v_proj") {
                attnEntries["v_proj"] = .value(makeAdapterLinear(wrapping: block.attn.vProj, config: config))
            }
            if suffixes.contains("o_proj") {
                attnEntries["o_proj"] = .value(makeAdapterLinear(wrapping: block.attn.oProj, config: config))
            }
            // HF MLP is SwiGLU — the targetable Linears are gate_proj /
            // up_proj / down_proj. Suffixes referencing fc_in / fc_out
            // (from-scratch naming) silently no-op for HF models.
            if suffixes.contains("gate_proj") {
                mlpEntries["gate_proj"] = .value(makeAdapterLinear(wrapping: block.mlp.fcGate, config: config))
            }
            if suffixes.contains("up_proj") {
                mlpEntries["up_proj"] = .value(makeAdapterLinear(wrapping: block.mlp.fcUp, config: config))
            }
            if suffixes.contains("down_proj") {
                mlpEntries["down_proj"] = .value(makeAdapterLinear(wrapping: block.mlp.fcDown, config: config))
            }

            var blockChildren: [String: NestedItem<String, Module>] = [:]
            if !attnEntries.isEmpty { blockChildren["self_attn"] = .dictionary(attnEntries) }
            if !mlpEntries.isEmpty { blockChildren["mlp"] = .dictionary(mlpEntries) }
            layersList.append(.dictionary(blockChildren))
        }
        var root = NestedDictionary<String, Module>()
        root["layers"] = .array(layersList)
        model.update(modules: root)
        return model
    }

    public static func trainableParamCount(in model: TinyGPTModelHF) -> Int {
        var n = 0
        for block in model.blocks {
            let leaves: [Linear] = [block.attn.qProj, block.attn.kProj,
                                    block.attn.vProj, block.attn.oProj,
                                    block.mlp.fcGate, block.mlp.fcUp, block.mlp.fcDown]
            for layer in leaves {
                if let lora = layer as? LoraLinear {
                    n += trainableElementCount(of: lora)
                } else if let dora = layer as? DoraLinear {
                    n += dora.loraA.shape.reduce(1, *) + dora.loraB.shape.reduce(1, *)
                        + dora.m.shape.reduce(1, *)
                }
            }
        }
        return n
    }

    public static func freezeBase(_ model: TinyGPTModelHF) {
        model.freeze(recursive: true)
        for block in model.blocks {
            let leaves: [Linear] = [block.attn.qProj, block.attn.kProj,
                                    block.attn.vProj, block.attn.oProj,
                                    block.mlp.fcGate, block.mlp.fcUp, block.mlp.fcDown]
            for layer in leaves {
                if let lora = layer as? LoraLinear {
                    unfreezeLoraLinear(lora)
                } else if let dora = layer as? DoraLinear {
                    dora.unfreeze(recursive: false, keys: ["loraA", "loraB", "m"])
                }
            }
        }
    }
}

/// LoRA-adapter writer for the HF model. Walks the targeted Linears
/// (which are LoraLinear instances after injection) and saves their
/// A/B matrices. Uses the same `.lora` format the from-scratch path
/// uses, with the architecture-validation header field carrying the
/// HF arch instead.
public enum LoraAdapterHFWriter {
    public static func write(model: TinyGPTModelHF, baseConfig: ModelConfig,
                              loraConfig: LoraConfig, finalLoss: Float?,
                              to url: URL) throws {
        var entries: [LoraAdapter.Entry] = []
        var matrices: [(loraA: [Float], loraB: [Float])] = []

        for (i, block) in model.blocks.enumerated() {
            let attnLinears: [(String, Linear)] = [
                ("self_attn.q_proj", block.attn.qProj),
                ("self_attn.k_proj", block.attn.kProj),
                ("self_attn.v_proj", block.attn.vProj),
                ("self_attn.o_proj", block.attn.oProj),
            ]
            let mlpLinears: [(String, Linear)] = [
                ("mlp.gate_proj", block.mlp.fcGate),
                ("mlp.up_proj",   block.mlp.fcUp),
                ("mlp.down_proj", block.mlp.fcDown),
            ]
            for (suffix, lin) in attnLinears + mlpLinears {
                guard let lora = lin as? LoraLinear else { continue }
                eval(lora.loraA, lora.loraB)
                entries.append(.init(name: "layers.\(i).\(suffix)",
                                     loraAShape: lora.loraA.shape,
                                     loraBShape: lora.loraB.shape))
                matrices.append((lora.loraA.asArray(Float.self),
                                 lora.loraB.asArray(Float.self)))
            }
        }
        let header = LoraAdapter.Header(
            rank: loraConfig.rank, alpha: loraConfig.alpha,
            targetSuffixes: loraConfig.targetSuffixes,
            baseLayers: baseConfig.nLayers, baseDModel: baseConfig.dModel,
            baseCtx: baseConfig.contextLength, baseHeads: baseConfig.nHeads,
            baseDMlp: baseConfig.dMlp,
            entries: entries,
            savedAt: ISO8601DateFormatter().string(from: Date()),
            finalLoss: finalLoss
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try encoder.encode(header)
        var out = Data()
        out.append(contentsOf: LoraAdapterFormat.magic)
        var v = LoraAdapterFormat.currentVersion.littleEndian
        withUnsafeBytes(of: &v) { out.append(contentsOf: $0) }
        var hl = UInt32(headerData.count).littleEndian
        withUnsafeBytes(of: &hl) { out.append(contentsOf: $0) }
        out.append(headerData)
        for (a, b) in matrices {
            a.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
            b.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        }
        try out.write(to: url, options: .atomic)
    }
}

public enum LoraAdapterHFReader {
    /// Load a .lora adapter onto an HF model. The adapter file's
    /// header.targetSuffixes determines what gets wrapped. Architecture
    /// is validated against the model's config (rejects mismatched bases).
    public static func apply(_ adapter: LoraAdapter, to model: TinyGPTModelHF) throws {
        let h = adapter.header
        let cfg = model.config
        guard h.baseLayers == cfg.nLayers,
              h.baseDModel == cfg.dModel,
              h.baseCtx == cfg.contextLength,
              h.baseHeads == cfg.nHeads,
              h.baseDMlp == cfg.dMlp else {
            throw NSError(domain: "TinyGPTLoRA", code: 4,
                          userInfo: [NSLocalizedDescriptionKey:
                            "adapter base config doesn't match loaded HF model"])
        }
        let loraCfg = LoraConfig(rank: h.rank, alpha: h.alpha, targetSuffixes: h.targetSuffixes)
        LoraInjectionHF.inject(model, config: loraCfg)

        // Overwrite the freshly-injected LoraLinears' A/B with the saved
        // values. Build a NestedDictionary of just the LoRA params.
        var layersList: [NestedItem<String, MLXArray>] = []
        var idx = 0
        for _ in model.blocks {
            var attn: [String: NestedItem<String, MLXArray>] = [:]
            var mlp:  [String: NestedItem<String, MLXArray>] = [:]
            let projs: [(name: String, isAttn: Bool, target: Bool)] = [
                ("q_proj",    true,  h.targetSuffixes.contains("q_proj")),
                ("k_proj",    true,  h.targetSuffixes.contains("k_proj")),
                ("v_proj",    true,  h.targetSuffixes.contains("v_proj")),
                ("o_proj",    true,  h.targetSuffixes.contains("o_proj")),
                ("gate_proj", false, h.targetSuffixes.contains("gate_proj")),
                ("up_proj",   false, h.targetSuffixes.contains("up_proj")),
                ("down_proj", false, h.targetSuffixes.contains("down_proj")),
            ]
            for p in projs where p.target {
                let entry = adapter.matrices[idx]
                let aShape = h.entries[idx].loraAShape
                let bShape = h.entries[idx].loraBShape
                let child: NestedItem<String, MLXArray> = .dictionary([
                    "loraA": .value(MLXArray(entry.loraA, aShape)),
                    "loraB": .value(MLXArray(entry.loraB, bShape)),
                ])
                if p.isAttn { attn[p.name] = child } else { mlp[p.name] = child }
                idx += 1
            }
            var entries: [String: NestedItem<String, MLXArray>] = [:]
            if !attn.isEmpty { entries["self_attn"] = .dictionary(attn) }
            if !mlp.isEmpty { entries["mlp"] = .dictionary(mlp) }
            layersList.append(.dictionary(entries))
        }
        var root = NestedDictionary<String, MLXArray>()
        root["layers"] = .array(layersList)
        try model.update(parameters: root, verify: [])
    }
}
