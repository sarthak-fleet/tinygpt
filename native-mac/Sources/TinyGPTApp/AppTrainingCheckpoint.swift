import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

enum AppTrainingCheckpoint {
    static var defaultURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
            .appendingPathComponent("tinygpt")
            .appendingPathComponent("app-training")
            .appendingPathComponent("paused.tinygpt")
    }

    static func save(
        model: TinyGPTModel,
        cfg: ModelConfig,
        trainer: Trainer,
        step: Int,
        loss: Float,
        to url: URL = defaultURL
    ) throws {
        let tmp = URL(fileURLWithPath: url.path + ".tmp")
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let entries = manifestEntries(cfg)
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let moments = optimizerMoments(from: trainer)
        var tensors: [TinyGPTTensor] = []
        tensors.reserveCapacity(entries.count)

        for entry in entries {
            guard let param = params[entry.name] else {
                throw NSError(
                    domain: "TinyGPTAppCheckpoint",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "missing param \(entry.name)"]
                )
            }
            var weight = param
            var m = moments[entry.name]?.m
            var v = moments[entry.name]?.v
            if isLinearWeightName(entry.name) && weight.shape.count == 2 {
                weight = weight.transposed()
                m = m?.transposed()
                v = v?.transposed()
            }
            eval(weight)
            if let m, let v { eval(m, v) }
            let wf: [Float] = weight.asArray(Float.self)
            let weightData = wf.withUnsafeBufferPointer { Data(buffer: $0) }
            let momentData: (Data, Data)
            if let m, let v {
                let mf: [Float] = m.asArray(Float.self)
                let vf: [Float] = v.asArray(Float.self)
                momentData = (
                    mf.withUnsafeBufferPointer { Data(buffer: $0) },
                    vf.withUnsafeBufferPointer { Data(buffer: $0) }
                )
            } else {
                let zeros = Data(count: weightData.count)
                momentData = (zeros, zeros)
            }
            tensors.append(TinyGPTTensor(
                entry: entry,
                weight: weightData,
                adamM: momentData.0,
                adamV: momentData.1,
                dtype: .fp32
            ))
        }

        let header = TinyGPTHeader(
            config: .init(
                layers: cfg.nLayers,
                dModel: cfg.dModel,
                ctx: cfg.contextLength,
                heads: cfg.nHeads,
                dMlp: cfg.dMlp,
                batchSize: 8,
                backend: "TinyGPTApp",
                vocabSize: cfg.vocabSize == 256 ? nil : cfg.vocabSize,
                tokenizerSource: cfg.tokenizerSource,
                useGradCheckpoint: cfg.useGradCheckpoint ? true : nil
            ),
            manifest: entries,
            savedAt: ISO8601DateFormatter().string(from: Date()),
            finalLoss: .init(step: step, train: Double(loss), val: nil),
            sample: nil,
            weightDtype: "fp32",
            includesOptimizerState: true,
            stateByteLength: 4 + tensors.reduce(0) { $0 + 3 * $1.weight.count }
        )
        let file = TinyGPTFile(
            version: TinyGPTFormat.currentVersion,
            header: header,
            step: Int32(step),
            tensors: tensors
        )
        try TinyGPTFileWriter.write(file, to: tmp)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        try FileManager.default.moveItem(at: tmp, to: url)
    }

    static func load(_ url: URL) throws -> (file: TinyGPTFile, cfg: ModelConfig, model: TinyGPTModel) {
        let file = try TinyGPTFileReader.read(url)
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: h.vocabSize ?? 256,
            contextLength: h.ctx ?? 128,
            nLayers: h.layers ?? 4,
            nHeads: h.heads ?? 4,
            dModel: h.dModel ?? 128,
            dMlp: h.dMlp ?? 512,
            tokenizerSource: h.tokenizerSource,
            useGradCheckpoint: h.useGradCheckpoint ?? false
        )
        let model = TinyGPTModel(cfg)
        try TinyGPTWeightLoader.load(file, into: model)
        return (file, cfg, model)
    }

    static func restoreOptimizerState(from file: TinyGPTFile, into trainer: Trainer) -> Bool {
        guard let adam = trainer.optimizer as? CompiledAdamW else { return false }
        let moments = optimizerMoments(from: file)
        guard !moments.isEmpty else { return false }
        return adam.importMoments(moments, matching: trainer.model)
    }

    private static func optimizerMoments(from trainer: Trainer) -> [String: (m: MLXArray, v: MLXArray)] {
        guard let adam = trainer.optimizer as? CompiledAdamW else { return [:] }
        return Dictionary(uniqueKeysWithValues: adam.exportMoments().map { ($0.name, (m: $0.m, v: $0.v)) })
    }

    private static func optimizerMoments(from file: TinyGPTFile) -> [(name: String, m: MLXArray, v: MLXArray)] {
        file.tensors.compactMap { tensor in
            guard tensor.adamM.contains(where: { $0 != 0 }) || tensor.adamV.contains(where: { $0 != 0 }) else {
                return nil
            }
            if isLinearWeightName(tensor.entry.name) && tensor.entry.shape.count == 2 {
                let shape = [tensor.entry.shape[1], tensor.entry.shape[0]]
                return (
                    tensor.entry.name,
                    MLXArray(tensor.adamM, shape, dtype: .float32).transposed(),
                    MLXArray(tensor.adamV, shape, dtype: .float32).transposed()
                )
            }
            return (
                tensor.entry.name,
                MLXArray(tensor.adamM, tensor.entry.shape, dtype: .float32),
                MLXArray(tensor.adamV, tensor.entry.shape, dtype: .float32)
            )
        }
    }

    private static func manifestEntries(_ cfg: ModelConfig) -> [TinyGPTHeader.TensorEntry] {
        var entries: [TinyGPTHeader.TensorEntry] = []
        var offset = 0
        func push(_ name: String, _ shape: [Int]) {
            let size = shape.reduce(1, *)
            entries.append(.init(name: name, shape: shape, floatOffset: offset))
            offset += size
        }
        let c = cfg.dModel
        let m = cfg.dMlp
        push("token_embedding.weight", [cfg.vocabSize, c])
        push("position_embedding.weight", [cfg.contextLength, c])
        push("ln_final.weight", [c])
        push("ln_final.bias", [c])
        for i in 0..<cfg.nLayers {
            push("blocks.\(i).ln1.weight", [c])
            push("blocks.\(i).ln1.bias", [c])
            push("blocks.\(i).attn.q_proj.weight", [c, c])
            push("blocks.\(i).attn.q_proj.bias", [c])
            push("blocks.\(i).attn.k_proj.weight", [c, c])
            push("blocks.\(i).attn.k_proj.bias", [c])
            push("blocks.\(i).attn.v_proj.weight", [c, c])
            push("blocks.\(i).attn.v_proj.bias", [c])
            push("blocks.\(i).attn.o_proj.weight", [c, c])
            push("blocks.\(i).attn.o_proj.bias", [c])
            push("blocks.\(i).ln2.weight", [c])
            push("blocks.\(i).ln2.bias", [c])
            push("blocks.\(i).mlp.fc_in.weight", [m, c])
            push("blocks.\(i).mlp.fc_in.bias", [m])
            push("blocks.\(i).mlp.fc_out.weight", [c, m])
            push("blocks.\(i).mlp.fc_out.bias", [c])
        }
        push("lm_head.weight", [cfg.vocabSize, c])
        push("lm_head.bias", [cfg.vocabSize])
        return entries
    }

    private static func isLinearWeightName(_ name: String) -> Bool {
        guard name.hasSuffix(".weight") else { return false }
        if name == "token_embedding.weight" || name == "position_embedding.weight" {
            return false
        }
        if name.hasSuffix(".ln1.weight") || name.hasSuffix(".ln2.weight") || name == "ln_final.weight" {
            return false
        }
        return true
    }
}

