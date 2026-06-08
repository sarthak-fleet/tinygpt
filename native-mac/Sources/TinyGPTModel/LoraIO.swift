import Foundation
import MLX
import MLXNN

/// On-disk format for a LoRA adapter — separate from `.tinygpt`. A small
/// header + the A/B matrices for each wrapped Linear. Bytes:
///
///     0    4    magic = "TGLA"  (TinyGPT LoRA Adapter)
///     4    4    version (u32; v1 = LoRA-only; v2 = LoRA+DoRA optional)
///     8    4    header_len (u32)
///     12   N    JSON header — { config, modelConfig, entries: [{name, shape, loraMShape?}, ...] }
///     12+N      raw fp32 matrices in manifest order. Per entry:
///                 [in × r] floats   — loraA
///                 [r × out] floats  — loraB
///                 [out] floats      — magnitude (ONLY if entry.loraMShape != null, v2+)
///
/// The header carries the BASE model's config so the adapter refuses to
/// load against an architecture-mismatched checkpoint.
///
/// v1 → v2 change (2026-06-09): added optional per-entry magnitude vector
/// for DoRA-trained adapters. v1 readers can load v1 files (no m), v2
/// readers can load both v1 (no m fields anywhere) and v2 (m present iff
/// loraMShape is non-nil in the entry).
public enum LoraAdapterFormat {
    public static let magic: [UInt8] = Array("TGLA".utf8)
    public static let currentVersion: UInt32 = 2
    public static let minSupportedVersion: UInt32 = 1
}

public struct LoraAdapter {
    public struct Entry: Codable, Equatable, Sendable {
        public var name: String           // "blocks.0.attn.q_proj"
        public var loraAShape: [Int]      // [in, r]
        public var loraBShape: [Int]      // [r, out]
        public var loraMShape: [Int]?     // [out] — DoRA magnitude. Nil for plain LoRA. (v2+)
        public init(name: String, loraAShape: [Int], loraBShape: [Int],
                    loraMShape: [Int]? = nil) {
            self.name = name; self.loraAShape = loraAShape
            self.loraBShape = loraBShape; self.loraMShape = loraMShape
        }
    }
    public struct Header: Codable, Sendable {
        public var rank: Int
        public var alpha: Float
        public var targetSuffixes: [String]
        public var baseLayers: Int
        public var baseDModel: Int
        public var baseCtx: Int
        public var baseHeads: Int
        public var baseDMlp: Int
        public var entries: [Entry]
        public var savedAt: String?
        public var finalLoss: Float?
    }

    public var header: Header
    /// Per-entry trio. `m` is non-nil iff the entry was trained as DoRA
    /// (i.e. `header.entries[i].loraMShape != nil`). For plain-LoRA
    /// entries, `m` is nil and the consumer treats the entry as standard
    /// `(loraA, loraB)` only. Matches `header.entries` order.
    public var matrices: [(loraA: [Float], loraB: [Float], m: [Float]?)]
}

public enum LoraAdapterWriter {
    public static func write(model: TinyGPTModel, baseConfig: ModelConfig,
                              loraConfig: LoraConfig, finalLoss: Float?,
                              to url: URL) throws {
        var entries: [LoraAdapter.Entry] = []
        var matrices: [(loraA: [Float], loraB: [Float], m: [Float]?)] = []

        // Per-block linears in target order. MoE blocks (where `mlp` is
        // nil) only contribute the four attention projections; MLP-LoRA
        // on MoE-expert weights is future work.
        for (i, block) in model.blocks.enumerated() {
            var projections: [(String, Linear)] = [
                ("attn.q_proj", block.attn.qProj),
                ("attn.k_proj", block.attn.kProj),
                ("attn.v_proj", block.attn.vProj),
                ("attn.o_proj", block.attn.oProj),
            ]
            if let dense = block.mlp {
                projections.append(("mlp.fc_in",  dense.fcIn))
                projections.append(("mlp.fc_out", dense.fcOut))
            }
            for (suffix, lin) in projections {
                // Accept both LoraLinear and DoraLinear. v2 format
                // preserves DoRA's magnitude vector; v1 readers silently
                // ignore the loraMShape field, so v2 files with no DoRA
                // entries are also v1-readable.
                let lA: MLXArray
                let lB: MLXArray
                var mFloats: [Float]? = nil
                var mShape: [Int]? = nil
                if let lora = lin as? LoraLinear {
                    lA = lora.loraA; lB = lora.loraB
                } else if let dora = lin as? DoraLinear {
                    lA = dora.loraA; lB = dora.loraB
                    eval(dora.m)
                    mFloats = dora.m.asArray(Float.self)
                    mShape = dora.m.shape
                } else { continue }
                eval(lA, lB)
                let aFloats = lA.asArray(Float.self)
                let bFloats = lB.asArray(Float.self)
                entries.append(.init(
                    name: "blocks.\(i).\(suffix)",
                    loraAShape: lA.shape,
                    loraBShape: lB.shape,
                    loraMShape: mShape
                ))
                matrices.append((loraA: aFloats, loraB: bFloats, m: mFloats))
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
        appendU32(&out, LoraAdapterFormat.currentVersion)
        appendU32(&out, UInt32(headerData.count))
        out.append(headerData)
        for (a, b, m) in matrices {
            a.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
            b.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
            if let m {
                m.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
            }
        }
        try out.write(to: url, options: .atomic)
    }

    private static func appendU32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }
}

public enum LoraAdapterReader {
    public static func read(_ url: URL) throws -> LoraAdapter {
        let data = try Data(contentsOf: url)
        guard data.count >= 12 else {
            throw NSError(domain: "TinyGPTLoRA", code: 1, userInfo: [NSLocalizedDescriptionKey: "file too small"])
        }
        let magicBytes = Array(data[0..<4])
        guard magicBytes == LoraAdapterFormat.magic else {
            throw NSError(domain: "TinyGPTLoRA", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "bad magic, expected 'TGLA'"])
        }
        let version = data[4..<8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
        guard version >= LoraAdapterFormat.minSupportedVersion,
              version <= LoraAdapterFormat.currentVersion else {
            throw NSError(domain: "TinyGPTLoRA", code: 3,
                          userInfo: [NSLocalizedDescriptionKey:
                            "unsupported version \(version) (need \(LoraAdapterFormat.minSupportedVersion)…\(LoraAdapterFormat.currentVersion))"])
        }
        let headerLen = Int(data[8..<12].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
        let header = try JSONDecoder().decode(LoraAdapter.Header.self,
                                              from: data.subdata(in: 12..<(12 + headerLen)))
        var cursor = 12 + headerLen
        var matrices: [(loraA: [Float], loraB: [Float], m: [Float]?)] = []
        for entry in header.entries {
            let aSize = entry.loraAShape.reduce(1, *) * 4
            let bSize = entry.loraBShape.reduce(1, *) * 4
            let aData = data.subdata(in: cursor..<(cursor + aSize)); cursor += aSize
            let bData = data.subdata(in: cursor..<(cursor + bSize)); cursor += bSize
            let aFloats = aData.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(
                start: $0.baseAddress?.assumingMemoryBound(to: Float.self),
                count: aSize / 4)) }
            let bFloats = bData.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(
                start: $0.baseAddress?.assumingMemoryBound(to: Float.self),
                count: bSize / 4)) }
            // v2: optional magnitude vector after loraB iff entry declared it.
            var mFloats: [Float]? = nil
            if let mShape = entry.loraMShape {
                let mSize = mShape.reduce(1, *) * 4
                let mData = data.subdata(in: cursor..<(cursor + mSize)); cursor += mSize
                mFloats = mData.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(
                    start: $0.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: mSize / 4)) }
            }
            matrices.append((loraA: aFloats, loraB: bFloats, m: mFloats))
        }
        return LoraAdapter(header: header, matrices: matrices)
    }

    /// Inject and load a saved adapter onto a model. Throws if the
    /// adapter's recorded base config doesn't match the model.
    public static func apply(_ adapter: LoraAdapter, to model: TinyGPTModel) throws {
        let h = adapter.header
        let cfg = model.config
        guard h.baseLayers == cfg.nLayers,
              h.baseDModel == cfg.dModel,
              h.baseCtx == cfg.contextLength,
              h.baseHeads == cfg.nHeads,
              h.baseDMlp == cfg.dMlp else {
            throw NSError(domain: "TinyGPTLoRA", code: 4,
                          userInfo: [NSLocalizedDescriptionKey:
                            "adapter base config doesn't match loaded model"])
        }
        // Detect whether saved adapter has DoRA magnitudes. ALL entries
        // must agree — mixed DoRA/LoRA in one adapter is not supported.
        let hasMagnitudes = adapter.matrices.allSatisfy { $0.m != nil }
        let hasPureLora = adapter.matrices.allSatisfy { $0.m == nil }
        guard hasMagnitudes || hasPureLora else {
            throw NSError(domain: "TinyGPTLoRA", code: 5,
                          userInfo: [NSLocalizedDescriptionKey:
                            "mixed DoRA/LoRA entries in one adapter — not supported"])
        }
        let loraCfg = LoraConfig(rank: h.rank, alpha: h.alpha,
                                  targetSuffixes: h.targetSuffixes,
                                  useDora: hasMagnitudes)
        LoraInjection.inject(model, config: loraCfg)
        // Now overwrite each LoraLinear / DoraLinear's A, B (and m if DoRA)
        // with the saved values. Build a NestedDictionary update.
        var blocksList: [NestedItem<String, MLXArray>] = []
        var idx = 0
        for (i, block) in model.blocks.enumerated() {
            _ = i
            var attn: [String: NestedItem<String, MLXArray>] = [:]
            var mlp: [String: NestedItem<String, MLXArray>] = [:]
            let projs: [(String, Linear, Bool)] = [
                ("q_proj", block.attn.qProj, h.targetSuffixes.contains("q_proj")),
                ("k_proj", block.attn.kProj, h.targetSuffixes.contains("k_proj")),
                ("v_proj", block.attn.vProj, h.targetSuffixes.contains("v_proj")),
                ("o_proj", block.attn.oProj, h.targetSuffixes.contains("o_proj")),
            ]
            for (name, _, isTarget) in projs where isTarget {
                let entry = adapter.matrices[idx]
                let hEntry = h.entries[idx]
                var dict: [String: NestedItem<String, MLXArray>] = [
                    "loraA": .value(MLXArray(entry.loraA, hEntry.loraAShape)),
                    "loraB": .value(MLXArray(entry.loraB, hEntry.loraBShape)),
                ]
                if let m = entry.m, let mShape = hEntry.loraMShape {
                    dict["m"] = .value(MLXArray(m, mShape))
                }
                attn[name] = .dictionary(dict)
                idx += 1
            }
            var mProjs: [(String, Linear, Bool)] = []
            if let dense = block.mlp {
                mProjs.append(("fc_in",  dense.fcIn,  h.targetSuffixes.contains("fc_in")))
                mProjs.append(("fc_out", dense.fcOut, h.targetSuffixes.contains("fc_out")))
            }
            for (name, _, isTarget) in mProjs where isTarget {
                let entry = adapter.matrices[idx]
                let hEntry = h.entries[idx]
                var dict: [String: NestedItem<String, MLXArray>] = [
                    "loraA": .value(MLXArray(entry.loraA, hEntry.loraAShape)),
                    "loraB": .value(MLXArray(entry.loraB, hEntry.loraBShape)),
                ]
                if let m = entry.m, let mShape = hEntry.loraMShape {
                    dict["m"] = .value(MLXArray(m, mShape))
                }
                mlp[name] = .dictionary(dict)
                idx += 1
            }
            var blockEntries: [String: NestedItem<String, MLXArray>] = [:]
            if !attn.isEmpty { blockEntries["attn"] = .dictionary(attn) }
            if !mlp.isEmpty { blockEntries["mlp"] = .dictionary(mlp) }
            blocksList.append(.dictionary(blockEntries))
        }
        var root = NestedDictionary<String, MLXArray>()
        root["blocks"] = .array(blocksList)
        try model.update(parameters: root, verify: [])
    }
}
