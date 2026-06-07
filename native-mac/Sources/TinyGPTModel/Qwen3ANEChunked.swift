import Foundation
import Accelerate
#if canImport(CoreML)
@preconcurrency import CoreML
import TinyGPTIO

/// Per-block stateful CoreML inference for Qwen3-class models.
///
/// Loads N separately-compiled .mlpackage files (one per transformer block)
/// and drives them in sequence per token. Embedding lookup + final RMSNorm
/// + tied lm_head run in Swift (Accelerate sgemv) since they're cheap
/// non-block ops.
///
/// Why per-block: the M6 bisect showed ANE runtime crashes (SIGTRAP) on any
/// multi-layer stateful Qwen3 graph compiled into one .mlpackage. Splitting
/// each block into its own 1-layer mlpackage (its own private k_cache /
/// v_cache MLState pair) sidesteps the crash entirely. The M8 result on
/// the Python driver: end-to-end correct, prefill 18.5 tok/s, decode
/// 17.3 tok/s. Swift removes Python ml.predict overhead (~1ms × 28 =
/// 28ms/token of waste) — projected 30-40 tok/s.
///
/// Files expected in the chunked dir:
///   m8-block-0.mlpackage ... m8-block-(N-1).mlpackage
///
/// HF dir provides `model.safetensors` (or sharded) for the embedding +
/// norm weights, plus `config.json` for vocab + hidden + n_layers.
@available(macOS 15.0, *)
public final class Qwen3ANEChunked: @unchecked Sendable {
    public let blocks: [MLModel]
    public let nLayers: Int
    public let hiddenSize: Int
    public let vocabSize: Int
    public let maxSeqLen: Int
    /// Embedding + lm_head are TIED for Qwen3-0.6B — one fp32 weight matrix
    /// [vocab, hidden] used twice: row gather for embedding lookup, matmul
    /// for the final lm_head projection.
    public let embedTokensWeight: [Float]
    public let finalNormWeight: [Float]
    public let rmsNormEps: Float

    public init(blocks: [MLModel], nLayers: Int, hiddenSize: Int, vocabSize: Int,
                maxSeqLen: Int, embedTokensWeight: [Float], finalNormWeight: [Float],
                rmsNormEps: Float) {
        self.blocks = blocks
        self.nLayers = nLayers
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.embedTokensWeight = embedTokensWeight
        self.finalNormWeight = finalNormWeight
        self.rmsNormEps = rmsNormEps
    }

    public enum LoadError: Error, CustomStringConvertible {
        case missingBlock(Int, URL)
        case missingHFFile(String)
        case malformedConfig(String)
        case missingTensor(String)
        case unsupportedDtype(String)
        public var description: String {
            switch self {
            case .missingBlock(let i, let u): return "missing block \(i) at \(u.path)"
            case .missingHFFile(let s): return "missing HF file: \(s)"
            case .malformedConfig(let s): return "malformed config.json: \(s)"
            case .missingTensor(let n): return "missing tensor '\(n)' in safetensors"
            case .unsupportedDtype(let s): return "unsupported safetensors dtype: \(s)"
            }
        }
    }

    /// Load N block .mlpackages from `chunkedDir` plus embedding + norm from
    /// `hfDir`. Returns ready-to-decode chunked model.
    public static func load(chunkedDir: URL, hfDir: URL,
                              computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                              defaultMaxSeq: Int = 128) async throws -> Qwen3ANEChunked {
        // 1. Parse config.json.
        let cfgURL = hfDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: cfgURL.path) else {
            throw LoadError.missingHFFile("config.json at \(cfgURL.path)")
        }
        let cfg = try JSONSerialization.jsonObject(with: try Data(contentsOf: cfgURL)) as? [String: Any]
        guard let cfg else { throw LoadError.malformedConfig("not a JSON object") }
        guard let nLayers = cfg["num_hidden_layers"] as? Int,
              let hiddenSize = cfg["hidden_size"] as? Int,
              let vocab = cfg["vocab_size"] as? Int
        else { throw LoadError.malformedConfig("missing num_hidden_layers / hidden_size / vocab_size") }
        let eps = (cfg["rms_norm_eps"] as? Double).map(Float.init) ?? 1e-6

        // 2. Discover safetensors shards. Pool embed_tokens.weight + norm.weight.
        let shards = try FileManager.default.contentsOfDirectory(at: hfDir,
                          includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        guard !shards.isEmpty else {
            throw LoadError.missingHFFile("*.safetensors in \(hfDir.path)")
        }
        var embedFP32: [Float] = []
        var normFP32: [Float] = []
        for shard in shards {
            let f = try SafetensorsReader.read(shard)
            for (name, info) in f.tensors {
                let baseName = name.hasPrefix("model.") ? String(name.dropFirst(6)) : name
                if baseName == "embed_tokens.weight" {
                    embedFP32 = try decodeFloatTensor(info: info, data: f.data)
                } else if baseName == "norm.weight" {
                    normFP32 = try decodeFloatTensor(info: info, data: f.data)
                }
            }
        }
        guard !embedFP32.isEmpty else { throw LoadError.missingTensor("embed_tokens.weight") }
        guard !normFP32.isEmpty else { throw LoadError.missingTensor("norm.weight") }

        // 3. Load N block .mlpackages.
        let mlcfg = MLModelConfiguration()
        mlcfg.computeUnits = computeUnits
        mlcfg.allowLowPrecisionAccumulationOnGPU = true
        var loaded: [MLModel] = []
        loaded.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let url = chunkedDir.appendingPathComponent("m8-block-\(i).mlpackage")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw LoadError.missingBlock(i, url)
            }
            let compiled = try await MLModel.compileModel(at: url)
            let m = try await MLModel.load(contentsOf: compiled, configuration: mlcfg)
            loaded.append(m)
        }

        return Qwen3ANEChunked(blocks: loaded, nLayers: nLayers, hiddenSize: hiddenSize,
                                vocabSize: vocab, maxSeqLen: defaultMaxSeq,
                                embedTokensWeight: embedFP32, finalNormWeight: normFP32,
                                rmsNormEps: eps)
    }

    /// Allocate fresh MLState for each block. Caller owns and reuses across
    /// forwards within a decode session.
    public func makeStates() -> [MLState] {
        return blocks.map { $0.makeState() }
    }

    /// One forward over `ids` (length T_new ≥ 1, typically 1 for decode).
    /// Returns logits at the LAST position.
    ///
    /// Flow:
    ///   1. Embedding lookup: hidden = embed_tokens[ids]      [1, T_new, H]
    ///   2. For block in 0..<N:
    ///        hidden = block.predict(hidden, mask, posOffset, state=states[block])
    ///   3. Final RMSNorm on hidden[last]                      [1, 1, H]
    ///   4. Tied lm_head: logits = h_norm @ embed_tokens.T     [vocab]
    public func forward(ids: [Int32], positionOffset: Int, states: [MLState]) async throws -> [Float] {
        precondition(states.count == nLayers, "states count \(states.count) != n_layers \(nLayers)")
        let T = ids.count
        precondition(T >= 1, "ids must be non-empty")
        let endStep = positionOffset + T
        precondition(endStep <= maxSeqLen, "endStep \(endStep) > maxSeqLen \(maxSeqLen)")

        // 1. Embedding lookup → MLMultiArray of shape [1, T, H] fp32.
        let hiddenArr = try MLMultiArray(
            shape: [1, NSNumber(value: T), NSNumber(value: hiddenSize)],
            dataType: .float32)
        let hp = hiddenArr.dataPointer.assumingMemoryBound(to: Float.self)
        for t in 0..<T {
            let tokenId = Int(ids[t])
            let srcBase = tokenId * hiddenSize
            let dstBase = t * hiddenSize
            for h in 0..<hiddenSize {
                hp[dstBase + h] = embedTokensWeight[srcBase + h]
            }
        }

        // 2. Causal mask [1, 1, T, endStep] fp32.
        let maskArr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: T), NSNumber(value: endStep)],
            dataType: .float32)
        let mp = maskArr.dataPointer.assumingMemoryBound(to: Float.self)
        let neg: Float = -1.0e4
        for j in 0..<T {
            let absRow = positionOffset + j
            let rowBase = j * endStep
            for k in 0..<endStep {
                mp[rowBase + k] = (k <= absRow) ? 0.0 : neg
            }
        }

        // 3. position_offset [1] int32.
        let posArr = try MLMultiArray(shape: [1], dataType: .int32)
        posArr.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(positionOffset)

        // 4. Loop through N blocks. Each predict reads + writes its own MLState.
        var current = hiddenArr
        for i in 0..<nLayers {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_state": current,
                "causal_mask": maskArr,
                "position_offset": posArr,
            ])
            let out = try await blocks[i].prediction(from: provider, using: states[i])
            guard let next = out.featureValue(for: "hidden_out")?.multiArrayValue else {
                throw NSError(domain: "Qwen3ANEChunked", code: 1,
                              userInfo: [NSLocalizedDescriptionKey:
                                "block \(i) returned no hidden_out"])
            }
            current = next
        }

        // 5. Slice last position hidden vector. shape: [1, T, H] → [H].
        let lastBase = (T - 1) * hiddenSize
        var lastHidden = [Float](repeating: 0, count: hiddenSize)
        if current.dataType == .float32 {
            let cp = current.dataPointer.assumingMemoryBound(to: Float.self)
            lastHidden.withUnsafeMutableBufferPointer { dst in
                _ = memcpy(dst.baseAddress, cp + lastBase, hiddenSize * MemoryLayout<Float>.size)
            }
        } else if current.dataType == .float16 {
            let cp = current.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<hiddenSize { lastHidden[i] = Float(Float16(bitPattern: cp[lastBase + i])) }
        } else {
            for i in 0..<hiddenSize { lastHidden[i] = Float(truncating: current[lastBase + i]) }
        }

        // 6. Final RMSNorm with norm.weight scaling.
        var meanSq: Float = 0
        for v in lastHidden { meanSq += v * v }
        meanSq /= Float(hiddenSize)
        let invRMS: Float = 1.0 / sqrt(meanSq + rmsNormEps)
        var normed = [Float](repeating: 0, count: hiddenSize)
        for h in 0..<hiddenSize {
            normed[h] = lastHidden[h] * invRMS * finalNormWeight[h]
        }

        // 7. Tied lm_head: logits[v] = sum_h normed[h] * embed_tokens[v, h].
        // embedTokensWeight is row-major [vocab, hidden], stored as Float.
        // GEMV: y[V] = A[V, H] @ x[H].  Use Accelerate's cblas_sgemv.
        var logits = [Float](repeating: 0, count: vocabSize)
        embedTokensWeight.withUnsafeBufferPointer { Aptr in
            normed.withUnsafeBufferPointer { xptr in
                logits.withUnsafeMutableBufferPointer { yptr in
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                Int32(vocabSize), Int32(hiddenSize),
                                1.0,
                                Aptr.baseAddress, Int32(hiddenSize),
                                xptr.baseAddress, 1,
                                0.0,
                                yptr.baseAddress, 1)
                }
            }
        }
        return logits
    }

    // MARK: - Internal helpers

    /// Decode a safetensors tensor blob into a `[Float]`. Supports F32, F16,
    /// BF16 — the dtypes Qwen3 base + LoRA-baked HF dirs use in 2026.
    private static func decodeFloatTensor(info: SafetensorsReader.TensorInfo,
                                            data: Data) throws -> [Float] {
        let slice = data.subdata(in: info.dataStart..<info.dataEnd)
        let nElements = slice.count / dtypeSize(info.dtype)
        var out = [Float](repeating: 0, count: nElements)
        switch info.dtype.uppercased() {
        case "F32":
            slice.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float.self)
                out.withUnsafeMutableBufferPointer { dst in
                    _ = memcpy(dst.baseAddress, src.baseAddress, slice.count)
                }
            }
        case "F16":
            slice.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: UInt16.self)
                for i in 0..<nElements {
                    out[i] = Float(Float16(bitPattern: src[i]))
                }
            }
        case "BF16":
            // bf16 = top 16 bits of fp32. Zero-extend to fp32.
            slice.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: UInt16.self)
                for i in 0..<nElements {
                    let bits = UInt32(src[i]) << 16
                    out[i] = Float(bitPattern: bits)
                }
            }
        default:
            throw LoadError.unsupportedDtype(info.dtype)
        }
        return out
    }

    private static func dtypeSize(_ dtype: String) -> Int {
        switch dtype.uppercased() {
        case "F32": return 4
        case "F16", "BF16": return 2
        case "I64": return 8
        case "I32": return 4
        default: return 1
        }
    }
}
#endif
