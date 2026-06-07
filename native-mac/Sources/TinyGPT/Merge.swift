import Foundation
import TinyGPTIO
import TinyGPTModel

/// `tinygpt merge` — combine 2+ same-architecture `.tinygpt` checkpoints
/// using TIES, DARE, or plain weighted averaging (mergekit-style).
enum Merge {
    enum Method: String {
        case ties, dare, linear
    }

    static func run(args: [String]) {
        var modelPaths: [String] = []
        var weights: [Float] = []
        var method = Method.ties
        var density: Float = 0.5
        var basePath: String? = nil
        var outPath: String? = nil
        var dareSeed: UInt64 = 42

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--models":
                i += 1
                while i < args.count && !args[i].hasPrefix("--") {
                    modelPaths.append(args[i]); i += 1
                }
            case "--weights":
                i += 1
                while i < args.count && !args[i].hasPrefix("--") {
                    weights.append(Float(args[i]) ?? 1.0); i += 1
                }
            case "--method":
                guard let m = Method(rawValue: args[i + 1].lowercased()) else {
                    fputs("--method must be ties|dare|linear\n", stderr); exitUsage()
                }
                method = m; i += 2
            case "--density":
                density = Float(args[i + 1]) ?? density; i += 2
            case "--base":
                basePath = args[i + 1]; i += 2
            case "--dare-seed":
                dareSeed = UInt64(args[i + 1]) ?? dareSeed; i += 2
            case "--out":
                outPath = args[i + 1]; i += 2
            case "-h", "--help":
                exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }

        guard modelPaths.count >= 2 else {
            fputs("--models requires at least 2 .tinygpt paths\n", stderr); exitUsage()
        }
        guard let outPath else { fputs("--out <merged.tinygpt> required\n", stderr); exitUsage() }

        if weights.isEmpty { weights = [Float](repeating: 1.0, count: modelPaths.count) }
        guard weights.count == modelPaths.count else {
            fputs("--weights count must match --models count\n", stderr); exit(2)
        }

        let files: [TinyGPTFile]
        do { files = try modelPaths.map { try TinyGPTFileReader.read(URL(fileURLWithPath: $0)) } }
        catch { fputs("model read failed: \(error)\n", stderr); exit(1) }

        let baseFile: TinyGPTFile?
        if let basePath {
            do { baseFile = try TinyGPTFileReader.read(URL(fileURLWithPath: basePath)) }
            catch { fputs("base read failed: \(error)\n", stderr); exit(1) }
        } else {
            baseFile = nil
        }

        do {
            try assertCompatible(files: files, base: baseFile)
        } catch {
            fputs("architecture mismatch: \(error)\n", stderr); exit(1)
        }

        let merged = tryMerge(
            files: files, weights: weights, base: baseFile,
            method: method, density: density, dareSeed: dareSeed
        )

        do {
            try TinyGPTFileWriter.write(merged, to: URL(fileURLWithPath: outPath))
            print("✓ wrote merged model → \(outPath) (\(merged.tensors.count) tensors, method=\(method.rawValue))")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
    }

    private static func assertCompatible(files: [TinyGPTFile], base: TinyGPTFile?) throws {
        let ref = files[0]
        for f in files.dropFirst() {
            guard f.tensors.count == ref.tensors.count else {
                throw NSError(domain: "Merge", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "tensor count differs"])
            }
            for (a, b) in zip(ref.tensors, f.tensors) {
                guard a.entry.name == b.entry.name,
                      a.entry.shape == b.entry.shape,
                      a.dtype == b.dtype else {
                    throw NSError(domain: "Merge", code: 2,
                                  userInfo: [NSLocalizedDescriptionKey:
                                             "tensor \(a.entry.name) shape/dtype mismatch"])
                }
            }
        }
        if let base {
            guard base.tensors.count == ref.tensors.count else {
                throw NSError(domain: "Merge", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "base tensor count differs"])
            }
        }
    }

    private static func tryMerge(
        files: [TinyGPTFile], weights: [Float], base: TinyGPTFile?,
        method: Method, density: Float, dareSeed: UInt64
    ) -> TinyGPTFile {
        var out = files[0]
        out.header = files[0].header
        out.step = 0
        out.header.finalLoss = nil
        out.header.sample = "merged(\(method.rawValue)) from \(files.count) models"

        var mergedTensors: [TinyGPTTensor] = []
        mergedTensors.reserveCapacity(files[0].tensors.count)

        for idx in 0..<files[0].tensors.count {
            let name = files[0].tensors[idx].entry.name
            let baseFloats: [Float]?
            if let base {
                baseFloats = tensorFloats(base.tensors[idx])
            } else {
                baseFloats = nil
            }

            let deltas: [[Float]] = files.map { file in
                let w = tensorFloats(file.tensors[idx])
                if let b = baseFloats {
                    return zip(w, b).map { $0 - $1 }
                }
                return w
            }

            let mergedDelta: [Float]
            switch method {
            case .linear:
                mergedDelta = weightedAverage(deltas, weights: weights)
            case .ties:
                mergedDelta = tiesMerge(deltas, weights: weights, density: density)
            case .dare:
                mergedDelta = dareMerge(deltas, weights: weights, density: density, seed: dareSeed)
            }

            let mergedWeights: [Float]
            if let b = baseFloats {
                mergedWeights = zip(mergedDelta, b).map { $0 + $1 }
            } else {
                mergedWeights = mergedDelta
            }

            var t = files[0].tensors[idx]
            t.weight = floatsToData(mergedWeights)
            // Keep training layout (fp32 triplets) so TinyGPTFileWriter
            // doesn't switch to inference-fp16 body encoding.
            let zero = Data(count: t.weight.count)
            t.adamM = zero
            t.adamV = zero
            t.dtype = .fp32
            mergedTensors.append(t)
            _ = name
        }
        out.tensors = mergedTensors
        out.header.savedAt = ISO8601DateFormatter().string(from: Date())
        out.header.weightDtype = "fp32"
        out.header.includesOptimizerState = true
        out.header.stateByteLength = 4 + mergedTensors.reduce(0) { $0 + 3 * $1.weight.count }
        return out
    }

    private static func tensorFloats(_ t: TinyGPTTensor) -> [Float] {
        switch t.dtype {
        case .fp32: return t.weightFloats
        case .fp16: return t.weightFP16AsFloat32()
        }
    }

    private static func floatsToData(_ floats: [Float]) -> Data {
        floats.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private static func weightedAverage(_ tensors: [[Float]], weights: [Float]) -> [Float] {
        let n = tensors[0].count
        var sumW: Float = 0
        var out = [Float](repeating: 0, count: n)
        for (t, w) in zip(tensors, weights) {
            sumW += w
            for i in 0..<n { out[i] += t[i] * w }
        }
        let denom = max(sumW, 1e-8)
        return out.map { $0 / denom }
    }

    /// TIES: trim bottom (1-density) magnitudes, majority sign vote, disjoint avg.
    private static func tiesMerge(_ deltas: [[Float]], weights: [Float], density: Float) -> [Float] {
        let n = deltas[0].count
        let keepFrac = max(0.01, min(1.0, density))
        var trimmed: [[Float]] = []
        trimmed.reserveCapacity(deltas.count)
        for d in deltas {
            trimmed.append(trimTopK(d, keepFraction: keepFrac))
        }
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var pos: Float = 0, neg: Float = 0
            var vals: [(Float, Float)] = []
            for (d, w) in zip(trimmed, weights) {
                let v = d[i]
                if v > 0 { pos += w } else if v < 0 { neg += w }
                if abs(v) > 1e-12 { vals.append((v, w)) }
            }
            if vals.isEmpty { continue }
            let sign: Float = pos >= neg ? 1 : -1
            var num: Float = 0, den: Float = 0
            for (v, w) in vals where (v > 0 && sign > 0) || (v < 0 && sign < 0) {
                num += v * w
                den += w
            }
            if den > 0 { out[i] = num / den }
        }
        return out
    }

    /// DARE: random Bernoulli mask on trimmed deltas + rescale to preserve expectation.
    private static func dareMerge(_ deltas: [[Float]], weights: [Float],
                                   density: Float, seed: UInt64) -> [Float] {
        let trimmed = deltas.map { trimTopK($0, keepFraction: max(0.01, min(1.0, density))) }
        var rng = SeededRNG(seed: seed)
        let maskProb = max(0.01, min(1.0, density))
        var masked: [[Float]] = []
        masked.reserveCapacity(trimmed.count)
        for d in trimmed {
            var row = [Float](repeating: 0, count: d.count)
            for i in 0..<d.count {
                if rng.nextUnit() < maskProb {
                    row[i] = d[i] / maskProb
                }
            }
            masked.append(row)
        }
        return weightedAverage(masked, weights: weights)
    }

    private static func trimTopK(_ v: [Float], keepFraction: Float) -> [Float] {
        let n = v.count
        guard n > 0 else { return v }
        let k = max(1, Int(Float(n) * keepFraction))
        let sorted = v.map(abs).sorted(by: >)
        let threshold = sorted[min(k - 1, sorted.count - 1)]
        return v.map { abs($0) >= threshold ? $0 : 0 }
    }

    private struct SeededRNG {
        private var state: UInt64
        init(seed: UInt64) { state = seed == 0 ? 0xDEADBEEF : seed }
        mutating func nextUnit() -> Float {
            state = state &* 6364136223846793005 &+ 1
            return Float(state >> 33) / Float(UInt32.max)
        }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt merge --models A.tinygpt B.tinygpt [C.tinygpt ...] --out merged.tinygpt [options]

          --weights W1 W2 ...     Per-model contribution (default: 1.0 each)
          --method ties|dare|linear   Merge algorithm (default: ties)
          --density F             TIES/DARE keep fraction by magnitude (default: 0.5)
          --base path.tinygpt     Optional base for task-vector deltas
          --dare-seed N           RNG seed for DARE masks (default: 42)
        """)
        exit(code)
    }
}
