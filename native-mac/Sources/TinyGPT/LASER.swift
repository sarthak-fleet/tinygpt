import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt laser` — selective rank reduction via SVD (Sharma et al.,
/// 2024, "The Truth Is in There").
///
/// LASER ("LAyer SElective Rank reduction") replaces a weight matrix
/// W with the rank-r truncated SVD of W. Counterintuitively, dropping
/// the lowest singular components often IMPROVES downstream-task
/// accuracy — the model's smaller "noise tail" had been adding wrong
/// signal that the higher-singular-value structure had to fight.
///
/// Typical usage targets the MLP `fc_out` weights of mid-to-late
/// layers: `--target mlp.fc_out --layers 8-11 --rank-fraction 0.85`.
///
/// Implementation: works at the `.tinygpt` file level — load, walk the
/// tensor manifest, replace the matching tensors with their SVD-truncated
/// versions, write a new file. The model class never round-trips into
/// MLX module state for this post-hoc surgery — keeps things simple and
/// keeps every other tensor bit-identical.
///
/// USAGE
///   tinygpt laser <input.tinygpt> \
///       --target mlp.fc_out --layers 8,9,10,11 \
///       --rank-fraction 0.85 \
///       --out reduced.tinygpt
enum LASER {
    static func run(args: [String]) {
        var inPath: String? = nil
        var outPath: String? = nil
        var target: String = "mlp.fc_out"
        var layersSpec: String = ""
        var rankFraction: Float = 0.85

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":            outPath = args[i+1]; i += 2
            case "--target":         target = args[i+1]; i += 2
            case "--layers":         layersSpec = args[i+1]; i += 2
            case "--rank-fraction":  rankFraction = Float(args[i+1]) ?? rankFraction; i += 2
            case "-h", "--help":     exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inPath = args[i]; i += 1
            }
        }
        guard let inPath = inPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path> required\n", stderr); exitUsage() }
        precondition(rankFraction > 0 && rankFraction <= 1.0,
                     "--rank-fraction must be in (0, 1]")

        print("loading \(inPath)…")
        let inputURL = URL(fileURLWithPath: inPath)
        var file: TinyGPTFile
        do { file = try TinyGPTFileReader.read(inputURL) }
        catch { fputs("read failed: \(error)\n", stderr); exit(1) }

        let nLayers = file.header.config.layers ?? 12
        let targetLayers = parseLayers(layersSpec, total: nLayers)

        print("""

        TinyGPT — LASER (selective rank reduction)
        ------------------------------------------
        input:          \(inPath)
        target:         \(target)
        layers:         \(targetLayers.map(String.init).joined(separator: ","))
        rank-fraction:  \(rankFraction)
        output:         \(outPath)

        """)

        // For each (layer, target) pair: find the tensor in the file,
        // re-shape its float bytes, run truncated SVD, write back.
        let targetNames = Set(targetLayers.map { "blocks.\($0).\(target).weight" })
        var reduced = 0
        for (idx, tensor) in file.tensors.enumerated() {
            guard targetNames.contains(tensor.entry.name) else { continue }
            let shape = tensor.entry.shape
            guard shape.count == 2 else {
                fputs("skip \(tensor.entry.name): expected 2-D weight, got \(shape)\n", stderr)
                continue
            }
            let floats = floatsFromData(tensor.weight, count: shape[0] * shape[1])
            let truncated = truncateRank(floats, m: shape[0], n: shape[1],
                                          keepFraction: rankFraction)
            file.tensors[idx].weight = truncated.withUnsafeBufferPointer { Data(buffer: $0) }
            reduced += 1
            print("  ✓ \(tensor.entry.name)  \(shape[0])×\(shape[1])")
        }
        if reduced == 0 {
            fputs("warning: 0 tensors matched (\(target) on layers \(targetLayers)). Nothing written.\n", stderr)
            exit(1)
        }

        do {
            try TinyGPTFileWriter.write(file, to: URL(fileURLWithPath: outPath))
            print("\n✓ wrote \(outPath)  (\(reduced) tensors reduced)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
    }

    /// Unpack `weight` bytes into a Float array of `count` elements.
    private static func floatsFromData(_ data: Data, count: Int) -> [Float] {
        return data.withUnsafeBytes { ptr -> [Float] in
            Array(UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                count: count))
        }
    }

    /// Reduce a `[m, n]` matrix to its top-`k` singular components via
    /// deflated power iteration. k = max(1, round(keepFraction · min(m, n))).
    ///
    /// We avoid a LAPACK SVD bridge and use power iteration: slower than
    /// LAPACK but no dependencies, and accurate enough for the rank-
    /// truncation use case (we discard low-singular-value components,
    /// so small approximation error on those is exactly what we want).
    private static func truncateRank(_ flat: [Float], m: Int, n: Int,
                                      keepFraction: Float) -> [Float] {
        let k = max(1, Int((Float(min(m, n)) * keepFraction).rounded()))
        if k >= min(m, n) { return flat }  // no-op

        // Materialise as [[Float]] for the iterative loop.
        var residual = (0..<m).map { row -> [Float] in
            Array(flat[row * n ..< (row + 1) * n])
        }
        var U: [[Float]] = []; U.reserveCapacity(k)
        var S: [Float] = [];   S.reserveCapacity(k)
        var V: [[Float]] = []; V.reserveCapacity(k)
        for _ in 0..<k {
            let (u, s, v) = topSingular(residual, iterations: 25)
            U.append(u); S.append(s); V.append(v)
            // Deflate: residual -= s · u ⊗ v.
            for i in 0..<m {
                let ui = u[i] * s
                for j in 0..<n { residual[i][j] -= ui * v[j] }
            }
        }
        // Reconstruct W_k = Σ s_r · u_r ⊗ v_r.
        var out = [Float](repeating: 0, count: m * n)
        for r in 0..<k {
            let s = S[r]
            for i in 0..<m {
                let ui = U[r][i] * s
                let rowBase = i * n
                for j in 0..<n { out[rowBase + j] += ui * V[r][j] }
            }
        }
        return out
    }

    /// Power iteration for the top singular triple. (u, s, v) of A.
    private static func topSingular(_ A: [[Float]], iterations: Int) -> ([Float], Float, [Float]) {
        let m = A.count
        let n = A[0].count
        var v = (0..<n).map { _ in Float.random(in: -1...1) }
        normaliseInPlace(&v)
        var u = [Float](repeating: 0, count: m)
        var s: Float = 0
        for _ in 0..<iterations {
            // u_new = A @ v
            for i in 0..<m {
                var acc: Float = 0
                let row = A[i]
                for j in 0..<n { acc += row[j] * v[j] }
                u[i] = acc
            }
            normaliseInPlace(&u)
            // v_new = Aᵀ @ u; s = ‖v_new‖ before normalisation.
            var vNew = [Float](repeating: 0, count: n)
            for i in 0..<m {
                let ui = u[i]
                let row = A[i]
                for j in 0..<n { vNew[j] += ui * row[j] }
            }
            s = norm(vNew)
            if s < 1e-12 { break }
            for j in 0..<n { v[j] = vNew[j] / s }
        }
        return (u, s, v)
    }

    private static func normaliseInPlace(_ x: inout [Float]) {
        let n = norm(x)
        if n > 1e-12 { for i in 0..<x.count { x[i] /= n } }
    }
    private static func norm(_ x: [Float]) -> Float {
        var s: Float = 0
        for v in x { s += v * v }
        return Foundation.sqrt(s)
    }

    /// Parse "1,2,5-8" → [1, 2, 5, 6, 7, 8]. Empty = all layers.
    private static func parseLayers(_ spec: String, total: Int) -> [Int] {
        if spec.isEmpty { return Array(0..<total) }
        var out: [Int] = []
        for part in spec.split(separator: ",") {
            let p = String(part).trimmingCharacters(in: .whitespaces)
            if p.contains("-") {
                let bits = p.split(separator: "-").map { Int($0) ?? -1 }
                if bits.count == 2, bits[0] >= 0, bits[1] >= bits[0] {
                    for i in bits[0]...bits[1] { out.append(i) }
                }
            } else if let v = Int(p) {
                out.append(v)
            }
        }
        return out.filter { $0 >= 0 && $0 < total }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt laser <input.tinygpt> [options]

        --out <path>         Where to save the rank-reduced model — required
        --target SUFFIX      Which matrix to reduce per layer (default mlp.fc_out)
        --layers SPEC        Layer indices to apply LASER to (e.g. "8-11" or "2,4,6").
                              Empty = all layers (rarely what you want — pick mid-to-late).
        --rank-fraction F    Keep this fraction of the singular components (default 0.85).

        LASER is post-hoc — apply it to a TRAINED model and re-evaluate.
        Often improves task accuracy by removing the noise tail.
        """)
        exit(code)
    }
}
