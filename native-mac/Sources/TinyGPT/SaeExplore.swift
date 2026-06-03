import Foundation
import MLX
import MLXNN
import TinyGPTModel

/// `tinygpt sae-explore` — load a trained SAE sidecar + a base model,
/// scan a corpus, and for each feature in the dictionary find the
/// input windows that activate it most. The standard interpretability
/// move once an SAE is trained: "what does feature 47 mean?" answered
/// by showing you the top-K windows whose encoded vector lights up
/// that feature.
///
/// Companion to `tinygpt sae`. The .sae sidecar produced by SAE.swift
/// is loaded here, the SaeModule is re-instantiated with the saved
/// weights, and we run encoded = ReLU(W_enc·(h - b_dec) + b_enc) over
/// every window of the corpus, tracking per-feature max-activation +
/// the window that produced it.
///
/// USAGE
///   tinygpt sae-explore <model.tinygpt> --probe <probe.sae> \
///       --corpus <text.txt> [--features 47,128,256 | --top 8]
///       [--top-k 3] [--window-ctx 32]
///
///   --probe <path.sae>      trained SAE sidecar (required)
///   --corpus <text.txt>     UTF-8 corpus to scan (required)
///   --features SPEC         comma list of feature indices to explore
///                           (e.g. "47,128,256"). Default: top 8 by
///                           overall max activation across the scan.
///   --top                   alias: print top-K features ranked by max
///                           activation (instead of a fixed list).
///   --top-k N               per-feature, how many top windows to show
///                           (default: 3)
///   --window-ctx N          window length when scanning (default: ctx
///                           from .sae header)
enum SaeExplore {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var probePath: String? = nil
        var corpusPath: String? = nil
        var featureSpec: String? = nil
        var topAuto = false
        var topAutoN = 8
        var topK = 3
        var windowCtx: Int? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--probe":       probePath = args[i+1]; i += 2
            case "--corpus":      corpusPath = args[i+1]; i += 2
            case "--features":    featureSpec = args[i+1]; i += 2
            case "--top":         topAuto = true; i += 1
            case "--top-n":       topAuto = true; topAutoN = Int(args[i+1]) ?? topAutoN; i += 2
            case "--top-k":       topK = Int(args[i+1]) ?? topK; i += 2
            case "--window-ctx":  windowCtx = Int(args[i+1]); i += 2
            case "-h", "--help":  exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let probePath = probePath else { fputs("--probe required\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }

        // Load the SAE sidecar.
        let probeURL = URL(fileURLWithPath: probePath)
        guard let probeData = try? Data(contentsOf: probeURL) else {
            fputs("could not read --probe \(probePath)\n", stderr); exit(1)
        }
        let (header, sae) = parseSaeSidecar(probeData)

        // Load base model.
        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("sae-explore first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        guard header.layer >= 0 && header.layer < cfg.nLayers,
              header.baseDModel == cfg.dModel
        else {
            fputs("sae probe was trained on a different model arch (layer \(header.layer), d_model \(header.baseDModel) — current is \(cfg.nLayers) layers, d_model \(cfg.dModel))\n", stderr); exit(2)
        }
        let layer = header.layer
        let D = cfg.dModel
        let F = sae.dFeatures
        let T = min(windowCtx ?? cfg.contextLength, cfg.contextLength)

        // Load corpus.
        guard let bytes = try? Data(contentsOf: URL(fileURLWithPath: corpusPath)) else {
            fputs("could not read --corpus\n", stderr); exit(1)
        }
        let raw = [UInt8](bytes)
        guard raw.count > T + 1 else {
            fputs("corpus too small (\(raw.count) bytes for window \(T))\n", stderr); exit(1)
        }

        // Walk the corpus in non-overlapping windows of length T;
        // for each window's last-token hidden state at the saved
        // layer, encode through the SAE and track per-feature max +
        // the (window-offset, decoded-text) that produced it.
        struct ActivationRecord {
            var value: Float
            var offset: Int
            var snippet: String
        }
        var topPerFeature = [[ActivationRecord]](repeating: [], count: F)
        let nWindows = (raw.count - 1) / T
        print("scanning \(nWindows) windows of length \(T)…")

        for w in 0..<nWindows {
            let lo = w * T
            let bytesArr: [Int32] = (lo..<(lo + T)).map { Int32(raw[$0]) }
            let idx = MLXArray(bytesArr, [1, T])
            let states = model.forwardLayerwise(idx)
            // Take the LAST-token hidden state. Could also take all-positions
            // and track per-position activations; last-token is cheaper.
            let h = states[layer][0, T - 1, 0...]    // [D]
            MLX.eval(h)
            let hVec = h.asArray(Float.self)

            // Encode: ReLU(W_enc · (h - b_dec) + b_enc). Hand-compute
            // since N=1 here — avoids a tiny MLX dispatch.
            let bDecVec = sae.bDec.asArray(Float.self)
            let wEncFlat = sae.wEnc.asArray(Float.self)   // [F, D] row-major
            let bEncVec = sae.bEnc.asArray(Float.self)
            for f in 0..<F {
                var dot: Float = 0
                let rowBase = f * D
                for d in 0..<D {
                    dot += wEncFlat[rowBase + d] * (hVec[d] - bDecVec[d])
                }
                let activation = max(dot + bEncVec[f], 0)
                if activation <= 0 { continue }
                let snippet = String(bytes: raw[lo..<min(lo + T, raw.count)],
                                      encoding: .utf8) ?? "<non-utf8>"
                let cleaned = snippet.replacingOccurrences(of: "\n", with: " ")
                let record = ActivationRecord(value: activation, offset: lo, snippet: cleaned)
                // Maintain a small top-K heap per feature (just append +
                // sort + truncate; F * K is small for typical K=3-10).
                topPerFeature[f].append(record)
                if topPerFeature[f].count > topK * 2 {
                    topPerFeature[f].sort { $0.value > $1.value }
                    topPerFeature[f] = Array(topPerFeature[f].prefix(topK))
                }
            }

            if (w + 1) % max(1, nWindows / 10) == 0 {
                fputs("  window \(w + 1)/\(nWindows)\n", stderr)
            }
        }
        // Final sort per feature.
        for f in 0..<F {
            topPerFeature[f].sort { $0.value > $1.value }
            topPerFeature[f] = Array(topPerFeature[f].prefix(topK))
        }

        // Decide which features to print.
        let featuresToShow: [Int]
        if let spec = featureSpec {
            featuresToShow = spec.split(separator: ",")
                .compactMap { Int($0) }
                .filter { $0 >= 0 && $0 < F }
        } else {
            // Auto: top-N features by maximum activation seen.
            let ranked: [(Int, Float)] = (0..<F).map { f in
                (f, topPerFeature[f].first?.value ?? 0)
            }
            .sorted { $0.1 > $1.1 }
            featuresToShow = Array(ranked.prefix(topAuto ? topAutoN : 8).map { $0.0 })
        }

        print("\nfeature explorer results")
        print("========================")
        for f in featuresToShow {
            let bucket = topPerFeature[f]
            if bucket.isEmpty {
                print("feature \(f): never activated")
                continue
            }
            print("feature \(f) (max \(String(format: "%.3f", bucket[0].value))):")
            for rec in bucket {
                let preview = String(rec.snippet.prefix(70))
                print(String(format: "  %.3f  off=%5d  »  %@", rec.value, rec.offset, preview))
            }
        }
    }

    /// Parse the .sae sidecar produced by `tinygpt sae`. Returns the
    /// JSON header + a fresh SaeModule with the saved weights bound
    /// into its @ParameterInfo slots.
    private static func parseSaeSidecar(_ data: Data) -> (SaeHeaderView, SaeModule) {
        precondition(data.count >= 12, "sae sidecar too small")
        let magic = Array(data.prefix(4))
        precondition(magic == Array("TGSA".utf8), "sae sidecar magic 'TGSA' mismatch")
        let version = data[4..<8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
        precondition(version == 1, "unsupported sae sidecar version \(version)")
        let headerLen = Int(data[8..<12].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
        let headerData = data.subdata(in: 12..<(12 + headerLen))
        let header = (try? JSONDecoder().decode(SaeHeaderView.self, from: headerData))
        guard let header = header else {
            fputs("sae sidecar header parse failed\n", stderr); exit(1)
        }
        var cursor = 12 + headerLen
        let D = header.dModel
        let F = header.dFeatures
        let sae = SaeModule(dModel: D, dFeatures: F)
        // Load arrays in order: wEnc[F,D], bEnc[F], wDec[D,F], bDec[D].
        func readFloats(_ count: Int) -> [Float] {
            let bytes = count * 4
            let slice = data.subdata(in: cursor..<(cursor + bytes))
            cursor += bytes
            return slice.withUnsafeBytes { ptr in
                Array(UnsafeBufferPointer<Float>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: count))
            }
        }
        let wEnc = readFloats(F * D)
        let bEnc = readFloats(F)
        let wDec = readFloats(D * F)
        let bDec = readFloats(D)
        // Overwrite the (random-init) SaeModule params with the saved values.
        var root = NestedDictionary<String, MLXArray>()
        root["w_enc"] = .value(MLXArray(wEnc, [F, D]))
        root["b_enc"] = .value(MLXArray(bEnc, [F]))
        root["w_dec"] = .value(MLXArray(wDec, [D, F]))
        root["b_dec"] = .value(MLXArray(bDec, [D]))
        _ = try? sae.update(parameters: root, verify: [])
        return (header, sae)
    }

    struct SaeHeaderView: Codable {
        let version: Int
        let dModel: Int
        let dFeatures: Int
        let layer: Int
        let baseLayers: Int
        let baseDModel: Int
        let baseCtx: Int
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt sae-explore <model.tinygpt> --probe <probe.sae> \\
                                   --corpus <text.txt> [options]

        For each feature in the trained SAE dictionary, surface the
        input windows in the corpus that activate it most. The standard
        "what does feature 47 mean?" interpretability tool.

        --probe <path.sae>      trained SAE sidecar (required)
        --corpus <path.txt>     UTF-8 corpus to scan (required)
        --features SPEC         comma list of feature indices (default: top 8)
        --top                   show top-N features by max activation
        --top-n N               how many features (default 8)
        --top-k N               windows per feature (default 3)
        --window-ctx N          window length (default: model's ctx)
        """)
        exit(code)
    }
}
