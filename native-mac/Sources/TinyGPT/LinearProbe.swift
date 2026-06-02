import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import TinyGPTModel

/// `tinygpt linear-probe` — train a small linear classifier on the
/// hidden-state representations at a specific layer of a frozen base.
///
/// Linear probes are the canonical interpretability tool for detecting
/// "does layer L *know* about property P?". Train Linear(d_model → C)
/// over (hidden_at_L, label) pairs from a labeled dataset; classification
/// accuracy is a lower bound on the linear separability of P at layer L.
///
/// Distinct mechanism from logit / tuned lens: those project a hidden
/// state to vocab via a fixed (or learned) map → next-token distribution.
/// Linear probes target an arbitrary external property (sentiment,
/// language, syntactic category, etc.) and tell you *where the model
/// represents it*, not *what it would emit*.
///
/// Reference: Alain & Bengio 2016, "Understanding intermediate layers
/// using linear classifier probes."
///
/// USAGE
///   tinygpt linear-probe <model.tinygpt> --data labels.jsonl --out probe.lp \
///       [--layer N | --all] [--steps 500] [--lr 1e-3]
///
/// DATA FORMAT (JSONL, one record per line):
///   {"text": "the input string", "label": "positive"}   (string label OK)
///   {"text": "...", "label": 0}                          (or integer)
///
/// OUTPUT
///   `.lp` sidecar: 8-byte magic "TGLP" + version + header JSON +
///   per-layer (weight[d_model,C], bias[C]) raw fp32. Header carries
///   the label map so inference can decode the class index back.
enum LinearProbe {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var dataPath: String? = nil
        var outPath: String? = nil
        var layerSpec: String? = nil   // "all" | integer
        var steps = 500
        var lr: Float = 1e-3

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--data":      dataPath = args[i+1]; i += 2
            case "--out":       outPath = args[i+1]; i += 2
            case "--layer":     layerSpec = args[i+1]; i += 2
            case "--all":       layerSpec = "all"; i += 1
            case "--steps":     steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":        lr = Float(args[i+1]) ?? lr; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let dataPath = dataPath else { fputs("--data required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }

        // Load model — from-scratch byte-level only in this first cut
        // (mirrors tuned-lens's scoping). BPE support adds ~5 lines of
        // tokenizer plumbing.
        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("linear-probe first-cut targets from-scratch byte-level models. " +
                  "HF-side support is queued.\n", stderr); exit(2)
        }
        let cfg = load.config

        // Read the labeled dataset. Labels can be strings or ints; we
        // build a label map on the fly. Empty labels / blank texts skip.
        let lines = readJsonl(path: dataPath)
        var labelMap: [String: Int32] = [:]
        var samples: [(text: String, label: Int32)] = []
        for raw in lines {
            guard let dict = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
            else { continue }
            guard let text = dict["text"] as? String, !text.isEmpty else { continue }
            let labelKey: String
            if let s = dict["label"] as? String { labelKey = s }
            else if let n = dict["label"] as? Int { labelKey = String(n) }
            else if let n = dict["label"] as? Double { labelKey = String(Int(n)) }
            else { continue }
            if labelMap[labelKey] == nil { labelMap[labelKey] = Int32(labelMap.count) }
            samples.append((text: text, label: labelMap[labelKey]!))
        }
        guard !samples.isEmpty else {
            fputs("no usable records in \(dataPath)\n", stderr); exit(1)
        }
        let numClasses = labelMap.count
        guard numClasses >= 2 else {
            fputs("linear probes need ≥ 2 distinct labels (found \(numClasses))\n", stderr); exit(1)
        }
        print("read \(samples.count) records, \(numClasses) classes (\(labelMap.keys.sorted().joined(separator: ", ")))")

        // Decide which layers to probe. Default = all blocks; otherwise
        // parse "0,3,7" or "5".
        let layerIdxs: [Int]
        if layerSpec == nil || layerSpec == "all" {
            layerIdxs = Array(0..<cfg.nLayers)
        } else {
            let parts = layerSpec!.split(separator: ",")
            var picked: [Int] = []
            for p in parts {
                guard let v = Int(p), v >= 0, v < cfg.nLayers else {
                    fputs("invalid --layer spec '\(layerSpec!)'\n", stderr); exit(2)
                }
                picked.append(v)
            }
            layerIdxs = picked
        }

        // Pre-compute hidden states: forward the entire dataset once
        // through the frozen base, capture per-layer last-token hidden
        // states. This is O(N · forward) memory and time, but
        // dramatically faster than re-running the base each train step.
        print("capturing hidden states…")
        var features: [[[Float]]] = Array(repeating: [], count: cfg.nLayers)
        // features[layer][sample] = [d_model]
        for s in samples {
            let bytes: [Int32] = s.text.utf8.prefix(cfg.contextLength).map { Int32($0) }
            let idx = MLXArray(bytes, [1, bytes.count])
            let states = model.forwardLayerwise(idx)
            // Last-token hidden state at each layer.
            for layer in 0..<cfg.nLayers {
                let h = states[layer][0..., (bytes.count - 1)..<bytes.count, 0...]
                MLX.eval(h)
                features[layer].append(h.asArray(Float.self))
            }
        }

        // Build labels tensor once.
        let yArr = samples.map { $0.label }

        // Train one probe per requested layer.
        var probes: [(layer: Int, weight: [Float], bias: [Float])] = []
        for layer in layerIdxs {
            print("training probe @ layer \(layer)…")
            let xMat = features[layer]   // [N][d_model]
            let probe = Linear(cfg.dModel, numClasses, bias: true)
            let optimizer = AdamW(learningRate: lr)

            // Flatten X into one [N, d_model] tensor.
            var flat: [Float] = []
            flat.reserveCapacity(xMat.count * cfg.dModel)
            for row in xMat { flat.append(contentsOf: row) }
            let X = MLXArray(flat, [xMat.count, cfg.dModel])
            let Y = MLXArray(yArr, [xMat.count])

            let lossFn = { (m: Linear, xb: MLXArray, yb: MLXArray) -> MLXArray in
                let logits = m(xb)
                return crossEntropy(logits: logits, targets: yb, reduction: .mean)
            }
            let lossGrad = valueAndGrad(model: probe, lossFn)
            for step in 1...steps {
                let (loss, grads) = lossGrad(probe, X, Y)
                optimizer.update(model: probe, gradients: grads)
                MLX.eval(loss, probe, optimizer)
                if step % max(1, steps / 10) == 0 || step == 1 {
                    print(String(format: "  step %4d  loss=%.4f", step, loss.item(Float.self)))
                }
            }

            // Final accuracy on the (same) training set — sanity, not
            // generalisation. Caller should hold out a validation split.
            let logits = probe(X)
            let preds = MLX.argMax(logits, axis: 1)
            let correct = MLX.sum((preds .== Y).asType(.float32)).item(Float.self)
            let acc = correct / Float(xMat.count)
            print(String(format: "  train acc = %.3f", acc))

            MLX.eval(probe.weight, probe.bias!)
            probes.append((
                layer: layer,
                weight: probe.weight.asArray(Float.self),
                bias: probe.bias!.asArray(Float.self),
            ))
        }

        // Persist.
        do { try LinearProbeWriter.write(probes: probes, labelMap: labelMap,
                                          cfg: cfg, to: URL(fileURLWithPath: outPath)) }
        catch { fputs("write failed: \(error)\n", stderr); exit(1) }
        print("wrote \(probes.count) probe(s) → \(outPath)")
    }

    private static func readJsonl(path: String) -> [String] {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let s = String(data: data, encoding: .utf8) else { return [] }
        return s.split(separator: "\n", omittingEmptySubsequences: true).map { String($0) }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt linear-probe <model.tinygpt> --data <jsonl> --out <probe.lp> [options]

        --data <path.jsonl>    JSONL of {text, label} records (required)
        --out <path.lp>        Where to save the trained probe(s) — required
        --layer N | --layer N,M | --all
                               Which layer(s) to probe (default: all)
        --steps N              Train steps per probe (default: 500)
        --lr F                 Learning rate (default: 1e-3)

        Distinct mechanism from tuned-lens: probes a labeled external
        property rather than next-token prediction. See Alain & Bengio
        2016 for the standard methodology.
        """)
        exit(code)
    }
}

// ============================================================================
// .lp file format — small sidecar, ASCII magic + JSON header + raw fp32.
// ============================================================================

private struct LinearProbeFile: Codable {
    var version: Int
    var nLayers: Int
    var dModel: Int
    var numClasses: Int
    var labelMap: [String: Int32]   // label name → class index
    var layers: [Int]                // which layer indices each probe targets
}

enum LinearProbeWriter {
    static func write(
        probes: [(layer: Int, weight: [Float], bias: [Float])],
        labelMap: [String: Int32],
        cfg: ModelConfig,
        to url: URL,
    ) throws {
        let numClasses = labelMap.count
        let header = LinearProbeFile(
            version: 1, nLayers: cfg.nLayers, dModel: cfg.dModel,
            numClasses: numClasses, labelMap: labelMap,
            layers: probes.map { $0.layer },
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try encoder.encode(header)
        var out = Data()
        out.append(contentsOf: Array("TGLP".utf8))                 // magic
        var version = UInt32(1).littleEndian
        withUnsafeBytes(of: &version) { out.append(contentsOf: $0) }
        var headerLen = UInt32(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { out.append(contentsOf: $0) }
        out.append(headerData)
        for p in probes {
            p.weight.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
            p.bias.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        }
        try out.write(to: url, options: .atomic)
    }
}
