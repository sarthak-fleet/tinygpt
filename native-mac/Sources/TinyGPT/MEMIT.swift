import Foundation
import MLX
import MLXNN
import TinyGPTIO
import TinyGPTModel

/// `tinygpt memit` — mass fact editing via rank-K update to one MLP's
/// down-projection. The batched generalisation of `tinygpt rome`.
///
/// Meng et al. 2023, "Mass-Editing Memory in a Transformer" (MEMIT).
/// Given N (prompt, target) pairs:
///   - capture k_i = gelu(fcIn(ln2(h_i))) at the last token of each prompt
///   - compute v*_i = embedding(target_i)            (target residual contribution)
///   - solve the least-squares  ΔW K ≈ R  for ΔW
///                          where K = [k_1, …, k_N],  R = V* − W K
///   - apply  W' = W + ΔW
///
/// The minimum-norm exact solution is ΔW = R (KᵀK + λI)⁻¹ Kᵀ. The λI
/// regulariser (Tikhonov) keeps the inverse stable when keys are
/// near-colinear — λ defaults to 1e-2 of the trace of KᵀK / N.
///
/// First-cut scoping: single layer, identity-Hessian (no corpus
/// covariance). Multi-layer MEMIT (distribute updates across L ≥ 5
/// mid-network layers in proportion to causal-trace influence) is the
/// natural follow-up — the math on each layer is unchanged, just with
/// the residual R partitioned by layer-influence weights.
///
/// Single-layer trade-off (verified empirically on shakespeare.tinygpt
/// with N=3 facts at layer 11):
///   - `--scale 1`  → least-squares EXACT (per-fact residual ~ 1e-4),
///                    but the engineered residual contribution gets
///                    averaged with the other 11 layers' contributions
///                    in the residual stream, so sampling output stays
///                    close to baseline.
///   - `--scale 5+` → target injection is visible in sampling, but the
///                    overshoot starts to interfere with the model's
///                    other behaviours (occasional gibberish, target
///                    collisions across facts whose keys overlap).
/// Multi-layer MEMIT splits the same total update across several
/// layers, each at modest scale — sampling becomes visible without
/// breaking the rest of the model. Queued as the second-cut.
///
/// USAGE
///   tinygpt memit <model.tinygpt> --facts facts.jsonl \
///                 --layer N --out edited.tinygpt [--scale F] [--lambda F]
///
/// FACTS FILE (JSONL):
///   {"prompt": "ROMEO:",    "target": "X"}
///   {"prompt": "JULIET:",   "target": "Y"}
///   {"prompt": "MERCUTIO:", "target": "Z"}
enum MEMIT {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var factsPath: String? = nil
        var layerSpec: String? = nil
        var outPath: String? = nil
        var scale: Float = 1.0
        var lambdaArg: Float? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--facts":      factsPath = args[i+1]; i += 2
            case "--layer":      layerSpec = args[i+1]; i += 2
            case "--layers":     layerSpec = args[i+1]; i += 2
            case "--out":        outPath = args[i+1]; i += 2
            case "--scale":      scale = Float(args[i+1]) ?? 1.0; i += 2
            case "--lambda":     lambdaArg = Float(args[i+1]); i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let factsPath = factsPath else { fputs("--facts required\n", stderr); exitUsage() }
        guard let layerSpec = layerSpec else { fputs("--layer or --layers required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }

        // Read facts. Lines that don't parse / lack prompt+target are skipped.
        struct Fact { let prompt: String; let target: UInt8 }
        var facts: [Fact] = []
        if let data = try? Data(contentsOf: URL(fileURLWithPath: factsPath)),
           let str = String(data: data, encoding: .utf8) {
            for raw in str.split(separator: "\n", omittingEmptySubsequences: true) {
                guard let dict = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
                else { continue }
                guard let p = dict["prompt"] as? String, !p.isEmpty,
                      let t = dict["target"] as? String, t.utf8.count == 1,
                      let tb = t.utf8.first
                else {
                    fputs("warn: skipping bad fact record: \(raw.prefix(60))\n", stderr); continue
                }
                facts.append(Fact(prompt: p, target: tb))
            }
        } else {
            fputs("could not read --facts file: \(factsPath)\n", stderr); exit(1)
        }
        guard !facts.isEmpty else { fputs("no usable facts in \(factsPath)\n", stderr); exit(1) }
        let N = facts.count

        // Load model — byte-level scoping mirrors ROME / linear-probe.
        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("memit first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        let layers = parseLayerSpec(layerSpec, nLayers: cfg.nLayers)
        guard !layers.isEmpty else {
            fputs("--layer/--layers spec '\(layerSpec)' produced no valid layers\n", stderr); exit(2)
        }
        // Per-layer share of the residual. Equal weighting is the
        // simplest distribution; Meng 2023 weights by causal-trace
        // influence per layer — the weights sum to 1 either way.
        let perLayerScale = scale / Float(layers.count)
        for L in layers {
            guard model.blocks[L].mlp != nil else {
                fputs("layer \(L) has no dense MLP (MoE expert layer? edit not applicable).\n", stderr); exit(2)
            }
        }

        let M = cfg.dModel
        let D = cfg.dMlp

        // For each fact: forward once, capture the pre-fcOut activation
        // at EVERY layer we're editing. Each layer's keys are
        // independent — what matters at layer L is gelu(fcIn(ln2(...)))
        // computed against THAT layer's input residual.
        // We cache the layerwise residual states and the target
        // embedding per fact; per-layer key capture happens inside the
        // edit loop below.
        var allLayerwise: [[MLXArray]] = []  // facts × layers (post-block residuals)
        var allTokEmb: [MLXArray] = []        // facts × [1, T, dModel] (block-0 input)
        var targets: [[Float]] = []           // N × dModel
        var promptLastT: [Int] = []           // N (last-token index per fact)
        for fact in facts {
            let bytes: [Int32] = fact.prompt.utf8.prefix(cfg.contextLength).map { Int32($0) }
            guard !bytes.isEmpty else {
                fputs("warn: empty prompt after byte encoding, skipping\n", stderr); continue
            }
            let idx = MLXArray(bytes, [1, bytes.count])
            let states = model.forwardLayerwise(idx)
            allLayerwise.append(states)
            promptLastT.append(bytes.count - 1)
            // Block-0 input (embed + posemb) — needed when editing layer 0.
            let positions = MLXArray((0..<bytes.count).map { Int32($0) })
            let posEmb = model.positionEmbedding(positions).expandedDimensions(axis: 0)
            allTokEmb.append(model.tokenEmbedding(idx) + posEmb)
            let targetEmbRow = model.tokenEmbedding.weight[Int(fact.target), 0...]
            MLX.eval(targetEmbRow)
            targets.append(targetEmbRow.asArray(Float.self))
        }
        precondition(targets.count == N && allLayerwise.count == N, "fact count mismatch")

        // Edit loop — one rank-K update per layer in `layers`. Each
        // layer's keys are captured against THAT layer's MLP input;
        // each layer's residual share is R/|layers|.
        print("editing \(layers.count) layer(s): \(layers.map(String.init).joined(separator: ","))")
        var totalDeltaFro: Float = 0
        var worstResidualAcrossLayers: Float = 0
        var maxRelSize: Float = 0
        var autoLambda: Float = 0

        for L in layers {
            let block = model.blocks[L]
            guard let mlp = block.mlp else { continue }   // guarded above
            // Per-fact key at layer L: gelu(fcIn(ln2(layer-L input)))
            // at the last token. Reuses cached layerwise / tok-emb so
            // we never re-run the forward pass.
            var keysL: [[Float]] = []
            for j in 0..<N {
                let inputResidual: MLXArray
                if L == 0 {
                    inputResidual = allTokEmb[j]
                } else {
                    inputResidual = allLayerwise[j][L - 1]
                }
                let attnOut = block.attn(block.ln1(inputResidual))
                let mlpInput = inputResidual + attnOut
                let kAll = gelu(mlp.fcIn(block.ln2(mlpInput)))
                let lastT = promptLastT[j]
                let k = kAll[0, lastT, 0...]
                MLX.eval(k)
                keysL.append(k.asArray(Float.self))
            }

            // V_curr = fcOut(keys) at this layer.
            var currents: [[Float]] = []
            for k in keysL {
                let kArr = MLXArray(k, [1, D])
                let out = mlp.fcOut(kArr)
                MLX.eval(out)
                currents.append(out.asArray(Float.self))
            }

            // R = perLayerScale * (V* - V_curr) — column-major [M, N].
            var R = [Float](repeating: 0, count: M * N)
            for j in 0..<N {
                for d in 0..<M {
                    R[d * N + j] = perLayerScale * (targets[j][d] - currents[j][d])
                }
            }

            // K matrix [D, N] column-major: K[m, j] = keysL[j][m].
            var K = [Float](repeating: 0, count: D * N)
            for j in 0..<N {
                for m in 0..<D {
                    K[m * N + j] = keysL[j][m]
                }
            }

            // KᵀK [N, N] + λI Tikhonov.
            var KtK = [Float](repeating: 0, count: N * N)
            for i in 0..<N {
                for j in 0..<N {
                    var s: Float = 0
                    for m in 0..<D { s += K[m * N + i] * K[m * N + j] }
                    KtK[i * N + j] = s
                }
            }
            var trace: Float = 0
            for i in 0..<N { trace += KtK[i * N + i] }
            let lambda = lambdaArg ?? (max(trace / Float(N), 1) * 1e-2)
            for i in 0..<N { KtK[i * N + i] += lambda }
            if L == layers.first { autoLambda = lambda }

            let inv = invertGaussJordan(KtK, n: N)

            // T = R @ inv → [M, N]
            var T = [Float](repeating: 0, count: M * N)
            for d in 0..<M {
                for j in 0..<N {
                    var s: Float = 0
                    for k in 0..<N { s += R[d * N + k] * inv[k * N + j] }
                    T[d * N + j] = s
                }
            }
            // ΔW[d, m] = sum_j T[d, j] · K[m, j] → [M, D]
            var deltaW = [Float](repeating: 0, count: M * D)
            for d in 0..<M {
                for m in 0..<D {
                    var s: Float = 0
                    for j in 0..<N { s += T[d * N + j] * K[m * N + j] }
                    deltaW[d * D + m] = s
                }
            }

            // Apply.
            let deltaArr = MLXArray(deltaW, [M, D])
            let newW = mlp.fcOut.weight + deltaArr
            MLX.eval(newW)
            var blocksList: [NestedItem<String, MLXArray>] = []
            for (i, _) in model.blocks.enumerated() {
                if i == L {
                    blocksList.append(.dictionary([
                        "mlp": .dictionary([
                            "fc_out": .dictionary(["weight": .value(newW)]),
                        ]),
                    ]))
                } else {
                    blocksList.append(.dictionary([:]))
                }
            }
            var root = NestedDictionary<String, MLXArray>()
            root["blocks"] = .array(blocksList)
            _ = try? model.update(parameters: root, verify: [])

            // Per-layer stats.
            var dFro: Float = 0
            for x: Float in deltaW { dFro += x * x }
            let origW = mlp.fcOut.weight.asArray(Float.self)
            var oFro: Float = 0
            for x: Float in origW { oFro += x * x }
            let rel = sqrt(dFro) / sqrt(oFro)
            totalDeltaFro += dFro
            maxRelSize = max(maxRelSize, rel)

            // Per-fact residual at THIS layer after the edit.
            var worstHere: Float = 0
            for j in 0..<N {
                let kArr = MLXArray(keysL[j], [1, D])
                let outNew = mlp.fcOut(kArr)
                MLX.eval(outNew)
                let outVec = outNew.asArray(Float.self)
                var r: Float = 0
                for d in 0..<M {
                    let diff = perLayerScale * targets[j][d] +
                                (1 - perLayerScale) * currents[j][d] - outVec[d]
                    r += diff * diff
                }
                worstHere = max(worstHere, sqrt(r))
            }
            worstResidualAcrossLayers = max(worstResidualAcrossLayers, worstHere)

            print(String(format: "  layer %2d: ‖ΔW‖_F %.3e (rel %.3f%%), worst residual %.3e",
                          L, sqrt(dFro), rel * 100, worstHere))
        }

        print(String(format: """
        memit summary
          facts:                \(N)
          layers:               \(layers.count)
          per-layer scale:      %.4f  (total scale \(scale))
          λ:                    %.4e (auto)
          aggregate ‖ΔW‖_F:     %.4e (sum across layers)
          worst per-layer rel:  %.4f%%
          worst residual:       %.4e
        """, perLayerScale, autoLambda, sqrt(totalDeltaFro),
            maxRelSize * 100, worstResidualAcrossLayers))

        do {
            try writeEditedCheckpoint(model: model, cfg: cfg,
                                        to: URL(fileURLWithPath: outPath))
            print("wrote edited model → \(outPath)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
    }

    /// Parse a --layer/--layers argument. Accepts a single integer
    /// ("11"), a comma-separated list ("3,5,7,9"), or an inclusive
    /// range ("4-8" → [4,5,6,7,8]). Out-of-range entries are dropped
    /// with a warning.
    static func parseLayerSpec(_ spec: String, nLayers: Int) -> [Int] {
        var out: [Int] = []
        for part in spec.split(separator: ",") {
            let s = String(part)
            if s.contains("-") {
                let pieces = s.split(separator: "-")
                guard pieces.count == 2,
                      let lo = Int(pieces[0]),
                      let hi = Int(pieces[1]), lo <= hi
                else { fputs("warn: bad range '\(s)', skipping\n", stderr); continue }
                for v in lo...hi where v >= 0 && v < nLayers { out.append(v) }
            } else if let v = Int(s) {
                if v >= 0 && v < nLayers { out.append(v) }
                else { fputs("warn: layer \(v) out of range [0,\(nLayers)), skipping\n", stderr) }
            }
        }
        return out
    }

    /// In-place Gauss-Jordan inverse for a small square matrix. N is
    /// typically 3-50 in practice — O(N³) is fine, no need for LAPACK.
    /// Returns the inverse as a fresh row-major Float array.
    private static func invertGaussJordan(_ src: [Float], n: Int) -> [Float] {
        // Augmented matrix [src | I], reduce to [I | inv].
        var m = [Float](repeating: 0, count: n * 2 * n)
        for i in 0..<n {
            for j in 0..<n { m[i * 2 * n + j] = src[i * n + j] }
            m[i * 2 * n + (n + i)] = 1
        }
        for col in 0..<n {
            // Partial-pivot: find row with max |m[r][col]| at or below diag.
            var pivot = col
            var pivotVal = abs(m[col * 2 * n + col])
            for r in (col + 1)..<n {
                let v = abs(m[r * 2 * n + col])
                if v > pivotVal { pivotVal = v; pivot = r }
            }
            if pivot != col {
                for k in 0..<(2 * n) {
                    let t = m[col * 2 * n + k]
                    m[col * 2 * n + k] = m[pivot * 2 * n + k]
                    m[pivot * 2 * n + k] = t
                }
            }
            let d = m[col * 2 * n + col]
            if abs(d) < 1e-12 {
                fputs("memit: KᵀK is singular (try larger --lambda)\n", stderr); exit(1)
            }
            let dInv = 1 / d
            for k in 0..<(2 * n) { m[col * 2 * n + k] *= dInv }
            for r in 0..<n where r != col {
                let f = m[r * 2 * n + col]
                if f == 0 { continue }
                for k in 0..<(2 * n) {
                    m[r * 2 * n + k] -= f * m[col * 2 * n + k]
                }
            }
        }
        var inv = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n { inv[i * n + j] = m[i * 2 * n + (n + j)] }
        }
        return inv
    }

    private static func writeEditedCheckpoint(model: TinyGPTModel,
                                                cfg: ModelConfig,
                                                to url: URL) throws {
        let entries = Train.manifestEntries(cfg)
        var tensors: [TinyGPTTensor] = []
        let params = model.parameters().flattened()
        for entry in entries {
            guard let w = params.first(where: { $0.0 == entry.name })?.1 else {
                throw NSError(domain: "memit", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "missing parameter \(entry.name) in model",
                ])
            }
            MLX.eval(w)
            let arr2 = Train.isLinearWeightName(entry.name) && w.shape.count == 2 ? w.transposed() : w
            let floats: [Float] = arr2.asArray(Float.self)
            let weightData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            let zeros = Data(count: weightData.count)
            tensors.append(TinyGPTTensor(
                entry: entry, weight: weightData, adamM: zeros, adamV: zeros, dtype: .fp32
            ))
        }
        let header = TinyGPTHeader(
            config: .init(
                layers: cfg.nLayers, dModel: cfg.dModel, ctx: cfg.contextLength,
                heads: cfg.nHeads, dMlp: cfg.dMlp, batchSize: 1, backend: "mlx-swift-memit"
            ),
            manifest: entries,
            weightDtype: "fp32",
            includesOptimizerState: true
        )
        let file = TinyGPTFile(header: header, step: 0, tensors: tensors)
        try TinyGPTFileWriter.write(file, to: url)
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt memit <model.tinygpt> --facts facts.jsonl \\
                             --layer N --out <edited.tinygpt> [--scale F] [--lambda F]

        Batched rank-K fact editing (Meng et al. 2023). Solves a
        least-squares ΔW K ≈ V* − W K via the closed-form
        ΔW = R (KᵀK + λI)⁻¹ Kᵀ. λ defaults to 1e-2 of mean(diag(KᵀK)).

        --facts <path.jsonl>   {prompt, target} records, one per line
        --layer N              single TransformerBlock to edit
        --layers SPEC          multi-layer edit. Spec is an integer,
                               comma list ("3,5,7"), or inclusive
                               range ("4-8"). Per-layer share = scale /
                               layer-count — softer per-layer, total
                               same.
        --out <path>           where to save the edited model
        --scale F              total edit strength (default 1.0)
        --lambda F             Tikhonov regulariser (default auto)

        Multi-layer mode is the second-cut Meng 2023 algorithm and
        produces cleaner sampling visibility than the single-layer
        first cut.
        """)
        exit(code)
    }
}
