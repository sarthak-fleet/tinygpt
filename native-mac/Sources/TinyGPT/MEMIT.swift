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
        var layer: Int? = nil
        var outPath: String? = nil
        var scale: Float = 1.0
        var lambdaArg: Float? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--facts":      factsPath = args[i+1]; i += 2
            case "--layer":      layer = Int(args[i+1]); i += 2
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
        guard let layer = layer else { fputs("--layer required\n", stderr); exitUsage() }
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
        guard layer >= 0, layer < cfg.nLayers else {
            fputs("--layer \(layer) out of range [0, \(cfg.nLayers))\n", stderr); exit(2)
        }
        let block = model.blocks[layer]
        guard let mlp = block.mlp else {
            fputs("layer \(layer) has no dense MLP (MoE expert layer? edit not applicable).\n", stderr); exit(2)
        }

        // For each fact: forward up to the chosen layer, capture k_i
        // (gelu(fcIn(ln2(mlpInput))) at the last token) and v*_i (the
        // target embedding row).
        var keys: [[Float]] = []          // N × dMlp
        var targets: [[Float]] = []       // N × dModel
        for fact in facts {
            let bytes: [Int32] = fact.prompt.utf8.prefix(cfg.contextLength).map { Int32($0) }
            guard !bytes.isEmpty else {
                fputs("warn: empty prompt after byte encoding, skipping\n", stderr); continue
            }
            let idx = MLXArray(bytes, [1, bytes.count])
            let states = model.forwardLayerwise(idx)
            let inputResidual: MLXArray
            if layer == 0 {
                let positions = MLXArray((0..<bytes.count).map { Int32($0) })
                let posEmb = model.positionEmbedding(positions).expandedDimensions(axis: 0)
                inputResidual = model.tokenEmbedding(idx) + posEmb
            } else {
                inputResidual = states[layer - 1]
            }
            let attnOut = block.attn(block.ln1(inputResidual))
            let mlpInput = inputResidual + attnOut
            let kAll = gelu(mlp.fcIn(block.ln2(mlpInput)))
            let lastT = bytes.count - 1
            let k = kAll[0, lastT, 0...]
            MLX.eval(k)
            keys.append(k.asArray(Float.self))

            let targetEmbRow = model.tokenEmbedding.weight[Int(fact.target), 0...]
            MLX.eval(targetEmbRow)
            targets.append(targetEmbRow.asArray(Float.self))
        }
        let M = cfg.dModel
        let D = cfg.dMlp
        precondition(keys.count == N && targets.count == N, "key/target count mismatch")

        // Stack K: dMlp × N. Stack V*: dModel × N. Compute V_curr = W·K
        // by running the SAME fcOut weight against each captured key.
        // currents[i] = fcOut(keys[i]) which is exactly what the model
        // would output today.
        var currents: [[Float]] = []
        for k in keys {
            let kArr = MLXArray(k, [1, D])
            let out = mlp.fcOut(kArr)
            MLX.eval(out)
            currents.append(out.asArray(Float.self))
        }

        // Residual R[d, j] = scale * (V*[d, j] - V_curr[d, j]).
        // We solve in column-major (per-fact) layout; flatten at the end.
        var R = [Float](repeating: 0, count: M * N)
        for j in 0..<N {
            for d in 0..<M {
                R[d * N + j] = scale * (targets[j][d] - currents[j][d])
            }
        }

        // K matrix in [dMlp, N] column-major: K[m, j] = keys[j][m].
        var K = [Float](repeating: 0, count: D * N)
        for j in 0..<N {
            for m in 0..<D {
                K[m * N + j] = keys[j][m]
            }
        }

        // K^T K is N × N (small). Build it.
        var KtK = [Float](repeating: 0, count: N * N)
        for i in 0..<N {
            for j in 0..<N {
                var s: Float = 0
                for m in 0..<D {
                    s += K[m * N + i] * K[m * N + j]
                }
                KtK[i * N + j] = s
            }
        }
        // Trace-scaled Tikhonov regulariser. lambda defaults to 1e-2 of
        // the mean diagonal — keeps the inverse stable when keys are
        // near-colinear without distorting the well-conditioned case.
        var trace: Float = 0
        for i in 0..<N { trace += KtK[i * N + i] }
        let lambda = lambdaArg ?? (max(trace / Float(N), 1) * 1e-2)
        for i in 0..<N { KtK[i * N + i] += lambda }

        // Invert KtK via Gauss-Jordan. N is small (typically 3-50), so
        // O(N³) is trivial. Hand-rolled to avoid pulling in a linalg
        // dependency for one routine.
        var inv = invertGaussJordan(KtK, n: N)

        // ΔW = R @ inv @ Kᵀ.
        // Shapes: R [M, N], inv [N, N], Kᵀ [N, D].
        //   (R @ inv) [M, N], then × Kᵀ → [M, D].
        // First compute T = R @ inv → [M, N].
        var T = [Float](repeating: 0, count: M * N)
        for d in 0..<M {
            for j in 0..<N {
                var s: Float = 0
                for k in 0..<N {
                    s += R[d * N + k] * inv[k * N + j]
                }
                T[d * N + j] = s
            }
        }
        _ = trace  // silence unused-warning if compiler ever flags

        // Then ΔW[d, m] = sum_j T[d, j] * K[m, j].
        var deltaW = [Float](repeating: 0, count: M * D)
        for d in 0..<M {
            for m in 0..<D {
                var s: Float = 0
                for j in 0..<N {
                    s += T[d * N + j] * K[m * N + j]
                }
                deltaW[d * D + m] = s
            }
        }

        // Apply to fcOut.
        let deltaArr = MLXArray(deltaW, [M, D])
        let newW = mlp.fcOut.weight + deltaArr
        MLX.eval(newW)
        var blocksList: [NestedItem<String, MLXArray>] = []
        for (i, _) in model.blocks.enumerated() {
            if i == layer {
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

        // Report. Per-fact residual norms tell you which facts the
        // edit struggled to land — large residuals = key collisions.
        var deltaFro: Float = 0
        for x: Float in deltaW { deltaFro += x * x }
        let origW = mlp.fcOut.weight.asArray(Float.self)
        var origFro: Float = 0
        for x: Float in origW { origFro += x * x }
        let relSize = sqrt(deltaFro) / sqrt(origFro)

        print(String(format: """
        memit edit applied
          facts:        \(N)
          layer:        \(layer)
          λ:            %.4e (auto)
          ‖ΔW‖_F:       %.4e  (rel to ‖W‖_F = %.4e → %.4f%%)
        """, lambda, sqrt(deltaFro), sqrt(origFro), relSize * 100))
        // Per-fact residuals after the edit — recompute V_new = W' · K
        // for each key and report ‖V*_j − V_new_j‖.
        var maxResidual: Float = 0
        for j in 0..<N {
            let kArr = MLXArray(keys[j], [1, D])
            let outNew = mlp.fcOut(kArr)
            MLX.eval(outNew)
            let outVec = outNew.asArray(Float.self)
            var resid: Float = 0
            for d in 0..<M {
                let diff = targets[j][d] - outVec[d]
                resid += diff * diff
            }
            let r = sqrt(resid)
            if r > maxResidual { maxResidual = r }
        }
        print(String(format: "  worst per-fact residual ‖v* - W'k‖: %.4e", maxResidual))

        do {
            try writeEditedCheckpoint(model: model, cfg: cfg,
                                        to: URL(fileURLWithPath: outPath))
            print("wrote edited model → \(outPath)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
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
        --layer N              which TransformerBlock's MLP to edit
        --out <path>           where to save the edited model
        --scale F              soften the edit (default 1.0)
        --lambda F             Tikhonov regulariser (default auto)
        """)
        exit(code)
    }
}
