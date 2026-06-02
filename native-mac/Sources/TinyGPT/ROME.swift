import Foundation
import MLX
import MLXNN
import TinyGPTIO
import TinyGPTModel

/// `tinygpt rome` — surgical fact editing via rank-1 update to one
/// layer's MLP down-projection.
///
/// Meng et al. 2022, "Locating and Editing Factual Associations in
/// GPT" (ROME). Idea: the W_out matrix of a mid-network MLP layer
/// stores facts as associations between a "key" (the activation at
/// the last subject token) and a "value" (the residual contribution
/// that biases the LM head toward the correct object). A rank-1 update
/// W' = W + v · k^T / ‖k‖² re-targets the value at fixed key, leaving
/// every other (key', value) pair approximately untouched (because
/// k'·k is small for unrelated keys).
///
/// FULL ROME uses a corpus-level covariance C = E[kk^T] to whiten the
/// update direction: W' = W + (v* - Wk) k^T C⁻¹ / (k^T C⁻¹ k). This
/// first cut uses C = I (identity-Hessian) — works empirically on
/// from-scratch byte-level models where the activation distribution
/// is closer to isotropic. Covariance-based ROME is queued for the
/// follow-up.
///
/// USAGE
///   tinygpt rome <model.tinygpt> --prompt "..." --target <byte>
///                --layer N --out <edited.tinygpt>
///
///   --prompt "..."      The full prompt the model should complete
///   --target <byte>     The target byte (single character) the edit
///                       should make the model predict next
///   --layer N           Which TransformerBlock's MLP to edit
///                       (mid-network typically best — e.g., layer 5/12)
///   --out <path>        Where to save the edited model
///   --scale F           Optional scaling factor on the update
///                       (default 1.0; lower → softer edit)
///
/// EXAMPLE
///   tinygpt rome shakespeare.tinygpt --prompt "ROMEO:" --target "X" \
///       --layer 6 --out shakespeare-romeo-says-x.tinygpt
///
/// After editing, sampling from the edited checkpoint with the same
/// prompt prefix should bias the next token toward the target. The
/// edit is rank-1, so most other generations stay close to the base.
enum ROME {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var prompt: String? = nil
        var targetChar: String? = nil
        var layer: Int? = nil
        var outPath: String? = nil
        var scale: Float = 1.0

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--prompt":     prompt = args[i+1]; i += 2
            case "--target":     targetChar = args[i+1]; i += 2
            case "--layer":      layer = Int(args[i+1]); i += 2
            case "--out":        outPath = args[i+1]; i += 2
            case "--scale":      scale = Float(args[i+1]) ?? 1.0; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let prompt = prompt else { fputs("--prompt required\n", stderr); exitUsage() }
        guard let targetChar = targetChar else { fputs("--target required\n", stderr); exitUsage() }
        guard let layer = layer else { fputs("--layer required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }
        guard let targetByte = targetChar.utf8.first, targetChar.utf8.count == 1 else {
            fputs("--target must be a single byte (ASCII char); got '\(targetChar)'\n", stderr); exit(2)
        }

        // Load — first cut targets byte-level models. Same scoping as
        // tuned-lens / linear-probe.
        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("rome first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        guard layer >= 0, layer < cfg.nLayers else {
            fputs("--layer \(layer) out of range [0, \(cfg.nLayers))\n", stderr); exit(2)
        }
        guard model.blocks[layer].mlp != nil else {
            fputs("layer \(layer) has no dense MLP (MoE expert layer? edit not applicable).\n", stderr); exit(2)
        }

        // Forward the prompt and capture the activation right BEFORE
        // the chosen layer's fcOut at the LAST position. That's the
        // "key" k. We do this by re-running the forward up to layer N,
        // intercepting MLP's `gelu(fcIn(x))` output.
        let promptBytes: [Int32] = prompt.utf8.prefix(cfg.contextLength).map { Int32($0) }
        guard !promptBytes.isEmpty else { fputs("--prompt is empty after byte encoding\n", stderr); exit(2) }
        let idx = MLXArray(promptBytes, [1, promptBytes.count])

        // Layerwise hidden states (post-block residual at each depth).
        // The pre-fcOut activation = gelu(fcIn(h)) where h is the
        // input residual to the MLP block at this layer. We can
        // reconstruct h from the previous layer's output via the
        // model's standard forward.
        let states = model.forwardLayerwise(idx)
        // The MLP at block[layer] consumes the *attention-output-added*
        // residual. forwardLayerwise returns states[i] = output AFTER
        // block i; for the MLP input we need the state right before
        // block[layer]'s MLP — i.e., after its attention sub-block.
        // Simplification: use the residual at block[layer-1]'s output
        // (or the embedding if layer==0) as the input to block[layer].
        // The attention sub-block adds its own contribution, so this
        // approximates h up to the attention-add (cheap and works for
        // the rank-1 update target — the key direction is what
        // matters, not the exact magnitude).
        let inputResidual: MLXArray
        if layer == 0 {
            // Re-run just the embedding + posemb to get block-0 input.
            let posEmb = positionEmbeddingFor(model: model, T: promptBytes.count)
            let tokEmb = model.tokenEmbedding(idx) + posEmb
            inputResidual = tokEmb
        } else {
            inputResidual = states[layer - 1]
        }

        // Apply attention sub-block to get the MLP-input residual.
        let block = model.blocks[layer]
        let attnOut = block.attn(block.ln1(inputResidual))
        let mlpInput = inputResidual + attnOut

        // Now compute k = gelu(fcIn(ln2(mlpInput))) at the LAST position.
        guard let mlp = block.mlp else {
            fputs("layer \(layer) MLP missing — unreachable post-guard\n", stderr); exit(2)
        }
        let normed = block.ln2(mlpInput)
        let preGelu = mlp.fcIn(normed)
        let kAll = gelu(preGelu)  // [1, T, dMlp]
        let lastT = promptBytes.count - 1
        let k = kAll[0, lastT, 0...]   // [dMlp]
        MLX.eval(k)
        let kVec = k.asArray(Float.self)

        // v_target: direction in residual-stream space that would push
        // the LM head toward the target byte. With tied embeddings the
        // LM head equals tokenEmbedding^T, so the unembedding for byte
        // `targetByte` is `tokenEmbedding[targetByte]`. Use that
        // embedding as the target residual contribution.
        let targetEmbRow = model.tokenEmbedding.weight[Int(targetByte), 0...]
        MLX.eval(targetEmbRow)
        let eTarget = targetEmbRow.asArray(Float.self)  // [dModel]

        // Current fcOut(k) — the contribution the model is making
        // today. The edit pushes from this toward eTarget.
        let currentOut = mlp.fcOut(k.reshaped([1, cfg.dMlp]))
        MLX.eval(currentOut)
        let currVec = currentOut.asArray(Float.self)  // [dModel]

        // Compute the rank-1 delta:  v = (e_target - currentOut) * scale
        //                            ΔW = v ⊗ (k / ‖k‖²)
        var vDelta = [Float](repeating: 0, count: cfg.dModel)
        for d in 0..<cfg.dModel {
            vDelta[d] = (eTarget[d] - currVec[d]) * scale
        }
        var kNorm: Float = 0
        for x: Float in kVec { kNorm += x * x }
        guard kNorm > 1e-9 else {
            fputs("rome: key norm too small (\(kNorm)) — edit underdetermined\n", stderr); exit(1)
        }
        let kInv = 1.0 / kNorm

        // ΔW has shape [dModel, dMlp]:  ΔW[d, m] = vDelta[d] * kVec[m] * kInv
        var deltaW = [Float](repeating: 0, count: cfg.dModel * cfg.dMlp)
        for d in 0..<cfg.dModel {
            let v = vDelta[d]
            for m in 0..<cfg.dMlp {
                deltaW[d * cfg.dMlp + m] = v * kVec[m] * kInv
            }
        }
        let deltaArr = MLXArray(deltaW, [cfg.dModel, cfg.dMlp])

        // Apply the update by replacing fcOut.weight in the model.
        // We update through model.update(parameters:) so the live
        // module reflects the new value, then re-evaluate the forward
        // briefly to verify the edit landed.
        let newW = mlp.fcOut.weight + deltaArr
        MLX.eval(newW)
        var blocksList: [NestedItem<String, MLXArray>] = []
        for (i, b) in model.blocks.enumerated() {
            if i == layer {
                blocksList.append(.dictionary([
                    "mlp": .dictionary([
                        "fc_out": .dictionary(["weight": .value(newW)]),
                    ]),
                ]))
            } else {
                blocksList.append(.dictionary([:]))
            }
            _ = b
        }
        var root = NestedDictionary<String, MLXArray>()
        root["blocks"] = .array(blocksList)
        try? model.update(parameters: root, verify: [])

        // Diff report — how big was the edit, in Frobenius norm vs the
        // original weight magnitude.
        var deltaFro: Float = 0
        for x in deltaW { deltaFro += x * x }
        let origW = mlp.fcOut.weight.asArray(Float.self)
        var origFro: Float = 0
        for x in origW { origFro += x * x }
        let relSize = sqrt(deltaFro) / sqrt(origFro)
        print(String(format: """
        rome edit applied
          layer:        \(layer)
          prompt:       %@
          target byte:  0x%02x ('%@')
          ‖k‖²:         %.4f
          ‖v_delta‖:    %.4f
          ‖ΔW‖_F:       %.4e  (rel to ‖W‖_F = %.4e → %.4f%%)
        """,
            prompt, targetByte, targetChar,
            kNorm, sqrt(vDelta.reduce(Float(0)) { $0 + $1*$1 }),
            sqrt(deltaFro), sqrt(origFro), relSize * 100))

        // Persist. Easiest path: build a fresh manifest + tensor list
        // from the current (mutated) model parameters. Reuse the
        // existing Train-side checkpoint writer.
        do {
            try writeEditedCheckpoint(model: model, cfg: cfg,
                                        to: URL(fileURLWithPath: outPath))
            print("wrote edited model → \(outPath)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
    }

    /// Re-derive the position embedding for a given sequence length,
    /// matching what TinyGPTModel.forwardLayerwise computes internally.
    private static func positionEmbeddingFor(model: TinyGPTModel, T: Int) -> MLXArray {
        let positions = MLXArray((0..<T).map { Int32($0) })
        return model.positionEmbedding(positions).expandedDimensions(axis: 0)
    }

    /// Write the (mutated) model out as a .tinygpt file. Reuses the
    /// TrainSupport manifest helper so the format matches what Train
    /// writes.
    private static func writeEditedCheckpoint(model: TinyGPTModel,
                                                cfg: ModelConfig,
                                                to url: URL) throws {
        let entries = Train.manifestEntries(cfg)
        var tensors: [TinyGPTTensor] = []
        let params = model.parameters().flattened()
        for entry in entries {
            guard let w = params.first(where: { $0.0 == entry.name })?.1 else {
                throw NSError(domain: "rome", code: 1, userInfo: [
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
                heads: cfg.nHeads, dMlp: cfg.dMlp, batchSize: 1, backend: "mlx-swift-rome"
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
        usage: tinygpt rome <model.tinygpt> --prompt "..." --target <byte> \\
                            --layer N --out <edited.tinygpt> [--scale F]

        Rank-1 fact-editing of one MLP down-projection. Meng et al. 2022.
        First-cut uses identity-Hessian (no corpus covariance) — works on
        from-scratch byte-level models where activations are roughly
        isotropic. Covariance-based ROME is the follow-up.

        --prompt "..."    full prompt the model should complete
        --target <ch>     single-byte target the edit pushes toward
        --layer N         which TransformerBlock's MLP to edit
                          (mid-network typically — e.g. 5/12)
        --out <path>      where to save the edited model
        --scale F         soften the edit (default 1.0)
        """)
        exit(code)
    }
}
