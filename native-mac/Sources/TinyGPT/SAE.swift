import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import TinyGPTModel

/// `tinygpt sae` — train a sparse autoencoder on per-layer hidden
/// states. Each decoder column becomes one "feature direction" in
/// residual space; the encoder + ReLU produces a sparse activation
/// pattern over those features for any input.
///
/// Anthropic, "Towards Monosemanticity" (Bricken et al. 2023) — the
/// SAE is the canonical mechanism for decomposing a model's
/// internal representations into interpretable features. Distinct
/// from lens (projects hidden state to vocab), tuned-lens (same with
/// learned projections), and linear-probe (classifies an external
/// label): SAEs *unsupervised-discover* directions in residual space.
///
/// Architecture (matches the standard fingerprint):
///   encoded = relu(W_enc @ (h - b_dec) + b_enc)
///   reconstructed = W_dec @ encoded + b_dec
///   loss = ‖h - reconstructed‖² + λ · ‖encoded‖₁
///
/// where W_enc is [F, D], W_dec is [D, F], D = d_model, F = d_features
/// (typically 4-8× D, overcomplete). The "pre-encoder bias subtract"
/// b_dec follows Bricken et al.'s formulation — it lets the decoder
/// learn the activation centroid separately from the per-feature
/// directions, which empirically gives cleaner features.
///
/// USAGE
///   tinygpt sae <model.tinygpt> --corpus <text.txt> --layer N \
///               --out probe.sae [--features F] [--steps N]
///               [--lr F] [--l1 F] [--batch N] [--ctx N]
///
/// OUTPUT (.sae sidecar)
///   8-byte magic "TGSA" + version u32 + header_len u32 + JSON header
///   + raw fp32 [W_enc, b_enc, W_dec, b_dec] in that order.
enum SAE {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var corpusPath: String? = nil
        var outPath: String? = nil
        var layer: Int? = nil
        var dFeatures: Int? = nil   // default: 4× d_model
        var steps = 2000
        var lr: Float = 1e-3
        var l1Penalty: Float = 5e-3
        var batchSize = 32
        var ctxOverride: Int? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus":     corpusPath = args[i+1]; i += 2
            case "--out":        outPath = args[i+1]; i += 2
            case "--layer":      layer = Int(args[i+1]); i += 2
            case "--features":   dFeatures = Int(args[i+1]); i += 2
            case "--steps":      steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":         lr = Float(args[i+1]) ?? lr; i += 2
            case "--l1":         l1Penalty = Float(args[i+1]) ?? l1Penalty; i += 2
            case "--batch":      batchSize = Int(args[i+1]) ?? batchSize; i += 2
            case "--ctx":        ctxOverride = Int(args[i+1]); i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }
        guard let layer = layer else { fputs("--layer required\n", stderr); exitUsage() }

        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("sae first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        guard layer >= 0, layer < cfg.nLayers else {
            fputs("--layer \(layer) out of range [0, \(cfg.nLayers))\n", stderr); exit(2)
        }
        let D = cfg.dModel
        let F = dFeatures ?? (4 * D)
        let T = min(ctxOverride ?? cfg.contextLength, cfg.contextLength)
        guard T >= 8 else { fputs("ctx too small (\(T))\n", stderr); exit(2) }

        // Load corpus as byte stream.
        let bytes: [UInt8]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: corpusPath))
            bytes = [UInt8](data)
        } catch { fputs("error reading \(corpusPath): \(error)\n", stderr); exit(1) }
        guard bytes.count > T + 1 else {
            fputs("corpus too small (\(bytes.count) bytes, need > \(T+1))\n", stderr); exit(1)
        }

        print("""

        TinyGPT — sparse autoencoder
        ----------------------------
        base:           \(modelPath)
        corpus:         \(corpusPath) (\(bytes.count) bytes)
        layer:          \(layer)
        d_model:        \(D)
        d_features:     \(F)  (\(String(format: "%.1f", Float(F)/Float(D)))× overcomplete)
        steps:          \(steps)
        lr:             \(lr)
        L1 penalty:     \(l1Penalty)
        batch:          \(batchSize)
        ctx:            \(T)
        """)

        // Build the SAE module.
        MLXRandom.seed(0xC0FFEE)
        let sae = SaeModule(dModel: D, dFeatures: F)
        let opt = AdamW(learningRate: lr)

        // The MLX valueAndGrad signature requires (model, x, y); SAE
        // reconstruction loss is unsupervised (target == input), so we
        // pass h twice and ignore the second arg in the closure.
        let lossFn = { (m: SaeModule, h: MLXArray, _: MLXArray) -> MLXArray in
            // h shape: [B, D]. The encoder subtracts the decoder bias
            // first (pre-encoder bias trick — see Bricken et al.).
            let centered = h - m.bDec
            let encoded = MLX.maximum(MLX.matmul(centered, m.wEnc.transposed()) + m.bEnc,
                                       MLXArray(Float(0)))
            let reconstructed = MLX.matmul(encoded, m.wDec.transposed()) + m.bDec
            let mse = MLX.mean(MLX.pow(reconstructed - h, 2))
            let l1 = MLX.mean(MLX.abs(encoded))
            return mse + MLXArray(l1Penalty) * l1
        }
        let gradFn = valueAndGrad(model: sae, lossFn)

        // Sample residual-stream training batches from random windows
        // of the corpus. For each window: run forward up to layer L,
        // take ALL T position hidden states, and feed them as B*T
        // independent training samples to the SAE.
        let t0 = Date()
        var lastLoss: Float = 0
        for step in 0..<steps {
            // Build a batch of `batchSize` random ctx-windows.
            var batchBytes: [Int32] = []
            batchBytes.reserveCapacity(batchSize * T)
            for _ in 0..<batchSize {
                let lo = Int.random(in: 0..<(bytes.count - T))
                for k in 0..<T {
                    batchBytes.append(Int32(bytes[lo + k]))
                }
            }
            let idx = MLXArray(batchBytes, [batchSize, T])
            let states = model.forwardLayerwise(idx)
            // states[layer] has shape [B, T, D]; flatten to [B*T, D].
            let h = states[layer].reshaped([batchSize * T, D])
            MLX.eval(h)

            let (loss, grads) = gradFn(sae, h, h)
            opt.update(model: sae, gradients: grads)
            MLX.eval(loss, sae, opt)
            lastLoss = loss.item(Float.self)

            if step == 0 || (step + 1) % max(1, steps / 25) == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.4f  · %.1f step/s · eta %.0fs\n",
                              step + 1, steps, lastLoss, sps,
                              Double(steps - step - 1) / sps), stderr)
            }
        }

        // Diagnostics: average sparsity (fraction of non-zero features
        // per sample) and reconstruction MSE on a fresh batch.
        var diagBytes: [Int32] = []
        for _ in 0..<batchSize {
            let lo = Int.random(in: 0..<(bytes.count - T))
            for k in 0..<T { diagBytes.append(Int32(bytes[lo + k])) }
        }
        let diagIdx = MLXArray(diagBytes, [batchSize, T])
        let diagStates = model.forwardLayerwise(diagIdx)
        let hDiag = diagStates[layer].reshaped([batchSize * T, D])
        MLX.eval(hDiag)
        let centered = hDiag - sae.bDec
        let encDiag = MLX.maximum(MLX.matmul(centered, sae.wEnc.transposed()) + sae.bEnc,
                                    MLXArray(Float(0)))
        let reconDiag = MLX.matmul(encDiag, sae.wDec.transposed()) + sae.bDec
        let mseDiag = MLX.mean(MLX.pow(reconDiag - hDiag, 2)).item(Float.self)
        // L0: fraction of non-zero features per sample.
        let nonzeroMask = (encDiag .> MLXArray(Float(0))).asType(.float32)
        let l0PerSample = MLX.mean(MLX.sum(nonzeroMask, axis: 1)).item(Float.self)
        let l0Frac = l0PerSample / Float(F)

        print("""

        diagnostics on held-out batch:
          reconstruction MSE:   \(String(format: "%.4e", mseDiag))
          mean L0 (active features per sample): \(String(format: "%.1f / %d  (%.2f%%)", l0PerSample, F, l0Frac * 100))
        """)

        // Persist.
        do {
            try SaeWriter.write(sae: sae, cfg: cfg, layer: layer,
                                 to: URL(fileURLWithPath: outPath))
            print("wrote SAE sidecar → \(outPath)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt sae <model.tinygpt> --corpus <text.txt> --layer N \\
                           --out <probe.sae> [options]

        Train a sparse autoencoder on per-layer hidden states for
        feature decomposition (Bricken et al. 2023). Distinct
        mechanism from lens / tuned-lens / linear-probe.

        --corpus <path.txt>   UTF-8 byte-level corpus to fit on (required)
        --layer N             which TransformerBlock's residual stream
        --out <path.sae>      where to save the trained SAE (required)
        --features F          dictionary size (default 4× d_model)
        --steps N             training steps (default 2000)
        --lr F                learning rate (default 1e-3)
        --l1 F                L1 sparsity penalty (default 5e-3 —
                              higher = sparser features, more
                              reconstruction error)
        --batch N             batch size (default 32; each window
                              contributes ctx × samples)
        --ctx N               override context length (default model's)
        """)
        exit(code)
    }
}

/// Sparse-autoencoder module. Encoder + decoder + a shared "decoder
/// bias subtracted before encode" trick (Bricken et al.).
final class SaeModule: Module {
    let dModel: Int
    let dFeatures: Int
    @ParameterInfo(key: "w_enc") var wEnc: MLXArray   // [F, D]
    @ParameterInfo(key: "b_enc") var bEnc: MLXArray   // [F]
    @ParameterInfo(key: "w_dec") var wDec: MLXArray   // [D, F]
    @ParameterInfo(key: "b_dec") var bDec: MLXArray   // [D]
    init(dModel: Int, dFeatures: Int) {
        self.dModel = dModel
        self.dFeatures = dFeatures
        // Encoder init: random gaussian std 1/√D.
        self._wEnc.wrappedValue = MLXRandom.normal(
            [dFeatures, dModel], scale: 1.0 / Foundation.sqrt(Float(dModel)))
        self._bEnc.wrappedValue = MLXArray.zeros([dFeatures])
        // Decoder init: random gaussian std 1/√F (so initial output
        // matches input magnitude). Some SAE recipes initialise W_dec
        // as W_enc^T (tied init) — both work; we keep separate.
        self._wDec.wrappedValue = MLXRandom.normal(
            [dModel, dFeatures], scale: 1.0 / Foundation.sqrt(Float(dFeatures)))
        self._bDec.wrappedValue = MLXArray.zeros([dModel])
        super.init()
    }
}

private struct SaeHeader: Codable {
    var version: Int
    var dModel: Int
    var dFeatures: Int
    var layer: Int
    var baseLayers: Int
    var baseDModel: Int
    var baseCtx: Int
}

enum SaeWriter {
    static func write(sae: SaeModule, cfg: ModelConfig, layer: Int,
                       to url: URL) throws {
        let header = SaeHeader(
            version: 1, dModel: sae.dModel, dFeatures: sae.dFeatures,
            layer: layer, baseLayers: cfg.nLayers, baseDModel: cfg.dModel,
            baseCtx: cfg.contextLength
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try encoder.encode(header)
        var out = Data()
        out.append(contentsOf: Array("TGSA".utf8))
        var version = UInt32(1).littleEndian
        withUnsafeBytes(of: &version) { out.append(contentsOf: $0) }
        var headerLen = UInt32(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { out.append(contentsOf: $0) }
        out.append(headerData)
        // Body: wEnc, bEnc, wDec, bDec, each as fp32.
        MLX.eval(sae.wEnc, sae.bEnc, sae.wDec, sae.bDec)
        for arr in [sae.wEnc, sae.bEnc, sae.wDec, sae.bDec] {
            let floats: [Float] = arr.asArray(Float.self)
            floats.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        }
        try out.write(to: url, options: .atomic)
    }
}
