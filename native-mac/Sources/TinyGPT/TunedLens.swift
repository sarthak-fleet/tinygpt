import Foundation
import MLX
import MLXNN
import MLXOptimizers
import TinyGPTIO
import TinyGPTModel

/// `tinygpt tuned-lens` — train per-layer projection probes that
/// improve on the raw logit lens (Belrose et al., 2023, "Eliciting
/// Latent Predictions from Transformers with the Tuned Lens").
///
/// The raw logit lens reuses the FINAL layernorm + tied LM head to
/// project every layer's residual stream into vocab space. That's
/// noisy for mid layers — the final LN's statistics were calibrated
/// for the FINAL layer's output, not intermediate ones. The tuned
/// lens replaces each layer's projection with a TRAINED probe
/// (`Linear(d_model → vocab)` per layer) fit on a held-out corpus.
/// The base model stays frozen; only the probes update.
///
/// File output: a small `.lenses` sidecar holding one (weight, bias)
/// per probe. Loadable via `attachTunedLens(from:)` for inference.
///
/// USAGE
///   tinygpt tuned-lens <model.tinygpt> --corpus <text.txt> \
///       --steps 500 --lr 1e-3 --out lenses.lenses
enum TunedLens {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var corpusPath: String? = nil
        var outPath: String? = nil
        var steps = 500
        var lr: Float = 1e-3
        var batchSize: Int? = nil
        var ctxOverride: Int? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus":     corpusPath = args[i+1]; i += 2
            case "--out":        outPath = args[i+1]; i += 2
            case "--steps":      steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":         lr = Float(args[i+1]) ?? lr; i += 2
            case "--batch":      batchSize = Int(args[i+1]); i += 2
            case "--ctx":        ctxOverride = Int(args[i+1]); i += 2
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }

        // Load model. Byte-level corpora only in this first cut — keeps
        // the sampler simple. Tuned lens for BPE models is the same
        // algorithm with a different sampler; ~5 lines of plumbing
        // when needed.
        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("tuned-lens first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        guard cfg.tokenizerSource == nil else {
            fputs("tuned-lens first-cut is byte-level only — BPE coming.\n", stderr); exit(2)
        }

        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: URL(fileURLWithPath: corpusPath)) }
        catch { fputs("corpus read failed: \(error)\n", stderr); exit(1) }

        // Freeze the base model; init the probes.
        model.freeze(recursive: true)
        model.initTunedLens()
        guard let probes = model.tunedLens else {
            fputs("internal error: probes not initialised\n", stderr); exit(1)
        }
        // Unfreeze only the probes — gradient flows through them, base
        // tensors stay frozen.
        for p in probes { p.unfreeze(recursive: true) }

        let B = batchSize ?? defaultBatch(cfg)
        let T = ctxOverride ?? cfg.contextLength

        print("""

        TinyGPT — tuned lens (probe training)
        -------------------------------------
        model:           \(modelPath)  (\(cfg.nLayers)L · d=\(cfg.dModel))
        corpus:          \(corpusPath) (\(corpus.bytes.count) bytes)
        probes:          \(probes.count)  (one Linear(\(cfg.dModel) → \(cfg.vocabSize)) per layer)
        steps / lr:      \(steps) / \(lr)
        batch / ctx:     \(B) / \(T)
        output:          \(outPath)

        """)

        // Build the train step. The loss is the MEAN of per-layer CE —
        // each probe pulls its OWN layer toward the next-token target.
        // Probes don't share gradients, so each one specialises to its
        // depth.
        let opt = AdamW(learningRate: lr, weightDecay: 0)
        let lossFn = { (m: TinyGPTModel, x: MLXArray, y: MLXArray) -> MLXArray in
            let perLayer = m.forwardTunedLens(x)   // [layers] of [B, T, V]
            var total = MLXArray(Float(0))
            for logits in perLayer {
                let v = logits.shape.last!
                total = total + crossEntropy(
                    logits: logits.reshaped([-1, v]),
                    targets: y.reshaped([-1]),
                    reduction: .mean
                )
            }
            return total / MLXArray(Float(perLayer.count))
        }
        let gradFn = valueAndGrad(model: model, lossFn)

        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()
        let t0 = Date()
        var lastLoss: Float = 0
        var lastStep = 0
        var stoppedEarly = false
        for step in 0..<steps {
            if TrainSupport.stopRequested.isSet { stoppedEarly = true; break }
            let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: T)
            let (loss, grads) = gradFn(model, x, y)
            opt.update(model: model, gradients: grads)
            MLX.eval(loss, model, opt)
            lastLoss = loss.item(Float.self)
            lastStep = step + 1
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.3f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(stoppedEarly
            ? String(format: "\ninterrupted at step %d · final mean loss %.3f", lastStep, lastLoss)
            : String(format: "\ndone — %d steps in %.1fs · final mean loss %.3f",
                      steps, elapsed, lastLoss))

        do {
            try saveLensFile(probes: probes, vocabSize: cfg.vocabSize,
                              dModel: cfg.dModel, to: URL(fileURLWithPath: outPath))
            print("✓ wrote \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    /// Write the trained probes as a compact sidecar file.
    /// Layout (little-endian):
    ///   magic "TGTL" (4 bytes)
    ///   version u32  (currently 1)
    ///   nLayers u32, vocabSize u32, dModel u32
    ///   per-layer (vocabSize·dModel Floats weight, vocabSize Floats bias)
    static func saveLensFile(probes: [Linear], vocabSize: Int, dModel: Int, to url: URL) throws {
        var buf = Data()
        buf.append(contentsOf: [0x54, 0x47, 0x54, 0x4c])    // "TGTL"
        var v: UInt32 = 1; withUnsafeBytes(of: &v) { buf.append(contentsOf: $0) }
        var nL = UInt32(probes.count); withUnsafeBytes(of: &nL) { buf.append(contentsOf: $0) }
        var vs = UInt32(vocabSize); withUnsafeBytes(of: &vs) { buf.append(contentsOf: $0) }
        var dm = UInt32(dModel); withUnsafeBytes(of: &dm) { buf.append(contentsOf: $0) }
        for probe in probes {
            MLX.eval(probe.weight)
            let wFloats: [Float] = probe.weight.asArray(Float.self)
            wFloats.withUnsafeBufferPointer { buf.append(Data(buffer: $0)) }
            if let b = probe.bias {
                MLX.eval(b)
                let bFloats: [Float] = b.asArray(Float.self)
                bFloats.withUnsafeBufferPointer { buf.append(Data(buffer: $0)) }
            } else {
                // No-bias case — write zeros of the expected length so
                // the file layout stays predictable.
                let zeros = [Float](repeating: 0, count: vocabSize)
                zeros.withUnsafeBufferPointer { buf.append(Data(buffer: $0)) }
            }
        }
        try buf.write(to: url, options: .atomic)
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 1024 { return 1 }
        if cfg.dModel >= 512 { return 2 }
        if cfg.dModel >= 256 { return 4 }
        return 8
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt tuned-lens <model.tinygpt> --corpus <text> [options]

        --corpus <text>     UTF-8 byte-level text to fit the probes on
        --out <path>        Where to save the .lenses sidecar (required)
        --steps N           Training steps (default 500)
        --lr F              Learning rate (default 1e-3 — small probes
                              don't need much)
        --batch N           Batch size (default by preset)
        --ctx N             Context length override

        Trains one Linear(d_model → vocab) per layer with the base
        model FROZEN. Output: a small sidecar file that any future
        sample / lens path can attach via `attachTunedLens(from:)`.
        """)
        exit(2)
    }
}
