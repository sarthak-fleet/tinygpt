import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt es` — Evolution Strategies trainer (Salimans et al., 2017,
/// "Evolution Strategies as a Scalable Alternative to Reinforcement
/// Learning"). Gradient-free training: at each step we sample K random
/// perturbations of the current weights, evaluate each perturbed model
/// on a shared batch, and update along the reward-weighted noise
/// direction.
///
/// Compared to AdamW + backprop:
///   - No autograd needed — only forward passes.
///   - Pleasantly parallel across the K samples (we ship the serial
///     version here; a parallel one is a future tweak).
///   - Often slower-to-converge than SGD in absolute terms, but a
///     completely different regime — useful for non-differentiable
///     reward signals and as an educational counterpoint to the main
///     SGD path.
///
/// Antithetic sampling halves the gradient variance for the same K:
/// for each noise vector ε, we evaluate BOTH `w + σ·ε` and `w - σ·ε`,
/// then estimate the gradient via `(R(+) - R(-)) / 2`. Standard ES
/// variance-reduction trick from Salimans 2017.
///
/// USAGE
///   tinygpt es <model.tinygpt> --corpus <text> \
///       --steps 200 --population 40 --sigma 0.02 --lr 0.01 \
///       --out es-trained.tinygpt
///
/// Population MUST be even (we pair them antithetically).
enum ES {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var corpusPath: String? = nil
        var outPath: String? = nil
        var steps = 100
        var populationSize = 40
        var sigma: Float = 0.02
        var lr: Float = 0.01
        var batchSize: Int? = nil
        var ctxOverride: Int? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus":     corpusPath = args[i+1]; i += 2
            case "--out":        outPath = args[i+1]; i += 2
            case "--steps":      steps = Int(args[i+1]) ?? steps; i += 2
            case "--population": populationSize = Int(args[i+1]) ?? populationSize; i += 2
            case "--sigma":      sigma = Float(args[i+1]) ?? sigma; i += 2
            case "--lr":         lr = Float(args[i+1]) ?? lr; i += 2
            case "--batch":      batchSize = Int(args[i+1]); i += 2
            case "--ctx":        ctxOverride = Int(args[i+1]); i += 2
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model.tinygpt>\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus <text> required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path> required\n", stderr); exitUsage() }
        precondition(populationSize % 2 == 0, "--population must be even (antithetic pairs)")

        // Load model. ES doesn't care what the model is — it just needs a
        // forward + a parameter tree. We restrict to the from-scratch
        // class because saving back as .tinygpt assumes that layout.
        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        let cfg = load.config
        guard case .fromScratch(let model) = load.model else {
            fputs("ES only supports from-scratch models (HF-class save isn't .tinygpt-compatible yet).\n", stderr)
            exit(2)
        }
        // Byte-level corpora only in this first cut — keeps the sampler
        // simple. BPE support would just plumb through the tokenizer.
        guard cfg.tokenizerSource == nil else {
            fputs("ES first-cut supports byte-level models only (BPE coming).\n", stderr); exit(2)
        }

        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: URL(fileURLWithPath: corpusPath)) }
        catch { fputs("corpus read failed: \(error)\n", stderr); exit(1) }

        let B = batchSize ?? defaultBatch(cfg)
        let T = ctxOverride ?? cfg.contextLength

        print("""

        TinyGPT — Evolution Strategies
        ------------------------------
        model:          \(modelPath) (\(cfg.nLayers)L · d=\(cfg.dModel))
        params:         \(formatLargeInt(model.numParameters()))
        corpus:         \(corpusPath) (\(corpus.bytes.count) bytes)
        population K:   \(populationSize)  (\(populationSize / 2) antithetic pairs)
        sigma:          \(sigma)  (perturbation scale)
        lr:             \(lr)
        steps:          \(steps)
        batch / ctx:    \(B) / \(T)
        device:         \(Device.defaultDevice())

        Each step does \(populationSize) forward passes (no backward).
        """)
        fflush(stdout)

        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()
        let t0 = Date()

        var stoppedEarly = false
        var lastMeanLoss: Float = 0
        var lastStep = 0
        for step in 0..<steps {
            if TrainSupport.stopRequested.isSet { stoppedEarly = true; break }
            let (loss, _) = esStep(model: model, corpus: corpus,
                                    B: B, T: T,
                                    populationSize: populationSize,
                                    sigma: sigma, lr: lr)
            lastMeanLoss = loss
            lastStep = step + 1
            if step == 0 || (step + 1) % 5 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  mean loss %.3f  · %.2f step/s · eta %.0fs\n",
                             step + 1, steps, lastMeanLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(stoppedEarly
            ? String(format: "\ninterrupted at step %d after %.1fs · loss %.3f",
                      lastStep, elapsed, lastMeanLoss)
            : String(format: "\ndone — %d steps in %.1fs (%.2f step/s) · final loss %.3f",
                      steps, elapsed, Double(steps) / elapsed, lastMeanLoss))

        // Save the final weights. Re-uses Train.swift's manifest helpers so
        // ES checkpoints are interoperable with the rest of the toolchain.
        do {
            try TrainSupport.atomicSave(
                model: model, cfg: cfg, step: lastStep, finalLoss: lastMeanLoss,
                weightTranspose: Train.isLinearWeightName,
                manifestEntries: Train.manifestEntries,
                to: URL(fileURLWithPath: outPath)
            )
            print("✓ wrote \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    /// One ES update step. Returns (meanLoss, _).
    ///
    /// Procedure (antithetic-pair ES):
    ///   1. Snapshot base params.
    ///   2. Sample a shared batch (same data for every population member).
    ///   3. For each of K/2 pairs:
    ///        - Draw noise ε ~ N(0, 1)
    ///        - Evaluate L(+) = loss(w + σε), L(-) = loss(w - σε)
    ///        - Reward = -loss
    ///   4. Centre rewards (subtract mean across all K samples).
    ///   5. Update direction: Σ_pairs (R(+) - R(-))/2 · ε
    ///   6. New params: w + (lr / (K·σ)) · update
    private static func esStep(model: TinyGPTModel, corpus: ByteCorpus,
                                B: Int, T: Int,
                                populationSize K: Int,
                                sigma: Float, lr: Float)
        -> (meanLoss: Float, _: Void)
    {
        // Snapshot base. ModuleParameters is the per-param dict — same
        // shape as what model.parameters() returns, with MLXArrays at
        // every leaf. We deep-copy by mapping each leaf through `+ 0` so
        // later updates don't mutate the snapshot in place.
        let basePV = model.parameters().mapValues { v in v + MLXArray(Float(0)) }
        MLX.eval(basePV)

        // Shared batch for the whole population — what ES needs to give
        // each sample the same reward landscape.
        let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: T)

        let nPairs = K / 2
        var rewards: [Float] = []
        rewards.reserveCapacity(K)
        // Store noise per pair; the update step reads them back.
        var noises: [ModuleParameters] = []
        noises.reserveCapacity(nPairs)
        let sigmaA = MLXArray(sigma)

        for _ in 0..<nPairs {
            // Generate one noise tree with the SAME shapes as basePV.
            // Each leaf is N(0, 1) — sigma is applied at perturbation time.
            let noise: ModuleParameters = basePV.mapValues { v in
                MLXRandom.normal(v.shape).asType(v.dtype)
            }
            noises.append(noise)

            // +ε branch
            let plus: ModuleParameters = basePV.mapValues(noise) { v, n in
                v + (n ?? MLXArray(Float(0))) * sigmaA
            }
            try? model.update(parameters: plus, verify: [])
            let lossPlus = model.loss(x, y)
            MLX.eval(lossPlus)
            rewards.append(-lossPlus.item(Float.self))

            // -ε branch
            let minus: ModuleParameters = basePV.mapValues(noise) { v, n in
                v - (n ?? MLXArray(Float(0))) * sigmaA
            }
            try? model.update(parameters: minus, verify: [])
            let lossMinus = model.loss(x, y)
            MLX.eval(lossMinus)
            rewards.append(-lossMinus.item(Float.self))
        }

        // Centre rewards. Salimans uses (R - mean) / std normalisation;
        // we use centring only — the std-divide is a tiny additional
        // variance reduction but adds a divide-by-near-zero risk on
        // early steps where rewards are all close to the same value.
        let meanR: Float = rewards.reduce(0, +) / Float(rewards.count)
        let centred: [Float] = rewards.map { $0 - meanR }

        // Antithetic gradient estimator: for each pair, dir = (R(+) - R(-)) / 2.
        // The +ε and -ε contributions partially cancel — cuts variance ~2×
        // for the same K samples vs IID one-sided estimation.
        var update: ModuleParameters = basePV.mapValues { v in
            MLXArray.zeros(v.shape).asType(v.dtype)
        }
        for (i, noise) in noises.enumerated() {
            let dir: Float = (centred[2 * i] - centred[2 * i + 1]) / 2.0
            let dirA = MLXArray(dir)
            update = update.mapValues(noise) { u, n in
                u + (n ?? MLXArray(Float(0))) * dirA
            }
        }
        // Final step: w = w + (lr / (K · σ)) · update
        let scaleA = MLXArray(lr / (Float(K) * sigma))
        let newParams: ModuleParameters = basePV.mapValues(update) { v, u in
            v + (u ?? MLXArray(Float(0))) * scaleA
        }
        try? model.update(parameters: newParams, verify: [])
        MLX.eval(model)

        return (meanLoss: -meanR, ())
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        // ES does K forward passes per step — keep batch modest so the
        // total per-step cost stays reasonable. Population is the main
        // throughput knob, not batch.
        if cfg.dModel >= 1024 { return 1 }
        if cfg.dModel >= 512 { return 2 }
        if cfg.dModel >= 256 { return 4 }
        return 8
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt es <model.tinygpt> --corpus <text> [options]

        --corpus <text>          UTF-8 text (byte-level only in this first cut)
        --out <path>             Where to save the trained checkpoint
        --steps N                ES iterations (default 100)
        --population K           Population size — MUST be even (default 40)
        --sigma F                Perturbation scale σ (default 0.02)
        --lr F                   Step size (default 0.01)
        --batch N                Batch (default by preset)
        --ctx N                  Context length override

        ES is gradient-free: each step does K forward passes (no backward).
        Slower per-step than SGD but useful as an educational counterpoint
        and for non-differentiable reward signals.
        """)
        exit(2)
    }
}
