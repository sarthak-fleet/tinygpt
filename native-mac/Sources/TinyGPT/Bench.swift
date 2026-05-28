import Foundation
import MLX
import MLXNN
import TinyGPTIO
import TinyGPTModel

/// `tinygpt bench` — runs the apples-to-apples training-throughput
/// benchmark against the browser's WebGPU baseline.
///
/// Reports steps/sec, tokens/sec, time-per-step, and the implied speedup
/// vs. the browser baseline documented in `roadmap.astro` lever 20
/// (~120 ms/step for the Huge preset on M-series, the gallery-training
/// reference).
enum Bench {
    /// Browser WebGPU baseline on M-series for Huge preset.
    /// Source: `browser/train_gallery_one.mjs` runs at ~100 steps/min on
    /// M-series Pro = ~600 ms/step. Lever 20 cites 60 min for 5000 steps
    /// → 720 ms/step. Use the more conservative number to avoid claiming
    /// inflated speedups.
    static let browserStepMillisHuge: Double = 720.0
    static let browserStepMillisMega: Double = 9000.0  // browser can't reach Mega

    static func run(args: [String]) {
        var preset = "huge"
        var steps = 200
        var batchSize = 8
        var dtype = "fp32"
        var warmup = 20
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--preset":
                guard i + 1 < args.count else { exitUsage() }
                preset = args[i + 1]; i += 2
            case "--steps":
                guard i + 1 < args.count else { exitUsage() }
                steps = Int(args[i + 1]) ?? steps; i += 2
            case "--batch":
                guard i + 1 < args.count else { exitUsage() }
                batchSize = Int(args[i + 1]) ?? batchSize; i += 2
            case "--dtype":
                guard i + 1 < args.count else { exitUsage() }
                dtype = args[i + 1]; i += 2
            case "--warmup":
                guard i + 1 < args.count else { exitUsage() }
                warmup = Int(args[i + 1]) ?? warmup; i += 2
            case "-h", "--help":
                exitUsage()
            default:
                fputs("unknown bench flag: \(args[i])\n", stderr); exitUsage()
            }
        }

        var cfg: ModelConfig
        switch preset.lowercased() {
        case "huge": cfg = ModelConfig.huge
        case "mega": cfg = ModelConfig.mega
        default:
            fputs("unknown preset: \(preset). Choose huge or mega.\n", stderr)
            exit(2)
        }
        cfg.dtype = dtype

        print("""

        TinyGPT — Mac training-throughput benchmark
        -------------------------------------------
        preset:       \(preset) (\(cfg.nLayers)L, d=\(cfg.dModel), ctx=\(cfg.contextLength), heads=\(cfg.nHeads), dMlp=\(cfg.dMlp))
        batch size:   \(batchSize)
        dtype:        \(dtype)
        warmup steps: \(warmup)
        timed steps:  \(steps)
        device:       \(Device.defaultDevice())
        """)

        // Build a small synthetic corpus of random bytes. We're measuring
        // time-per-step, not training quality — but the bytes MUST be random,
        // not zeros (which would let the model trivially memorise 0→0→0 and
        // drop loss to ~0 in two steps, giving the false impression of an
        // optimisation bug).
        let randomBytes = (0..<1_000_000).map { _ in UInt8.random(in: 0...255) }
        let corpus = ByteCorpus(Data(randomBytes))
        let model = TinyGPTModel(cfg)
        // Move model to dtype if fp16 requested.
        if cfg.mlxDType != .float32 {
            do {
                try model.update(parameters: castParametersToDtype(model, cfg.mlxDType), verify: [])
            } catch {
                fputs("warn: dtype cast failed (\(error)); continuing fp32\n", stderr)
            }
        }
        let trainer = Trainer(model: model)

        print("  params:       \(formatLargeInt(model.numParameters()))")
        // Sanity check: random model on random data should land near ln(256) ≈ 5.54.
        // A loss of ~0 indicates a graph/eval bug; a loss of NaN/inf indicates init or dtype.
        do {
            let (x, y) = corpus.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            let rawLoss = model.loss(x, y)
            eval(rawLoss)
            print("  init loss:    \(String(format: "%.3f", rawLoss.item(Float.self))) (expect ~\(String(format: "%.2f", log(Float(cfg.vocabSize)))))")
        }
        fflush(stdout)

        // Warmup — the first few steps include JIT-compilation overhead.
        print("\nwarmup…")
        for _ in 0..<warmup {
            let (x, y) = corpus.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            _ = trainer.step(inputs: x, targets: y)
        }

        print("timing \(steps) steps…")
        let t0 = Date()
        var lossSum: Float = 0
        for _ in 0..<steps {
            let (x, y) = corpus.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            lossSum += trainer.step(inputs: x, targets: y)
        }
        let elapsed = -t0.timeIntervalSinceNow
        let stepsPerSec = Double(steps) / elapsed
        let msPerStep = elapsed * 1000 / Double(steps)
        let tokensPerStep = batchSize * cfg.contextLength
        let tokensPerSec = stepsPerSec * Double(tokensPerStep)

        let browserMillis = preset == "huge" ? browserStepMillisHuge : browserStepMillisMega
        let speedup = browserMillis / msPerStep

        print("""

        RESULTS
        -------
        time per step:    \(String(format: "%.1f", msPerStep)) ms
        steps / second:   \(String(format: "%.1f", stepsPerSec))
        tokens / second:  \(formatLargeInt(Int(tokensPerSec)))
        avg loss:         \(String(format: "%.3f", lossSum / Float(steps)))
        elapsed:          \(String(format: "%.1f", elapsed)) s for \(steps) steps

        vs. browser baseline (\(String(format: "%.0f", browserMillis)) ms/step):
          speedup:        \(String(format: "%.1fx", speedup))

        """)

        if speedup >= 100 {
            print("🎯  >100x — the headline target.")
        } else if speedup >= 50 {
            print("✓  >50x — solid Mac-native lift; bump --preset mega or --dtype float16 to chase 100x.")
        } else if speedup >= 25 {
            print("•  >25x — expected MLX-Swift baseline. fp16 + larger batch should add another 1.5-2x.")
        } else {
            print("⚠  under 25x — something is wrong; check that Device.default is .gpu and dtype isn't tying us to CPU.")
        }
    }

    private static func castParametersToDtype(_ model: TinyGPTModel, _ dtype: DType) -> ModuleParameters {
        var result = NestedDictionary<String, MLXArray>()
        let params = model.parameters()
        for (key, item) in params.flattened() {
            result[key] = .value(item.asType(dtype))
        }
        return result
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt bench [options]

        --preset huge|mega    Model size (default huge — matches gallery)
        --steps N             Timed steps (default 200)
        --batch N             Batch size (default 8)
        --dtype float32|float16  Training dtype (default float32)
        --warmup N            Warmup steps (default 20, excluded from timing)
        """)
        exit(2)
    }
}
