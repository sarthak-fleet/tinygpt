import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt finetune` — LoRA-fine-tune a checkpoint on a small text
/// corpus. The base weights stay frozen; only the rank-r adapter
/// matrices are trained. Adapter files are tiny (~100KB-1MB) and
/// portable: load a base + multiple adapters to switch "voices"
/// without reloading the base.
///
/// USAGE
///   tinygpt finetune base.tinygpt --corpus my-text.txt --out mine.lora
///   tinygpt finetune shakespeare.bin --corpus my-blog.txt --rank 8 --steps 200 --out blog.lora
enum Finetune {
    static func run(args: [String]) {
        var basePath: String?
        var corpusPath: String?
        var outPath: String?
        var rank = 4
        var alpha: Float = 8.0
        var steps = 200
        var lr: Float = 1e-3  // higher than full-finetune since adapter params are few
        var targetSuffixesArg = "q_proj,v_proj"
        var batchSize: Int? = nil
        var sampleEvery = 100
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus":  corpusPath = args[i+1]; i += 2
            case "--out":     outPath = args[i+1]; i += 2
            case "--rank":    rank = Int(args[i+1]) ?? rank; i += 2
            case "--alpha":   alpha = Float(args[i+1]) ?? alpha; i += 2
            case "--steps":   steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":      lr = Float(args[i+1]) ?? lr; i += 2
            case "--targets": targetSuffixesArg = args[i+1]; i += 2
            case "--batch":   batchSize = Int(args[i+1]); i += 2
            case "--sample-every": sampleEvery = Int(args[i+1]) ?? sampleEvery; i += 2
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                basePath = args[i]; i += 1
            }
        }
        guard let basePath = basePath else { fputs("missing base.tinygpt or HF dir\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }

        // Default LoRA targets differ between from-scratch (q+v on plain
        // MLP) and HF (q+v on SwiGLU MLP). User-supplied --targets always
        // wins; otherwise we pick sensible defaults per model kind below.
        var url = URL(fileURLWithPath: basePath); var isDir: ObjCBool = false
        _ = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
        let userSetTargets = (targetSuffixesArg != "q_proj,v_proj")
        if isDir.boolValue && !userSetTargets {
            // HF models use SwiGLU — q_proj + v_proj still works and is
            // the LoRA paper's recommendation. Keep the default.
        }
        let targetSuffixes = targetSuffixesArg.split(separator: ",").map(String.init)
        let loraCfg = LoraConfig(rank: rank, alpha: alpha, targetSuffixes: targetSuffixes)

        // Load model (auto-detects .tinygpt file vs HF directory)
        print("loading base from \(basePath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(basePath) }
        catch { fputs("error loading base: \(error)\n", stderr); exit(1) }
        let cfg = load.config

        // Inject LoRA into whichever model variant we got. The wrapper
        // freezes the base and unfreezes just the adapter matrices.
        let nTrainable = load.model.injectLora(config: loraCfg)
        let nTotal = load.model.numParameters()

        // Load corpus
        let corpusURL = URL(fileURLWithPath: corpusPath)
        let corpus: ByteCorpus
        do { corpus = try ByteCorpus(contentsOf: corpusURL) }
        catch { fputs("error reading corpus: \(error)\n", stderr); exit(1) }

        let B = batchSize ?? defaultBatch(cfg)
        print("""

        TinyGPT — LoRA fine-tune
        ------------------------
        base:           \(basePath)
        corpus:         \(corpusPath) (\(formatBytes(corpus.bytes.count)))
        config:         \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength)
        LoRA:           rank=\(rank) alpha=\(alpha) targets=\(targetSuffixes.joined(separator: ","))
        trainable:      \(formatNum(nTrainable))  /  total \(formatNum(nTotal))  (\(String(format: "%.2f%%", 100 * Float(nTrainable) / Float(nTotal))))
        steps:          \(steps)
        batch / lr:     \(B) / \(lr)
        device:         \(Device.defaultDevice())

        """)
        fflush(stdout)

        // One step function works for either model variant — the
        // AnyModel wrapper hides the underlying type and dispatches.
        let stepFn = makeStepFn(load.model, lr: lr)
        let t0 = Date()
        var lastLoss: Float = 0
        for step in 0..<steps {
            let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: cfg.contextLength)
            lastLoss = stepFn(x, y)
            if step == 0 || (step + 1) % 25 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.3f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
            if (step + 1) % sampleEvery == 0 || step == steps - 1 {
                fputs("    [step \(step + 1)] (sample skipped during HF fine-tune; use `tinygpt sample` after)\n", stderr)
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(String(format: "\ndone — %d steps in %.1fs (%.1f step/s) · final loss %.3f",
                     steps, elapsed, Double(steps) / elapsed, lastLoss))

        // Save the adapter (small A/B matrices only).
        do {
            try load.model.saveLora(baseConfig: cfg, loraConfig: loraCfg,
                                     finalLoss: lastLoss,
                                     to: URL(fileURLWithPath: outPath))
            let attrs = try FileManager.default.attributesOfItem(atPath: outPath)
            let sz = attrs[.size] as? Int ?? 0
            print("✓ wrote \(outPath)  (\(formatBytes(sz)))")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
    }

    /// Build a per-step train function bound to the right model class.
    /// We dispatch once here (at trainer-build time) rather than per-step.
    private static func makeStepFn(_ model: AnyModel, lr: Float) -> (MLXArray, MLXArray) -> Float {
        switch model {
        case .fromScratch(let m):
            let trainer = Trainer(model: m, learningRate: lr, weightDecay: 0.0, compileStep: false)
            return { x, y in trainer.step(inputs: x, targets: y) }
        case .huggingFace(let m):
            let trainer = TrainerHF(model: m, learningRate: lr, weightDecay: 0.0, compileStep: false)
            return { x, y in trainer.step(inputs: x, targets: y) }
        }
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 512 { return 4 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }
    private static func formatNum(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }
    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt finetune <base.tinygpt> [options]

        --corpus path.txt        UTF-8 text to fine-tune on (required)
        --out path.lora          Where to save the adapter (required)
        --rank N                 LoRA rank (default 4; try 8 for more capacity)
        --alpha F                LoRA scale (default 8.0; usually 2× rank)
        --steps N                Training steps (default 200)
        --lr F                   Learning rate (default 1e-3; higher than full-finetune)
        --targets q,v[,k,o,...]  Which Linear modules to wrap (default: q_proj,v_proj)
        --batch N                Batch size (default by preset)
        --sample-every N         Print sample every N steps (default 100)
        """)
        exit(2)
    }
}
