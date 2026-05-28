import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt sample` — load a browser-trained `.tinygpt` file and generate
/// text. The cross-path interop demo: the model trained in the browser,
/// run here on Metal at native speeds.
enum Sample {
    static func run(args: [String]) {
        var path: String?
        var prompt = "ROMEO:"
        var maxTokens = 200
        var temperature: Float = 0.8
        var useKVCache = true
        var loraPaths: [String] = []
        var loraWeights: [Float] = []
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--prompt":
                guard i + 1 < args.count else { exitUsage() }
                prompt = args[i + 1]; i += 2
            case "--tokens":
                guard i + 1 < args.count else { exitUsage() }
                maxTokens = Int(args[i + 1]) ?? maxTokens; i += 2
            case "--temperature", "--temp":
                guard i + 1 < args.count else { exitUsage() }
                temperature = Float(args[i + 1]) ?? temperature; i += 2
            case "--no-cache":
                useKVCache = false; i += 1
            case "--cache":
                useKVCache = true; i += 1
            case "--lora":
                guard i + 1 < args.count else { exitUsage() }
                loraPaths.append(args[i + 1]); i += 2
            case "--lora-weight":
                // Per-adapter mix weight when composing multiple LoRAs.
                // Supply once per --lora, same order. Defaults to 1.0 each.
                guard i + 1 < args.count else { exitUsage() }
                loraWeights.append(Float(args[i + 1]) ?? 1.0); i += 2
            case "-h", "--help":
                exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                path = args[i]; i += 1
            }
        }
        guard let path = path else {
            fputs("sample: missing <path> to .tinygpt file\n", stderr)
            exitUsage()
        }
        let url = URL(fileURLWithPath: path)

        // Unified loader — accepts .tinygpt files or HF model dirs.
        print("loading \(url.lastPathComponent)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(path) }
        catch { fputs("error loading: \(error)\n", stderr); exit(1) }
        let cfg = load.config
        let model = load.model

        // Apply one OR MORE LoRA adapters on top. Adapters carry their
        // base architecture in the header so a from-scratch adapter
        // can't accidentally load on an HF base, and vice versa.
        if !loraPaths.isEmpty {
            do {
                let adapters = try loraPaths.map { try LoraAdapterReader.read(URL(fileURLWithPath: $0)) }
                while loraWeights.count < adapters.count { loraWeights.append(1.0) }
                if adapters.count == 1 {
                    try model.applyLora(adapters[0])
                    print("loaded LoRA: rank=\(adapters[0].header.rank) targets=\(adapters[0].header.targetSuffixes.joined(separator: ","))")
                } else {
                    // Stacked composition currently wired for the from-scratch
                    // path; HF stacking is the next step.
                    if case .fromScratch(let m) = model {
                        try LoraStackInjection.apply(adapters, weights: loraWeights, to: m)
                        let blend = zip(loraPaths, loraWeights).map { "\($0.0.split(separator: "/").last ?? "") @ \($0.1)" }.joined(separator: " + ")
                        print("composed \(adapters.count) LoRAs: \(blend)")
                    } else {
                        fputs("multi-LoRA composition isn't wired for HF models yet — apply one adapter at a time\n", stderr)
                        exit(1)
                    }
                }
            } catch {
                fputs("error loading LoRA adapter(s): \(error)\n", stderr)
                exit(1)
            }
        }
        print("ready — \(formatLargeInt(model.numParameters())) params on \(Device.defaultDevice())")
        print("")

        // Encode the prompt as bytes (byte-level tokenizer, matches the browser).
        let promptBytes = [UInt8](prompt.utf8)
        let promptIds = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])

        // Print the prompt first, then stream generated tokens.
        print(prompt, terminator: "")
        fflush(stdout)

        let t0 = Date()
        // KV cache currently only implemented for TinyGPTModel (from-scratch).
        // HF path falls back to the uncached forward — slower per token but
        // correct. Adding KV cache to TinyGPTModelHF is a follow-up.
        let canUseCache: Bool
        var fromScratchModel: TinyGPTModel? = nil
        switch model {
        case .fromScratch(let m): canUseCache = true; fromScratchModel = m
        case .huggingFace: canUseCache = false
        }
        let useActualCache = useKVCache && canUseCache
        let cache = useActualCache ? KVCache(nLayers: cfg.nLayers) : nil

        if useActualCache, let cache, let m = fromScratchModel {
            let prefillLogits = m.forwardCached(promptIds, cache: cache)
            var lastLogits = prefillLogits[0..., prefillLogits.shape[1] - 1, 0...]
            for _ in 0..<maxTokens {
                let nextId: MLXArray
                if temperature <= 0 {
                    nextId = argMax(lastLogits, axis: -1).reshaped([1, 1])
                } else {
                    let scaled = lastLogits / MLXArray(temperature)
                    nextId = MLXRandomCategorical(scaled).reshaped([1, 1])
                }
                eval(nextId)
                let id = Int(nextId.item(Int32.self))
                if let scalar = UnicodeScalar(id) {
                    print(String(scalar), terminator: "")
                    fflush(stdout)
                }
                if cache.currentLength >= cfg.contextLength { break }
                let logits = m.forwardCached(nextId.asType(promptIds.dtype), cache: cache)
                lastLogits = logits[0..., 0, 0...]
            }
        } else {
            // Uncached forward — works on either model variant via AnyModel.
            var idx = promptIds
            for _ in 0..<maxTokens {
                let T = idx.shape.last!
                let lo = max(0, T - cfg.contextLength)
                let cond = idx[0..., lo..<T]
                let logits = model(cond)
                let last = logits[0..., logits.shape[1] - 1, 0...]
                let nextId: MLXArray
                if temperature <= 0 {
                    nextId = argMax(last, axis: -1).reshaped([1, 1])
                } else {
                    let scaled = last / MLXArray(temperature)
                    nextId = MLXRandomCategorical(scaled).reshaped([1, 1])
                }
                eval(nextId)
                let id = Int(nextId.item(Int32.self))
                if let scalar = UnicodeScalar(id) {
                    print(String(scalar), terminator: "")
                    fflush(stdout)
                }
                idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        let tokensPerSec = Double(maxTokens) / elapsed
        print("\n")
        let cacheLabel = useActualCache ? "KV-cached" : "uncached"
        print("(\(maxTokens) tokens in \(String(format: "%.2f", elapsed))s — \(String(format: "%.0f", tokensPerSec)) tok/s · \(cacheLabel))")
    }

    private static func MLXRandomCategorical(_ logits: MLXArray) -> MLXArray {
        // Sample one id per leading row from the unnormalized logits.
        return MLXRandom.categorical(logits)
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt sample <path.tinygpt> [options]

        --prompt "..."        Starting text (default: "ROMEO:")
        --tokens N            Max new tokens (default: 200)
        --temperature F       Sampling temperature (default: 0.8; 0 = greedy)
        """)
        exit(2)
    }
}
