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

        // Read header to determine model config.
        let file: TinyGPTFile
        do {
            file = try TinyGPTFileReader.read(url)
        } catch {
            fputs("error reading \(path): \(error)\n", stderr)
            exit(1)
        }
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )

        print("loading \(url.lastPathComponent) (\(cfg.nLayers)L, d=\(cfg.dModel), ctx=\(cfg.contextLength))…")
        let model = TinyGPTModel(cfg)
        do {
            try TinyGPTWeightLoader.load(file, into: model)
        } catch {
            fputs("error loading weights: \(error)\n", stderr)
            exit(1)
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
        // Generate one token at a time so we can stream output. (Calling
        // model.generate in one shot would buffer until completion.)
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
            // 0-255 valid byte; lone bytes inside multibyte UTF-8 sequences
            // will print as the replacement char. Acceptable for streaming
            // demo; collect-then-decode would block until end.
            if let scalar = UnicodeScalar(id) {
                print(String(scalar), terminator: "")
                fflush(stdout)
            }
            idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
        }
        let elapsed = -t0.timeIntervalSinceNow
        let tokensPerSec = Double(maxTokens) / elapsed
        print("\n")
        print("(\(maxTokens) tokens in \(String(format: "%.2f", elapsed))s — \(String(format: "%.0f", tokensPerSec)) tok/s)")
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
