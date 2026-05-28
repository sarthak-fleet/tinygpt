import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt hf-load <dir> [--sample] [--prompt "..."]` — instantiate a
/// TinyGPTModelHF from a downloaded HuggingFace model directory, load
/// the safetensors weights, optionally generate a sample to verify
/// everything wires up.
///
/// USAGE
///   huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ~/Models/llama-3.2-1b
///   tinygpt hf-load ~/Models/llama-3.2-1b --sample --prompt "The capital of France is"
///
/// The dir must contain:
///   config.json                                — architecture description
///   tokenizer.json (+ tokenizer_config.json)   — BPE / SentencePiece vocab
///   model.safetensors (or sharded variants)    — weights
enum HFLoad {
    static func run(args: [String]) {
        var dirPath: String?
        var doSample = false
        var prompt = "The capital of France is"
        var maxTokens = 60
        var temperature: Float = 0.7
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--sample":      doSample = true; i += 1
            case "--prompt":      prompt = args[i+1]; i += 2
            case "--tokens":      maxTokens = Int(args[i+1]) ?? maxTokens; i += 2
            case "--temperature": temperature = Float(args[i+1]) ?? temperature; i += 2
            case "-h", "--help":  exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                dirPath = args[i]; i += 1
            }
        }
        guard let dirPath = dirPath else {
            fputs("missing <dir>\n", stderr); exitUsage()
        }
        let dir = URL(fileURLWithPath: dirPath)

        // Load the model
        print("loading HF model from \(dir.path)…")
        let result: HFModelLoader.LoadResult
        do { result = try HFModelLoader.load(from: dir) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        let model = result.model
        let cfg = result.config

        print("""

        ✓ loaded \(result.hfConfig.architectures.first ?? "unknown")
          params:   \(formatLargeInt(model.numParameters()))
          layers:   \(cfg.nLayers) · d=\(cfg.dModel) · ctx=\(cfg.contextLength)
          heads:    \(cfg.nHeads) Q / \(cfg.nKvHeads) KV (GQA: \(cfg.nHeads != cfg.nKvHeads))
          rope:     base=\(cfg.ropeBase) (extrapolation-friendly)
          vocab:    \(cfg.vocabSize) (needs BPE tokenizer for real text)
          device:   \(Device.defaultDevice())
        """)

        if doSample {
            print("\nattempting sample with byte-level tokenizer (placeholder — text will look like noise for")
            print("any HF model since they use BPE; this is the integration smoke test):\n")
            // Byte-level fallback because hf-load doesn't yet wire the
            // BPE tokenizer through. When we finish the tokenizer attach,
            // this changes to `HFTokenizer.load(from: dir)`.
            let bytes = [UInt8](prompt.utf8)
            var idx = MLXArray(bytes.map { Int32($0) }, [1, bytes.count])

            print(prompt, terminator: "")
            fflush(stdout)
            for _ in 0..<maxTokens {
                let T = idx.shape.last!
                let lo = max(0, T - cfg.contextLength)
                let cond = idx[0..., lo..<T]
                let logits = model(cond)
                let last = logits[0..., logits.shape[1] - 1, 0...]
                let next: MLXArray
                if temperature <= 0 {
                    next = argMax(last, axis: -1).reshaped([1, 1])
                } else {
                    next = MLXRandom.categorical(last / MLXArray(temperature))
                        .reshaped([1, 1])
                }
                eval(next)
                let id = Int(next.item(Int32.self))
                // Byte-level decode is wrong for BPE tokens; output will
                // be Unicode noise. Genuine decode needs HFTokenizer.
                if id < 256, let scalar = UnicodeScalar(id) {
                    print(String(scalar), terminator: "")
                } else {
                    print("·", terminator: "")
                }
                fflush(stdout)
                idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
            }
            print()
        }

        print("\nNext steps once tokenizer is wired through:")
        print("  tinygpt finetune <hf-dir> --corpus my.txt --out my.lora")
        print("  tinygpt sample <hf-dir> --lora my.lora --prompt \"...\"")
    }

    private static func formatLargeInt(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.1f B", Double(n) / 1_000_000_000) }
        if n >= 1_000_000 { return String(format: "%.1f M", Double(n) / 1_000_000) }
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt hf-load <hf-model-dir> [options]

        --sample                Run a quick sample after loading (smoke test)
        --prompt "..."          Sampling prompt (default: "The capital of France is")
        --tokens N              Max new tokens (default 60)
        --temperature F         Sampling temperature (default 0.7)
        """)
        exit(2)
    }
}
