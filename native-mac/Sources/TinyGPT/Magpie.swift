import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt magpie` — bootstrap synthetic SFT data from any chat-format
/// base model (Xu et al., 2024).
///
/// Trick: a base model trained on chat-template tokens will, when
/// prompted with JUST the "start of user turn" marker, autocomplete it
/// into a plausible user query. We then re-prompt with the same marker
/// + the captured query + the "start of assistant turn" marker, sample
/// the assistant response, and record both as a single `(instruction,
/// response)` pair. Repeat N times, get a synthetic SFT dataset.
///
/// The base model needs ChatML-aware tokenization (BPE tokenizer with
/// `<|im_start|>` / `<|im_end|>` special tokens). Byte-level models
/// don't have those tokens — the script falls back to a simpler
/// "USER:" / "ASSISTANT:" plain-text template in that case, which
/// works on any model but yields less varied data.
///
/// USAGE
///   tinygpt magpie <model> --count 500 --out sft-data.jsonl
///   tinygpt magpie <model> --count 500 --template chatml \
///       --max-user 64 --max-assistant 256 --temperature 0.9 \
///       --out sft.jsonl
enum Magpie {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var outPath: String? = nil
        var count = 100
        var templateName = "chatml"
        var maxUserTokens = 64
        var maxAssistantTokens = 256
        var temperature: Float = 0.9

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":              outPath = args[i+1]; i += 2
            case "--count":            count = Int(args[i+1]) ?? count; i += 2
            case "--template":         templateName = args[i+1]; i += 2
            case "--max-user":         maxUserTokens = Int(args[i+1]) ?? maxUserTokens; i += 2
            case "--max-assistant":    maxAssistantTokens = Int(args[i+1]) ?? maxAssistantTokens; i += 2
            case "--temperature":      temperature = Float(args[i+1]) ?? temperature; i += 2
            case "-h", "--help":       exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path> required\n", stderr); exitUsage() }

        // Load the model + its tokenizer (or fall back to byte level).
        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        let cfg = load.config
        let model = load.model
        let tokenizer: HFTokenizer?
        if let td = load.hfTokenizerDir {
            tokenizer = try? HFTokenizer.loadBlocking(from: td)
        } else {
            tokenizer = nil
        }

        // Resolve markers. ChatML markers only make sense when the
        // tokenizer encodes them as discrete special tokens; on
        // byte-level fall back to plain prefixes that any model can
        // continue.
        let usingChatML = templateName == "chatml" && tokenizer != nil
        let userOpen: String
        let userClose: String
        let assistantOpen: String
        let assistantClose: String
        if usingChatML {
            userOpen      = "<|im_start|>user\n"
            userClose     = "<|im_end|>\n"
            assistantOpen = "<|im_start|>assistant\n"
            assistantClose = "<|im_end|>"
        } else {
            userOpen      = "USER: "
            userClose     = "\nASSISTANT: "
            assistantOpen = ""
            assistantClose = "\n\nUSER: "  // stop marker — used to terminate sampling
        }

        print("""

        TinyGPT — Magpie (synthetic SFT data)
        -------------------------------------
        model:          \(modelPath)
        count:          \(count) pairs
        template:       \(usingChatML ? "ChatML (BPE)" : "plain USER:/ASSISTANT: (any tokeniser)")
        max user:       \(maxUserTokens) tokens
        max assistant:  \(maxAssistantTokens) tokens
        temperature:    \(temperature)
        output:         \(outPath)

        """)

        guard let outStream = OutputStream(toFileAtPath: outPath, append: false) else {
            fputs("could not open \(outPath) for writing\n", stderr); exit(1)
        }
        outStream.open()
        defer { outStream.close() }

        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        var emitted = 0
        for _ in 0..<count {
            if TrainSupport.stopRequested.isSet { break }
            // 1. Sample a user query — prompt with the open marker, stop
            //    at the close marker (or maxUserTokens).
            let userText = sampleSegment(
                prompt: userOpen, stopAt: userClose, maxTokens: maxUserTokens,
                temperature: temperature, model: model, tokenizer: tokenizer, cfg: cfg
            )
            if userText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }

            // 2. Sample an assistant response — re-prompt with user
            //    query + assistant open marker.
            let assistantPrompt = userOpen + userText + userClose + assistantOpen
            let assistantText = sampleSegment(
                prompt: assistantPrompt, stopAt: assistantClose, maxTokens: maxAssistantTokens,
                temperature: temperature, model: model, tokenizer: tokenizer, cfg: cfg
            )
            if assistantText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }

            // 3. Emit JSONL line.
            let record: [String: String] = [
                "instruction": userText.trimmingCharacters(in: .whitespacesAndNewlines),
                "response": assistantText.trimmingCharacters(in: .whitespacesAndNewlines),
            ]
            let line = jsonLine(record) + "\n"
            if let data = line.data(using: .utf8) {
                _ = data.withUnsafeBytes { ptr in
                    outStream.write(ptr.baseAddress!.assumingMemoryBound(to: UInt8.self),
                                     maxLength: data.count)
                }
            }
            emitted += 1
            if emitted == 1 || emitted % 10 == 0 || emitted == count {
                fputs("  \(emitted) / \(count) pairs\n", stderr)
            }
        }
        print("\n✓ wrote \(emitted) pairs to \(outPath)")
    }

    /// Sample text from `model` starting from `prompt`, stopping at the
    /// first occurrence of `stopAt` in the decoded output (or after
    /// maxTokens). Returns just the GENERATED text (prompt stripped).
    private static func sampleSegment(
        prompt: String, stopAt: String, maxTokens: Int,
        temperature: Float, model: AnyModel, tokenizer: HFTokenizer?, cfg: ModelConfig
    ) -> String {
        // Encode prompt.
        let promptIds: [Int32]
        if let tok = tokenizer {
            guard let ids = try? tok.encode(prompt) else { return "" }
            promptIds = ids.map { Int32($0) }
        } else {
            promptIds = [UInt8](prompt.utf8).map { Int32($0) }
        }
        var idx = MLXArray(promptIds, [1, promptIds.count])
        var generatedTail: [Int32] = []

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
                nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
            }
            eval(nextId)
            let id = nextId.item(Int32.self)
            generatedTail.append(id)
            idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)

            // Stop condition — decode the running tail and check for
            // the stop marker. Cheap for short tails (typical case).
            let decoded: String
            if let tok = tokenizer {
                decoded = tok.decode(generatedTail.map { Int($0) })
            } else {
                let bytes = generatedTail.map { UInt8($0 & 0xff) }
                decoded = String(bytes: bytes, encoding: .utf8) ?? ""
            }
            if decoded.contains(stopAt) {
                // Trim everything from the stop marker onward.
                if let r = decoded.range(of: stopAt) {
                    return String(decoded[..<r.lowerBound])
                }
            }
        }
        // Reached maxTokens without seeing the stop marker — return what
        // we have (caller decides whether to keep or drop).
        if let tok = tokenizer {
            return tok.decode(generatedTail.map { Int($0) })
        }
        let bytes = generatedTail.map { UInt8($0 & 0xff) }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    /// Minimal JSON-line encoder for the `{instruction, response}`
    /// shape. JSONSerialization gives sorted keys + UTF-8; that's all
    /// we need for a JSONL writer.
    private static func jsonLine(_ obj: [String: String]) -> String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: obj, options: [.sortedKeys]
        ), let s = String(data: data, encoding: .utf8) else { return "{}" }
        return s
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt magpie <model> [options]

        --out <path>             Where to write the JSONL (required)
        --count N                Number of (instruction, response) pairs (default 100)
        --template chatml|plain  Chat template (default chatml — requires BPE tokenizer)
        --max-user N             Max user-query tokens per pair (default 64)
        --max-assistant N        Max assistant-response tokens (default 256)
        --temperature F          Sampling temperature (default 0.9 — higher than usual
                                   for diversity)

        Works best on chat-tuned models (Llama-Instruct, SmolLM2-Instruct, etc.).
        Base models often produce off-topic completions; the resulting
        dataset is then filtered by the user via simple rules
        ("length > 20" / "no repeating loops") before SFT.
        """)
        exit(code)
    }
}
