import Foundation
import TinyGPTModel

/// `tinygpt gguf-extract` — extract the tokenizer + arch config from a
/// GGUF file into a directory that the HF loader can consume.
///
/// Produces (when run with `--out-dir <dir>`):
///   <dir>/tokenizer.json         — swift-transformers-compatible BPE
///                                   vocab + merges + special tokens
///   <dir>/tokenizer_config.json   — auxiliary tokenizer metadata
///   <dir>/config.json             — HF-style model config (arch
///                                   hyperparams: layers, heads, dim,
///                                   vocab, etc.)
///   <dir>/gguf_manifest.json      — tensor inventory pointing back at
///                                   the source GGUF + the dequant
///                                   recipe (which `tinygpt gguf-load`
///                                   already validates)
///
/// Companion to `tinygpt gguf-load` (the validator) and `tinygpt
/// gguf-inspect` (the metadata browser). After this, the missing
/// piece for end-to-end runnable loading is **weight materialization**
/// (dequant the K-quants → write as safetensors so the existing HF
/// loader picks them up). That's the natural follow-up; this CLI
/// produces the tokenizer + config that the loader needs to even
/// instantiate the model class.
///
/// USAGE
///   tinygpt gguf-extract <input.gguf> --out-dir <dir>
enum GGUFExtract {
    static func run(args: [String]) {
        var inputPath: String? = nil
        var outDir: String? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out-dir":     outDir = args[i+1]; i += 2
            case "-h", "--help":  exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inputPath = args[i]; i += 1
            }
        }
        guard let inputPath = inputPath else { fputs("missing <input.gguf>\n", stderr); exitUsage() }
        guard let outDir = outDir else { fputs("--out-dir required\n", stderr); exitUsage() }

        let parsed: GGUFReader.ParsedFile
        do { parsed = try GGUFReader.parse(url: URL(fileURLWithPath: inputPath)) }
        catch { fputs("parse failed: \(error)\n", stderr); exit(1) }

        let outURL = URL(fileURLWithPath: outDir)
        try? FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)

        // ---- tokenizer.json ----
        // GGUF tokenizer metadata uses these keys:
        //   tokenizer.ggml.model              — "llama" | "gpt2" | …
        //   tokenizer.ggml.tokens             — [String]
        //   tokenizer.ggml.scores             — [Float]    (sentencepiece)
        //   tokenizer.ggml.token_type         — [Int]
        //   tokenizer.ggml.merges             — [String]    "a b"
        //   tokenizer.ggml.bos_token_id       — Int
        //   tokenizer.ggml.eos_token_id       — Int
        //   tokenizer.ggml.unknown_token_id   — Int (optional)
        //   tokenizer.ggml.padding_token_id   — Int (optional)
        let tokens = (parsed.metadata["tokenizer.ggml.tokens"] as? [Any])?
            .compactMap { $0 as? String } ?? []
        let merges = (parsed.metadata["tokenizer.ggml.merges"] as? [Any])?
            .compactMap { $0 as? String } ?? []
        let bosId = intOrNil(parsed.metadata["tokenizer.ggml.bos_token_id"])
        let eosId = intOrNil(parsed.metadata["tokenizer.ggml.eos_token_id"])
        let unkId = intOrNil(parsed.metadata["tokenizer.ggml.unknown_token_id"])
        let padId = intOrNil(parsed.metadata["tokenizer.ggml.padding_token_id"])
        let tokenizerModel = (parsed.metadata["tokenizer.ggml.model"] as? String) ?? "llama"

        guard !tokens.isEmpty else {
            fputs("GGUF has no tokenizer.ggml.tokens — can't extract a tokenizer\n", stderr)
            exit(1)
        }

        // Build vocab map for tokenizer.json:  {"<token>": id}
        var vocabMap: [String: Int] = [:]
        for (i, tok) in tokens.enumerated() { vocabMap[tok] = i }

        // Special tokens block. We include the ones whose IDs we know;
        // the tokenizer.json format puts them in "added_tokens."
        var addedTokens: [[String: Any]] = []
        func addSpecial(_ id: Int?, content: String?) {
            guard let id = id, id < tokens.count else { return }
            let c = content ?? tokens[id]
            addedTokens.append([
                "id": id, "content": c, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false,
                "special": true,
            ])
        }
        addSpecial(bosId, content: nil)
        addSpecial(eosId, content: nil)
        addSpecial(unkId, content: nil)
        addSpecial(padId, content: nil)

        // tokenizer.json schema. Pre-tokenizer is byte-level (Llama
        // family standard); model is BPE; merges as listed.
        let tokenizerJSON: [String: Any] = [
            "version": "1.0",
            "added_tokens": addedTokens,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true,
            ],
            "post_processor": NSNull(),
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": unkId.flatMap { tokens[$0] } as Any? ?? NSNull(),
                "continuing_subword_prefix": NSNull(),
                "end_of_word_suffix": NSNull(),
                "fuse_unk": false,
                "byte_fallback": (tokenizerModel == "llama"),
                "vocab": vocabMap,
                "merges": merges,
            ],
        ]
        try? writeJSON(tokenizerJSON, to: outURL.appendingPathComponent("tokenizer.json"))

        // ---- tokenizer_config.json ----
        let tokConfig: [String: Any] = [
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": (parsed.metadata["llama.context_length"] as? UInt32).map { Int($0) }
                                ?? (parsed.metadata["llama.context_length"] as? Int32).map { Int($0) }
                                ?? 4096,
            "bos_token": bosId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "eos_token": eosId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "unk_token": unkId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "pad_token": padId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "add_bos_token": true,
            "add_eos_token": false,
        ]
        try? writeJSON(tokConfig, to: outURL.appendingPathComponent("tokenizer_config.json"))

        // ---- config.json (HF-style model config) ----
        let arch = (parsed.metadata["general.architecture"] as? String) ?? "llama"
        let prefix = arch + "."
        func intMeta(_ key: String) -> Int? { intOrNil(parsed.metadata[prefix + key]) }
        let modelConfig: [String: Any] = [
            "architectures": ["LlamaForCausalLM"],
            "model_type": arch,
            "hidden_size": intMeta("embedding_length") as Any? ?? NSNull(),
            "intermediate_size": intMeta("feed_forward_length") as Any? ?? NSNull(),
            "num_hidden_layers": intMeta("block_count") as Any? ?? NSNull(),
            "num_attention_heads": intMeta("attention.head_count") as Any? ?? NSNull(),
            "num_key_value_heads": intMeta("attention.head_count_kv")
                                    ?? intMeta("attention.head_count") as Any? ?? NSNull(),
            "max_position_embeddings": intMeta("context_length") as Any? ?? NSNull(),
            "vocab_size": tokens.count,
            "hidden_act": "silu",
            "rms_norm_eps": (parsed.metadata[prefix + "attention.layer_norm_rms_epsilon"] as? Float)
                            ?? 1e-5,
            "rope_theta": (parsed.metadata[prefix + "rope.freq_base"] as? Float) ?? 10000.0,
            "tie_word_embeddings": false,
        ]
        try? writeJSON(modelConfig, to: outURL.appendingPathComponent("config.json"))

        // ---- gguf_manifest.json: weight-load instructions ----
        let manifest: [String: Any] = [
            "source_gguf": (URL(fileURLWithPath: inputPath).absoluteString),
            "tensor_data_base": parsed.tensorDataBase,
            "tensors": parsed.tensors.map { t -> [String: Any] in
                [
                    "name": t.name,
                    "shape": t.shape,
                    "type": Int(t.type),
                    "offset": Int(t.offset),
                ]
            },
        ]
        try? writeJSON(manifest, to: outURL.appendingPathComponent("gguf_manifest.json"))

        print("""

        TinyGPT — GGUF extract
        ----------------------
        source:           \(inputPath)
        out:              \(outDir)
        wrote:            tokenizer.json (\(tokens.count) tokens, \(merges.count) merges)
                          tokenizer_config.json
                          config.json
                          gguf_manifest.json (\(parsed.tensors.count) tensors)

        Tokenizer + config are now in HF format; can be consumed by
        swift-transformers / HFModelLoader.

        Next step for end-to-end runnable model: convert the dequant'd
        weights to safetensors so the HF loader picks them up. The
        gguf_manifest.json points at the source GGUF + offsets; a
        weight-materializer reads via GGUFReader.loadTensor, applies
        the standard llama tensor-name → HF param mapping (see
        `tinygpt gguf-load` for the canonical list), and writes
        model.safetensors. ~1 day focused.
        """)
    }

    private static func intOrNil(_ v: Any?) -> Int? {
        if let v = v as? UInt32 { return Int(v) }
        if let v = v as? Int32 { return Int(v) }
        if let v = v as? UInt64 { return Int(v) }
        if let v = v as? Int64 { return Int(v) }
        if let v = v as? Int { return v }
        return nil
    }

    private static func writeJSON(_ obj: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: obj,
                                                options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt gguf-extract <input.gguf> --out-dir <dir>

        Extract tokenizer + arch config from a GGUF file into a
        directory consumable by swift-transformers / HFModelLoader.
        Closes the tokenizer-extraction gap left by `gguf-load`.

        Outputs (in --out-dir):
          tokenizer.json          BPE vocab + merges + special tokens
          tokenizer_config.json   tokenizer auxiliaries
          config.json             HF-style model config
          gguf_manifest.json      tensor inventory back-pointer

        Weight materialization to safetensors is the natural
        follow-up; the manifest carries everything needed to do it.
        """)
        exit(code)
    }
}
