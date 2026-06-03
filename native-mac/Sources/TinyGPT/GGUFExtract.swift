import Foundation
import MLX
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

        // ---- model.safetensors: dequant every tensor + write ----
        // Map GGUF names to HF Llama convention, dequantize via
        // GGUFReader.loadTensor (covers F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K),
        // and stream out as a single safetensors file. This is the
        // missing piece that turns the validator (gguf-load) into a
        // runnable model directory.
        var weightEntries: [SafetensorsWriter.Entry] = []
        var skipped: [String] = []
        for t in parsed.tensors {
            let hfName = mapGGUFToHF(t.name)
            do {
                let arr = try GGUFReader.loadTensor(t, from: parsed)
                MLX.eval(arr)
                let floats = arr.asArray(Float.self)
                weightEntries.append(SafetensorsWriter.Entry(
                    name: hfName, data: floats, shape: t.shape
                ))
            } catch {
                skipped.append("\(t.name): \(error)")
            }
        }
        let safetensorsURL = outURL.appendingPathComponent("model.safetensors")
        do {
            try SafetensorsWriter.write(entries: weightEntries, to: safetensorsURL)
        } catch {
            fputs("safetensors write failed: \(error)\n", stderr)
        }
        if !skipped.isEmpty {
            fputs("\nNOTE: \(skipped.count) tensor(s) skipped (unsupported dtype or read error):\n", stderr)
            for s in skipped.prefix(5) { fputs("  \(s)\n", stderr) }
            if skipped.count > 5 { fputs("  … +\(skipped.count - 5) more\n", stderr) }
        }
        var totalBytes = 0
        for e in weightEntries { totalBytes += e.data.count * 4 }

        print("""

        TinyGPT — GGUF extract
        ----------------------
        source:           \(inputPath)
        out:              \(outDir)
        wrote:            tokenizer.json       (\(tokens.count) tokens, \(merges.count) merges)
                          tokenizer_config.json
                          config.json
                          gguf_manifest.json   (\(parsed.tensors.count) tensors)
                          model.safetensors    (\(weightEntries.count) weights, \(formatBytes(totalBytes)))

        Output dir is a complete HuggingFace model bundle. Load via:
          import transformers
          model = transformers.AutoModelForCausalLM.from_pretrained(\"\(outDir)\")
          tok   = transformers.AutoTokenizer.from_pretrained(\"\(outDir)\")
        """)
    }

    /// Map GGUF tensor names → HuggingFace Llama-family conventions.
    /// Mirrors the dequant + weight-load conventions documented in
    /// `tinygpt gguf-load`'s validator.
    private static func mapGGUFToHF(_ name: String) -> String {
        // Top-level renames.
        if name == "token_embd.weight" { return "model.embed_tokens.weight" }
        if name == "output_norm.weight" { return "model.norm.weight" }
        if name == "output.weight" { return "lm_head.weight" }
        // Block-prefixed renames.  blk.N. → model.layers.N.
        guard name.hasPrefix("blk.") else { return name }
        let stripped = name.dropFirst(4) // remove "blk."
        // Find the layer index up to the next '.'
        guard let dot = stripped.firstIndex(of: ".") else { return name }
        let layerIdx = String(stripped[..<dot])
        let rest = String(stripped[stripped.index(after: dot)...])
        let suffix: String
        switch rest {
        case "attn_norm.weight":   suffix = "input_layernorm.weight"
        case "ffn_norm.weight":    suffix = "post_attention_layernorm.weight"
        case "attn_q.weight":      suffix = "self_attn.q_proj.weight"
        case "attn_k.weight":      suffix = "self_attn.k_proj.weight"
        case "attn_v.weight":      suffix = "self_attn.v_proj.weight"
        case "attn_output.weight": suffix = "self_attn.o_proj.weight"
        case "ffn_gate.weight":    suffix = "mlp.gate_proj.weight"
        case "ffn_up.weight":      suffix = "mlp.up_proj.weight"
        case "ffn_down.weight":    suffix = "mlp.down_proj.weight"
        default:                   suffix = rest  // pass-through for arch-specific extras
        }
        return "model.layers.\(layerIdx).\(suffix)"
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.2f GB", Double(n) / 1e9) }
        if n >= 1_000_000     { return String(format: "%.2f MB", Double(n) / 1e6) }
        if n >= 1_000         { return String(format: "%.1f KB", Double(n) / 1e3) }
        return "\(n) B"
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
