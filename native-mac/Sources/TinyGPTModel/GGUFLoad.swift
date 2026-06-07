import Foundation
import MLX

/// Materialize a GGUF file into an HF-compatible directory that the existing
/// `HFModelLoader` can consume. This keeps `.gguf` support as a loader shim:
/// GGUFReader owns parsing/dequant, SafetensorsWriter owns the output format,
/// and the mature HF path owns model construction.
public enum GGUFHFMaterializer {
    public static func materializeIfNeeded(gguf url: URL) throws -> URL {
        let cacheDir = try cacheDirectory(for: url)
        let config = cacheDir.appendingPathComponent("config.json")
        let weights = cacheDir.appendingPathComponent("model.safetensors")
        let tokenizer = cacheDir.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: config.path),
           FileManager.default.fileExists(atPath: weights.path),
           FileManager.default.fileExists(atPath: tokenizer.path) {
            return cacheDir
        }

        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let parsed = try GGUFReader.parse(url: url)
        try writeTokenizer(parsed: parsed, to: cacheDir)
        try writeConfig(parsed: parsed, to: cacheDir)
        try writeManifest(parsed: parsed, source: url, to: cacheDir)
        try writeSafetensors(parsed: parsed, to: cacheDir)
        return cacheDir
    }

    private static func cacheDirectory(for url: URL) throws -> URL {
        let fm = FileManager.default
        let base = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/tinygpt/gguf-hf", isDirectory: true)
        let attrs = try? fm.attributesOfItem(atPath: url.path)
        let size = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
        let mtime = ((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0) * 1000
        let raw = "\(url.lastPathComponent)-\(size)-\(Int(mtime))"
        let safe = raw.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) || scalar == "." || scalar == "-" || scalar == "_"
            ? Character(scalar)
            : "-"
        }
        return base.appendingPathComponent(String(safe), isDirectory: true)
    }

    private static func writeTokenizer(parsed: GGUFReader.ParsedFile, to dir: URL) throws {
        let tokens = (parsed.metadata["tokenizer.ggml.tokens"] as? [Any])?
            .compactMap { $0 as? String } ?? []
        let merges = (parsed.metadata["tokenizer.ggml.merges"] as? [Any])?
            .compactMap { $0 as? String } ?? []
        guard !tokens.isEmpty else {
            throw NSError(domain: "tinygpt.gguf", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "GGUF has no tokenizer.ggml.tokens"])
        }
        let bosId = intOrNil(parsed.metadata["tokenizer.ggml.bos_token_id"])
        let eosId = intOrNil(parsed.metadata["tokenizer.ggml.eos_token_id"])
        let unkId = intOrNil(parsed.metadata["tokenizer.ggml.unknown_token_id"])
        let padId = intOrNil(parsed.metadata["tokenizer.ggml.padding_token_id"])
        let tokenizerModel = (parsed.metadata["tokenizer.ggml.model"] as? String) ?? "llama"

        var vocab: [String: Int] = [:]
        for (idx, token) in tokens.enumerated() { vocab[token] = idx }

        var added: [[String: Any]] = []
        func addSpecial(_ id: Int?) {
            guard let id, id >= 0, id < tokens.count else { return }
            added.append([
                "id": id,
                "content": tokens[id],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            ])
        }
        addSpecial(bosId)
        addSpecial(eosId)
        addSpecial(unkId)
        addSpecial(padId)

        let tokenizerJSON: [String: Any] = [
            "version": "1.0",
            "added_tokens": added,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true,
            ],
            "post_processor": NSNull(),
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true,
            ],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": unkId.flatMap { tokens[$0] } as Any? ?? NSNull(),
                "continuing_subword_prefix": NSNull(),
                "end_of_word_suffix": NSNull(),
                "fuse_unk": false,
                "byte_fallback": tokenizerModel == "llama",
                "vocab": vocab,
                "merges": merges,
            ],
        ]
        try writeJSON(tokenizerJSON, to: dir.appendingPathComponent("tokenizer.json"))

        let tokConfig: [String: Any] = [
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": intOrNil(parsed.metadata["llama.context_length"]) ?? 4096,
            "bos_token": bosId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "eos_token": eosId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "unk_token": unkId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "pad_token": padId.flatMap { tokens[$0] } as Any? ?? NSNull(),
            "add_bos_token": true,
            "add_eos_token": false,
        ]
        try writeJSON(tokConfig, to: dir.appendingPathComponent("tokenizer_config.json"))
    }

    private static func writeConfig(parsed: GGUFReader.ParsedFile, to dir: URL) throws {
        let arch = (parsed.metadata["general.architecture"] as? String) ?? "llama"
        let prefix = arch + "."
        func intMeta(_ key: String) -> Int? { intOrNil(parsed.metadata[prefix + key]) }
        let tokens = (parsed.metadata["tokenizer.ggml.tokens"] as? [Any]) ?? []
        let config: [String: Any] = [
            "architectures": ["LlamaForCausalLM"],
            "model_type": arch,
            "hidden_size": intMeta("embedding_length") as Any? ?? NSNull(),
            "intermediate_size": intMeta("feed_forward_length") as Any? ?? NSNull(),
            "num_hidden_layers": intMeta("block_count") as Any? ?? NSNull(),
            "num_attention_heads": intMeta("attention.head_count") as Any? ?? NSNull(),
            "num_key_value_heads": intMeta("attention.head_count_kv")
                ?? intMeta("attention.head_count") as Any? ?? NSNull(),
            "head_dim": intMeta("attention.key_length") as Any? ?? NSNull(),
            "max_position_embeddings": intMeta("context_length") as Any? ?? NSNull(),
            "vocab_size": tokens.count,
            "hidden_act": "silu",
            "rms_norm_eps": floatOrNil(parsed.metadata[prefix + "attention.layer_norm_rms_epsilon"]) ?? 1e-5,
            "rope_theta": floatOrNil(parsed.metadata[prefix + "rope.freq_base"]) ?? 10000.0,
            "tie_word_embeddings": false,
        ]
        try writeJSON(config, to: dir.appendingPathComponent("config.json"))
    }

    private static func writeManifest(parsed: GGUFReader.ParsedFile, source: URL, to dir: URL) throws {
        let manifest: [String: Any] = [
            "source_gguf": source.absoluteString,
            "tensor_data_base": parsed.tensorDataBase,
            "tensors": parsed.tensors.map { tensor in
                [
                    "name": tensor.name,
                    "shape": tensor.shape,
                    "type": Int(tensor.type),
                    "offset": Int(tensor.offset),
                ] as [String: Any]
            },
        ]
        try writeJSON(manifest, to: dir.appendingPathComponent("gguf_manifest.json"))
    }

    private static func writeSafetensors(parsed: GGUFReader.ParsedFile, to dir: URL) throws {
        var entries: [SafetensorsWriter.Entry] = []
        entries.reserveCapacity(parsed.tensors.count)
        for tensor in parsed.tensors {
            let arr = try GGUFReader.loadTensor(tensor, from: parsed)
            MLX.eval(arr)
            entries.append(SafetensorsWriter.Entry(
                name: mapGGUFToHF(tensor.name),
                data: arr.asArray(Float.self),
                shape: tensor.shape
            ))
        }
        try SafetensorsWriter.write(entries: entries,
                                    to: dir.appendingPathComponent("model.safetensors"))
    }

    private static func mapGGUFToHF(_ name: String) -> String {
        if name == "token_embd.weight" { return "model.embed_tokens.weight" }
        if name == "output_norm.weight" { return "model.norm.weight" }
        if name == "output.weight" { return "lm_head.weight" }
        guard name.hasPrefix("blk.") else { return name }
        let stripped = name.dropFirst(4)
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
        default:                   suffix = rest
        }
        return "model.layers.\(layerIdx).\(suffix)"
    }

    private static func intOrNil(_ v: Any?) -> Int? {
        if let v = v as? UInt32 { return Int(v) }
        if let v = v as? Int32 { return Int(v) }
        if let v = v as? UInt64 { return Int(v) }
        if let v = v as? Int64 { return Int(v) }
        if let v = v as? Int { return v }
        return nil
    }

    private static func floatOrNil(_ v: Any?) -> Float? {
        if let v = v as? Float { return v }
        if let v = v as? Double { return Float(v) }
        return nil
    }

    private static func writeJSON(_ obj: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: obj,
                                              options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }
}
