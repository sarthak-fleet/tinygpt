import Foundation
import MLX
import TinyGPTModel

/// `tinygpt gguf-load` — parse a GGUF model file, extract the
/// llama-arch metadata, map GGUF tensor names onto TinyGPT's HF op
/// tree, and report what would load.
///
/// First-cut deliverable: the validator pass. Reports:
///   - which model shape the GGUF declares (n_layers, d_model, etc.)
///   - which tensor names are present + their types
///   - which TinyGPT-HF parameters they map to
///   - which (if any) expected parameters are missing
///   - whether dequantising every tensor would actually work
///
/// What this does NOT do (yet): build a runnable model from the
/// dequantised weights. That's blocked on tokenizer extraction —
/// GGUF embeds the BPE vocab inline (`tokenizer.ggml.tokens` array)
/// while our tokenizer surface expects swift-transformers'
/// tokenizer.json layout. Tokenizer-from-GGUF is the natural
/// follow-up (~1 day), after which a real `--out` path can persist
/// the loaded model as `.tinygpt` for ordinary `sample` use.
///
/// USAGE
///   tinygpt gguf-load <path.gguf> [--verbose]
///
/// Llama-family GGUF naming convention (covered today):
///   token_embd.weight                 → tok embeddings
///   output_norm.weight                → final norm
///   output.weight                     → LM head
///   blk.{i}.attn_norm.weight          → block i pre-attn norm
///   blk.{i}.attn_q.weight             → q projection
///   blk.{i}.attn_k.weight             → k projection
///   blk.{i}.attn_v.weight             → v projection
///   blk.{i}.attn_output.weight        → o projection
///   blk.{i}.ffn_norm.weight           → block i pre-mlp norm
///   blk.{i}.ffn_gate.weight           → SwiGLU gate
///   blk.{i}.ffn_up.weight             → SwiGLU up
///   blk.{i}.ffn_down.weight           → SwiGLU down
enum GGUFLoad {
    static func run(args: [String]) {
        var path: String? = nil
        var verbose = false
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--verbose", "-v": verbose = true; i += 1
            case "-h", "--help":    exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                path = args[i]; i += 1
            }
        }
        guard let path = path else { fputs("missing <path.gguf>\n", stderr); exitUsage() }

        let parsed: GGUFReader.ParsedFile
        do { parsed = try GGUFReader.parse(url: URL(fileURLWithPath: path)) }
        catch { fputs("parse failed: \(error)\n", stderr); exit(1) }

        // Pull the llama-arch config out of metadata. GGUF's
        // convention puts these under a per-arch namespace —
        // `llama.embedding_length`, `llama.block_count`, etc.
        // Other archs (`gemma.`, `qwen2.`) follow the same pattern.
        let arch = (parsed.metadata["general.architecture"] as? String) ?? "llama"
        let prefix = arch + "."
        func meta(_ key: String) -> Any? { return parsed.metadata[prefix + key] }
        func intMeta(_ key: String) -> Int? {
            if let v = meta(key) as? UInt32 { return Int(v) }
            if let v = meta(key) as? Int32  { return Int(v) }
            if let v = meta(key) as? UInt64 { return Int(v) }
            if let v = meta(key) as? Int64  { return Int(v) }
            return nil
        }
        let dModel = intMeta("embedding_length")
        let dMlp   = intMeta("feed_forward_length")
        let nLayers = intMeta("block_count")
        let nHeads = intMeta("attention.head_count")
        let nKvHeads = intMeta("attention.head_count_kv") ?? nHeads
        let ctx = intMeta("context_length")
        let vocabFromArr = (parsed.metadata["tokenizer.ggml.tokens"] as? [Any])?.count
        let vocabFromMeta = intMeta("vocab_size") ?? vocabFromArr

        print("""

        TinyGPT — GGUF load report
        --------------------------
        path:           \(path)
        architecture:   \(arch)
        n_layers:       \(stringOr(nLayers))
        n_heads:        \(stringOr(nHeads))
        n_kv_heads:     \(stringOr(nKvHeads))
        d_model:        \(stringOr(dModel))
        d_mlp:          \(stringOr(dMlp))
        context:        \(stringOr(ctx))
        vocab_size:     \(stringOr(vocabFromMeta))
        tensor count:   \(parsed.tensors.count)
        """)

        guard let L = nLayers, let H = nHeads, let D = dModel, let M = dMlp else {
            fputs("\nrequired metadata missing — can't validate without n_layers / n_heads / d_model / d_mlp\n", stderr)
            exit(2)
        }
        _ = (nKvHeads, ctx, vocabFromMeta, H, M)  // referenced in the mapping below

        // Build the expected parameter inventory for a llama-family
        // model AT THIS SHAPE. For each, find the matching GGUF tensor
        // by canonical name. Report present / missing / shape-mismatch.
        struct Expectation {
            let ggufName: String
            let role: String
            let expectedShape: [Int]   // GGUF stores weights as [in, out] for matmul; we report what we'd expect
        }
        var expectations: [Expectation] = [
            Expectation(ggufName: "token_embd.weight",
                         role: "token embedding",
                         expectedShape: [vocabFromMeta ?? 0, D]),
            Expectation(ggufName: "output_norm.weight",
                         role: "final norm",
                         expectedShape: [D]),
            Expectation(ggufName: "output.weight",
                         role: "LM head",
                         expectedShape: [vocabFromMeta ?? 0, D]),
        ]
        for li in 0..<L {
            let p = "blk.\(li)"
            expectations += [
                Expectation(ggufName: "\(p).attn_norm.weight", role: "block \(li) attn norm", expectedShape: [D]),
                Expectation(ggufName: "\(p).attn_q.weight",     role: "block \(li) q proj",   expectedShape: [D, D]),
                Expectation(ggufName: "\(p).attn_k.weight",     role: "block \(li) k proj",   expectedShape: [D * nKvHeads! / H, D]),
                Expectation(ggufName: "\(p).attn_v.weight",     role: "block \(li) v proj",   expectedShape: [D * nKvHeads! / H, D]),
                Expectation(ggufName: "\(p).attn_output.weight", role: "block \(li) o proj",  expectedShape: [D, D]),
                Expectation(ggufName: "\(p).ffn_norm.weight",   role: "block \(li) ffn norm", expectedShape: [D]),
                Expectation(ggufName: "\(p).ffn_gate.weight",   role: "block \(li) ffn gate", expectedShape: [M, D]),
                Expectation(ggufName: "\(p).ffn_up.weight",     role: "block \(li) ffn up",   expectedShape: [M, D]),
                Expectation(ggufName: "\(p).ffn_down.weight",   role: "block \(li) ffn down", expectedShape: [D, M]),
            ]
        }

        // Index tensors by name for O(1) lookup.
        let byName = Dictionary(uniqueKeysWithValues: parsed.tensors.map { ($0.name, $0) })
        var missing: [Expectation] = []
        var shapeMismatches: [(Expectation, [Int])] = []
        var unrecognizedTypes: [(Expectation, UInt32)] = []
        var ok: [Expectation] = []
        let supportedTypes: Set<UInt32> = [0, 1, 2, 8, 12, 13, 14, 15]  // F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K
        for exp in expectations {
            guard let t = byName[exp.ggufName] else { missing.append(exp); continue }
            if !supportedTypes.contains(t.type) {
                unrecognizedTypes.append((exp, t.type)); continue
            }
            // Validate shape: GGUF stores as [in, out] (or [N] for vectors).
            // We compare against the expected shape; allow either ordering
            // for 2-D weight matrices since transposing happens at load.
            let shapeMatch = exp.expectedShape == t.shape ||
                             (exp.expectedShape.count == 2 && t.shape.count == 2 &&
                              exp.expectedShape == [t.shape[1], t.shape[0]])
            if !shapeMatch {
                shapeMismatches.append((exp, t.shape)); continue
            }
            ok.append(exp)
        }

        print("""

        weight-load validation
        ----------------------
          expected tensors:        \(expectations.count)
          present + shape-match:   \(ok.count)
          missing:                 \(missing.count)
          shape mismatch:          \(shapeMismatches.count)
          unsupported dtype:       \(unrecognizedTypes.count)
        """)

        if !missing.isEmpty {
            print("\n  MISSING (first 10):")
            for exp in missing.prefix(10) {
                print("    \(exp.ggufName)  expected shape \(exp.expectedShape)")
            }
            if missing.count > 10 { print("    … +\(missing.count - 10) more") }
        }
        if !shapeMismatches.isEmpty {
            print("\n  SHAPE MISMATCH (first 10):")
            for (exp, got) in shapeMismatches.prefix(10) {
                print("    \(exp.ggufName)  expected \(exp.expectedShape) got \(got)")
            }
        }
        if !unrecognizedTypes.isEmpty {
            print("\n  UNSUPPORTED DTYPE (first 10):")
            for (exp, t) in unrecognizedTypes.prefix(10) {
                print("    \(exp.ggufName)  ggml type \(t) (need Q4_K/Q5_K/Q6_K dequant or add the type)")
            }
        }

        // Listing all bytes — quantisation type breakdown.
        var typeCounts: [UInt32: Int] = [:]
        for t in parsed.tensors {
            typeCounts[t.type, default: 0] += 1
        }
        print("\n  tensor type histogram:")
        for (type, n) in typeCounts.sorted(by: { $0.key < $1.key }) {
            let label = typeLabel(type)
            print("    \(label.padding(toLength: 8, withPad: " ", startingAt: 0))  \(n)")
        }

        // Unrecognised metadata — useful when debugging a new arch.
        if verbose {
            print("\n  raw metadata keys (\(parsed.metadata.count)):")
            for k in parsed.metadata.keys.sorted() { print("    \(k)") }
        }

        // Final verdict.
        let allClean = missing.isEmpty && shapeMismatches.isEmpty && unrecognizedTypes.isEmpty
        if allClean {
            print("""

            ✅ all tensors present, shapes match, all dtypes supported.
            This GGUF would load into a TinyGPT-HF model at:
              ModelConfig(nLayers: \(L), nHeads: \(H), nKvHeads: \(nKvHeads!),
                          dModel: \(D), dMlp: \(M),
                          contextLength: \(ctx ?? 0),
                          vocabSize: \(vocabFromMeta ?? 0),
                          useRoPE: true, useRMSNorm: true, useSwiGLU: true)

            Still missing for an end-to-end runnable model:
              - tokenizer extraction from GGUF inline vocab
                (`tokenizer.ggml.tokens` + `tokenizer.ggml.merges`)
                → adapter to swift-transformers Tokenizer protocol.
            """)
        } else {
            print("""

            ⚠️  GGUF would NOT load cleanly into the current TinyGPT-HF
                model surface. See the missing / mismatch lists above.
            """)
            exit(1)
        }
    }

    private static func typeLabel(_ t: UInt32) -> String {
        switch GGUFReader.GGMLType(rawValue: t) {
        case .f32:  return "F32"
        case .f16:  return "F16"
        case .q4_0: return "Q4_0"
        case .q8_0: return "Q8_0"
        case .q4_K: return "Q4_K"
        case .q5_K: return "Q5_K"
        case .q6_K: return "Q6_K"
        case .q8_K: return "Q8_K"
        case nil:   return "T\(t)"
        }
    }

    private static func stringOr(_ v: Int?) -> String {
        if let v = v { return String(v) }
        return "—"
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt gguf-load <path.gguf> [--verbose]

        Parse a GGUF model file and validate that its tensor inventory +
        shapes would load into TinyGPT's HF model surface. Reports
        which tensors are present / missing / shape-mismatched / use
        unsupported quant types. Llama-family naming convention covered;
        other archs (qwen, gemma) typically follow the same shape.

        --verbose          dump all metadata keys (debugging new archs)
        """)
        exit(code)
    }
}
