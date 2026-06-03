import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt to-safetensors` — exports a `.tinygpt` checkpoint as a
/// HuggingFace-compatible `model.safetensors` file.
///
/// Bridges two ecosystems:
///   • TinyGPT's native `.tinygpt` format → universal HF safetensors
///   • Lets you load TinyGPT-trained models in PyTorch / transformers
///   • Acts as the weight-loading hop for `tinygpt to-coreml` (the
///     generated Python script now does `safetensors.torch.load_file`)
///
/// Name remapping: `.tinygpt` parameters live under the TinyGPT-Swift
/// op tree (`blocks.0.attn.q_proj.weight`, etc.). We can either
/// preserve those names or translate to HF Llama-style conventions
/// (`model.layers.0.self_attn.q_proj.weight`). The default is HF-
/// compatible so the file drops into transformers' AutoModel surface.
///
/// USAGE
///   tinygpt to-safetensors <model.tinygpt> --out model.safetensors [--keep-names]
///
/// --keep-names   write tensors with the TinyGPT-native names instead
///                of remapping to HF Llama conventions
enum ToSafetensors {
    static func run(args: [String]) {
        var inputPath: String? = nil
        var outPath: String? = nil
        var keepNames = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":         outPath = args[i+1]; i += 2
            case "--keep-names":  keepNames = true; i += 1
            case "-h", "--help":  exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inputPath = args[i]; i += 1
            }
        }
        guard let inputPath = inputPath else { fputs("missing <model.tinygpt>\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }

        // Load the model so we get the canonical parameter tree
        // (the .tinygpt reader handles all the manifest + dtype
        // bookkeeping that ad-hoc parsing would have to reproduce).
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(inputPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("to-safetensors first-cut targets from-scratch models. " +
                  "HF-loaded models already use safetensors on disk.\n", stderr); exit(2)
        }

        // Collect every parameter (name, MLXArray) pair. The model's
        // `parameters().flattened()` returns the live tree — what's
        // actually in memory after load + any post-load mutations.
        var entries: [SafetensorsWriter.Entry] = []
        for (name, arr) in model.parameters().flattened() {
            MLX.eval(arr)
            let mapped = keepNames ? name : remapToHF(name)
            // safetensors uses HF's `[out, in]` Linear convention; our
            // in-memory weights match that (MLXNN.Linear stores
            // weight as [out, in]). No transpose needed.
            entries.append(SafetensorsWriter.Entry(
                name: mapped,
                data: arr.asArray(Float.self),
                shape: arr.shape
            ))
        }

        let url = URL(fileURLWithPath: outPath)
        do { try SafetensorsWriter.write(entries: entries, to: url) }
        catch { fputs("write failed: \(error)\n", stderr); exit(1) }

        var totalBytes = 0
        for e in entries { totalBytes += e.data.count * 4 }
        print("""

        TinyGPT — safetensors export
        ----------------------------
        source:           \(inputPath)
        tensors written:  \(entries.count)
        body size:        \(formatBytes(totalBytes))
        names:            \(keepNames ? "TinyGPT-native" : "HF Llama convention")
        out:              \(outPath)

        Loadable from Python via:
          from safetensors.torch import load_file
          weights = load_file(\"\(outPath)\")
          # weights is a {name: torch.Tensor} dict in fp32.
        """)
    }

    /// Remap a TinyGPT parameter name into HF Llama convention.
    /// Standard mapping:
    ///   tokenEmbedding.weight             → model.embed_tokens.weight
    ///   positionEmbedding.weight          → model.embed_positions.weight  (TinyGPT-specific; HF Llama uses RoPE-only — keep)
    ///   lnFinal.weight / .bias            → model.norm.weight / .bias
    ///   blocks.{i}.ln1.weight             → model.layers.{i}.input_layernorm.weight
    ///   blocks.{i}.ln2.weight             → model.layers.{i}.post_attention_layernorm.weight
    ///   blocks.{i}.attn.{q,k,v,o}_proj    → model.layers.{i}.self_attn.{q,k,v,o}_proj
    ///   blocks.{i}.mlp.fc_in              → model.layers.{i}.mlp.gate_proj (SwiGLU) OR fc_in (when SwiGLU off)
    ///   blocks.{i}.mlp.fc_out             → model.layers.{i}.mlp.down_proj
    private static func remapToHF(_ name: String) -> String {
        var s = name
        // Top-level renames.
        s = s.replacingOccurrences(of: "tokenEmbedding.weight",
                                     with: "model.embed_tokens.weight")
        s = s.replacingOccurrences(of: "lnFinal.weight",
                                     with: "model.norm.weight")
        s = s.replacingOccurrences(of: "lnFinal.bias",
                                     with: "model.norm.bias")
        // Block-level renames.
        s = s.replacingOccurrences(of: "blocks.", with: "model.layers.")
        s = s.replacingOccurrences(of: ".ln1.", with: ".input_layernorm.")
        s = s.replacingOccurrences(of: ".ln2.", with: ".post_attention_layernorm.")
        s = s.replacingOccurrences(of: ".attn.q_proj", with: ".self_attn.q_proj")
        s = s.replacingOccurrences(of: ".attn.k_proj", with: ".self_attn.k_proj")
        s = s.replacingOccurrences(of: ".attn.v_proj", with: ".self_attn.v_proj")
        s = s.replacingOccurrences(of: ".attn.o_proj", with: ".self_attn.o_proj")
        // MLP renames. fc_in → gate_proj is the SwiGLU-equivalent; HF
        // ALSO has up_proj for the second SwiGLU half. Our default GELU
        // MLP only has fc_in + fc_out, so we map directly. Callers
        // who need SwiGLU-shape output can write a thin transform on
        // top of this baseline.
        s = s.replacingOccurrences(of: ".mlp.fc_in",  with: ".mlp.gate_proj")
        s = s.replacingOccurrences(of: ".mlp.fc_out", with: ".mlp.down_proj")
        return s
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.2f GB", Double(n) / 1e9) }
        if n >= 1_000_000     { return String(format: "%.2f MB", Double(n) / 1e6) }
        if n >= 1_000         { return String(format: "%.1f KB", Double(n) / 1e3) }
        return "\(n) B"
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt to-safetensors <model.tinygpt> --out <model.safetensors> [--keep-names]

        Convert a .tinygpt checkpoint to HuggingFace safetensors. Tensor
        names are remapped to HF Llama convention by default; pass
        --keep-names to preserve TinyGPT-native names.

        After conversion, load with PyTorch:
          from safetensors.torch import load_file
          weights = load_file("model.safetensors")

        Or feed into the to-coreml-generated convert.py script — it
        loads weights via safetensors.torch.load_file directly.
        """)
        exit(code)
    }
}
