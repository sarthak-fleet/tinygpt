import Foundation
import MLX
import TinyGPTModel

/// `tinygpt sae-to-saelens` — convert a TinyGPT `.sae` sidecar to the
/// directory layout SAELens (decoderesearch/SAELens) consumes.
///
/// The destination directory ends up with:
///   <out>/sae_weights.safetensors    keys: W_enc, b_enc, W_dec, b_dec
///   <out>/cfg.json                   d_in, d_sae, dtype, ... + metadata
///
/// Once written, the SAE loads in Python via:
///
///   from sae_lens import SAE
///   sae = SAE.load_from_disk("<out>")
///
/// That also unlocks Neuronpedia upload (their tooling consumes the
/// same on-disk shape) — TinyGPT-trained SAEs become first-class in
/// the interp ecosystem without us reinventing visualisation.
///
/// Conversion is metadata-faithful: the `metadata` block in cfg.json
/// preserves the .sae's training context — base layer, base model
/// shape, and (B19) the full `tinygpt_layers` list for group SAEs.
enum SaeToSaelens {
    static func run(args: [String]) {
        var inPath: String? = nil
        var outDir: String? = nil
        var modelName: String = "tinygpt-from-scratch"
        var hookTemplate: String = "blocks.{layer}.hook_resid_post"

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":          outDir = args[i+1]; i += 2
            case "--model-name":   modelName = args[i+1]; i += 2
            case "--hook":         hookTemplate = args[i+1]; i += 2
            case "-h", "--help":   exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inPath = args[i]; i += 1
            }
        }
        guard let inPath = inPath else { fputs("missing <in.sae>\n", stderr); exitUsage() }
        guard let outDir = outDir else { fputs("--out <dir> required\n", stderr); exitUsage() }

        let url = URL(fileURLWithPath: inPath)
        guard let data = try? Data(contentsOf: url) else {
            fputs("could not read \(inPath)\n", stderr); exit(1)
        }
        let parsed = parse(data: data)

        let outURL = URL(fileURLWithPath: outDir)
        let fm = FileManager.default
        do { try fm.createDirectory(at: outURL, withIntermediateDirectories: true) }
        catch { fputs("could not create \(outDir): \(error)\n", stderr); exit(1) }

        // Weights — SAELens uses capitalized W_enc/W_dec; our reader
        // produced w_enc/w_dec with the same shape conventions (encoder
        // weight is [d_sae, d_in], decoder weight is [d_in, d_sae]).
        // No transpose needed.
        let weightsURL = outURL.appendingPathComponent("sae_weights.safetensors")
        do {
            try SafetensorsWriter.write(named: [
                ("W_enc", parsed.wEnc),
                ("b_enc", parsed.bEnc),
                ("W_dec", parsed.wDec),
                ("b_dec", parsed.bDec),
            ], to: weightsURL)
        } catch {
            fputs("safetensors write failed: \(error)\n", stderr); exit(1)
        }

        // cfg.json — match SAELens's expected schema. The "metadata"
        // block carries TinyGPT-specific provenance so a future round-
        // trip back to our format isn't lossy.
        let hookLayer = parsed.layer
        let hookName = hookTemplate.replacingOccurrences(of: "{layer}", with: "\(hookLayer)")
        var metadata: [String: Any] = [
            "model_name": modelName,
            "hook_name": hookName,
            "hook_layer": hookLayer,
            "context_size": parsed.baseCtx,
            "prepend_bos": false,
            "dataset_path": "tinygpt-pretrain",
            "tinygpt_layer": parsed.layer,
            "tinygpt_base_layers": parsed.baseLayers,
            "tinygpt_base_d_model": parsed.baseDModel,
            "tinygpt_base_ctx": parsed.baseCtx,
        ]
        if let group = parsed.layers, group.count > 1 {
            metadata["tinygpt_group_layers"] = group
            metadata["tinygpt_is_group_sae"] = true
        }

        let cfg: [String: Any] = [
            "architecture": "standard",
            "d_in": parsed.dModel,
            "d_sae": parsed.dFeatures,
            "dtype": "float32",
            "device": "cpu",
            "apply_b_dec_to_input": true,    // matches our pre-encoder bias-subtract
            "normalize_activations": "none",
            "reshape_activations": "none",
            "metadata": metadata,
        ]
        guard JSONSerialization.isValidJSONObject(cfg),
              let cfgData = try? JSONSerialization.data(
                withJSONObject: cfg, options: [.prettyPrinted, .sortedKeys])
        else {
            fputs("cfg.json encode failed\n", stderr); exit(1)
        }
        let cfgURL = outURL.appendingPathComponent("cfg.json")
        do { try cfgData.write(to: cfgURL, options: .atomic) }
        catch { fputs("cfg.json write failed: \(error)\n", stderr); exit(1) }

        let wSize = (try? weightsURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let cSize = cfgData.count

        print("""

        wrote SAELens-format SAE to \(outDir):
          sae_weights.safetensors  \(wSize) bytes  (W_enc \(parsed.dFeatures)×\(parsed.dModel), b_enc \(parsed.dFeatures), W_dec \(parsed.dModel)×\(parsed.dFeatures), b_dec \(parsed.dModel))
          cfg.json                 \(cSize) bytes  (architecture=standard, d_in=\(parsed.dModel), d_sae=\(parsed.dFeatures), hook=\(hookName))

        load in Python:
          from sae_lens import SAE
          sae = SAE.load_from_disk("\(outDir)")

        """)
    }

    // MARK: - .sae parser

    private struct Parsed {
        let dModel: Int
        let dFeatures: Int
        let layer: Int
        let layers: [Int]?   // group SAE only
        let baseLayers: Int
        let baseDModel: Int
        let baseCtx: Int
        let wEnc: MLXArray   // [F, D]
        let bEnc: MLXArray   // [F]
        let wDec: MLXArray   // [D, F]
        let bDec: MLXArray   // [D]
    }

    private struct HeaderView: Codable {
        let version: Int
        let dModel: Int
        let dFeatures: Int
        let layer: Int
        let layers: [Int]?
        let baseLayers: Int
        let baseDModel: Int
        let baseCtx: Int
    }

    private static func parse(data: Data) -> Parsed {
        precondition(data.count >= 12, "sae sidecar too small")
        let magic = Array(data.prefix(4))
        precondition(magic == Array("TGSA".utf8), "sae sidecar magic 'TGSA' mismatch")
        let version = data[4..<8].withUnsafeBytes {
            $0.loadUnaligned(as: UInt32.self).littleEndian
        }
        precondition(version == 1, "unsupported sae sidecar version \(version)")
        let headerLen = Int(data[8..<12].withUnsafeBytes {
            $0.loadUnaligned(as: UInt32.self).littleEndian
        })
        let headerData = data.subdata(in: 12..<(12 + headerLen))
        guard let header = try? JSONDecoder().decode(HeaderView.self, from: headerData) else {
            fputs("sae sidecar header parse failed\n", stderr); exit(1)
        }
        var cursor = 12 + headerLen
        let D = header.dModel
        let F = header.dFeatures
        func readFloats(_ count: Int) -> [Float] {
            let bytes = count * 4
            let slice = data.subdata(in: cursor..<(cursor + bytes))
            cursor += bytes
            return slice.withUnsafeBytes { ptr in
                Array(UnsafeBufferPointer<Float>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: count))
            }
        }
        let wEnc = readFloats(F * D)
        let bEnc = readFloats(F)
        let wDec = readFloats(D * F)
        let bDec = readFloats(D)
        return Parsed(
            dModel: D, dFeatures: F,
            layer: header.layer, layers: header.layers,
            baseLayers: header.baseLayers,
            baseDModel: header.baseDModel,
            baseCtx: header.baseCtx,
            wEnc: MLXArray(wEnc, [F, D]),
            bEnc: MLXArray(bEnc, [F]),
            wDec: MLXArray(wDec, [D, F]),
            bDec: MLXArray(bDec, [D])
        )
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt sae-to-saelens <in.sae> --out <dir> [options]

        Convert a TinyGPT-trained .sae sidecar to SAELens's on-disk
        format (sae_weights.safetensors + cfg.json under <dir>). Once
        written, the SAE loads in Python via:

          from sae_lens import SAE
          sae = SAE.load_from_disk("<dir>")

        That unlocks SAELens's analysis tooling and Neuronpedia upload
        on TinyGPT-trained SAEs without reimplementing either ourselves.

        --out <dir>           destination directory (created if absent)
        --model-name NAME     written into cfg.json metadata.model_name
                              (default: "tinygpt-from-scratch")
        --hook TEMPLATE       hook-name template; {layer} is substituted.
                              (default: "blocks.{layer}.hook_resid_post";
                               TransformerLens convention, matches what
                               SAELens expects to load against)

        For a group SAE (B19), metadata gets `tinygpt_group_layers` +
        `tinygpt_is_group_sae=true` so the provenance round-trips.
        """)
        exit(code)
    }
}
