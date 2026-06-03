import Foundation
import MLX
import MLXRandom
import TinyGPTModel

/// `tinygpt patch` — activation patching (Meng et al. 2022).
///
/// The Mac-CLI version of the browser playground's Inspect-tab
/// patching. Two variants:
///
///   --zero  → zero out the residual at (layer, position) in the
///             recipient's forward and sample from there. Causal
///             intervention: which downstream tokens depended on
///             this representation being intact?
///   --donor → forward a SEPARATE prompt and capture its residual at
///             (donor-layer, donor-position), then inject that vector
///             into the recipient's forward at (recipient-layer,
///             recipient-position). Causal swap: which outputs
///             causally depend on the DONOR's representation at this
///             coord?
///
/// USAGE
///   tinygpt patch <model.tinygpt> --recipient "..." --layer N \
///                 --position P [--zero | --donor "..." --donor-layer N --donor-position P] \
///                 [--tokens N] [--temperature F]
enum Patch {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var recipient: String? = nil
        var layer: Int? = nil
        var position: Int? = nil
        var zeroPatch = false
        var donor: String? = nil
        var donorLayer: Int? = nil
        var donorPosition: Int? = nil
        var maxTokens = 40
        var temperature: Float = 0.0

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--recipient":       recipient = args[i+1]; i += 2
            case "--layer":           layer = Int(args[i+1]); i += 2
            case "--position":        position = Int(args[i+1]); i += 2
            case "--zero":            zeroPatch = true; i += 1
            case "--donor":           donor = args[i+1]; i += 2
            case "--donor-layer":     donorLayer = Int(args[i+1]); i += 2
            case "--donor-position":  donorPosition = Int(args[i+1]); i += 2
            case "--tokens":          maxTokens = Int(args[i+1]) ?? maxTokens; i += 2
            case "--temperature", "--temp":
                                      temperature = Float(args[i+1]) ?? 0; i += 2
            case "-h", "--help":      exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let recipient = recipient else { fputs("--recipient required\n", stderr); exitUsage() }
        guard let layer = layer else { fputs("--layer required\n", stderr); exitUsage() }
        guard let position = position else { fputs("--position required\n", stderr); exitUsage() }
        if !zeroPatch && donor == nil {
            fputs("either --zero or --donor (with --donor-layer/--donor-position) required\n", stderr); exitUsage()
        }
        if let _ = donor {
            guard donorLayer != nil, donorPosition != nil else {
                fputs("--donor needs --donor-layer and --donor-position\n", stderr); exitUsage()
            }
        }

        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("patch first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        guard layer >= 0, layer < cfg.nLayers else {
            fputs("--layer \(layer) out of range [0, \(cfg.nLayers))\n", stderr); exit(2)
        }

        // Donor hidden capture (donor-swap mode only).
        var donorHidden: MLXArray? = nil
        if let donorPrompt = donor, let dL = donorLayer, let dP = donorPosition {
            let donorBytes: [Int32] = donorPrompt.utf8.prefix(cfg.contextLength).map { Int32($0) }
            guard !donorBytes.isEmpty else { fputs("--donor empty after byte encoding\n", stderr); exit(2) }
            guard dL >= 0, dL < cfg.nLayers else {
                fputs("--donor-layer \(dL) out of range\n", stderr); exit(2)
            }
            guard dP >= 0, dP < donorBytes.count else {
                fputs("--donor-position \(dP) out of range for donor T=\(donorBytes.count)\n", stderr); exit(2)
            }
            let donorIdx = MLXArray(donorBytes, [1, donorBytes.count])
            let states = model.forwardLayerwise(donorIdx)
            let h = states[dL][0, dP, 0...]
            MLX.eval(h)
            donorHidden = h
            print("donor captured: prompt=\"\(donorPrompt.prefix(40))\" layer=\(dL) position=\(dP)")
        } else {
            print("zero patch: layer=\(layer) position=\(position)")
        }

        // Autoregressive sample loop. At each step, run forwardWithPatch
        // on the current (prompt + already-generated) byte sequence.
        // The patch coordinate (layer, position) stays fixed relative
        // to the recipient sequence — that matches the browser
        // playground's behaviour.
        let recipientBytes: [Int32] = recipient.utf8.prefix(cfg.contextLength).map { Int32($0) }
        guard !recipientBytes.isEmpty else { fputs("--recipient empty after byte encoding\n", stderr); exit(2) }
        guard position < recipientBytes.count else {
            fputs("--position \(position) out of range for recipient T=\(recipientBytes.count)\n", stderr); exit(2)
        }

        MLXRandom.seed(0x1337)
        var generated: [Int32] = []
        var current = recipientBytes
        for _ in 0..<maxTokens {
            let T = min(current.count, cfg.contextLength)
            let trimmed = Array(current.suffix(T))
            let idx = MLXArray(trimmed, [1, T])
            // The patch position is relative to recipientBytes; ensure
            // it still falls within the trimmed window.
            let posInTrimmed = position - (current.count - T)
            guard posInTrimmed >= 0 && posInTrimmed < T else { break }
            let logits = model.forwardWithPatch(idx,
                                                  donorHidden: donorHidden,
                                                  patchLayer: layer,
                                                  patchPosition: posInTrimmed)
            let lastLogits = logits[0, T - 1, 0...]
            let nextId: Int32
            if temperature <= 0 {
                let arg = MLX.argMax(lastLogits, axis: -1)
                MLX.eval(arg)
                nextId = arg.item(Int32.self)
            } else {
                let scaled = lastLogits / MLXArray(temperature)
                let sample = MLXRandom.categorical(scaled)
                MLX.eval(sample)
                nextId = sample.item(Int32.self)
            }
            generated.append(nextId)
            current.append(nextId)
        }

        let bytes: [UInt8] = generated.compactMap {
            ($0 >= 0 && $0 < 256) ? UInt8($0) : nil
        }
        let completion = String(bytes: bytes, encoding: .utf8) ?? "<non-utf8>"
        print("\nrecipient: \(recipient)")
        print("patch:     layer=\(layer) position=\(position) " +
              "\(donorHidden != nil ? "(donor)" : "(zero)")")
        print("output:    \(completion)")
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt patch <model.tinygpt> --recipient "..." \\
                             --layer N --position P \\
                             [--zero | --donor "..." --donor-layer N --donor-position P] \\
                             [--tokens N] [--temperature F]

        Mac-CLI activation patching (Meng et al. 2022).
        --zero   zero out the residual at (layer, position) and sample.
        --donor  capture the donor prompt's residual at
                 (--donor-layer, --donor-position) and inject it.
        """)
        exit(code)
    }
}
