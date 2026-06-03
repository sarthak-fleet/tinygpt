import Foundation
import MLX
import MLXRandom
import TinyGPTModel

/// `tinygpt causal-trace` — Meng et al. 2022's causal-trace methodology.
///
/// For a fact (prompt, target-token), measure per-layer how important
/// each layer's residual state at the subject position is to the
/// model's prediction of the target. The procedure:
///
///   1. Forward CLEAN prompt → record clean per-layer states +
///      target token's log-prob P_clean.
///   2. CORRUPT the subject's input embedding (zero it out — equivalent
///      to "the model never saw the subject") → forward → P_corrupt.
///      Typically P_corrupt ≪ P_clean.
///   3. For each layer L: forward with corrupted subject embedding,
///      BUT after block L, restore the residual at the subject
///      position to its clean value. Forward through the remaining
///      blocks → P_restored_L.
///   4. Causal effect of layer L = P_restored_L − P_corrupt.
///      Layers where restoring the clean state recovers the target
///      probability are the ones that "store" the fact.
///
/// Output: a per-layer table of (P_restored, causal_effect) so you can
/// see where the fact lives in the network.
///
/// Use case 1 — interpretability. Where does the model store
/// "(France, capital, Paris)"?
///
/// Use case 2 — MEMIT v2 layer weighting. The causal-trace influence
/// weights are exactly what Meng 2023's full MEMIT uses to distribute
/// the rank-K update across layers.
///
/// USAGE
///   tinygpt causal-trace <model.tinygpt> --prompt "..." --target <byte> \
///       [--subject-position N | --subject "..."]
enum CausalTrace {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var prompt: String? = nil
        var targetChar: String? = nil
        var subjectPos: Int? = nil
        var subjectStr: String? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--prompt":           prompt = args[i+1]; i += 2
            case "--target":           targetChar = args[i+1]; i += 2
            case "--subject-position": subjectPos = Int(args[i+1]); i += 2
            case "--subject":          subjectStr = args[i+1]; i += 2
            case "-h", "--help":       exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        guard let prompt = prompt else { fputs("--prompt required\n", stderr); exitUsage() }
        guard let targetChar = targetChar, let tb = targetChar.utf8.first,
              targetChar.utf8.count == 1
        else { fputs("--target must be a single ASCII byte\n", stderr); exit(2) }

        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("causal-trace first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        let promptBytes: [Int32] = prompt.utf8.prefix(cfg.contextLength).map { Int32($0) }
        guard !promptBytes.isEmpty else { fputs("--prompt empty\n", stderr); exit(2) }
        let T = promptBytes.count

        // Resolve subject position. Default = last token. If
        // --subject is given, find its last occurrence in the prompt
        // and use the position right after.
        let subPos: Int
        if let p = subjectPos {
            subPos = p
        } else if let s = subjectStr {
            guard let r = prompt.range(of: s, options: .backwards) else {
                fputs("--subject \"\(s)\" not found in --prompt\n", stderr); exit(2)
            }
            // Byte index of the last char of the subject in the prompt's utf8.
            let upper = prompt.utf8.distance(from: prompt.utf8.startIndex,
                                              to: s.utf8.endIndex == prompt.utf8.endIndex
                                                    ? prompt.utf8.endIndex
                                                    : r.upperBound.samePosition(in: prompt.utf8)
                                                      ?? prompt.utf8.endIndex)
            subPos = max(0, upper - 1)
        } else {
            subPos = T - 1
        }
        guard subPos >= 0 && subPos < T else {
            fputs("subject position \(subPos) out of range for T=\(T)\n", stderr); exit(2)
        }
        let target = Int(tb)
        print("prompt:           \(prompt)")
        print("target byte:      0x\(String(target, radix: 16)) ('\(targetChar)')")
        print("subject position: \(subPos)")
        print()

        // ----- Clean forward: capture per-layer hidden states + P_clean -----
        let idx = MLXArray(promptBytes, [1, T])
        let cleanStates = model.forwardLayerwise(idx)
        let cleanLogits = model(idx)
        MLX.eval(cleanLogits)
        let pClean = lastTokenTargetLogProb(logits: cleanLogits,
                                              T: T, vocab: cfg.vocabSize, target: target)

        // ----- Corrupted forward: zero the subject's input embedding -----
        let pCorruptOnly = forwardCorruptedThenRestore(
            model: model, cfg: cfg, idx: idx,
            subjectPosition: subPos, restoreLayer: nil, restoreHidden: nil,
            target: target)

        print(String(format: "P_clean(target)   = %.4f", pClean))
        print(String(format: "P_corrupt(target) = %.4f", pCorruptOnly))
        print()
        print("per-layer causal effect (P_restored_L − P_corrupt):")
        print("layer  P_restored    Δ vs corrupt")
        print("-----  -----------   ------------")

        var traceValues = [Float](repeating: 0, count: cfg.nLayers)
        for L in 0..<cfg.nLayers {
            let cleanAtL = cleanStates[L][0, subPos, 0...]
            MLX.eval(cleanAtL)
            let pRestored = forwardCorruptedThenRestore(
                model: model, cfg: cfg, idx: idx,
                subjectPosition: subPos, restoreLayer: L, restoreHidden: cleanAtL,
                target: target)
            traceValues[L] = pRestored - pCorruptOnly
            print(String(format: "  %2d   %.4f       %+.4f", L, pRestored, traceValues[L]))
        }

        // Identify the peak layer and overall summary.
        if let (peakL, peak) = traceValues.enumerated().max(by: { $0.1 < $1.1 }) {
            print()
            print(String(format: "peak causal layer: %d  (Δ = %+.4f)", peakL, peak))
            print("  ↑ this is where the fact is most strongly stored")
            print("  use this for MEMIT --layer N if doing a single-layer edit,")
            print("  or as the centre of a --layers range for multi-layer edits.")
        }
    }

    /// Run forward with the subject's INPUT embedding zeroed. Optionally
    /// after block `restoreLayer`, replace the residual at
    /// `[0, subjectPosition, :]` with `restoreHidden` (the clean value).
    /// Returns the target token's log-prob at the last-position logits.
    static func forwardCorruptedThenRestore(
        model: TinyGPTModel, cfg: ModelConfig, idx: MLXArray,
        subjectPosition: Int, restoreLayer: Int?, restoreHidden: MLXArray?,
        target: Int
    ) -> Float {
        let T = idx.shape[1]
        let positions = MLXArray((0..<T).map { Int32($0) })
        let posEmb = model.positionEmbedding(positions).expandedDimensions(axis: 0)
        var tokEmb = model.tokenEmbedding(idx)
        // Zero the subject's token embedding (the corruption).
        let zerosRow = MLXArray.zeros([1, 1, cfg.dModel])
        let preTok = tokEmb[0..., 0..<subjectPosition, 0...]
        let postTok = tokEmb[0..., (subjectPosition + 1)..<T, 0...]
        tokEmb = concatenated([preTok, zerosRow, postTok], axis: 1)
        var x = tokEmb + posEmb
        for (i, block) in model.blocks.enumerated() {
            x = block(x)
            if let L = restoreLayer, i == L, let h = restoreHidden {
                let donor = h.reshaped([1, 1, cfg.dModel])
                let pre = x[0..., 0..<subjectPosition, 0...]
                let post = x[0..., (subjectPosition + 1)..<T, 0...]
                x = concatenated([pre, donor, post], axis: 1)
            }
        }
        // Final norm + LM head.
        let logits = projectViaModelHead(model: model, hidden: model.lnFinal(x))
        MLX.eval(logits)
        return lastTokenTargetLogProb(logits: logits, T: T,
                                       vocab: cfg.vocabSize, target: target)
    }

    /// Compute log P(target | context) for the LAST position of a
    /// [1, T, vocab] logits tensor.
    static func lastTokenTargetLogProb(logits: MLXArray, T: Int,
                                         vocab: Int, target: Int) -> Float {
        let last = logits[0, T - 1, 0...]
        MLX.eval(last)
        let arr = last.asArray(Float.self)
        var maxLogit: Float = -Float.greatestFiniteMagnitude
        for v in arr where v > maxLogit { maxLogit = v }
        var sumExp: Float = 0
        for v in arr { sumExp += expf(v - maxLogit) }
        let logZ = maxLogit + logf(sumExp)
        return arr[target] - logZ
    }

    /// Apply the model's LM-head projection to a [1, T, dModel] hidden
    /// tensor. Goes through the same path as `callAsFunction` minus the
    /// initial blocks (already done).
    static func projectViaModelHead(model: TinyGPTModel, hidden: MLXArray) -> MLXArray {
        // TinyGPTModel's projectLogits is private; cheapest equivalent:
        // matmul against the tokenEmbedding weight (tied embedding) or
        // call the model's full forward starting from this hidden.
        // Tied path covers our default config.
        return MLX.matmul(hidden, model.tokenEmbedding.weight.transposed())
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt causal-trace <model.tinygpt> --prompt "..." --target <byte> \\
                                    [--subject-position N | --subject "..."]

        Per-layer causal-trace (Meng et al. 2022). Reports where in
        the model the fact (prompt → target) is stored. Use the peak
        layer as MEMIT's --layer argument for single-layer edits, or
        the peak region as a --layers range for multi-layer edits.

        --prompt "..."         the fact's surface form (required)
        --target <ch>          single-byte continuation we test for
        --subject-position N   index of last subject token (default: T-1)
        --subject "..."        alternative: find subject string in prompt
        """)
        exit(code)
    }
}
