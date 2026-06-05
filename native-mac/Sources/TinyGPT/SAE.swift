import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt sae` — train a sparse autoencoder on per-layer hidden
/// states. Each decoder column becomes one "feature direction" in
/// residual space; the encoder + ReLU produces a sparse activation
/// pattern over those features for any input.
///
/// Anthropic, "Towards Monosemanticity" (Bricken et al. 2023) — the
/// SAE is the canonical mechanism for decomposing a model's
/// internal representations into interpretable features. Distinct
/// from lens (projects hidden state to vocab), tuned-lens (same with
/// learned projections), and linear-probe (classifies an external
/// label): SAEs *unsupervised-discover* directions in residual space.
///
/// Architecture (matches the standard fingerprint):
///   encoded = relu(W_enc @ (h - b_dec) + b_enc)
///   reconstructed = W_dec @ encoded + b_dec
///   loss = ‖h - reconstructed‖² + λ · ‖encoded‖₁
///
/// where W_enc is [F, D], W_dec is [D, F], D = d_model, F = d_features
/// (typically 4-8× D, overcomplete). The "pre-encoder bias subtract"
/// b_dec follows Bricken et al.'s formulation — it lets the decoder
/// learn the activation centroid separately from the per-feature
/// directions, which empirically gives cleaner features.
///
/// USAGE
///   tinygpt sae <model.tinygpt> --corpus <text.txt> --layer N \
///               --out probe.sae [--features F] [--steps N]
///               [--lr F] [--l1 F] [--batch N] [--ctx N]
///
/// OUTPUT (.sae sidecar)
///   8-byte magic "TGSA" + version u32 + header_len u32 + JSON header
///   + raw fp32 [W_enc, b_enc, W_dec, b_dec] in that order.
enum SAE {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var corpusPath: String? = nil
        var outPath: String? = nil
        var layer: Int? = nil
        var dFeatures: Int? = nil   // default: 4× d_model
        var steps = 2000
        var lr: Float = 1e-3
        var l1Penalty: Float = 5e-3
        var batchSize = 32
        var ctxOverride: Int? = nil
        // B13: interp-on-checkpoints. When --checkpoint-dir is set, glob
        // *.tinygpt under it, sort by training step (read from header),
        // run the SAE training body on each, emit a JSONL timeline row
        // per checkpoint to --timeline-out. The positional <model> arg
        // is ignored in this mode; outPath becomes a stem (each ckpt
        // gets `<stem>.step-N.sae`).
        var checkpointDir: String? = nil
        var timelineOut: String? = nil
        // B19 Group-SAE: train one SAE on the union of residuals from a
        // group of (typically contiguous) layers. The encoder/decoder
        // weights are shared across the group. Comma-separated list,
        // e.g. `--layer-group 0,1,2`. Mutually exclusive with --layer.
        // Source: Wang et al., 2024 (arxiv 2410.21508).
        var layerGroup: [Int]? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus":     corpusPath = args[i+1]; i += 2
            case "--out":        outPath = args[i+1]; i += 2
            case "--layer":      layer = Int(args[i+1]); i += 2
            case "--features":   dFeatures = Int(args[i+1]); i += 2
            case "--steps":      steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":         lr = Float(args[i+1]) ?? lr; i += 2
            case "--l1":         l1Penalty = Float(args[i+1]) ?? l1Penalty; i += 2
            case "--batch":      batchSize = Int(args[i+1]) ?? batchSize; i += 2
            case "--ctx":        ctxOverride = Int(args[i+1]); i += 2
            case "--checkpoint-dir": checkpointDir = args[i+1]; i += 2
            case "--timeline-out":   timelineOut = args[i+1]; i += 2
            case "--layer-group":
                layerGroup = args[i+1].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                if (layerGroup ?? []).isEmpty {
                    fputs("--layer-group needs a comma-separated list of layer indices (got '\(args[i+1])')\n", stderr); exit(2)
                }
                i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let corpusPath = corpusPath else { fputs("--corpus required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }

        // Resolve the layer set: either --layer-group A,B,C (group mode) or
        // --layer N (singleton). Exactly one must be set.
        let layers: [Int]
        if layer == nil && layerGroup == nil {
            fputs("either --layer N or --layer-group A,B,C is required\n", stderr); exitUsage()
        }
        if layer != nil && layerGroup != nil {
            fputs("--layer and --layer-group are mutually exclusive\n", stderr); exitUsage()
        }
        if let one = layer {
            layers = [one]
        } else if let group = layerGroup {
            layers = group
        } else {
            fputs("internal: failed to resolve layer set\n", stderr); exit(2)
        }

        // B13 dispatch: --checkpoint-dir runs the SAE training loop across
        // every .tinygpt file under <dir>, sorted by header step.
        if let ckptDir = checkpointDir {
            runTimeline(checkpointDir: ckptDir, corpusPath: corpusPath, layers: layers,
                         outStem: outPath, timelineOut: timelineOut,
                         dFeatures: dFeatures, steps: steps, lr: lr,
                         l1Penalty: l1Penalty, batchSize: batchSize,
                         ctxOverride: ctxOverride)
            return
        }

        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }

        _ = trainOne(modelPath: modelPath, corpusPath: corpusPath, layers: layers,
                      dFeatures: dFeatures, steps: steps, lr: lr,
                      l1Penalty: l1Penalty, batchSize: batchSize,
                      ctxOverride: ctxOverride, outPath: outPath)
    }

    /// Run the SAE training body for a single checkpoint. Returns the
    /// final-batch diagnostic stats so the timeline driver can record
    /// them. Exits the process on any error (mirrors the legacy single-
    /// shot behaviour — a bad checkpoint in a timeline run aborts the
    /// whole loop).
    @discardableResult
    private static func trainOne(
        modelPath: String, corpusPath: String, layers: [Int],
        dFeatures: Int?, steps: Int, lr: Float, l1Penalty: Float,
        batchSize: Int, ctxOverride: Int?, outPath: String
    ) -> (mse: Float, l0PerSample: Float, l0Frac: Float,
          dModel: Int, dFeatures: Int, sourceStep: Int) {

        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("sae first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config
        for l in layers {
            guard l >= 0, l < cfg.nLayers else {
                fputs("layer \(l) out of range [0, \(cfg.nLayers))\n", stderr); exit(2)
            }
        }
        let isGroup = layers.count > 1
        let D = cfg.dModel
        let F = dFeatures ?? (4 * D)
        let T = min(ctxOverride ?? cfg.contextLength, cfg.contextLength)
        guard T >= 8 else { fputs("ctx too small (\(T))\n", stderr); exit(2) }

        // Load corpus as byte stream.
        let bytes: [UInt8]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: corpusPath))
            bytes = [UInt8](data)
        } catch { fputs("error reading \(corpusPath): \(error)\n", stderr); exit(1) }
        guard bytes.count > T + 1 else {
            fputs("corpus too small (\(bytes.count) bytes, need > \(T+1))\n", stderr); exit(1)
        }

        // Surface the checkpoint's training step (read from header) so the
        // timeline driver can emit a step column without re-parsing.
        let sourceStep = (try? TinyGPTFileReader.read(URL(fileURLWithPath: modelPath)).step) ?? -1

        print("""

        TinyGPT — sparse autoencoder
        ----------------------------
        base:           \(modelPath)
        corpus:         \(corpusPath) (\(bytes.count) bytes)
        \(isGroup ? "group:          \(layers) (\(layers.count) layers; shared encoder/decoder)"
                  : "layer:          \(layers[0])")
        d_model:        \(D)
        d_features:     \(F)  (\(String(format: "%.1f", Float(F)/Float(D)))× overcomplete)
        steps:          \(steps)
        lr:             \(lr)
        L1 penalty:     \(l1Penalty)
        batch:          \(batchSize)
        ctx:            \(T)
        """)

        // Build the SAE module.
        MLXRandom.seed(0xC0FFEE)
        let sae = SaeModule(dModel: D, dFeatures: F)
        let opt = AdamW(learningRate: lr)

        // The MLX valueAndGrad signature requires (model, x, y); SAE
        // reconstruction loss is unsupervised (target == input), so we
        // pass h twice and ignore the second arg in the closure.
        let lossFn = { (m: SaeModule, h: MLXArray, _: MLXArray) -> MLXArray in
            // h shape: [B, D]. The encoder subtracts the decoder bias
            // first (pre-encoder bias trick — see Bricken et al.).
            let centered = h - m.bDec
            let encoded = MLX.maximum(MLX.matmul(centered, m.wEnc.transposed()) + m.bEnc,
                                       MLXArray(Float(0)))
            let reconstructed = MLX.matmul(encoded, m.wDec.transposed()) + m.bDec
            let mse = MLX.mean(MLX.pow(reconstructed - h, 2))
            let l1 = MLX.mean(MLX.abs(encoded))
            return mse + MLXArray(l1Penalty) * l1
        }
        let gradFn = valueAndGrad(model: sae, lossFn)

        // Sample residual-stream training batches from random windows
        // of the corpus. For each window: run forward up to layer L,
        // take ALL T position hidden states, and feed them as B*T
        // independent training samples to the SAE.
        let t0 = Date()
        var lastLoss: Float = 0
        for step in 0..<steps {
            // Build a batch of `batchSize` random ctx-windows.
            var batchBytes: [Int32] = []
            batchBytes.reserveCapacity(batchSize * T)
            for _ in 0..<batchSize {
                let lo = Int.random(in: 0..<(bytes.count - T))
                for k in 0..<T {
                    batchBytes.append(Int32(bytes[lo + k]))
                }
            }
            let idx = MLXArray(batchBytes, [batchSize, T])
            let states = model.forwardLayerwise(idx)
            // Single-layer: states[layer] → [B*T, D].
            // Group-SAE: concat states[i] for each i in `layers` → [B*T*|G|, D].
            //   Each row is one residual sample; the SAE sees the union of
            //   the group's activations and learns features that reconstruct
            //   all of them with shared parameters.
            let h: MLXArray
            if isGroup {
                let parts = layers.map { states[$0].reshaped([batchSize * T, D]) }
                h = MLX.concatenated(parts, axis: 0)
            } else {
                h = states[layers[0]].reshaped([batchSize * T, D])
            }
            MLX.eval(h)

            let (loss, grads) = gradFn(sae, h, h)
            opt.update(model: sae, gradients: grads)
            MLX.eval(loss, sae, opt)
            lastLoss = loss.item(Float.self)

            if step == 0 || (step + 1) % max(1, steps / 25) == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %4d/%4d  loss %.4f  · %.1f step/s · eta %.0fs\n",
                              step + 1, steps, lastLoss, sps,
                              Double(steps - step - 1) / sps), stderr)
            }
        }

        // Diagnostics: average sparsity (fraction of non-zero features
        // per sample) and reconstruction MSE on a fresh batch.
        var diagBytes: [Int32] = []
        for _ in 0..<batchSize {
            let lo = Int.random(in: 0..<(bytes.count - T))
            for k in 0..<T { diagBytes.append(Int32(bytes[lo + k])) }
        }
        let diagIdx = MLXArray(diagBytes, [batchSize, T])
        let diagStates = model.forwardLayerwise(diagIdx)
        // Same stacking as training: held-out diagnostics reflect group
        // behaviour, not single-layer.
        let hDiag: MLXArray
        if isGroup {
            let parts = layers.map { diagStates[$0].reshaped([batchSize * T, D]) }
            hDiag = MLX.concatenated(parts, axis: 0)
        } else {
            hDiag = diagStates[layers[0]].reshaped([batchSize * T, D])
        }
        MLX.eval(hDiag)
        let centered = hDiag - sae.bDec
        let encDiag = MLX.maximum(MLX.matmul(centered, sae.wEnc.transposed()) + sae.bEnc,
                                    MLXArray(Float(0)))
        let reconDiag = MLX.matmul(encDiag, sae.wDec.transposed()) + sae.bDec
        let mseDiag = MLX.mean(MLX.pow(reconDiag - hDiag, 2)).item(Float.self)
        // L0: fraction of non-zero features per sample.
        let nonzeroMask = (encDiag .> MLXArray(Float(0))).asType(.float32)
        let l0PerSample = MLX.mean(MLX.sum(nonzeroMask, axis: 1)).item(Float.self)
        let l0Frac = l0PerSample / Float(F)

        print("""

        diagnostics on held-out batch:
          reconstruction MSE:   \(String(format: "%.4e", mseDiag))
          mean L0 (active features per sample): \(String(format: "%.1f / %d  (%.2f%%)", l0PerSample, F, l0Frac * 100))
        """)

        // Persist. Single-layer SAE: store the layer index as before.
        // Group SAE: store the first layer in the legacy `layer` field
        // (so old readers still see a valid value) plus the full layer
        // list in the new `layers` field via SaeWriter's layers param.
        do {
            try SaeWriter.write(sae: sae, cfg: cfg,
                                 layer: layers[0],
                                 layers: layers.count > 1 ? layers : nil,
                                 to: URL(fileURLWithPath: outPath))
            print("wrote SAE sidecar → \(outPath)")
        } catch {
            fputs("write failed: \(error)\n", stderr); exit(1)
        }

        return (mse: mseDiag, l0PerSample: l0PerSample, l0Frac: l0Frac,
                dModel: D, dFeatures: F, sourceStep: Int(sourceStep))
    }

    /// B13: run `trainOne` across every `.tinygpt` checkpoint in `checkpointDir`,
    /// sorted by training step (read from each file's header). Emits one JSONL
    /// line per checkpoint to `timelineOut` (or stdout if omitted) so a viewer
    /// can plot SAE feature emergence over training time.
    ///
    /// Output paths: each checkpoint writes its `.sae` to `<outStem>.step-N.sae`.
    /// The timeline file is append-safe — re-running with a longer list of
    /// checkpoints rewrites it.
    private static func runTimeline(
        checkpointDir: String, corpusPath: String, layers: [Int],
        outStem: String, timelineOut: String?,
        dFeatures: Int?, steps: Int, lr: Float, l1Penalty: Float,
        batchSize: Int, ctxOverride: Int?
    ) {
        let dirURL = URL(fileURLWithPath: checkpointDir)
        let fm = FileManager.default
        let entries: [URL]
        do {
            entries = try fm.contentsOfDirectory(at: dirURL,
                                                  includingPropertiesForKeys: nil)
                              .filter { $0.pathExtension == "tinygpt" }
        } catch {
            fputs("could not list \(checkpointDir): \(error)\n", stderr); exit(1)
        }
        guard !entries.isEmpty else {
            fputs("no .tinygpt files under \(checkpointDir)\n", stderr); exit(1)
        }

        // Pair each checkpoint with its training step (header), sort ascending.
        struct CkPt { let url: URL; let step: Int }
        var ckpts: [CkPt] = []
        for url in entries {
            // Best effort — checkpoints with unreadable headers are skipped
            // with a stderr note, not aborted, so a partially-corrupt dir
            // doesn't kill the whole timeline run.
            guard let file = try? TinyGPTFileReader.read(url) else {
                fputs("warning: could not read \(url.lastPathComponent), skipping\n", stderr)
                continue
            }
            ckpts.append(CkPt(url: url, step: Int(file.step)))
        }
        ckpts.sort { $0.step < $1.step }
        guard !ckpts.isEmpty else {
            fputs("no readable .tinygpt files under \(checkpointDir)\n", stderr); exit(1)
        }

        print("""

        TinyGPT — SAE timeline
        ----------------------
        checkpoints: \(ckpts.count) (steps \(ckpts.first!.step) → \(ckpts.last!.step))
        corpus:      \(corpusPath)
        \(layers.count > 1 ? "group:       \(layers)" : "layer:       \(layers[0])")
        steps/SAE:   \(steps)
        out stem:    \(outStem)
        timeline:    \(timelineOut ?? "(stdout)")
        """)

        // Open the timeline output. Use a file handle so each row is flushed
        // immediately — a crash mid-loop preserves earlier rows.
        let outFH: FileHandle?
        if let p = timelineOut {
            let url = URL(fileURLWithPath: p)
            try? fm.createDirectory(at: url.deletingLastPathComponent(),
                                     withIntermediateDirectories: true)
            // Truncate any previous timeline; we don't want to append stale rows.
            _ = fm.createFile(atPath: url.path, contents: nil)
            outFH = try? FileHandle(forWritingTo: url)
        } else {
            outFH = nil
        }

        // outStem can be either a directory ("/tmp/saes") or a path stem
        // ("/tmp/saes/probe"). Treat trailing-slash or existing-directory as
        // the former: derive `<dir>/sae.step-N.sae`.
        var isDir: ObjCBool = false
        let stemIsDir = fm.fileExists(atPath: outStem, isDirectory: &isDir) && isDir.boolValue
        let stemDir = stemIsDir ? outStem : URL(fileURLWithPath: outStem).deletingLastPathComponent().path
        let stemBase = stemIsDir ? "sae" : URL(fileURLWithPath: outStem).deletingPathExtension().lastPathComponent
        try? fm.createDirectory(atPath: stemDir, withIntermediateDirectories: true)

        // Group SAEs get a different per-checkpoint filename suffix so
        // single-layer and grouped timelines don't collide if both are
        // run against the same checkpoint dir.
        let layerTag = layers.count > 1
            ? "group-" + layers.map(String.init).joined(separator: "_")
            : "layer-\(layers[0])"

        for (i, ckpt) in ckpts.enumerated() {
            let perOut = "\(stemDir)/\(stemBase).\(layerTag).step-\(ckpt.step).sae"
            print("\n[\(i+1)/\(ckpts.count)] step \(ckpt.step) — training SAE on \(ckpt.url.lastPathComponent)")
            let stats = trainOne(modelPath: ckpt.url.path, corpusPath: corpusPath,
                                  layers: layers, dFeatures: dFeatures, steps: steps,
                                  lr: lr, l1Penalty: l1Penalty, batchSize: batchSize,
                                  ctxOverride: ctxOverride, outPath: perOut)
            // Schema kept stable for the future browser viewer; "layers"
            // is the new field for group runs, "layer" stays for backcompat.
            let row: [String: Any] = [
                "step": ckpt.step,
                "layer": layers[0],
                "layers": layers,
                "d_model": stats.dModel,
                "d_features": stats.dFeatures,
                "mse": stats.mse.isFinite ? stats.mse : 0,
                "l0_per_sample": stats.l0PerSample.isFinite ? stats.l0PerSample : 0,
                "l0_frac": stats.l0Frac.isFinite ? stats.l0Frac : 0,
                "sae_path": perOut,
                "ckpt_path": ckpt.url.path,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: row,
                                                      options: [.sortedKeys]) {
                var payload = data
                payload.append(0x0A)
                if let fh = outFH {
                    try? fh.write(contentsOf: payload)
                    try? fh.synchronize()
                } else if let s = String(data: payload, encoding: .utf8) {
                    print(s, terminator: "")
                }
            }
        }
        try? outFH?.close()
        print("\ntimeline complete — \(ckpts.count) checkpoints")
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt sae <model.tinygpt> --corpus <text.txt> --layer N \\
                           --out <probe.sae> [options]

        Train a sparse autoencoder on per-layer hidden states for
        feature decomposition (Bricken et al. 2023). Distinct
        mechanism from lens / tuned-lens / linear-probe.

        --corpus <path.txt>   UTF-8 byte-level corpus to fit on (required)
        --layer N             which TransformerBlock's residual stream
        --out <path.sae>      where to save the trained SAE (required)
        --features F          dictionary size (default 4× d_model)
        --steps N             training steps (default 2000)
        --lr F                learning rate (default 1e-3)
        --l1 F                L1 sparsity penalty (default 5e-3 —
                              higher = sparser features, more
                              reconstruction error)
        --batch N             batch size (default 32; each window
                              contributes ctx × samples)
        --ctx N               override context length (default model's)

        Timeline mode (B13 — interp-on-checkpoints):
        --checkpoint-dir DIR  train an SAE per .tinygpt under DIR (sorted
                              by header step). The positional <model> arg
                              is ignored when this is set.
        --timeline-out PATH   write one JSONL row per checkpoint (step +
                              MSE + L0 + per-checkpoint .sae path) to
                              PATH. Defaults to stdout when omitted.
                              In timeline mode --out becomes a path stem
                              (or directory); each checkpoint produces
                              `<stem>.<layer-tag>.step-N.sae`.

        Group-SAE mode (B19 — Wang et al., 2024 · arxiv 2410.21508):
        --layer-group A,B,C   train ONE SAE on the union of residuals
                              from layers A, B, C (encoder/decoder
                              weights shared). Mutually exclusive with
                              --layer. Typically picks contiguous layers
                              (e.g. 2,3,4). ~3× faster than three
                              per-layer SAEs at the cost of slightly
                              higher MSE — useful when you'd otherwise
                              train many similar SAEs. Combines with
                              --checkpoint-dir to get one timeline per
                              group.
        """)
        exit(code)
    }
}

/// Sparse-autoencoder module. Encoder + decoder + a shared "decoder
/// bias subtracted before encode" trick (Bricken et al.).
final class SaeModule: Module {
    let dModel: Int
    let dFeatures: Int
    @ParameterInfo(key: "w_enc") var wEnc: MLXArray   // [F, D]
    @ParameterInfo(key: "b_enc") var bEnc: MLXArray   // [F]
    @ParameterInfo(key: "w_dec") var wDec: MLXArray   // [D, F]
    @ParameterInfo(key: "b_dec") var bDec: MLXArray   // [D]
    init(dModel: Int, dFeatures: Int) {
        self.dModel = dModel
        self.dFeatures = dFeatures
        // Encoder init: random gaussian std 1/√D.
        self._wEnc.wrappedValue = MLXRandom.normal(
            [dFeatures, dModel], scale: 1.0 / Foundation.sqrt(Float(dModel)))
        self._bEnc.wrappedValue = MLXArray.zeros([dFeatures])
        // Decoder init: random gaussian std 1/√F (so initial output
        // matches input magnitude). Some SAE recipes initialise W_dec
        // as W_enc^T (tied init) — both work; we keep separate.
        self._wDec.wrappedValue = MLXRandom.normal(
            [dModel, dFeatures], scale: 1.0 / Foundation.sqrt(Float(dFeatures)))
        self._bDec.wrappedValue = MLXArray.zeros([dModel])
        super.init()
    }
}

private struct SaeHeader: Codable {
    var version: Int
    var dModel: Int
    var dFeatures: Int
    var layer: Int
    /// Group-SAE only — full layer list when the SAE was trained on the
    /// union of multiple layers' residuals (B19 Group-SAE). Nil for
    /// single-layer SAEs so older readers round-trip cleanly.
    var layers: [Int]?
    var baseLayers: Int
    var baseDModel: Int
    var baseCtx: Int
}

enum SaeWriter {
    static func write(sae: SaeModule, cfg: ModelConfig,
                       layer: Int, layers: [Int]? = nil,
                       to url: URL) throws {
        let header = SaeHeader(
            version: 1, dModel: sae.dModel, dFeatures: sae.dFeatures,
            layer: layer, layers: layers,
            baseLayers: cfg.nLayers, baseDModel: cfg.dModel,
            baseCtx: cfg.contextLength
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try encoder.encode(header)
        var out = Data()
        out.append(contentsOf: Array("TGSA".utf8))
        var version = UInt32(1).littleEndian
        withUnsafeBytes(of: &version) { out.append(contentsOf: $0) }
        var headerLen = UInt32(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { out.append(contentsOf: $0) }
        out.append(headerData)
        // Body: wEnc, bEnc, wDec, bDec, each as fp32.
        MLX.eval(sae.wEnc, sae.bEnc, sae.wDec, sae.bDec)
        for arr in [sae.wEnc, sae.bEnc, sae.wDec, sae.bDec] {
            let floats: [Float] = arr.asArray(Float.self)
            floats.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        }
        try out.write(to: url, options: .atomic)
    }
}
