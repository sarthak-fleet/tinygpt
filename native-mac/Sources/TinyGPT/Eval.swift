import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt eval` — measure how well a checkpoint predicts a held-out text.
/// Reports:
///   - cross-entropy loss (lower = better)
///   - bits per token (loss / ln(2))
///   - perplexity (exp(loss))
///   - a few generated samples
///
/// Two corpus paths are auto-selected based on the model header:
///   - **byte-level** (vocabSize=256, no tokenizer): raw bytes → ByteCorpus.
///     Uniform baseline is ln(256) ≈ 5.55. Reports bits-per-byte (BPB).
///   - **BPE** (vocabSize from HF config, tokenizer pinned in header):
///     UTF-8 → HFTokenizer.encode → TokenizedCorpus (cached on disk).
///     Uniform baseline is ln(vocabSize). Reports bits-per-token.
///
/// USAGE
///
///   tinygpt eval path/to/model.tinygpt --corpus held-out.txt
///   tinygpt eval shakespeare.bin --corpus shakespeare-complete.txt --batches 100
enum Eval {
    static func run(args: [String]) {
        var path: String?
        var corpusPath: String?
        var loraPath: String? = nil
        var nBatches = 50
        var batchSize: Int? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--corpus": corpusPath = args[i+1]; i += 2
            case "--lora":   loraPath = args[i+1]; i += 2
            case "--batches": nBatches = Int(args[i+1]) ?? nBatches; i += 2
            case "--batch": batchSize = Int(args[i+1]); i += 2
            case "--seed": _ = UInt32(args[i+1]); i += 2  // accepted for compat; sampleBatch is internal-random
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                path = args[i]; i += 1
            }
        }
        guard let path = path else {
            fputs("eval: missing <model.tinygpt>\n", stderr); exitUsage()
        }
        guard let corpusPath = corpusPath else {
            fputs("eval: --corpus is required\n", stderr); exitUsage()
        }
        let corpusURL = URL(fileURLWithPath: corpusPath)

        // Unified loader — picks byte-level vs BPE from header.vocabSize and
        // header.tokenizerSource, builds the right model variant (from-scratch
        // dense / MoE / DiffAttn / MoD / HF), and returns the tokenizer dir
        // if one is pinned in the header.
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(path) }
        catch { fputs("error loading \(path): \(error)\n", stderr); exit(1) }
        let model = load.model
        let cfg = load.config

        // Apply LoRA adapter on the from-scratch variant. HF-LoRA composition
        // lives in LoraCompositionHF and isn't wired through eval — the eval
        // path for HF models exists but adapter application is sample-side.
        if let loraPath = loraPath {
            switch model {
            case .fromScratch(let m):
                do {
                    let adapter = try LoraAdapterReader.read(URL(fileURLWithPath: loraPath))
                    try LoraAdapterReader.apply(adapter, to: m)
                    print("• with LoRA adapter: rank=\(adapter.header.rank) alpha=\(adapter.header.alpha) targets=\(adapter.header.targetSuffixes.joined(separator: ","))")
                } catch {
                    fputs("error loading LoRA: \(error)\n", stderr); exit(1)
                }
            case .huggingFace:
                fputs("warning: --lora on HF-loaded models isn't wired through eval yet; ignoring.\n", stderr)
            }
        }

        // Load + maybe-tokenize the corpus. Both branches end with the same
        // closure shape so the scoring loop is corpus-flavor-agnostic.
        let sampleBatch: (Int, Int) -> (MLXArray, MLXArray)
        let corpusSummary: String
        let unitLabel: String              // "byte" or "token"
        let tokenizer: HFTokenizer?        // only set when BPE

        if let tokDir = load.hfTokenizerDir {
            // BPE path — same logic as Train.swift, with the persistent
            // TokenCache so repeated evals on the same corpus are instant.
            print("loading BPE tokenizer from \(tokDir.lastPathComponent)…")
            let tok: HFTokenizer
            do { tok = try HFTokenizer.loadBlocking(from: tokDir) }
            catch { fputs("tokenizer load failed: \(error)\n", stderr); exit(1) }
            tokenizer = tok
            let cacheURL = TokenCache.cacheURL(corpus: corpusURL,
                                                tokenizerDir: tokDir,
                                                vocabSize: cfg.vocabSize)
            let tokens: [Int32]
            if let cu = cacheURL, let cached = TokenCache.read(cu) {
                tokens = cached
                print("loaded \(formatLargeInt(tokens.count)) tokens from cache: \(cu.lastPathComponent)")
            } else {
                let text: String
                do { text = try String(contentsOf: corpusURL, encoding: .utf8) }
                catch { fputs("error reading corpus: \(error)\n", stderr); exit(1) }
                print("encoding corpus (\(formatBytes(text.utf8.count)))…")
                let ids: [Int]
                do { ids = try tok.encode(text) }
                catch { fputs("tokenize failed: \(error)\n", stderr); exit(1) }
                tokens = ids.map { Int32($0) }
                if let cu = cacheURL {
                    try? TokenCache.write(tokens, to: cu)
                }
            }
            let corpus = TokenizedCorpus(tokens: tokens, vocabSize: cfg.vocabSize)
            sampleBatch = { B, T in corpus.sampleBatch(batchSize: B, contextLength: T) }
            corpusSummary = "\(corpusPath) (\(formatBytes(corpusURL.fileSizeBytes())) · \(formatLargeInt(tokens.count)) BPE tokens · vocab=\(cfg.vocabSize))"
            unitLabel = "token"
        } else {
            // Byte-level path — original Eval behaviour, preserved unchanged.
            let corpus: ByteCorpus
            do { corpus = try ByteCorpus(contentsOf: corpusURL) }
            catch { fputs("error reading corpus: \(error)\n", stderr); exit(1) }
            tokenizer = nil
            sampleBatch = { B, T in corpus.sampleBatch(batchSize: B, contextLength: T) }
            corpusSummary = "\(corpusPath) (\(formatBytes(corpus.bytes.count)) · byte-level)"
            unitLabel = "byte"
        }

        let B = batchSize ?? 8
        print("""

        TinyGPT — eval
        --------------
        model:    \(path)
        corpus:   \(corpusSummary)
        config:   \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength) · vocab=\(cfg.vocabSize)
        batches:  \(nBatches) × batch \(B) × ctx \(cfg.contextLength)
                  = \(formatLargeInt(nBatches * B * cfg.contextLength)) \(unitLabel)s scored

        """)

        // Score across N random windows. Mean per-token cross-entropy.
        var lossSum: Float = 0
        var count = 0
        for k in 0..<nBatches {
            let (x, y) = sampleBatch(B, cfg.contextLength)
            let loss = model.loss(x, y)
            eval(loss)
            let lv = loss.item(Float.self)
            lossSum += lv
            count += 1
            if k < 3 || k % 10 == 0 || k == nBatches - 1 {
                fputs(String(format: "  batch %3d  loss %.3f  running avg %.3f\n",
                             k + 1, lv, lossSum / Float(count)), stderr)
            }
        }
        let avgLoss = lossSum / Float(count)
        let bits = avgLoss / log(Float(2))  // ln → log2 — per-token bits
        let ppl = exp(avgLoss)
        let uniform = log(Float(cfg.vocabSize))

        // Per-unit metric name depends on corpus flavour:
        //   - byte-level: bits-per-byte (BPB), the literature standard
        //   - BPE: bits-per-token (informative but not directly comparable
        //     to BPB; perplexity is the cross-corpus comparable number)
        let bitsLabel = (unitLabel == "byte") ? "bits per byte  (BPB)" : "bits per token"
        print("""

        RESULTS
        -------
        cross-entropy loss:    \(String(format: "%.4f", avgLoss))   (uniform baseline: \(String(format: "%.2f", uniform)))
        \(bitsLabel):   \(String(format: "%.4f", bits))
        perplexity:            \(String(format: "%.2f", ppl))

        """)
        // Quality bands. Byte-level (vocab=256) and BPE (vocab=50k+) have
        // very different absolute scales — we report position relative to the
        // uniform baseline, which works for either.
        let ratio = avgLoss / uniform
        if ratio < 0.20 {
            print("✓ very strong — well below random; the model has learned distribution structure")
        } else if ratio < 0.30 {
            print("✓ strong — clear improvement over random; grammar should emerge in samples")
        } else if ratio < 0.50 {
            print("· OK — model has learned something but is under-trained or too small")
        } else if ratio < 0.80 {
            print("· weak — close to random; samples will be incoherent")
        } else {
            print("⚠ near random — the model isn't doing useful work")
        }

        // A few quick samples to anchor the numbers in observed output.
        // BPE branch decodes through the tokenizer; byte branch uses raw chars.
        print("\nSAMPLES")
        for prompt in ["The ", "He said, \"", "Once "] {
            print("  prompt: \(prompt.debugDescription)")
            let promptIds: [Int32]
            if let tok = tokenizer {
                let ids: [Int]
                do { ids = try tok.encode(prompt) }
                catch { print("    <tokenize failed: \(error)>"); continue }
                promptIds = ids.map { Int32($0) }
            } else {
                promptIds = [UInt8](prompt.utf8).map { Int32($0) }
            }
            var idx = MLXArray(promptIds, [1, promptIds.count])
            var generatedIds: [Int32] = promptIds
            for _ in 0..<80 {
                let T = idx.shape.last!
                let lo = max(0, T - cfg.contextLength)
                let cond = idx[0..., lo..<T]
                let logits = model(cond)
                let last = logits[0..., logits.shape[1] - 1, 0...]
                let scaled = last / MLXArray(Float(0.7))
                let next = MLX.argMax(scaled, axis: -1).reshaped([1, 1])
                eval(next)
                let id = Int32(next.item(Int32.self))
                generatedIds.append(id)
                idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
            }
            let rendered: String
            if let tok = tokenizer {
                rendered = tok.decode(generatedIds.map { Int($0) })
            } else {
                // Byte-level: only render printable / common-control chars.
                var s = ""
                for id in generatedIds {
                    if let scalar = UnicodeScalar(Int(id)), id >= 9 {
                        s.append(Character(scalar))
                    }
                }
                rendered = s
            }
            let clipped = rendered.prefix(150).replacingOccurrences(of: "\n", with: "\\n")
            print("    \(clipped)")
        }
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt eval <model.tinygpt> --corpus path.txt [options]

        --corpus path.txt    Held-out UTF-8 text to score (required)
        --batches N          Number of random windows to score (default: 50)
        --batch N            Tokens per window batch (default: 8)
        --lora <path.lora>   Apply a LoRA adapter on top of the base
        --seed N             Random seed (default: 0)

        Auto-detects byte-level vs BPE from the model's header (vocabSize
        + tokenizerSource). Byte-level reports bits-per-byte; BPE reports
        bits-per-token. Perplexity = exp(loss) for cross-corpus comparison.
        """)
        exit(2)
    }
}

// File-size helper that matches Train.swift's behavior — fileSize attribute
// straight from FileManager, defaulting to 0 if unreadable.
private extension URL {
    func fileSizeBytes() -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: self.path))?[.size]
         as? NSNumber)?.intValue ?? 0
    }
}
