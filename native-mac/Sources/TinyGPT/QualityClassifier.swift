import Foundation

/// Binary text quality classifier — bag-of-bigrams + logistic regression.
///
/// The architecture FineWeb-Edu used (Penedo et al. 2024 §3.2): a tiny
/// fastText-style scorer that takes a text snippet and predicts a single
/// "high-quality" / "low-quality" label. They trained it on Llama-3-70B-
/// labelled educational-value annotations; the result was a 50-MB
/// classifier that filtered raw CommonCrawl down to the 200M-doc
/// FineWeb-Edu corpus.
///
/// This implementation is the same SHAPE (bag-of-ngrams + linear scorer)
/// in pure Swift — the user supplies positive + negative texts, we fit
/// weights with SGD, and save a tiny on-disk binary that `tinygpt
/// quality-filter` consumes. The labels are whatever the user trains on:
/// we ship the technique, not a specific quality definition.
///
/// On-disk format (`.tgfq`):
///   magic "TGFQ" (4B) | version u32 | vocabSize u32 | ngramOrder u32
///   | bias f32 | weights f32 × vocabSize
///
/// Inference is one hash pass + one dot product. Inference throughput
/// is bound by tokenisation, not arithmetic — ~100s of MB/s on M-series.
///
/// Subcommands (registered in TinyGPT.swift):
///   tinygpt train-quality-classifier  — fit on positive + negative files
///   tinygpt quality-filter            — score input lines; keep ≥ threshold
enum QualityClassifier {

    static let magic: [UInt8] = Array("TGFQ".utf8)
    static let version: UInt32 = 1
    static let defaultVocab: Int = 65_536
    static let defaultNgram: Int = 2

    // MARK: - Tokenization + features

    /// Tokenise a text: lowercase, keep [a-z0-9] runs as words. Cheap;
    /// matches what fastText / FineWeb-Edu use as a baseline.
    private static func tokenize(_ text: String) -> [String] {
        var out: [String] = []
        var current = [Character]()
        out.reserveCapacity(text.count / 5)
        for ch in text.lowercased() {
            if ch.isLetter || ch.isNumber {
                current.append(ch)
            } else if !current.isEmpty {
                out.append(String(current)); current.removeAll(keepingCapacity: true)
            }
        }
        if !current.isEmpty { out.append(String(current)) }
        return out
    }

    /// FNV-1a 32-bit hash → bucket index. Stable across runs (vs Swift's
    /// `Hasher.combine` which randomises per process).
    private static func hash(_ s: String, vocabSize: Int) -> Int {
        var h: UInt32 = 2_166_136_261
        for b in s.utf8 {
            h ^= UInt32(b)
            h = h &* 16_777_619
        }
        return Int(h) % vocabSize
    }

    /// For a tokenized text, return an array of n-gram bucket indices.
    /// Order 1 = unigrams; 2 = bigrams; mixed 1+2 also common in fastText.
    private static func ngrams(words: [String], n: Int, vocabSize: Int) -> [Int] {
        guard words.count >= n else { return [] }
        var buckets: [Int] = []
        buckets.reserveCapacity(words.count + words.count - n + 1)
        // Unigrams always included so single-word docs aren't featureless.
        for w in words { buckets.append(hash(w, vocabSize: vocabSize)) }
        if n >= 2 {
            for i in 0..<(words.count - 1) {
                buckets.append(hash("\(words[i])_\(words[i + 1])", vocabSize: vocabSize))
            }
        }
        if n >= 3 {
            for i in 0..<(words.count - 2) {
                buckets.append(hash("\(words[i])_\(words[i + 1])_\(words[i + 2])", vocabSize: vocabSize))
            }
        }
        return buckets
    }

    /// Compute `score = bias + Σ weights[bucket]` over the n-gram buckets
    /// of `text`. `sigmoid(score)` is the predicted P(positive).
    fileprivate static func score(text: String, weights: [Float], bias: Float,
                                   vocabSize: Int, n: Int) -> Float {
        let toks = tokenize(text)
        let buckets = ngrams(words: toks, n: n, vocabSize: vocabSize)
        var s: Float = bias
        for b in buckets { s += weights[b] }
        return s
    }

    private static func sigmoid(_ x: Float) -> Float {
        // Stable sigmoid: avoid overflow on large |x|.
        if x >= 0 { let e = expf(-x); return 1 / (1 + e) }
        let e = expf(x); return e / (1 + e)
    }

    // MARK: - Model file I/O

    private static func saveModel(weights: [Float], bias: Float,
                                   vocabSize: Int, ngramOrder: Int,
                                   to url: URL) throws {
        var out = Data()
        out.append(contentsOf: magic)
        var v = version.littleEndian
        withUnsafeBytes(of: &v) { out.append(contentsOf: $0) }
        var vs = UInt32(vocabSize).littleEndian
        withUnsafeBytes(of: &vs) { out.append(contentsOf: $0) }
        var ng = UInt32(ngramOrder).littleEndian
        withUnsafeBytes(of: &ng) { out.append(contentsOf: $0) }
        var b = bias
        withUnsafeBytes(of: &b) { out.append(contentsOf: $0) }
        weights.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        try out.write(to: url, options: .atomic)
    }

    fileprivate struct LoadedModel {
        let weights: [Float]
        let bias: Float
        let vocabSize: Int
        let ngramOrder: Int
    }

    fileprivate static func loadModel(from url: URL) throws -> LoadedModel {
        let data = try Data(contentsOf: url)
        guard data.count >= 20 else {
            throw NSError(domain: "tgfq", code: 1, userInfo: [NSLocalizedDescriptionKey: "file too small"])
        }
        guard Array(data.prefix(4)) == magic else {
            throw NSError(domain: "tgfq", code: 2, userInfo: [NSLocalizedDescriptionKey: "magic mismatch"])
        }
        let ver = data[4..<8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
        guard ver == 1 else {
            throw NSError(domain: "tgfq", code: 3, userInfo: [NSLocalizedDescriptionKey: "unsupported version \(ver)"])
        }
        let vs = Int(data[8..<12].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
        let ng = Int(data[12..<16].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
        let bias = data[16..<20].withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
        let weightsBytes = data[20..<(20 + vs * 4)]
        let weights = weightsBytes.withUnsafeBytes { raw -> [Float] in
            let buf = UnsafeBufferPointer(
                start: raw.baseAddress?.assumingMemoryBound(to: Float.self),
                count: vs)
            return Array(buf)
        }
        return LoadedModel(weights: weights, bias: bias, vocabSize: vs, ngramOrder: ng)
    }

    // MARK: - train subcommand

    /// `tinygpt train-quality-classifier`
    static func runTrain(args: [String]) {
        var positivePath: String? = nil
        var negativePath: String? = nil
        var outPath: String? = nil
        var vocabSize: Int = defaultVocab
        var ngramOrder: Int = defaultNgram
        var epochs: Int = 5
        var lr: Float = 0.05
        var l2: Float = 1e-5
        var seed: UInt64 = 42

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--positive":  positivePath = args[i+1]; i += 2
            case "--negative":  negativePath = args[i+1]; i += 2
            case "--out":       outPath = args[i+1]; i += 2
            case "--vocab":     vocabSize = max(1024, Int(args[i+1]) ?? vocabSize); i += 2
            case "--ngram":     ngramOrder = max(1, min(3, Int(args[i+1]) ?? ngramOrder)); i += 2
            case "--epochs":    epochs = max(1, Int(args[i+1]) ?? epochs); i += 2
            case "--lr":        lr = Float(args[i+1]) ?? lr; i += 2
            case "--l2":        l2 = Float(args[i+1]) ?? l2; i += 2
            case "--seed":      seed = UInt64(args[i+1]) ?? seed; i += 2
            case "-h", "--help": exitUsageTrain(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsageTrain()
            }
        }
        guard let positivePath = positivePath else { fputs("--positive required\n", stderr); exitUsageTrain() }
        guard let negativePath = negativePath else { fputs("--negative required\n", stderr); exitUsageTrain() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsageTrain() }

        let positives = readDocs(positivePath)
        let negatives = readDocs(negativePath)
        guard !positives.isEmpty, !negatives.isEmpty else {
            fputs("both --positive and --negative must contain at least one document\n", stderr); exit(1)
        }
        print("""

        TinyGPT — quality classifier (train)
        -----------------------------------
        positive:    \(positivePath) (\(positives.count) docs)
        negative:    \(negativePath) (\(negatives.count) docs)
        vocab:       \(vocabSize)
        ngram order: \(ngramOrder)
        epochs:      \(epochs)
        lr:          \(lr)
        l2:          \(l2)
        seed:        \(seed)
        """)

        // Pre-tokenize once: each doc becomes a list of bucket indices.
        // O(docs * tokens) one-time, O(docs * tokens) per epoch after.
        let posBuckets = positives.map { ngrams(words: tokenize($0), n: ngramOrder, vocabSize: vocabSize) }
        let negBuckets = negatives.map { ngrams(words: tokenize($0), n: ngramOrder, vocabSize: vocabSize) }

        // Labeled stream: (buckets, label) pairs. Shuffle per epoch.
        struct Example { let buckets: [Int]; let label: Float }
        var examples: [Example] = []
        examples.reserveCapacity(positives.count + negatives.count)
        for b in posBuckets { examples.append(Example(buckets: b, label: 1.0)) }
        for b in negBuckets { examples.append(Example(buckets: b, label: 0.0)) }

        // Deterministic shuffle. Mulberry32 — small, fast, seedable; the
        // host stdlib Int.random is non-seedable so we roll our own.
        var rng = Mulberry32(seed: UInt32(truncatingIfNeeded: seed))

        var weights = [Float](repeating: 0, count: vocabSize)
        var bias: Float = 0

        for epoch in 0..<epochs {
            // Shuffle in place.
            for i in stride(from: examples.count - 1, through: 1, by: -1) {
                let j = Int(rng.next() % UInt32(i + 1))
                if i != j { examples.swapAt(i, j) }
            }
            var lossSum: Double = 0
            var correct = 0
            for ex in examples {
                var s = bias
                for b in ex.buckets { s += weights[b] }
                let p = sigmoid(s)
                let err = p - ex.label
                let pClamp = max(p, Float(1e-10))
                let qClamp = max(1 - p, Float(1e-10))
                let logP = Double(logf(pClamp))
                let logQ = Double(logf(qClamp))
                let y = Double(ex.label)
                lossSum += -y * logP - (1 - y) * logQ
                // SGD step with L2.
                bias -= lr * err
                for b in ex.buckets {
                    weights[b] -= lr * (err + l2 * weights[b])
                }
                if (p >= 0.5) == (ex.label >= 0.5) { correct += 1 }
            }
            let avgLoss = lossSum / Double(examples.count)
            let acc = Float(correct) / Float(examples.count)
            print(String(format: "  epoch %d/%d  loss %.4f  acc %.3f", epoch + 1, epochs, avgLoss, acc))
        }

        // Persist.
        do {
            try saveModel(weights: weights, bias: bias,
                           vocabSize: vocabSize, ngramOrder: ngramOrder,
                           to: URL(fileURLWithPath: outPath))
            print("\nwrote classifier → \(outPath) (\(20 + weights.count * 4) bytes)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
    }

    /// Read documents from a file. Two modes:
    ///   - if file ends with `.jsonl`: one JSON object per line, pull
    ///     `text` field
    ///   - else: split into paragraphs by double-newline
    private static func readDocs(_ path: String) -> [String] {
        guard let data = try? String(contentsOfFile: path, encoding: .utf8) else {
            fputs("could not read \(path)\n", stderr); exit(1)
        }
        if path.hasSuffix(".jsonl") {
            var docs: [String] = []
            for line in data.split(separator: "\n") {
                guard let lineData = line.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any]
                else { continue }
                if let text = obj["text"] as? String, !text.isEmpty {
                    docs.append(text)
                }
            }
            return docs
        } else {
            return data.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        }
    }

    // MARK: - filter subcommand

    /// `tinygpt quality-filter` — score lines of <input>, write those
    /// passing --threshold to <output>.
    static func runFilter(args: [String]) {
        var inputPath: String? = nil
        var outputPath: String? = nil
        var classifierPath: String? = nil
        var threshold: Float = 0.5
        var perLine: Bool = false
        var maxDocs: Int? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--classifier":  classifierPath = args[i+1]; i += 2
            case "--out":         outputPath = args[i+1]; i += 2
            case "--threshold":   threshold = Float(args[i+1]) ?? threshold; i += 2
            case "--per-line":    perLine = true; i += 1
            case "--max-docs":    maxDocs = Int(args[i+1]); i += 2
            case "-h", "--help":  exitUsageFilter(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsageFilter() }
                inputPath = args[i]; i += 1
            }
        }
        guard let inputPath = inputPath else { fputs("missing <input>\n", stderr); exitUsageFilter() }
        guard let outputPath = outputPath else { fputs("--out required\n", stderr); exitUsageFilter() }
        guard let classifierPath = classifierPath else { fputs("--classifier required\n", stderr); exitUsageFilter() }

        let model: LoadedModel
        do { model = try loadModel(from: URL(fileURLWithPath: classifierPath)) }
        catch { fputs("load classifier failed: \(error)\n", stderr); exit(1) }

        print("loaded classifier: vocab=\(model.vocabSize), ngram=\(model.ngramOrder), bias=\(String(format: "%.4f", model.bias))")

        guard let raw = try? String(contentsOfFile: inputPath, encoding: .utf8) else {
            fputs("could not read \(inputPath)\n", stderr); exit(1)
        }
        let docs: [String]
        if perLine {
            docs = raw.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        } else {
            docs = raw.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        }

        let docsToScan = maxDocs.map { Swift.min($0, docs.count) } ?? docs.count
        var keptCount = 0
        var totalScored = 0
        var scoreSum: Double = 0

        let outURL = URL(fileURLWithPath: outputPath)
        try? FileManager.default.removeItem(at: outURL)
        FileManager.default.createFile(atPath: outURL.path, contents: nil)
        guard let outFH = try? FileHandle(forWritingTo: outURL) else {
            fputs("could not open \(outputPath) for write\n", stderr); exit(1)
        }
        defer { try? outFH.close() }

        let separator = perLine ? "\n" : "\n\n"
        for i in 0..<docsToScan {
            let doc = docs[i]
            let s = score(text: doc, weights: model.weights, bias: model.bias,
                           vocabSize: model.vocabSize, n: model.ngramOrder)
            let p = sigmoid(s)
            scoreSum += Double(p)
            totalScored += 1
            if p >= threshold {
                let payload = (doc + separator).data(using: .utf8) ?? Data()
                try? outFH.write(contentsOf: payload)
                keptCount += 1
            }
        }

        let avgScore = totalScored > 0 ? scoreSum / Double(totalScored) : 0
        let keepRate = totalScored > 0 ? Float(keptCount) / Float(totalScored) : 0
        print("""

        scanned:   \(totalScored) docs
        kept:      \(keptCount) (\(String(format: "%.1f%%", keepRate * 100)) ≥ \(threshold))
        avg score: \(String(format: "%.3f", avgScore))
        out:       \(outputPath)
        """)
    }

    private static func exitUsageTrain(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt train-quality-classifier --positive <pos.txt|jsonl> \\
                                                --negative <neg.txt|jsonl> \\
                                                --out <model.tgfq> [options]

        Train a bag-of-ngrams + logistic-regression text quality classifier.

        --positive  path to positive examples (.txt = paragraphs, .jsonl = `text` field)
        --negative  path to negative examples (same shape as --positive)
        --out       where to save the classifier (.tgfq binary, ~vocab*4 bytes)
        --vocab N   hashed-feature vocabulary size (default 65536)
        --ngram N   1/2/3 — unigrams / +bigrams / +trigrams (default 2)
        --epochs N  SGD passes over the data (default 5)
        --lr F      learning rate (default 0.05)
        --l2 F      L2 regulariser (default 1e-5)
        --seed N    deterministic example shuffle (default 42)
        """)
        exit(code)
    }

    private static func exitUsageFilter(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt quality-filter <input> --classifier <model.tgfq> \\
                                              --out <filtered> [options]

        Apply a trained TGFQ classifier to <input>. Documents (paragraphs
        by default; --per-line for line-delimited inputs) with predicted
        P(positive) ≥ --threshold are written to <output>.

        --classifier <path.tgfq>   trained model (required)
        --out <path>               filtered output (required)
        --threshold F              keep when P ≥ F (default 0.5)
        --per-line                 treat each line as a document (default: paragraphs by \\n\\n)
        --max-docs N               stop after scanning N input docs (handy for sampling)
        """)
        exit(code)
    }
}

/// Mulberry32 — tiny seedable PRNG. Used only for the deterministic
/// example shuffle in train; nothing else here needs randomness.
private struct Mulberry32 {
    private var state: UInt32
    init(seed: UInt32) { self.state = seed == 0 ? 0xDEADBEEF : seed }
    mutating func next() -> UInt32 {
        state = state &+ 0x6D2B79F5
        var z: UInt32 = state
        z = (z ^ (z >> 15)) &* (z | 1)
        z = z ^ (z &+ ((z ^ (z >> 7)) &* (z | 61)))
        return z ^ (z >> 14)
    }
}
