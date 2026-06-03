import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTModel

/// `tinygpt bon` — best-of-N sampling with a self-likelihood verifier
/// (test-time compute scaling, Tier-5 §5.2).
///
/// Methodology mirrors Snell et al. 2024 ("Scaling LLM Test-Time
/// Compute Optimally can be More Effective than Scaling Model
/// Parameters"). For each N:
///   1. Sample N candidate completions from a fixed prompt at
///      temperature > 0 (so they diverge).
///   2. Score each candidate by its average per-token log-probability
///      under the SAME model — this is the cheapest verifier and the
///      one that has theoretical grounding for at-scale models. For
///      a tiny from-scratch model the self-likelihood signal is
///      noisier, so the curve we plot is a lower bound on what a real
///      verifier (separate small reward model, perplexity-on-held-out
///      eval) would extract.
///   3. Return the highest-scoring candidate.
///
/// `--scan N1,N2,N3,...` produces the quality-vs-compute curve as
/// JSON, with one row per N value (best score, mean score, text of
/// the chosen completion).
///
/// USAGE
///   tinygpt bon <model.tinygpt> --prompt "..." --tokens 30 \
///               --temperature 0.8 [--n 16 | --scan 1,2,4,8,16,32]
///               [--seed 42] [--out curve.json]
enum BestOfN {
    static func run(args: [String]) {
        var modelPath: String? = nil
        var prompt: String = "ROMEO:"
        var maxTokens = 30
        var temperature: Float = 0.8
        var nSingle: Int? = nil
        var scanSpec: String? = nil
        var seed: UInt64 = 42
        var outPath: String? = nil
        var verifier: String = "self"      // "self" | "corpus-ppl"
        var corpusPath: String? = nil

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--prompt":      prompt = args[i+1]; i += 2
            case "--tokens":      maxTokens = Int(args[i+1]) ?? maxTokens; i += 2
            case "--temperature", "--temp":
                                  temperature = Float(args[i+1]) ?? temperature; i += 2
            case "--n":           nSingle = Int(args[i+1]); i += 2
            case "--scan":        scanSpec = args[i+1]; i += 2
            case "--seed":        seed = UInt64(args[i+1]) ?? seed; i += 2
            case "--out":         outPath = args[i+1]; i += 2
            case "--verifier":    verifier = args[i+1]; i += 2
            case "--corpus":      corpusPath = args[i+1]; i += 2
            case "-h", "--help":  exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else { fputs("missing <model>\n", stderr); exitUsage() }
        let nList: [Int]
        if let s = scanSpec {
            nList = s.split(separator: ",").compactMap { Int($0) }.filter { $0 > 0 }
            guard !nList.isEmpty else { fputs("bad --scan '\(s)'\n", stderr); exit(2) }
        } else if let n = nSingle {
            nList = [n]
        } else {
            nList = [16]
        }

        print("loading model from \(modelPath)…")
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        guard case .fromScratch(let model) = load.model else {
            fputs("bon first-cut targets from-scratch byte-level models.\n", stderr); exit(2)
        }
        let cfg = load.config

        let promptBytes: [Int32] = prompt.utf8.prefix(cfg.contextLength).map { Int32($0) }
        guard !promptBytes.isEmpty else { fputs("--prompt empty after byte encoding\n", stderr); exit(2) }

        // Corpus-PPL verifier prep: read the held-out text upfront.
        // Score = -PPL_change_when_completion_replaces_a_window, where
        // we measure how well the COMPLETION continues a sliding-window
        // baseline. Approximation: for each completion, prefix the
        // first 32 bytes of the corpus, then score the prompt+completion
        // tail. Higher = the completion's "shape" matches the corpus's
        // distribution. Distinct from self-likelihood because the
        // verifier model never saw the COMPLETION during sampling.
        var corpusPrefix: [Int32] = []
        if verifier == "corpus-ppl" {
            guard let cp = corpusPath else {
                fputs("--verifier corpus-ppl requires --corpus <text.txt>\n", stderr); exit(2)
            }
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: cp)) else {
                fputs("could not read --corpus \(cp)\n", stderr); exit(1)
            }
            // Take the first up-to-64 bytes of the corpus as a
            // distribution anchor. The PPL of (anchor || prompt ||
            // completion) under the model is the score; lower PPL =
            // better. Negated so higher = better matches self-mode.
            let anchor = data.prefix(64)
            corpusPrefix = anchor.map { Int32($0) }
        } else if verifier != "self" {
            fputs("--verifier must be 'self' or 'corpus-ppl' (got '\(verifier)')\n", stderr); exit(2)
        }

        struct ScanRow: Codable {
            let n: Int
            let bestScore: Float
            let meanScore: Float
            let bestText: String
            let allScores: [Float]
        }
        var rows: [ScanRow] = []

        // Sample once for the largest N, score all candidates, then for
        // each smaller N take the best-of the first N candidates. This
        // is the standard methodology — fixed sample budget, sliding
        // best-of-N window. Saves N_max forward passes vs sampling
        // fresh per scan point.
        let nMax = nList.max() ?? 16
        MLXRandom.seed(seed)
        var candidates: [(text: String, score: Float)] = []
        for k in 0..<nMax {
            let bytes = generateBytes(model: model, cfg: cfg,
                                       promptBytes: promptBytes,
                                       maxTokens: maxTokens,
                                       temperature: temperature)
            let score: Float
            if verifier == "corpus-ppl" {
                score = scoreCorpusPPL(model: model, cfg: cfg,
                                        anchorBytes: corpusPrefix,
                                        promptBytes: promptBytes,
                                        completion: bytes)
            } else {
                score = scoreSelfLikelihood(model: model, cfg: cfg,
                                             promptBytes: promptBytes,
                                             completion: bytes)
            }
            let text = String(bytes: bytes.map { UInt8($0) },
                               encoding: .utf8) ?? "<non-utf8>"
            candidates.append((text: text, score: score))
            if (k + 1) % max(1, nMax / 8) == 0 {
                fputs("  sampled \(k + 1)/\(nMax) candidates (score=\(String(format: "%.3f", score)))\n", stderr)
            }
        }

        for n in nList {
            let window = Array(candidates.prefix(n))
            let scores = window.map { $0.score }
            let best = window.max(by: { $0.score < $1.score })!
            let mean = scores.reduce(0, +) / Float(scores.count)
            rows.append(ScanRow(n: n, bestScore: best.score, meanScore: mean,
                                  bestText: best.text, allScores: scores))
        }

        print("\nN     mean-score   best-score   best text (first 60 chars)")
        print("---   ----------   ----------   ----------------------------")
        for r in rows {
            let preview = String(r.bestText.prefix(60))
            print(String(format: "%-5d %10.4f   %10.4f   %@", r.n, r.meanScore, r.bestScore, preview))
        }

        if let path = outPath {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            do {
                let data = try enc.encode([
                    "prompt": AnyEncodable(prompt),
                    "tokens": AnyEncodable(maxTokens),
                    "temperature": AnyEncodable(temperature),
                    "seed": AnyEncodable(seed),
                    "rows": AnyEncodable(rows),
                ])
                try data.write(to: URL(fileURLWithPath: path), options: .atomic)
                print("\nwrote curve → \(path)")
            } catch {
                fputs("--out write failed: \(error)\n", stderr); exit(1)
            }
        }
    }

    /// One sampling pass — autoregressive byte-by-byte at the given
    /// temperature. Returns the GENERATED bytes (not the prompt).
    private static func generateBytes(model: TinyGPTModel, cfg: ModelConfig,
                                        promptBytes: [Int32],
                                        maxTokens: Int,
                                        temperature: Float) -> [Int32] {
        let promptArr = MLXArray(promptBytes, [1, promptBytes.count])
        let result = model.generate(prompt: promptArr, maxNewTokens: maxTokens,
                                      temperature: temperature)
        MLX.eval(result)
        let allFlat = result[0, 0...].asArray(Int32.self)
        return Array(allFlat.dropFirst(promptBytes.count))
    }

    /// Self-likelihood verifier: average per-token log-probability of
    /// the (prompt + completion) sequence, restricted to the COMPLETION
    /// positions. Higher = the model considers this continuation more
    /// likely under its own distribution. For best-of-N this rewards
    /// "the model's confident pick" out of N temperature-driven draws.
    private static func scoreSelfLikelihood(model: TinyGPTModel, cfg: ModelConfig,
                                              promptBytes: [Int32],
                                              completion: [Int32]) -> Float {
        let full = promptBytes + completion
        let T = min(full.count, cfg.contextLength)
        let idx = MLXArray(Array(full.prefix(T)), [1, T])
        let logits = model(idx)
        MLX.eval(logits)
        // For each position p in [promptLen, T-1], the prediction for
        // token (p+1) is at logits[0, p, :]. The target is full[p+1].
        let vocab = cfg.vocabSize
        let logitsFlat = logits.asArray(Float.self)
        let promptLen = promptBytes.count
        var sumLogProb: Float = 0
        var count = 0
        for p in promptLen..<(T - 1) {
            let base = p * vocab
            // Compute log_softmax(logits[p, :]) at target = full[p+1].
            var maxLogit: Float = -Float.greatestFiniteMagnitude
            for v in 0..<vocab { if logitsFlat[base + v] > maxLogit { maxLogit = logitsFlat[base + v] } }
            var sumExp: Float = 0
            for v in 0..<vocab { sumExp += expf(logitsFlat[base + v] - maxLogit) }
            let logZ = maxLogit + logf(sumExp)
            let target = Int(full[p + 1])
            if target >= 0 && target < vocab {
                let lp = logitsFlat[base + target] - logZ
                sumLogProb += lp
                count += 1
            }
        }
        return count > 0 ? sumLogProb / Float(count) : -Float.infinity
    }

    /// Corpus-PPL verifier: score = -mean PPL of (anchor || prompt ||
    /// completion) under the model, restricted to the COMPLETION
    /// positions. The anchor (a held-out corpus prefix) sits ahead of
    /// the prompt as a distribution conditioner — by feeding the
    /// completion through a context that includes a clean piece of
    /// corpus, we get a score that's less self-referential than pure
    /// self-likelihood. Distinct verifier signal at the cost of one
    /// extra forward pass per candidate.
    private static func scoreCorpusPPL(model: TinyGPTModel, cfg: ModelConfig,
                                         anchorBytes: [Int32],
                                         promptBytes: [Int32],
                                         completion: [Int32]) -> Float {
        let full = anchorBytes + promptBytes + completion
        let T = min(full.count, cfg.contextLength)
        let idx = MLXArray(Array(full.prefix(T)), [1, T])
        let logits = model(idx)
        MLX.eval(logits)
        let vocab = cfg.vocabSize
        let logitsFlat = logits.asArray(Float.self)
        let scoredStart = anchorBytes.count + promptBytes.count
        var sumLogProb: Float = 0
        var count = 0
        for p in (scoredStart - 1)..<(T - 1) where p >= 0 {
            let base = p * vocab
            var maxLogit: Float = -Float.greatestFiniteMagnitude
            for v in 0..<vocab { if logitsFlat[base + v] > maxLogit { maxLogit = logitsFlat[base + v] } }
            var sumExp: Float = 0
            for v in 0..<vocab { sumExp += expf(logitsFlat[base + v] - maxLogit) }
            let logZ = maxLogit + logf(sumExp)
            let target = Int(full[p + 1])
            if target >= 0 && target < vocab {
                sumLogProb += logitsFlat[base + target] - logZ
                count += 1
            }
        }
        return count > 0 ? sumLogProb / Float(count) : -Float.infinity
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt bon <model.tinygpt> [options]

        Best-of-N sampling with self-likelihood verifier. Implements the
        Tier-5 §5.2 test-time-compute-scaling methodology (Snell et al.
        2024) at byte-level scale: sample N candidates, score each by
        its own avg log-prob, return the best. `--scan` produces the
        quality-vs-compute curve as JSON.

        --prompt "..."         prompt to continue (default "ROMEO:")
        --tokens N             how many new tokens per completion (default 30)
        --temperature F        sampling temperature (default 0.8 — needs >0 for diversity)
        --n N                  single best-of-N (default 16 when --scan absent)
        --scan N1,N2,N3,...    quality-vs-compute scan (e.g. 1,2,4,8,16,32)
        --seed S               RNG seed (default 42)
        --out <curve.json>     persist the scan rows + best texts to JSON
        """)
        exit(code)
    }
}

// Small AnyEncodable shim so we can dump a heterogeneous dict to JSON
// without committing to a typed schema (the scan output is meant for
// downstream plotting / spreadsheets, not for typed re-loading).
private struct AnyEncodable: Encodable {
    let value: Any
    init(_ value: Any) { self.value = value }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch value {
        case let v as Int:    try c.encode(v)
        case let v as Float:  try c.encode(v)
        case let v as Double: try c.encode(v)
        case let v as String: try c.encode(v)
        case let v as Bool:   try c.encode(v)
        case let v as Encodable:
            try v.encode(to: encoder)
        default:
            try c.encode("\(value)")
        }
    }
}
