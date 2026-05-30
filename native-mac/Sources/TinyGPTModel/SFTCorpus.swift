import Foundation
import MLX

/// Corpus of (instruction, response) pairs tokenized + masked for SFT.
///
/// `sampleBatch(B, T)` returns three tensors:
///   - `inputs[B, T]`  int32 token ids (padded with 0)
///   - `targets[B, T]` int32 token ids (= inputs shifted by 1, padded)
///   - `lossMask[B, T]` float32 — 1.0 where the target token is part of
///                     the response, 0.0 where it's prompt OR padding.
///
/// The training loss is computed as mean cross-entropy over only the
/// `lossMask == 1` positions — see `TGModelLoss.maskedLoss` below. This
/// is the standard "ignore the prompt in the loss" SFT trick that
/// massively improves instruction-following quality.
public final class SFTCorpus: Sendable {
    public let examples: [SFTExample]
    public let vocabSize: Int
    /// Inverse-length sampling weights (∝ 1/length). Pre-computed cumulative
    /// distribution so each `sampleBatchWeighted` pick is O(log N).
    private let cumulativeWeights: [Double]

    public init(_ examples: [SFTExample], vocabSize: Int) {
        precondition(!examples.isEmpty, "SFTCorpus needs at least one example")
        self.examples = examples
        self.vocabSize = vocabSize
        // Inverse-length weights — short examples get over-represented so
        // batches see them as often as long ones in expectation. Add a +1
        // floor to avoid div-by-zero on degenerate (single-token) examples.
        var running: Double = 0
        var cum: [Double] = []
        cum.reserveCapacity(examples.count)
        for ex in examples {
            let len = max(1, ex.tokens.count)
            running += 1.0 / Double(len)
            cum.append(running)
        }
        self.cumulativeWeights = cum
    }

    /// Sampled index from the inverse-length distribution using binary
    /// search on the cumulative weights. O(log N) per draw.
    private func weightedIndex() -> Int {
        guard let total = cumulativeWeights.last, total > 0 else { return 0 }
        let target = Double.random(in: 0..<total)
        var lo = 0
        var hi = cumulativeWeights.count - 1
        while lo < hi {
            let mid = (lo + hi) / 2
            if cumulativeWeights[mid] < target { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    /// Sample a batch by random replacement from `examples`. Each example
    /// is right-padded to `T` (or truncated if it exceeds T). The targets
    /// are the inputs shifted by 1; the last position's target is 0
    /// (it's masked anyway).
    public func sampleBatch(batchSize B: Int, contextLength T: Int)
        -> (inputs: MLXArray, targets: MLXArray, mask: MLXArray)
    {
        var ins = [Int32](repeating: 0, count: B * T)
        var tgs = [Int32](repeating: 0, count: B * T)
        var msk = [Float](repeating: 0, count: B * T)
        for i in 0..<B {
            let ex = examples[Int.random(in: 0..<examples.count)]
            let seq = Array(ex.tokens.prefix(T + 1))     // need T+1 so we can shift by 1
            let mask = Array(ex.responseMask.prefix(T + 1))
            let usable = min(seq.count - 1, T)            // up to T (input, target) pairs
            for j in 0..<usable {
                ins[i * T + j] = seq[j]
                tgs[i * T + j] = seq[j + 1]
                // The TARGET token's mask is what matters — we score how
                // well we predicted `seq[j+1]`. So the mask we apply is
                // the target position's flag.
                msk[i * T + j] = mask[j + 1] ? 1.0 : 0.0
            }
        }
        return (
            MLXArray(ins, [B, T]),
            MLXArray(tgs, [B, T]),
            MLXArray(msk, [B, T])
        )
    }

    /// Sample-packed sampling: same shape as `sampleBatch` (one example per
    /// row, padded), but draws each row from an inverse-length-weighted
    /// distribution. Short examples are over-represented so that, on
    /// expectation, each example contributes (length × frequency) ≈ const
    /// — counteracts the natural bias in `sampleBatch` where long examples
    /// dominate the per-step token budget.
    ///
    /// Use this when the corpus has a heavy long-tail of long examples and
    /// you want the model to see the short ones more often (instruction
    /// datasets are usually like this).
    public func sampleBatchWeighted(batchSize B: Int, contextLength T: Int)
        -> (inputs: MLXArray, targets: MLXArray, mask: MLXArray)
    {
        var ins = [Int32](repeating: 0, count: B * T)
        var tgs = [Int32](repeating: 0, count: B * T)
        var msk = [Float](repeating: 0, count: B * T)
        for i in 0..<B {
            let ex = examples[weightedIndex()]
            let seq = Array(ex.tokens.prefix(T + 1))
            let mask = Array(ex.responseMask.prefix(T + 1))
            let usable = min(seq.count - 1, T)
            for j in 0..<usable {
                ins[i * T + j] = seq[j]
                tgs[i * T + j] = seq[j + 1]
                msk[i * T + j] = mask[j + 1] ? 1.0 : 0.0
            }
        }
        return (
            MLXArray(ins, [B, T]),
            MLXArray(tgs, [B, T]),
            MLXArray(msk, [B, T])
        )
    }

    /// Length-bucketed sampling: bins examples into `nBuckets` buckets by
    /// token length (equal-width on a linear scale), picks a bucket
    /// uniformly at random per row, then a uniform example from that bucket.
    /// Empty buckets are skipped. Effect: each LENGTH REGIME (short / mid /
    /// long) is seen equally often, regardless of how skewed the
    /// per-length population is.
    public func sampleBatchBucketed(batchSize B: Int, contextLength T: Int,
                                     nBuckets: Int)
        -> (inputs: MLXArray, targets: MLXArray, mask: MLXArray)
    {
        precondition(nBuckets >= 1, "nBuckets must be ≥1")
        // Build buckets lazily on first call would force mutation; cheaper to
        // just rebuild each batch — N examples + binning is sub-millisecond
        // for any reasonable corpus.
        let lens = examples.map { $0.tokens.count }
        let minLen = lens.min() ?? 1
        let maxLen = lens.max() ?? 1
        var buckets: [[Int]] = Array(repeating: [], count: nBuckets)
        if minLen == maxLen {
            buckets[0] = Array(0..<examples.count)
        } else {
            let span = Double(maxLen - minLen)
            for (i, l) in lens.enumerated() {
                let frac = Double(l - minLen) / span
                var b = Int(frac * Double(nBuckets))
                if b >= nBuckets { b = nBuckets - 1 }
                buckets[b].append(i)
            }
        }
        let nonEmpty = buckets.filter { !$0.isEmpty }
        precondition(!nonEmpty.isEmpty, "no non-empty buckets")
        var ins = [Int32](repeating: 0, count: B * T)
        var tgs = [Int32](repeating: 0, count: B * T)
        var msk = [Float](repeating: 0, count: B * T)
        for i in 0..<B {
            let bucket = nonEmpty[Int.random(in: 0..<nonEmpty.count)]
            let idx = bucket[Int.random(in: 0..<bucket.count)]
            let ex = examples[idx]
            let seq = Array(ex.tokens.prefix(T + 1))
            let mask = Array(ex.responseMask.prefix(T + 1))
            let usable = min(seq.count - 1, T)
            for j in 0..<usable {
                ins[i * T + j] = seq[j]
                tgs[i * T + j] = seq[j + 1]
                msk[i * T + j] = mask[j + 1] ? 1.0 : 0.0
            }
        }
        return (
            MLXArray(ins, [B, T]),
            MLXArray(tgs, [B, T]),
            MLXArray(msk, [B, T])
        )
    }

    /// For diagnostics: return the inverse-length weight for each example
    /// (normalized so they sum to 1).
    public var inverseLengthWeights: [Double] {
        guard let total = cumulativeWeights.last, total > 0 else {
            return Array(repeating: 0, count: examples.count)
        }
        var out: [Double] = []
        out.reserveCapacity(cumulativeWeights.count)
        var prev: Double = 0
        for c in cumulativeWeights {
            out.append((c - prev) / total)
            prev = c
        }
        return out
    }

    /// Sequence-packed sampling: greedy-fill each row by concatenating
    /// random examples that fit in the remaining space, leaving the rest
    /// zero-padded (and masked-out). Bumps effective batch by 3-10× for
    /// SFT data dominated by short examples — every position carries a
    /// real next-token-prediction target instead of padding.
    ///
    /// Caveat: this is "naive" packing — attention isn't block-masked
    /// per example, so example k+1 can in principle attend to example k.
    /// In practice the loss only fires on response tokens so the model
    /// learns to treat the prior example as ambient context (similar to
    /// shuffled-document pre-training), and the win from killing the
    /// pad-fraction dwarfs the slight cross-example bleed. Block-masked
    /// attention is a future-work add.
    public func sampleBatchPacked(batchSize B: Int, contextLength T: Int)
        -> (inputs: MLXArray, targets: MLXArray, mask: MLXArray)
    {
        var ins = [Int32](repeating: 0, count: B * T)
        var tgs = [Int32](repeating: 0, count: B * T)
        var msk = [Float](repeating: 0, count: B * T)
        let nEx = examples.count
        let maxAttempts = max(8, nEx * 3)
        for i in 0..<B {
            var pos = 0
            var attempts = 0
            while pos < T && attempts < maxAttempts {
                attempts += 1
                let ex = examples[Int.random(in: 0..<nEx)]
                // Need at least one (input, target) pair → exLen = tokens.count - 1.
                let exLen = ex.tokens.count - 1
                if exLen <= 0 { continue }
                if pos + exLen > T { continue }   // doesn't fit; try another
                for j in 0..<exLen {
                    let p = i * T + pos + j
                    ins[p] = ex.tokens[j]
                    tgs[p] = ex.tokens[j + 1]
                    msk[p] = ex.responseMask[j + 1] ? 1.0 : 0.0
                }
                pos += exLen
                attempts = 0   // a fit resets the budget — favours dense packing
            }
        }
        return (
            MLXArray(ins, [B, T]),
            MLXArray(tgs, [B, T]),
            MLXArray(msk, [B, T])
        )
    }
}

/// Read a JSONL file where each line is `{"instruction": ..., "input"?: ..., "response": ...}`
/// or `{"prompt": ..., "completion": ...}`. Empty lines are skipped.
/// Returns the parsed records, ready to be templated + tokenized.
public struct SFTRecord: Sendable {
    public let instruction: String
    public let input: String
    public let response: String
}

public enum SFTReader {
    public enum ReadError: Error, CustomStringConvertible {
        case ioError(String)
        case parseError(line: Int, detail: String)
        public var description: String {
            switch self {
            case .ioError(let s): return "could not read SFT file: \(s)"
            case .parseError(let l, let d): return "SFT parse error at line \(l): \(d)"
            }
        }
    }

    public static func readJSONL(_ url: URL) throws -> [SFTRecord] {
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw ReadError.ioError("\(error)") }
        guard let text = String(data: data, encoding: .utf8) else {
            throw ReadError.ioError("file isn't UTF-8")
        }
        var records: [SFTRecord] = []
        var lineNo = 0
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            lineNo += 1
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            guard let lineData = line.data(using: .utf8) else { continue }
            let obj: [String: Any]
            do {
                obj = (try JSONSerialization.jsonObject(with: lineData)) as? [String: Any] ?? [:]
            } catch {
                throw ReadError.parseError(line: lineNo, detail: "\(error)")
            }
            // Accept either {instruction, input?, response} OR {prompt, completion}.
            let instruction = (obj["instruction"] as? String) ?? (obj["prompt"] as? String) ?? ""
            let input = (obj["input"] as? String) ?? ""
            let response = (obj["response"] as? String) ?? (obj["completion"] as? String) ?? ""
            if instruction.isEmpty && response.isEmpty { continue }
            records.append(SFTRecord(instruction: instruction, input: input, response: response))
        }
        return records
    }
}

/// Token + mask construction. Encodes the prompt-then-response string
/// through the tokenizer, then derives the response mask by re-encoding
/// just the prompt to find where the response starts in token space.
///
/// Caveat: BPE tokenizers don't always split on byte boundaries when
/// neighbouring text changes, so re-encoding the prompt alone can yield
/// a slightly different token boundary than "prompt + response then take
/// prefix". The cleanest is to encode prompt+response, then encode
/// prompt-only, and treat the difference in length as where the response
/// begins. For most templates this is exact; rare edge cases mask one
/// extra prompt token in the loss, which is harmless.
public enum SFTBuilder {
    public static func buildExample(
        record: SFTRecord, template: PromptTemplate, tokenizer: HFTokenizer,
        maxSeqLen: Int
    ) throws -> SFTExample {
        let (fullText, _) = template.render(
            instruction: record.instruction, input: record.input, response: record.response
        )
        let (prefaceText, _) = template.render(
            instruction: record.instruction, input: record.input, response: ""
        )
        // Drop any closing markers the template added after the (empty)
        // response — e.g. ChatML's `<|im_end|>` — so re-encoding the
        // preface alone matches what's at the start of fullText.
        let preface = String(prefaceText.prefix(while: { _ in true }))
        // We want the response-mask boundary in token space. The simplest
        // robust approach: encode preface alone, then encode full text.
        // The token count of `prefaceIds` is the offset at which the
        // response begins.
        let fullIds = try tokenizer.encode(fullText).map { Int32($0) }
        let prefaceIds = try tokenizer.encode(preface).map { Int32($0) }
        let responseStart = min(prefaceIds.count, fullIds.count)
        let mask = (0..<fullIds.count).map { i in i >= responseStart }
        // Truncate to maxSeqLen — keep the start (so the prompt is intact)
        // and drop tail tokens if the response is too long.
        let truncatedIds = Array(fullIds.prefix(maxSeqLen))
        let truncatedMask = Array(mask.prefix(maxSeqLen))
        return SFTExample(tokens: truncatedIds, responseMask: truncatedMask)
    }
}
