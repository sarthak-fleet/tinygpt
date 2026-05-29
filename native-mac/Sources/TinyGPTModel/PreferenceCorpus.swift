import Foundation
import MLX

/// A preference triplet used by DPO: same prompt, two completions, the
/// `chosen` is what the model should prefer over the `rejected`.
///
/// Common public sources:
///   - HuggingFaceH4/ultrafeedback_binarized  (~60K pairs from GPT-4 judgments)
///   - anthropic/hh-rlhf                       (human-labeled helpful/harmless)
///   - argilla/dpo-mix-7k                      (cleaned mix)
///
/// All publish JSONL with `prompt`, `chosen`, `rejected` fields; readers
/// that hit other schemas (e.g., conversation arrays) extract the
/// last-turn pair and warn.
public struct PreferenceRecord: Sendable {
    public let prompt: String
    public let chosen: String
    public let rejected: String
}

/// Tokenized, ready-to-batch preference example. The DPO loss scores
/// the policy and reference models' average log-probability over the
/// `responseMask=1` positions of the two completions and back-propagates
/// through the policy adapter only — same masking convention as
/// `SFTExample` so the same helpers can be re-used.
public struct PreferenceExample: Sendable {
    public let chosenTokens: [Int32]
    public let chosenMask: [Bool]
    public let rejectedTokens: [Int32]
    public let rejectedMask: [Bool]
    public init(chosenTokens: [Int32], chosenMask: [Bool],
                rejectedTokens: [Int32], rejectedMask: [Bool]) {
        precondition(chosenTokens.count == chosenMask.count, "chosen len mismatch")
        precondition(rejectedTokens.count == rejectedMask.count, "rejected len mismatch")
        self.chosenTokens = chosenTokens
        self.chosenMask = chosenMask
        self.rejectedTokens = rejectedTokens
        self.rejectedMask = rejectedMask
    }
}

/// Corpus of preference examples — drop-in shape for the DPO trainer's
/// batch sampler.
public final class PreferenceCorpus: Sendable {
    public let examples: [PreferenceExample]
    public let vocabSize: Int
    public init(_ examples: [PreferenceExample], vocabSize: Int) {
        precondition(!examples.isEmpty, "PreferenceCorpus needs ≥1 example")
        self.examples = examples
        self.vocabSize = vocabSize
    }

    /// Sample a batch with each side flattened to `[B, T]` plus a float
    /// mask `[B, T]` flagging the response positions. The two sides come
    /// out the same shape so a single tensor pair makes both forward
    /// passes downstream.
    public func sampleBatch(batchSize B: Int, contextLength T: Int)
        -> (chosen: (MLXArray, MLXArray, MLXArray),
            rejected: (MLXArray, MLXArray, MLXArray))
    {
        var cIn = [Int32](repeating: 0, count: B * T)
        var cTg = [Int32](repeating: 0, count: B * T)
        var cMs = [Float](repeating: 0, count: B * T)
        var rIn = [Int32](repeating: 0, count: B * T)
        var rTg = [Int32](repeating: 0, count: B * T)
        var rMs = [Float](repeating: 0, count: B * T)
        for i in 0..<B {
            let ex = examples[Int.random(in: 0..<examples.count)]
            fill(rowStart: i * T, tokens: ex.chosenTokens, mask: ex.chosenMask,
                 T: T, ins: &cIn, tgs: &cTg, msk: &cMs)
            fill(rowStart: i * T, tokens: ex.rejectedTokens, mask: ex.rejectedMask,
                 T: T, ins: &rIn, tgs: &rTg, msk: &rMs)
        }
        return (
            (MLXArray(cIn, [B, T]), MLXArray(cTg, [B, T]), MLXArray(cMs, [B, T])),
            (MLXArray(rIn, [B, T]), MLXArray(rTg, [B, T]), MLXArray(rMs, [B, T]))
        )
    }

    private func fill(rowStart: Int, tokens: [Int32], mask: [Bool], T: Int,
                       ins: inout [Int32], tgs: inout [Int32], msk: inout [Float]) {
        let seq = Array(tokens.prefix(T + 1))
        let mk = Array(mask.prefix(T + 1))
        let usable = min(seq.count - 1, T)
        for j in 0..<usable {
            ins[rowStart + j] = seq[j]
            tgs[rowStart + j] = seq[j + 1]
            msk[rowStart + j] = mk[j + 1] ? 1.0 : 0.0
        }
    }
}

public enum PreferenceReader {
    public enum ReadError: Error, CustomStringConvertible {
        case ioError(String)
        case parseError(line: Int, detail: String)
        public var description: String {
            switch self {
            case .ioError(let s): return "could not read preference file: \(s)"
            case .parseError(let l, let d): return "preference parse error at line \(l): \(d)"
            }
        }
    }

    /// Read a JSONL file. Each line: `{prompt, chosen, rejected}` (the
    /// ultrafeedback / Argilla shape) OR `{chosen: [{role, content}…], rejected: [...]}`
    /// (the chat-array shape used by some HF datasets — the reader walks
    /// the array and joins by role).
    public static func readJSONL(_ url: URL) throws -> [PreferenceRecord] {
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw ReadError.ioError("\(error)") }
        guard let text = String(data: data, encoding: .utf8) else {
            throw ReadError.ioError("file isn't UTF-8")
        }
        var records: [PreferenceRecord] = []
        var lineNo = 0
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            lineNo += 1
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            guard let lineData = line.data(using: .utf8) else { continue }
            let obj: [String: Any]
            do { obj = (try JSONSerialization.jsonObject(with: lineData)) as? [String: Any] ?? [:] }
            catch { throw ReadError.parseError(line: lineNo, detail: "\(error)") }

            let prompt = (obj["prompt"] as? String) ?? ""
            // Plain string form.
            if let chosen = obj["chosen"] as? String,
               let rejected = obj["rejected"] as? String,
               !prompt.isEmpty {
                records.append(PreferenceRecord(prompt: prompt, chosen: chosen, rejected: rejected))
                continue
            }
            // Chat-array form. Walk the messages and take the last assistant
            // turn as the response; everything before is the "prompt".
            if let chosenArr = obj["chosen"] as? [[String: Any]],
               let rejectedArr = obj["rejected"] as? [[String: Any]] {
                let (chPrompt, chResponse) = extractLastAssistant(chosenArr)
                let (_,        reResponse) = extractLastAssistant(rejectedArr)
                if !chPrompt.isEmpty && !chResponse.isEmpty && !reResponse.isEmpty {
                    records.append(PreferenceRecord(prompt: chPrompt, chosen: chResponse, rejected: reResponse))
                    continue
                }
            }
        }
        return records
    }

    /// Walk the messages, join all non-final-assistant turns into a single
    /// `prompt` string (with ChatML-style markers), and return the last
    /// assistant turn as the `response`.
    private static func extractLastAssistant(_ messages: [[String: Any]]) -> (prompt: String, response: String) {
        var prefix: [String] = []
        var response = ""
        for (i, msg) in messages.enumerated() {
            let role = (msg["role"] as? String) ?? ""
            let content = (msg["content"] as? String) ?? ""
            if role == "assistant" && i == messages.count - 1 {
                response = content
            } else if !role.isEmpty {
                prefix.append("<|im_start|>\(role)\n\(content)<|im_end|>")
            }
        }
        return (prefix.joined(separator: "\n"), response)
    }
}

/// Build a PreferenceExample from a PreferenceRecord using a prompt
/// template. Tokenization mirrors `SFTBuilder.buildExample`:
///   - Render `(prompt, "", chosen)` → full text + token-space response-start
///   - Render `(prompt, "", rejected)` → same
///   - Truncate to maxSeqLen
public enum PreferenceBuilder {
    public static func buildExample(
        record: PreferenceRecord, template: PromptTemplate, tokenizer: HFTokenizer,
        maxSeqLen: Int
    ) throws -> PreferenceExample {
        let (chosenFull, _)   = template.render(instruction: record.prompt, input: "", response: record.chosen)
        let (rejectedFull, _) = template.render(instruction: record.prompt, input: "", response: record.rejected)
        let (preface, _)      = template.render(instruction: record.prompt, input: "", response: "")
        let prefaceIds = try tokenizer.encode(preface).map { Int32($0) }
        let chosenIds  = try tokenizer.encode(chosenFull).map { Int32($0) }
        let rejectedIds = try tokenizer.encode(rejectedFull).map { Int32($0) }
        let responseStart = min(prefaceIds.count, min(chosenIds.count, rejectedIds.count))
        let chosenMask  = (0..<chosenIds.count).map  { i in i >= responseStart }
        let rejectedMask = (0..<rejectedIds.count).map { i in i >= responseStart }
        return PreferenceExample(
            chosenTokens:  Array(chosenIds.prefix(maxSeqLen)),
            chosenMask:    Array(chosenMask.prefix(maxSeqLen)),
            rejectedTokens: Array(rejectedIds.prefix(maxSeqLen)),
            rejectedMask:   Array(rejectedMask.prefix(maxSeqLen))
        )
    }
}
