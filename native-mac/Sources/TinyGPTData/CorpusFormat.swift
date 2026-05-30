import Foundation

/// CorpusFormat — convert HF dataset rows into tinygpt's training formats.
///
/// Three target formats:
///   - .sft    → JSONL with `{instruction, input?, response}` (per
///               SFTReader in TinyGPTModel/SFTCorpus.swift line 239)
///   - .dpo    → JSONL with `{prompt, chosen, rejected}` (per
///               PreferenceCorpus in TinyGPTModel)
///   - .plain  → one example per line, plain text (pretraining)
///
/// HF datasets land here as Swift dictionaries (already JSON-decoded from
/// JSONL/parquet/arrow by the row reader). This file maps a row to a
/// concrete record — and the adapter sniffs at the first row to pick a
/// format when the user doesn't pass --format.
public enum CorpusFormat: String, Sendable, CaseIterable {
    case sft, dpo, plain

    public init?(parsing s: String) {
        switch s.lowercased() {
        case "sft", "instruct", "instruction": self = .sft
        case "dpo", "simpo", "preference", "rlhf": self = .dpo
        case "plain", "text", "raw", "pretrain": self = .plain
        default: return nil
        }
    }
}

/// A single output record. The variant matches the user's target format.
public enum CorpusRecord: Sendable {
    case sft(instruction: String, input: String, response: String)
    case dpo(prompt: String, chosen: String, rejected: String)
    case plain(text: String)
}

/// User-controlled field aliasing. When auto-detection picks the wrong
/// columns (or the dataset has weird names), the user can pass
/// `--map src:dst,src2:dst2` to override.
public struct FieldMap: Sendable {
    public var mapping: [String: String]   // dataset-field → canonical-field
    public init(_ mapping: [String: String] = [:]) { self.mapping = mapping }

    /// Parse a comma-separated `src:dst,src2:dst2` string.
    public static func parse(_ raw: String) -> FieldMap {
        var out: [String: String] = [:]
        for pair in raw.split(separator: ",") {
            let kv = pair.split(separator: ":", maxSplits: 1).map { String($0).trimmingCharacters(in: .whitespaces) }
            if kv.count == 2 { out[kv[0]] = kv[1] }
        }
        return FieldMap(out)
    }
}

/// Heuristic schema detector. Looks at one or more sample rows and
/// returns the best-guess format + the field map needed to extract it.
public enum FormatDetector {
    /// Common synonyms for each canonical SFT/DPO/plain field. The first
    /// hit wins, so order matters (most specific first).
    public static let sftInstructionAliases = ["instruction", "prompt", "question", "query", "input_text"]
    public static let sftInputAliases       = ["input", "context", "background"]
    public static let sftResponseAliases    = ["response", "completion", "answer", "output", "target", "label", "responses"]
    public static let dpoPromptAliases      = ["prompt", "question", "instruction", "query"]
    public static let dpoChosenAliases      = ["chosen", "chosen_response", "preferred", "win", "winner"]
    public static let dpoRejectedAliases    = ["rejected", "rejected_response", "dispreferred", "loss", "loser"]
    public static let plainAliases          = ["text", "content", "document", "raw_text", "body"]

    public struct Detection: Sendable {
        public let format: CorpusFormat
        public let map: [String: String]     // canonical → dataset-field
        public let confidence: Double         // 0..1
        public let rationale: String
    }

    public static func detect(sampleRow: [String: Any], userMap: FieldMap = FieldMap()) -> Detection {
        let keys = Set(sampleRow.keys.map { $0.lowercased() })
        let lcRow: [String: Any] = Dictionary(uniqueKeysWithValues: sampleRow.map { ($0.key.lowercased(), $0.value) })

        // Apply user overrides first — if they pass --map prompt:instruction
        // we treat "prompt" as if it were "instruction".
        let aliasedKeys: Set<String> = {
            var out = keys
            for (src, dst) in userMap.mapping { if keys.contains(src.lowercased()) { out.insert(dst.lowercased()) } }
            return out
        }()
        func userAlias(_ canonical: String) -> String? {
            // Reverse lookup: did the user remap something INTO this canonical?
            userMap.mapping.first(where: { $0.value.lowercased() == canonical })?.key
        }

        // DPO check first — `chosen` + `rejected` is unambiguous.
        let chosenKey = firstMatch(in: aliasedKeys, against: Self.dpoChosenAliases) ?? userAlias("chosen")
        let rejectedKey = firstMatch(in: aliasedKeys, against: Self.dpoRejectedAliases) ?? userAlias("rejected")
        if let _ = chosenKey, let _ = rejectedKey {
            let promptKey = firstMatch(in: aliasedKeys, against: Self.dpoPromptAliases) ?? userAlias("prompt")
            // Even if prompt is missing, ultrafeedback-style rows hide it
            // inside `chosen[].content` — flag confidence accordingly.
            let conf = promptKey != nil ? 0.95 : 0.70
            var map: [String: String] = [:]
            if let p = promptKey { map["prompt"] = sourceKey(in: sampleRow, alias: p) }
            if let c = chosenKey { map["chosen"] = sourceKey(in: sampleRow, alias: c) }
            if let r = rejectedKey { map["rejected"] = sourceKey(in: sampleRow, alias: r) }
            return Detection(format: .dpo, map: map, confidence: conf,
                             rationale: "found chosen+rejected fields → DPO")
        }

        // SFT check — instruction-ish + response-ish.
        let instructionKey = firstMatch(in: aliasedKeys, against: Self.sftInstructionAliases) ?? userAlias("instruction")
        let responseKey = firstMatch(in: aliasedKeys, against: Self.sftResponseAliases) ?? userAlias("response")
        if instructionKey != nil && responseKey != nil {
            let inputKey = firstMatch(in: aliasedKeys, against: Self.sftInputAliases) ?? userAlias("input")
            var map: [String: String] = [:]
            if let i = instructionKey { map["instruction"] = sourceKey(in: sampleRow, alias: i) }
            if let i = inputKey        { map["input"]       = sourceKey(in: sampleRow, alias: i) }
            if let r = responseKey     { map["response"]    = sourceKey(in: sampleRow, alias: r) }
            return Detection(format: .sft, map: map, confidence: 0.90,
                             rationale: "found \(instructionKey!)+\(responseKey!) → SFT")
        }

        // ShareGPT-style multi-turn: `conversations: [{from, value}]`
        // or `messages: [{role, content}]`. We coerce to SFT by taking
        // the last user turn as instruction and the last assistant as response.
        if lcRow["conversations"] is [[String: Any]] || lcRow["messages"] is [[String: Any]] {
            var map: [String: String] = [:]
            if lcRow["conversations"] is [[String: Any]] {
                map["__chat_array"] = sourceKey(in: sampleRow, alias: "conversations") ?? "conversations"
            } else {
                map["__chat_array"] = sourceKey(in: sampleRow, alias: "messages") ?? "messages"
            }
            return Detection(format: .sft, map: map, confidence: 0.75,
                             rationale: "chat array field → SFT (last user / last assistant)")
        }

        // Plain text fallback.
        let plainKey = firstMatch(in: aliasedKeys, against: Self.plainAliases) ?? userAlias("text")
        if let pk = plainKey {
            return Detection(format: .plain, map: ["text": sourceKey(in: sampleRow, alias: pk) ?? pk],
                             confidence: 0.85, rationale: "single \(pk) field → plain text")
        }

        // Last-ditch: if there's exactly one string-valued field, use it as plain text.
        let stringFields = sampleRow.filter { $0.value is String }
        if stringFields.count == 1 {
            let only = stringFields.first!.key
            return Detection(format: .plain, map: ["text": only], confidence: 0.55,
                             rationale: "single string-valued field '\(only)' → plain text")
        }

        return Detection(format: .plain, map: [:], confidence: 0.0,
                         rationale: "could not detect schema; pass --format and --map")
    }

    /// Find the first dataset-field key whose lowercased name matches any
    /// of the given aliases.
    private static func firstMatch(in keys: Set<String>, against aliases: [String]) -> String? {
        for a in aliases where keys.contains(a) { return a }
        return nil
    }

    /// Recover the actual case-preserving key from a row given a lowercased alias.
    private static func sourceKey(in row: [String: Any], alias: String) -> String? {
        row.keys.first(where: { $0.lowercased() == alias.lowercased() })
    }
}

/// Convert a single dataset row + a detected schema into a CorpusRecord.
/// Returns nil for rows that don't have valid content (missing fields,
/// empty strings, etc.) — caller should skip these.
public enum CorpusConverter {
    public static func convert(row: [String: Any], format: CorpusFormat, map: [String: String]) -> CorpusRecord? {
        switch format {
        case .sft:
            // Chat-array path.
            if let chatKey = map["__chat_array"], let arr = row[chatKey] as? [[String: Any]] {
                let (instruction, response) = lastTurnsFromChat(arr)
                if instruction.isEmpty || response.isEmpty { return nil }
                return .sft(instruction: instruction, input: "", response: response)
            }
            let instruction = string(row, map["instruction"])
            let input       = string(row, map["input"])
            let response    = string(row, map["response"])
            if instruction.isEmpty && response.isEmpty { return nil }
            return .sft(instruction: instruction, input: input, response: response)

        case .dpo:
            var prompt = string(row, map["prompt"])
            var chosen = stringOrChat(row, map["chosen"])
            var rejected = stringOrChat(row, map["rejected"])
            // Chat-array DPO (ultrafeedback): chosen/rejected are
            // `[{role, content}]` arrays — extract the last assistant turn.
            if let arr = row[map["chosen"] ?? ""] as? [[String: Any]] {
                let (p, r) = lastTurnsFromChat(arr)
                if prompt.isEmpty { prompt = p }
                chosen = r
            }
            if let arr = row[map["rejected"] ?? ""] as? [[String: Any]] {
                let (_, r) = lastTurnsFromChat(arr)
                rejected = r
            }
            if prompt.isEmpty || chosen.isEmpty || rejected.isEmpty { return nil }
            return .dpo(prompt: prompt, chosen: chosen, rejected: rejected)

        case .plain:
            let text = string(row, map["text"])
            if text.isEmpty { return nil }
            return .plain(text: text)
        }
    }

    /// Walk a chat array (`[{role, content}]` or ShareGPT `[{from, value}]`)
    /// and return (joinedPrior, lastAssistant). For SFT we use the joined
    /// prior as the instruction and the last assistant as the response.
    private static func lastTurnsFromChat(_ messages: [[String: Any]]) -> (String, String) {
        var prefix: [String] = []
        var lastAssistant = ""
        for (i, msg) in messages.enumerated() {
            let role = (msg["role"] as? String)
                ?? (msg["from"] as? String).map { mapShareGPTRole($0) }
                ?? ""
            let content = (msg["content"] as? String) ?? (msg["value"] as? String) ?? ""
            if role == "assistant" && i == messages.count - 1 {
                lastAssistant = content
            } else {
                prefix.append("\(role): \(content)")
            }
        }
        return (prefix.joined(separator: "\n"), lastAssistant)
    }

    private static func mapShareGPTRole(_ from: String) -> String {
        switch from.lowercased() {
        case "human", "user": return "user"
        case "gpt", "assistant", "chatgpt": return "assistant"
        case "system": return "system"
        default: return from
        }
    }

    /// Extract a string from a row by key. JSON values that aren't strings
    /// get JSON-encoded back to a string (e.g. arrays of tool calls in
    /// function-calling datasets — we serialise them so downstream
    /// trainers see a deterministic representation).
    private static func string(_ row: [String: Any], _ key: String?) -> String {
        guard let key = key, let v = row[key] else { return "" }
        if let s = v as? String { return s }
        if v is NSNull { return "" }
        if let data = try? JSONSerialization.data(withJSONObject: v, options: [.sortedKeys]) {
            return String(data: data, encoding: .utf8) ?? ""
        }
        return "\(v)"
    }

    private static func stringOrChat(_ row: [String: Any], _ key: String?) -> String {
        guard let key = key, let v = row[key] else { return "" }
        if let s = v as? String { return s }
        if let arr = v as? [[String: Any]] {
            let (_, last) = lastTurnsFromChat(arr)
            return last
        }
        return string(row, key)
    }
}

/// JSONL writer for the three target formats. Append-mode so multi-shard
/// downloads can stream rows into a single output file.
public final class JSONLWriter {
    public let url: URL
    private let handle: FileHandle
    private var count: Int = 0

    public init(url: URL, append: Bool = false) throws {
        self.url = url
        let fm = FileManager.default
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !append || !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil)
        }
        guard let h = try? FileHandle(forWritingTo: url) else {
            throw HFDatasets.HFError.ioError("could not open \(url.path) for writing")
        }
        self.handle = h
        if append {
            try h.seekToEnd()
        }
    }

    public func write(_ record: CorpusRecord) throws {
        let obj: [String: Any]
        switch record {
        case .sft(let instruction, let input, let response):
            var d: [String: Any] = ["instruction": instruction, "response": response]
            if !input.isEmpty { d["input"] = input }
            obj = d
        case .dpo(let prompt, let chosen, let rejected):
            obj = ["prompt": prompt, "chosen": chosen, "rejected": rejected]
        case .plain(let text):
            obj = ["text": text]
        }
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes])
        handle.write(data)
        handle.write(Data([0x0a])) // \n
        count += 1
    }

    /// Plain-text variant — one example per line, no JSON wrapper. Only
    /// usable for `.plain` records (others throw).
    public func writePlainLine(_ text: String) throws {
        let stripped = text.replacingOccurrences(of: "\n", with: " ")
        handle.write(Data(stripped.utf8))
        handle.write(Data([0x0a]))
        count += 1
    }

    public func close() {
        try? handle.close()
    }

    public var recordCount: Int { count }
}
