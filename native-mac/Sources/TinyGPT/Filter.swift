import Foundation

/// `tinygpt filter` — lightweight data-safety filtering for training corpora.
///
/// v1 deliberately stays dependency-free: built-in regex PII redaction plus a
/// transparent heuristic toxicity score. External toxicity models such as
/// Detoxify are not bundled here because that would add a large dependency and
/// model artifact to this learning repo.
enum Filter {
    struct Config {
        var inputPath: String?
        var outputPath: String?
        var sidecarPath: String?
        var piiTypes: Set<PIIType> = Set(PIIType.auto)
        var toxicityThreshold: Float?
        var toxicityModel = "heuristic"
        var dropPII = false
    }

    enum PIIType: String, CaseIterable {
        case email
        case phone
        case ssn
        case ip
        case creditCard = "credit-card"
        case name

        static let auto: [PIIType] = [.email, .phone, .ssn, .ip, .creditCard]
    }

    struct FilterResult {
        var text: String
        var reasons: [String]
        var toxicityScore: Float
    }

    static func run(args: [String]) {
        var cfg = Config()
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--in": cfg.inputPath = args[i + 1]; i += 2
            case "--out": cfg.outputPath = args[i + 1]; i += 2
            case "--sidecar": cfg.sidecarPath = args[i + 1]; i += 2
            case "--pii":
                cfg.piiTypes = parsePIITypes(args[i + 1])
                i += 2
            case "--drop-pii":
                cfg.dropPII = true
                i += 1
            case "--toxicity":
                cfg.toxicityThreshold = Float(args[i + 1])
                i += 2
            case "--toxicity-model":
                cfg.toxicityModel = args[i + 1].lowercased()
                i += 2
            case "-h", "--help":
                exitUsage(0)
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr)
                    exitUsage()
                }
                if cfg.inputPath == nil {
                    cfg.inputPath = args[i]
                    i += 1
                } else {
                    fputs("unexpected argument: \(args[i])\n", stderr)
                    exitUsage()
                }
            }
        }

        guard let inputPath = cfg.inputPath else { fputs("--in <path> required\n", stderr); exitUsage() }
        guard let outputPath = cfg.outputPath else { fputs("--out <path> required\n", stderr); exitUsage() }
        if cfg.toxicityModel != "heuristic" && cfg.toxicityModel != "off" {
            fputs("--toxicity-model \(cfg.toxicityModel) is not bundled; use heuristic or off\n", stderr)
            exit(2)
        }

        guard let raw = try? String(contentsOfFile: inputPath, encoding: .utf8) else {
            fputs("could not read \(inputPath)\n", stderr)
            exit(1)
        }
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)

        let outURL = URL(fileURLWithPath: outputPath)
        try? FileManager.default.removeItem(at: outURL)
        FileManager.default.createFile(atPath: outURL.path, contents: nil)
        guard let outFH = try? FileHandle(forWritingTo: outURL) else {
            fputs("could not open \(outputPath)\n", stderr)
            exit(1)
        }
        defer { try? outFH.close() }

        var sidecarFH: FileHandle?
        if let sidecarPath = cfg.sidecarPath {
            let url = URL(fileURLWithPath: sidecarPath)
            try? FileManager.default.removeItem(at: url)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            sidecarFH = try? FileHandle(forWritingTo: url)
        }
        defer { try? sidecarFH?.close() }

        var scanned = 0
        var kept = 0
        var redacted = 0
        var droppedPII = 0
        var droppedToxic = 0

        for (idx, line) in lines.enumerated() {
            if line.isEmpty && idx == lines.count - 1 { continue }
            scanned += 1
            let result = filterLine(line, cfg: cfg)
            let hasPII = result.reasons.contains { $0.hasPrefix("pii.") }
            let toxic = cfg.toxicityThreshold.map { result.toxicityScore >= $0 } ?? false
            if cfg.dropPII && hasPII {
                droppedPII += 1
                writeSidecar(sidecarFH, line: scanned, reasons: result.reasons,
                             toxicityScore: result.toxicityScore, redacted: result.text)
                continue
            }
            if toxic {
                droppedToxic += 1
                var reasons = result.reasons
                reasons.append("toxicity")
                writeSidecar(sidecarFH, line: scanned, reasons: reasons,
                             toxicityScore: result.toxicityScore, redacted: result.text)
                continue
            }
            if hasPII { redacted += 1 }
            writeLine(result.text, to: outFH)
            kept += 1
        }

        print("""

        TinyGPT — filter
        ----------------
        input:          \(inputPath)
        out:            \(outputPath)
        scanned:        \(scanned)
        kept:           \(kept)
        redacted PII:   \(redacted)
        dropped PII:    \(droppedPII)
        dropped toxic:  \(droppedToxic)
        sidecar:        \(cfg.sidecarPath ?? "off")
        """)
    }

    private static func filterLine(_ line: String, cfg: Config) -> FilterResult {
        let toxicity = cfg.toxicityModel == "off" ? 0 : toxicityScore(line)
        if let obj = parseJSON(line) {
            var reasons: [String] = []
            let redacted = redactJSON(obj, types: cfg.piiTypes, reasons: &reasons)
            let text = encodeJSONLine(redacted) ?? line
            return FilterResult(text: text, reasons: unique(reasons), toxicityScore: toxicity)
        }
        var reasons: [String] = []
        let redacted = redactText(line, types: cfg.piiTypes, reasons: &reasons)
        return FilterResult(text: redacted, reasons: unique(reasons), toxicityScore: toxicity)
    }

    private static func parsePIITypes(_ raw: String) -> Set<PIIType> {
        let lowered = raw.lowercased()
        if lowered == "none" || lowered == "off" { return [] }
        if lowered == "auto" { return Set(PIIType.auto) }
        var out: Set<PIIType> = []
        for part in lowered.split(separator: ",") {
            let name = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let t = PIIType(rawValue: name) else {
                fputs("unknown PII type '\(name)'. Options: auto, none, \(PIIType.allCases.map(\.rawValue).joined(separator: ","))\n", stderr)
                exit(2)
            }
            out.insert(t)
        }
        return out
    }

    private static func redactJSON(_ value: Any, types: Set<PIIType>, reasons: inout [String]) -> Any {
        if let s = value as? String { return redactText(s, types: types, reasons: &reasons) }
        if let arr = value as? [Any] {
            return arr.map { redactJSON($0, types: types, reasons: &reasons) }
        }
        if let dict = value as? [String: Any] {
            var out: [String: Any] = [:]
            for (k, v) in dict {
                out[k] = redactJSON(v, types: types, reasons: &reasons)
            }
            return out
        }
        return value
    }

    private static func redactText(_ text: String, types: Set<PIIType>, reasons: inout [String]) -> String {
        var out = text
        if types.contains(.email) {
            out = replace(pattern: #"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"#,
                          in: out, with: "[EMAIL]", options: [.caseInsensitive]) {
                reasons.append("pii.email")
            }
        }
        if types.contains(.ssn) {
            out = replace(pattern: #"\b\d{3}-\d{2}-\d{4}\b"#, in: out, with: "[SSN]") {
                reasons.append("pii.ssn")
            }
        }
        if types.contains(.ip) {
            out = replace(pattern: #"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"#,
                          in: out, with: "[IP]") {
                reasons.append("pii.ip")
            }
            out = replace(pattern: #"\b(?:[A-F0-9]{1,4}:){2,7}[A-F0-9]{1,4}\b"#,
                          in: out, with: "[IP]", options: [.caseInsensitive]) {
                reasons.append("pii.ip")
            }
        }
        if types.contains(.phone) {
            out = replace(pattern: #"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"#,
                          in: out, with: "[PHONE]") {
                reasons.append("pii.phone")
            }
        }
        if types.contains(.creditCard) {
            out = redactCreditCards(out, reasons: &reasons)
        }
        if types.contains(.name), likelyContainsName(out) {
            reasons.append("pii.name-review")
        }
        return out
    }

    private static func replace(
        pattern: String,
        in text: String,
        with replacement: String,
        options: NSRegularExpression.Options = [],
        onMatch: () -> Void
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { return text }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        if !matches.isEmpty { onMatch() }
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: replacement)
    }

    private static func redactCreditCards(_ text: String, reasons: inout [String]) -> String {
        let pattern = #"\b(?:\d[ -]?){13,19}\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).reversed()
        var out = text
        for m in matches {
            let candidate = ns.substring(with: m.range)
            let digits = candidate.filter(\.isNumber).compactMap { Int(String($0)) }
            if digits.count >= 13 && digits.count <= 19 && luhn(digits), let r = Range(m.range, in: out) {
                out.replaceSubrange(r, with: "[CREDIT_CARD]")
                reasons.append("pii.credit-card")
            }
        }
        return out
    }

    private static func luhn(_ digits: [Int]) -> Bool {
        var sum = 0
        for (offset, d) in digits.reversed().enumerated() {
            if offset % 2 == 1 {
                let doubled = d * 2
                sum += doubled > 9 ? doubled - 9 : doubled
            } else {
                sum += d
            }
        }
        return sum % 10 == 0
    }

    private static func likelyContainsName(_ text: String) -> Bool {
        let pattern = #"\b(?:my name is|name:\s*)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"#
        return (try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]))
            .map { $0.firstMatch(in: text, range: NSRange(text.startIndex..<text.endIndex, in: text)) != nil }
            ?? false
    }

    private static func toxicityScore(_ text: String) -> Float {
        let s = text.lowercased()
        var score: Float = 0
        let phrases = ["kill yourself", "you should die", "i hate you"]
        for p in phrases where s.contains(p) { score += 0.85 }
        let words = ["idiot", "stupid", "moron", "trash", "worthless", "disgusting"]
        let tokens = s.split { !$0.isLetter }
        for t in tokens where words.contains(String(t)) { score += 0.18 }
        return min(score, 1.0)
    }

    private static func parseJSON(_ line: String) -> Any? {
        guard let data = line.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }

    private static func encodeJSONLine(_ obj: Any) -> String? {
        guard JSONSerialization.isValidJSONObject(obj),
              let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8)
        else { return nil }
        return s
    }

    private static func writeLine(_ line: String, to fh: FileHandle) {
        try? fh.write(contentsOf: Data((line + "\n").utf8))
    }

    private static func writeSidecar(
        _ fh: FileHandle?, line: Int, reasons: [String], toxicityScore: Float, redacted: String
    ) {
        guard let fh else { return }
        let obj: [String: Any] = [
            "line": line,
            "reasons": reasons,
            "toxicity_score": toxicityScore,
            "redacted": redacted,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
           let s = String(data: data, encoding: .utf8) {
            try? fh.write(contentsOf: Data((s + "\n").utf8))
        }
    }

    private static func unique(_ reasons: [String]) -> [String] {
        var seen: Set<String> = []
        var out: [String] = []
        for r in reasons where !seen.contains(r) {
            seen.insert(r)
            out.append(r)
        }
        return out
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt filter --in raw.jsonl --out cleaned.jsonl [options]

        Redact PII and optionally drop toxic rows. JSONL rows are preserved
        as JSON with string fields redacted; non-JSON lines are treated as text.

        --in <path>                  input text/jsonl
        --out <path>                 cleaned output
        --pii auto|none|list         default auto. list: email,phone,ssn,ip,credit-card,name
        --drop-pii                   drop rows with PII instead of redacting them
        --toxicity F                 drop rows with heuristic toxicity score >= F
        --toxicity-model heuristic   only bundled option; external Detoxify is not bundled
        --sidecar <path>             write dropped-row audit JSONL with redacted text
        """)
        exit(code)
    }
}
