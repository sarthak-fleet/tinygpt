import Foundation

/// `tinygpt quickstart` (B33) decision core — pure, unit-tested.
///
/// Turns a user's raw data file into a concrete (base model, training
/// recipe) plan with zero ML knowledge required. The CLI wizard in
/// `TinyGPT.Quickstart` is the orchestration shell; the judgement lives
/// here so it is testable without a GPU and shareable with B6's GUI
/// Factory tab.
public enum RecipeResolver {
    // MARK: - Data inspection

    public enum DataShape: String, Codable, Sendable {
        case chat // {"messages":[{role,content},...]}
        case toolCall // chat/instruction carrying tools or tool_calls
        case instruction // {"instruction","output"} / {"prompt","completion"}
        case rawText // not JSONL — a plain-text corpus
        case unknown
    }

    public struct DataInspection: Codable, Sendable {
        public let shape: DataShape
        public let rowCount: Int
        public let sample: String? // first non-empty line, truncated
        public let parsedAsJSON: Int // how many sampled lines parsed as JSON objects
        public let unparsable: Bool // true when nothing parsed as JSON (→ rawText)

        public init(shape: DataShape, rowCount: Int, sample: String?, parsedAsJSON: Int, unparsable: Bool) {
            self.shape = shape
            self.rowCount = rowCount
            self.sample = sample
            self.parsedAsJSON = parsedAsJSON
            self.unparsable = unparsable
        }
    }

    /// Inspect already-read lines (pure). The CLI reads the file and
    /// hands the lines in, so this stays filesystem-free for tests.
    public static func inspect(lines: [String], sampleLimit: Int = 50) -> DataInspection {
        let nonEmpty = lines
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let sample = nonEmpty.first.map { String($0.prefix(200)) }
        guard !nonEmpty.isEmpty else {
            return DataInspection(shape: .unknown, rowCount: 0, sample: nil, parsedAsJSON: 0, unparsable: true)
        }

        var parsed = 0
        var sawMessages = false, sawTool = false, sawInstruction = false
        for line in nonEmpty.prefix(sampleLimit) {
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data),
                  let dict = obj as? [String: Any]
            else { continue }
            parsed += 1
            if hasToolSignals(dict) { sawTool = true }
            if dict["messages"] is [Any] { sawMessages = true }
            if dict["instruction"] != nil
                || (dict["prompt"] != nil && dict["completion"] != nil)
                || (dict["input"] != nil && dict["output"] != nil) {
                sawInstruction = true
            }
        }

        let unparsable = parsed == 0
        let shape: DataShape
        if unparsable {
            shape = .rawText
        } else if sawTool {
            shape = .toolCall
        } else if sawMessages {
            shape = .chat
        } else if sawInstruction {
            shape = .instruction
        } else {
            shape = .unknown
        }
        return DataInspection(
            shape: shape, rowCount: nonEmpty.count, sample: sample,
            parsedAsJSON: parsed, unparsable: unparsable)
    }

    private static func hasToolSignals(_ dict: [String: Any]) -> Bool {
        if dict["tools"] != nil || dict["tool_calls"] != nil || dict["function_call"] != nil {
            return true
        }
        if let msgs = dict["messages"] as? [[String: Any]] {
            for m in msgs {
                if m["tool_calls"] != nil { return true }
                if let role = m["role"] as? String, role == "tool" || role == "function" { return true }
            }
        }
        return false
    }

    // MARK: - Recipe

    public enum TrainMode: String, Codable, Sendable {
        case loraFinetune
        case fromScratch
    }

    public struct Recipe: Codable, Sendable {
        public let mode: TrainMode
        public let template: String? // SFT chat template (chatml…); nil = SFT default / from-scratch
        public let rank: Int
        public let alpha: Int
        public let steps: Int
        public let lr: Double
        public let batch: Int
        public let maxSeq: Int
        public let neftuneAlpha: Int
        public let pack: Bool
        public let dora: Bool

        /// Flags for `tinygpt sft <base> <flags…>`. The base path plus
        /// `--data`/`--out` are supplied by the orchestration layer.
        public func sftFlags() -> [String] {
            var a: [String] = [
                "--rank", String(rank),
                "--alpha", String(alpha),
                "--steps", String(steps),
                "--lr", String(format: "%g", lr),
                "--batch", String(batch),
                "--max-seq", String(maxSeq),
                "--neftune-alpha", String(neftuneAlpha),
            ]
            if pack { a.append("--pack") }
            if let t = template { a += ["--template", t] }
            a.append(dora ? "--dora" : "--no-dora")
            return a
        }
    }

    public struct BaseChoice: Codable, Sendable {
        public let galleryId: String? // nil when from-scratch or unresolved
        public let fromScratch: Bool
        public let reason: String

        public init(galleryId: String?, fromScratch: Bool, reason: String) {
            self.galleryId = galleryId
            self.fromScratch = fromScratch
            self.reason = reason
        }
    }

    public struct ResolvedPlan: Codable, Sendable {
        public let inspection: DataInspection
        public let base: BaseChoice
        public let recipe: Recipe
        public let warnings: [String]
    }

    // MARK: - Resolution

    /// Resolve a full plan from an inspection plus the available gallery
    /// bases. `overrideBase` (a gallery id or local path) wins if given.
    public static func resolve(inspection: DataInspection,
                               gallery: [GalleryModel],
                               overrideBase: String? = nil) -> ResolvedPlan {
        var warnings: [String] = []
        if inspection.rowCount < 50, inspection.shape != .rawText {
            warnings.append(
                "only \(inspection.rowCount) rows — small datasets often don't beat the base; consider 200+ examples")
        }
        if inspection.shape == .unknown {
            warnings.append(
                "couldn't classify the data shape; defaulting to a chat fine-tune — pass --base or convert to chat JSONL if results look off")
        }

        let mode: TrainMode = inspection.shape == .rawText ? .fromScratch : .loraFinetune
        let base = pickBase(
            shape: inspection.shape, gallery: gallery,
            overrideBase: overrideBase, warnings: &warnings)
        let recipe = makeRecipe(shape: inspection.shape, rowCount: inspection.rowCount, mode: mode)
        return ResolvedPlan(inspection: inspection, base: base, recipe: recipe, warnings: warnings)
    }

    private static let preferredTags: [DataShape: [String]] = [
        .toolCall: ["tool", "tools", "agent", "function", "function-calling", "bfcl"],
        .chat: ["chat", "instruct", "instruction", "it"],
        .instruction: ["instruct", "instruction", "it", "chat"],
    ]

    private static func pickBase(shape: DataShape, gallery: [GalleryModel],
                                 overrideBase: String?, warnings: inout [String]) -> BaseChoice {
        if let ov = overrideBase {
            if !gallery.contains(where: { $0.id == ov }) {
                warnings.append("--base '\(ov)' is not in the gallery; assuming it's a local path or HF id")
            }
            return BaseChoice(galleryId: ov, fromScratch: false, reason: "explicit --base \(ov)")
        }
        if shape == .rawText {
            return BaseChoice(
                galleryId: nil, fromScratch: true,
                reason: "raw text → from-scratch pretraining (no base)")
        }

        let baseable = gallery.filter {
            switch $0.resolvedKind {
            case .macSafetensorsHf, .macGguf, .macTinygpt: return true
            case .macAdapter, .browserBin: return false
            }
        }
        guard !baseable.isEmpty else {
            warnings.append("no fine-tunable base in the gallery — pass --base <hf-id-or-path>")
            return BaseChoice(galleryId: nil, fromScratch: false, reason: "no base found in gallery")
        }

        let want = preferredTags[shape] ?? ["instruct", "chat"]
        func score(_ m: GalleryModel) -> Int {
            let tags = (m.tags ?? []).map { $0.lowercased() }
            return want.reduce(0) { $0 + (tags.contains($1) ? 1 : 0) }
        }
        // Highest tag-match wins; tiebreak toward the smaller (laptop-friendly) base.
        let best = baseable.max { a, b in
            let sa = score(a), sb = score(b)
            if sa != sb { return sa < sb }
            return (a.paramCount ?? Int.max) > (b.paramCount ?? Int.max)
        }!
        if score(best) == 0 {
            warnings.append(
                "no gallery base tagged for \(shape.rawValue) data; picked '\(best.id)' by size — override with --base if needed")
            return BaseChoice(
                galleryId: best.id, fromScratch: false,
                reason: "gallery base '\(best.id)' (smallest fine-tunable; no tag match)")
        }
        return BaseChoice(
            galleryId: best.id, fromScratch: false,
            reason: "gallery base '\(best.id)' matched \(shape.rawValue) tags")
    }

    private static func makeRecipe(shape: DataShape, rowCount: Int, mode: TrainMode) -> Recipe {
        // LoRA rank + step budget scale with dataset size.
        let rank: Int, steps: Int
        switch rowCount {
        case ..<500: (rank, steps) = (8, 300)
        case 500 ..< 5000: (rank, steps) = (16, 800)
        default: (rank, steps) = (32, 1500)
        }
        let template: String? = (shape == .chat || shape == .toolCall || shape == .unknown) ? "chatml" : nil
        let maxSeq = shape == .toolCall ? 2048 : 1024
        return Recipe(
            mode: mode, template: template,
            rank: rank, alpha: rank * 2, steps: steps,
            lr: 0.0002, batch: 4, maxSeq: maxSeq,
            neftuneAlpha: 5, pack: true, dora: false)
    }
}
