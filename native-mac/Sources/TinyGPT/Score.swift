import Foundation
import MLX
import MLXNN
import TinyGPTIO
import TinyGPTModel

/// `tinygpt score-bench` — Mac-side offline benchmark scorer for the
/// browser leaderboard.
///
/// Why this lives on the Mac:
///   The original `browser/score_gallery.ts` ran every benchmark through
///   the byte-level WASM module — fine for the five 9.6M browser-trained
///   gallery models, but silently broken on any BPE checkpoint (it
///   would create a vocab=256 WASM model and try to import vocab=49152
///   weights). Rather than embed a tokenizer into the Node tooling, we
///   score natively here (where `ModelLoader` already auto-detects
///   BPE-vs-byte from the header) and emit the JSON the leaderboard
///   reads.
///
/// Inputs:
///   - <model.tinygpt> (or HF model directory) — the checkpoint to score.
///   - --benchmarks bench/benchmarks.json — descriptor of which evals
///     to run (id, kind, vocabType, holdout path…).
///   - --manifest browser/public/gallery/manifest.json — leaderboard
///     manifest to update in place.
///
/// Output: rewrites the manifest's matching entry's `benchmarks` map,
///   inserting (or replacing) one score per descriptor. Models that
///   fail a vocab-type compatibility check get `null` (the existing
///   convention: `null` = "ran but incompatible", absence = "not yet
///   scored", number = "actual score").
///
/// Dispatch isn't wired into TinyGPT.swift's case-switch on purpose —
/// see // TODO(score-bench-merge) below for the insertion point.
enum Score {

    // MARK: - Benchmark descriptors (mirrors bench/benchmarks.json)

    private struct BenchmarksFile: Codable {
        let version: Int
        let benchmarks: [BenchSpec]
    }

    /// Source-of-truth shape for one row in `bench/benchmarks.json`.
    /// The scorer dispatches on `kind` ("perplexity" vs "task-exact-match")
    /// and uses `vocabType` to decide compatibility ("byte-only",
    /// "bpe-only", "any").
    private struct BenchSpec: Codable {
        let id: String
        let name: String
        let kind: String                 // "perplexity" | "task-exact-match"
        let lowerIsBetter: Bool
        let vocabType: String            // "byte-only" | "bpe-only" | "any"
        // Perplexity fields
        let holdoutCorpus: String?
        let holdoutFormat: String?       // "stories-json" | "raw-text"
        let batches: Int?
        let batchSize: Int?
        // Task fields
        let task: String?                // "sort-6" | "reverse-16"
        let trials: Int?
        let seed: Int?
        let description: String?
    }

    // MARK: - Entry point

    static func run(args: [String]) {
        var modelPath: String?
        var benchPath = "bench/benchmarks.json"
        var manifestPath = "browser/public/gallery/manifest.json"
        var modelId: String?              // override the manifest key
        var dryRun = false
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--benchmarks":
                guard i + 1 < args.count else { exitUsage() }
                benchPath = args[i + 1]; i += 2
            case "--manifest":
                guard i + 1 < args.count else { exitUsage() }
                manifestPath = args[i + 1]; i += 2
            case "--id":
                guard i + 1 < args.count else { exitUsage() }
                modelId = args[i + 1]; i += 2
            case "--dry-run":
                dryRun = true; i += 1
            case "-h", "--help": exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("score-bench: unknown flag: \(args[i])\n", stderr)
                    exitUsage()
                }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else {
            fputs("score-bench: missing <model.tinygpt>\n", stderr); exitUsage()
        }

        // Load benchmark descriptors first — fail fast if the JSON is
        // malformed before we eat the model load cost.
        let benchURL = URL(fileURLWithPath: benchPath)
        let benchData: Data
        do { benchData = try Data(contentsOf: benchURL) }
        catch { fputs("score-bench: can't read \(benchPath): \(error)\n", stderr); exit(1) }
        let benchFile: BenchmarksFile
        do {
            benchFile = try JSONDecoder().decode(BenchmarksFile.self, from: benchData)
        } catch {
            fputs("score-bench: bad descriptor JSON: \(error)\n", stderr); exit(1)
        }
        print("• loaded \(benchFile.benchmarks.count) benchmark descriptors from \(benchPath)")

        // Load the model. ModelLoader already discriminates byte-level
        // vs BPE from the header, so we just inspect `hfTokenizerDir`
        // to know which path each benchmark gets.
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("score-bench: model load failed: \(error)\n", stderr); exit(1) }
        let model = load.model
        let cfg = load.config
        let isBpe = (load.hfTokenizerDir != nil)
        print("• model: \(modelPath)  \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength) · vocab=\(cfg.vocabSize) · \(isBpe ? "BPE" : "byte-level")")

        // Cache the tokenizer once if we'll need it.
        var tokenizer: HFTokenizer? = nil
        if let tokDir = load.hfTokenizerDir {
            do { tokenizer = try HFTokenizer.loadBlocking(from: tokDir) }
            catch { fputs("score-bench: tokenizer load failed: \(error)\n", stderr); exit(1) }
        }

        // Derive a default gallery id from the model filename so the
        // manifest patch finds the right row without --id.
        let derivedId = modelId ?? URL(fileURLWithPath: modelPath)
            .deletingPathExtension().lastPathComponent
        print("• gallery id (manifest key): \(derivedId)")

        // Run each benchmark. Score=nil means "ran but incompatible" —
        // we still write it to the manifest as `null` so the leaderboard
        // can distinguish "skipped" from "never tried".
        struct Outcome { let id: String; let score: Double?; let details: [String: Any] }
        var outcomes: [Outcome] = []

        for spec in benchFile.benchmarks {
            print("\n→ \(spec.id)  (\(spec.kind), vocabType=\(spec.vocabType))")
            if let skip = vocabSkipReason(spec: spec, isBpe: isBpe) {
                print("  · skip: \(skip)")
                outcomes.append(Outcome(id: spec.id, score: nil,
                                        details: ["skipped": skip]))
                continue
            }
            switch spec.kind {
            case "perplexity":
                let r = scorePerplexity(spec: spec, model: model, cfg: cfg,
                                        tokenizer: tokenizer)
                outcomes.append(Outcome(id: spec.id, score: r.score,
                                        details: r.details))
            case "task-exact-match":
                let r = scoreTask(spec: spec, model: model, cfg: cfg,
                                  tokenizer: tokenizer)
                outcomes.append(Outcome(id: spec.id, score: r.score,
                                        details: r.details))
            default:
                print("  ! unknown kind '\(spec.kind)' — skipping")
                outcomes.append(Outcome(id: spec.id, score: nil,
                                        details: ["error": "unknown kind \(spec.kind)"]))
            }
        }

        // Manifest patch. We open the file as a free-form JSON
        // dictionary so we can read/write entries without binding to
        // the full Swift schema (which would couple this command to
        // the GalleryModel struct).
        if dryRun {
            print("\n[dry-run] would write to \(manifestPath):")
            for o in outcomes {
                let s = o.score.map { String(format: "%.4f", $0) } ?? "null"
                print("  \(o.id) -> \(s)")
            }
            return
        }
        do {
            try patchManifest(at: manifestPath, modelId: derivedId,
                              outcomes: outcomes.map { ($0.id, $0.score) },
                              modelPath: modelPath, cfg: cfg, isBpe: isBpe)
        } catch {
            fputs("score-bench: failed to patch manifest: \(error)\n", stderr)
            exit(1)
        }

        print("\nSUMMARY")
        print("-------")
        for o in outcomes {
            // Avoid `String(format: "%-Ns", ...)` here — it operates on
            // C-string byte length which mangles UTF-8 (the gallery
            // benchmark ids are pure ASCII, but the reason strings can
            // contain — long dashes, so we use the Swift-native pad()).
            let label = pad(o.id, 22)
            if let s = o.score {
                print(String(format: "  \(label)  %.4f", s))
            } else {
                let reason = (o.details["skipped"] as? String) ?? "—"
                print("  \(label)  null   (\(reason))")
            }
        }
        print("\nwrote \(manifestPath)")
    }

    // MARK: - Vocab compat

    /// Returns nil if the model and benchmark are compatible, otherwise
    /// a one-line reason to write into the manifest's details.
    private static func vocabSkipReason(spec: BenchSpec, isBpe: Bool) -> String? {
        switch spec.vocabType {
        case "any": return nil
        case "byte-only":
            return isBpe ? "byte-only benchmark; model is BPE" : nil
        case "bpe-only":
            return isBpe ? nil : "BPE-only benchmark; model is byte-level"
        default:
            return "unknown vocabType '\(spec.vocabType)'"
        }
    }

    // MARK: - Perplexity scoring

    private struct ScoreResult {
        let score: Double?
        let details: [String: Any]
    }

    private static func scorePerplexity(spec: BenchSpec, model: AnyModel,
                                        cfg: ModelConfig,
                                        tokenizer: HFTokenizer?) -> ScoreResult {
        guard let corpusPath = spec.holdoutCorpus else {
            return ScoreResult(score: nil, details: ["error": "missing holdoutCorpus"])
        }
        let corpusURL = URL(fileURLWithPath: corpusPath)
        let text: String
        do {
            text = try loadCorpusText(from: corpusURL,
                                       format: spec.holdoutFormat ?? "raw-text")
        } catch {
            return ScoreResult(score: nil, details: ["error": "load failed: \(error)"])
        }
        print("  · corpus: \(corpusPath)  (\(formatBytes(text.utf8.count)))")

        // Build the per-window sampler. BPE path encodes once and
        // reuses the token buffer; byte path slices the raw bytes.
        let sampleBatch: (Int, Int) -> (MLXArray, MLXArray)
        let unit: String
        if let tok = tokenizer {
            let ids: [Int]
            do { ids = try tok.encode(text) }
            catch {
                return ScoreResult(score: nil, details: ["error": "tokenize failed: \(error)"])
            }
            let tokens = ids.map { Int32($0) }
            print("  · encoded: \(tokens.count) BPE tokens")
            if tokens.count < cfg.contextLength + 2 {
                return ScoreResult(score: nil,
                    details: ["error": "corpus too small (\(tokens.count) tokens) for ctx=\(cfg.contextLength)"])
            }
            let corpus = TokenizedCorpus(tokens: tokens, vocabSize: cfg.vocabSize)
            sampleBatch = { B, T in corpus.sampleBatch(batchSize: B, contextLength: T) }
            unit = "token"
        } else {
            let data = Data(text.utf8)
            if data.count < cfg.contextLength + 2 {
                return ScoreResult(score: nil,
                    details: ["error": "corpus too small (\(data.count) bytes) for ctx=\(cfg.contextLength)"])
            }
            let corpus = ByteCorpus(data)
            sampleBatch = { B, T in corpus.sampleBatch(batchSize: B, contextLength: T) }
            unit = "byte"
        }

        let nBatches = spec.batches ?? 32
        let batchSize = spec.batchSize ?? 8

        var lossSum: Float = 0
        var count = 0
        for k in 0..<nBatches {
            let (x, y) = sampleBatch(batchSize, cfg.contextLength)
            let loss = model.loss(x, y)
            eval(loss)
            lossSum += loss.item(Float.self)
            count += 1
            if k == 0 || (k + 1) % 8 == 0 || k == nBatches - 1 {
                fputs(String(format: "    batch %3d  running avg %.4f\n",
                             k + 1, lossSum / Float(count)), stderr)
            }
        }
        let avgLoss = lossSum / Float(count)
        let ppl = exp(avgLoss)
        print(String(format: "  · result: loss=%.4f  ppl=%.4f", avgLoss, ppl))
        return ScoreResult(
            score: Double(ppl),
            details: [
                "loss": Double(avgLoss),
                "perplexity": Double(ppl),
                "batches": nBatches,
                "batchSize": batchSize,
                "tokens": nBatches * batchSize * cfg.contextLength,
                "unit": unit,
                "vocabSize": cfg.vocabSize,
            ]
        )
    }

    private static func loadCorpusText(from url: URL, format: String) throws -> String {
        let raw = try Data(contentsOf: url)
        switch format {
        case "stories-json":
            // Schema mirrors browser/public/benchmarks/tinystories-eval.json:
            // { source, count, totalBytes, stories: [String] }. We join on
            // \n\n the same way `browser/score_gallery.ts` did.
            struct StoriesFile: Decodable { let stories: [String] }
            let parsed = try JSONDecoder().decode(StoriesFile.self, from: raw)
            return parsed.stories.joined(separator: "\n\n")
        case "raw-text":
            return String(decoding: raw, as: UTF8.self)
        default:
            throw NSError(domain: "Score", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "unknown holdoutFormat: \(format)"])
        }
    }

    // MARK: - Task scoring (sort-6, reverse-16)

    private static func scoreTask(spec: BenchSpec, model: AnyModel,
                                  cfg: ModelConfig,
                                  tokenizer: HFTokenizer?) -> ScoreResult {
        guard let task = spec.task else {
            return ScoreResult(score: nil, details: ["error": "missing task"])
        }
        let trials: [(prompt: String, expected: String)]
        switch task {
        case "sort-6":
            trials = buildSort6(trials: spec.trials ?? 200,
                                seed: UInt32(spec.seed ?? 0x517))
        case "reverse-16":
            trials = buildReverse16(trials: spec.trials ?? 200,
                                    seed: UInt32(spec.seed ?? 0x3EE5))
        default:
            return ScoreResult(score: nil, details: ["error": "unknown task \(task)"])
        }

        var correct = 0
        var failures: [String] = []
        for (k, trial) in trials.enumerated() {
            let got = greedyGenerate(model: model, cfg: cfg, tokenizer: tokenizer,
                                     prompt: trial.prompt,
                                     maxNewTokens: trial.expected.count + 2)
            // Match the FIRST `expected.count` chars after leading whitespace —
            // same semantics as the browser score_gallery_tasks.ts loop.
            let trimmed = got.drop(while: { $0.isWhitespace })
            let prefix = String(trimmed.prefix(trial.expected.count))
            if prefix == trial.expected {
                correct += 1
            } else if failures.count < 5 {
                failures.append("\(trial.prompt)→ \"\(prefix)\" (expected \"\(trial.expected)\")")
            }
            if (k + 1) % 50 == 0 {
                fputs("    \(k + 1)/\(trials.count)  correct=\(correct)\n", stderr)
            }
        }
        let accuracy = 100.0 * Double(correct) / Double(trials.count)
        print(String(format: "  · result: %.2f%% (%d/%d)",
                     accuracy, correct, trials.count))
        return ScoreResult(
            score: accuracy,
            details: [
                "trials": trials.count,
                "correct": correct,
                "failures": failures
            ]
        )
    }

    /// Deterministic Mulberry32 — byte-for-byte equivalent to the
    /// browser score_gallery_tasks.ts implementation so the trial sets
    /// match exactly.
    private struct Mulberry32 {
        var s: UInt32
        init(_ seed: UInt32) { self.s = seed }
        mutating func next() -> Double {
            s &+= 0x6D2B79F5
            var t = s
            t = (t ^ (t >> 15)) &* (t | 1)
            t ^= t &+ ((t ^ (t >> 7)) &* (t | 61))
            return Double((t ^ (t >> 14)) & 0xFFFF_FFFF) / 4294967296.0
        }
    }

    private static func buildSort6(trials: Int, seed: UInt32)
        -> [(prompt: String, expected: String)]
    {
        var rng = Mulberry32(seed)
        var out: [(String, String)] = []
        for _ in 0..<trials {
            var digits: [Int] = []
            for _ in 0..<6 { digits.append(Int(rng.next() * 10)) }
            let prompt = "sort: \(digits.map(String.init).joined(separator: " ")) = "
            let expected = digits.sorted().map(String.init).joined(separator: " ")
            out.append((prompt, expected))
        }
        return out
    }

    private static func buildReverse16(trials: Int, seed: UInt32)
        -> [(prompt: String, expected: String)]
    {
        var rng = Mulberry32(seed)
        let alphabet = Array("abcdefghijklmnopqrstuvwxyz")
        var out: [(String, String)] = []
        for _ in 0..<trials {
            let len = 4 + Int(rng.next() * 13)
            var s = ""
            for _ in 0..<len { s.append(alphabet[Int(rng.next() * 26)]) }
            out.append(("reverse: \(s) = ", String(s.reversed())))
        }
        return out
    }

    /// Greedy generate `maxNewTokens` tokens (or bytes) and decode back
    /// to UTF-8. Used by task-based benchmarks where exact-match is
    /// computed on the rendered output. Mirrors the Sample.swift inner
    /// loop but stripped of the streaming / kv-cache / lora plumbing —
    /// scoring tasks are short (sub-30 tokens), so the simpler O(n²)
    /// path is fast enough and avoids any kv-cache reset bugs.
    private static func greedyGenerate(model: AnyModel, cfg: ModelConfig,
                                       tokenizer: HFTokenizer?,
                                       prompt: String,
                                       maxNewTokens: Int) -> String {
        let promptIds: [Int32]
        if let tok = tokenizer {
            do {
                let ids = try tok.encode(prompt)
                promptIds = ids.map { Int32($0) }
            } catch { return "" }
        } else {
            promptIds = [UInt8](prompt.utf8).map { Int32($0) }
        }
        var idx = MLXArray(promptIds, [1, promptIds.count])
        var generated: [Int32] = []
        for _ in 0..<maxNewTokens {
            let T = idx.shape.last!
            let lo = max(0, T - cfg.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let next = MLX.argMax(last, axis: -1).reshaped([1, 1])
            eval(next)
            let id = Int32(next.item(Int32.self))
            generated.append(id)
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }
        if let tok = tokenizer {
            return tok.decode(generated.map { Int($0) })
        } else {
            var s = ""
            for id in generated {
                if let scalar = UnicodeScalar(Int(id)), id >= 9 {
                    s.append(Character(scalar))
                }
            }
            return s
        }
    }

    // MARK: - Manifest I/O

    /// In-place patch: find (or insert) the entry whose `id == modelId`
    /// inside `manifest.json` and update its `benchmarks` map.
    ///
    /// Implementation note — we go to some lengths to preserve the
    /// **exact** existing JSON formatting (key order, whitespace, the
    /// idiomatic 2-space pretty-print the rest of the file uses). The
    /// alternative — round-tripping through `JSONSerialization` — would
    /// re-order every key alphabetically AND insert spaces around the
    /// `:` separator AND escape forward slashes, producing a massive
    /// non-reviewable diff that overwhelms the actual score change.
    ///
    /// Strategy:
    ///   - Parse to a dict tree to LOCATE existing entries by `id`.
    ///   - For each existing entry, surgically rewrite its
    ///     `"benchmarks": { … }` block as a text substring (matching
    ///     brace balance).
    ///   - For new entries (the rare path: first-time score from Mac),
    ///     append a fresh model object with hand-crafted formatting
    ///     that matches the existing entries.
    private static func patchManifest(at path: String, modelId: String,
                                      outcomes: [(String, Double?)],
                                      modelPath: String,
                                      cfg: ModelConfig,
                                      isBpe: Bool) throws {
        let url = URL(fileURLWithPath: path)
        var raw: String
        if FileManager.default.fileExists(atPath: url.path) {
            raw = try String(contentsOf: url, encoding: .utf8)
        } else {
            raw = """
            {
              "version": 1,
              "models": []
            }
            """
        }

        // Parse just for navigation — we never serialize this back.
        let data = Data(raw.utf8)
        let parsed = try JSONSerialization.jsonObject(with: data)
        guard let root = parsed as? [String: Any] else {
            throw NSError(domain: "Score", code: 100,
                          userInfo: [NSLocalizedDescriptionKey: "manifest root is not an object"])
        }
        let models = (root["models"] as? [[String: Any]]) ?? []
        let exists = models.contains { ($0["id"] as? String) == modelId }

        let benchJson = formatBenchmarksBlock(outcomes: outcomes, indent: 6)

        if exists {
            // Replace the existing entry's "benchmarks": { … } block.
            raw = try replaceBenchmarksBlock(in: raw, modelId: modelId,
                                              newBlock: benchJson)
        } else {
            // Append a new model object before the closing `]`.
            let newEntry = formatNewModelEntry(
                modelId: modelId, outcomes: outcomes,
                modelPath: modelPath, cfg: cfg, isBpe: isBpe
            )
            raw = try appendModelEntry(in: raw, newEntry: newEntry,
                                        haveExisting: !models.isEmpty)
        }
        try raw.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Build the text "benchmarks": {…} block — already indented as if
    /// inserted at `indent` columns. Matches the file's existing style:
    ///   2-space indent, ":" with one space after, no trailing comma.
    private static func formatBenchmarksBlock(outcomes: [(String, Double?)],
                                              indent: Int) -> String {
        let pad = String(repeating: " ", count: indent)
        let inner = String(repeating: " ", count: indent + 2)
        var lines: [String] = []
        for (k, score) in outcomes {
            let v = score.map { String(format: "%.6g", $0) } ?? "null"
            lines.append("\(inner)\"\(k)\": \(v)")
        }
        return "\"benchmarks\": {\n\(lines.joined(separator: ",\n"))\n\(pad)}"
    }

    /// Surgical text edit: find the model object whose `"id": "<modelId>"`
    /// matches, locate the `"benchmarks": { … }` block inside it
    /// (with brace balance), and replace it with `newBlock`.
    /// If the object doesn't yet have a `benchmarks` key, append one
    /// just before the model object's closing brace.
    private static func replaceBenchmarksBlock(in raw: String, modelId: String,
                                                newBlock: String) throws -> String {
        // Locate the model object: find `"id": "<modelId>"`, then walk
        // back to the opening `{` and forward to the matching `}`.
        let needle = "\"id\": \"\(modelId)\""
        guard let idRange = raw.range(of: needle) else {
            throw NSError(domain: "Score", code: 101,
                          userInfo: [NSLocalizedDescriptionKey:
                            "could not locate \(needle) in manifest"])
        }
        // Walk backwards to the enclosing `{`.
        var i = idRange.lowerBound
        var depth = 0
        var objStart: String.Index? = nil
        while i > raw.startIndex {
            i = raw.index(before: i)
            let c = raw[i]
            if c == "}" { depth += 1 }
            else if c == "{" {
                if depth == 0 { objStart = i; break }
                depth -= 1
            }
        }
        guard let start = objStart else {
            throw NSError(domain: "Score", code: 102,
                          userInfo: [NSLocalizedDescriptionKey: "no enclosing { for \(modelId)"])
        }
        // Walk forwards to the matching `}` to bound the object.
        var j = start
        depth = 0
        var objEnd: String.Index? = nil
        while j < raw.endIndex {
            let c = raw[j]
            if c == "{" { depth += 1 }
            else if c == "}" {
                depth -= 1
                if depth == 0 { objEnd = j; break }
            }
            j = raw.index(after: j)
        }
        guard let end = objEnd else {
            throw NSError(domain: "Score", code: 103,
                          userInfo: [NSLocalizedDescriptionKey: "no closing } for \(modelId)"])
        }
        let objText = String(raw[start...end])

        // Find the existing "benchmarks": { ... } block within objText.
        let bKey = "\"benchmarks\":"
        if let bKeyRange = objText.range(of: bKey) {
            // Find the opening `{` after the key.
            guard let openBrace = objText[bKeyRange.upperBound...].firstIndex(of: "{") else {
                throw NSError(domain: "Score", code: 104,
                              userInfo: [NSLocalizedDescriptionKey: "malformed benchmarks block"])
            }
            // Match the closing brace from openBrace.
            var k = openBrace
            depth = 0
            var bEnd: String.Index? = nil
            while k < objText.endIndex {
                let c = objText[k]
                if c == "{" { depth += 1 }
                else if c == "}" {
                    depth -= 1
                    if depth == 0 { bEnd = k; break }
                }
                k = objText.index(after: k)
            }
            guard let bClose = bEnd else {
                throw NSError(domain: "Score", code: 105,
                              userInfo: [NSLocalizedDescriptionKey: "unbalanced benchmarks block"])
            }
            let newObjText = objText.replacingCharacters(
                in: bKeyRange.lowerBound...bClose, with: newBlock)
            return raw.replacingCharacters(in: start...end, with: newObjText)
        } else {
            // No benchmarks block — append one before the closing `}`.
            // Find the last comma-or-newline before `}` and insert.
            // We do a simpler injection: put `,\n      benchmarks…\n    `
            // right before the final `}`.
            var newObjText = objText
            // Remove any trailing whitespace before `}`.
            let beforeClose = newObjText.index(before: newObjText.endIndex)
            // Insert: ",\n      " + block + "\n    "
            let indent = "      "
            let inject = ",\n\(indent)\(newBlock)\n    "
            newObjText.insert(contentsOf: inject, at: beforeClose)
            return raw.replacingCharacters(in: start...end, with: newObjText)
        }
    }

    /// Append a brand-new model entry into the `models` array. We
    /// insert before the closing `]`, with a leading `,` if there are
    /// already entries, matching the existing file's 4-space block
    /// indent.
    private static func appendModelEntry(in raw: String, newEntry: String,
                                          haveExisting: Bool) throws -> String {
        // Find the `"models"` key, then the opening `[`, then walk to
        // its matching `]`.
        guard let modelsKeyRange = raw.range(of: "\"models\":") else {
            throw NSError(domain: "Score", code: 106,
                          userInfo: [NSLocalizedDescriptionKey: "manifest has no models array"])
        }
        guard let openBracket = raw[modelsKeyRange.upperBound...].firstIndex(of: "[") else {
            throw NSError(domain: "Score", code: 107,
                          userInfo: [NSLocalizedDescriptionKey: "no [ after models key"])
        }
        var k = openBracket
        var depth = 0
        var closeBracket: String.Index? = nil
        while k < raw.endIndex {
            let c = raw[k]
            if c == "[" { depth += 1 }
            else if c == "]" {
                depth -= 1
                if depth == 0 { closeBracket = k; break }
            }
            k = raw.index(after: k)
        }
        guard let close = closeBracket else {
            throw NSError(domain: "Score", code: 108,
                          userInfo: [NSLocalizedDescriptionKey: "unbalanced models []"])
        }
        // The closing `]` sits on its own line at 2-space indent in the
        // existing file. We need to inject:
        //   <last existing entry>,
        //   <newEntry>
        //   ]
        // The trick is the closing `}` of the prior entry already
        // ends a line; we just need ",\n" before the new entry and a
        // bare newline+indent before the `]`.
        var out = raw
        // Walk back from `]` to the last non-whitespace char (which is
        // either `[` for an empty list, or `}` for the closing brace of
        // the previous-last entry).
        var p = out.index(before: close)
        while p > out.startIndex, out[p].isWhitespace {
            p = out.index(before: p)
        }
        // Replace the run [p+1 .. close-1] (trailing whitespace before
        // `]`) with: `,\n    <newEntry>\n  ` for appending, or
        // `\n    <newEntry>\n  ` for a fresh list.
        let trailingStart = out.index(after: p)
        let trailingRange = trailingStart..<close
        let separator = haveExisting ? "," : ""
        out.replaceSubrange(trailingRange,
                            with: "\(separator)\n\(newEntry)\n  ")
        return out
    }

    private static func formatNewModelEntry(modelId: String,
                                            outcomes: [(String, Double?)],
                                            modelPath: String,
                                            cfg: ModelConfig,
                                            isBpe: Bool) -> String {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let params = formatParams(estimateParams(cfg: cfg))
        let paramCount = estimateParams(cfg: cfg)
        let corpus = isBpe ? "BPE-trained (vocab \(cfg.vocabSize))" : "byte-level"
        let file = (modelId as NSString).lastPathComponent + ".tinygpt"
        let bench = formatBenchmarksBlock(outcomes: outcomes, indent: 6)
        return """
            {
              "id": "\(modelId)",
              "name": "\(modelId)",
              "file": "\(file)",
              "corpus": "\(corpus)",
              "params": "\(params)",
              "paramCount": \(paramCount),
              "submission": {
                "author": "TinyGPT (Mac)",
                "submittedAt": "\(timestamp)",
                "browserTrained": false,
                "featured": false
              },
              \(bench)
            }
        """
    }

    private static func estimateParams(cfg: ModelConfig) -> Int {
        // Rough Transformer parameter count — same formula the existing
        // gallery JSON uses. Doesn't matter for scoring; only the
        // leaderboard display column cares about it.
        let attn = cfg.dModel * cfg.dModel * 4
        let mlp = cfg.dModel * cfg.dMlp * 2
        let block = attn + mlp + cfg.dModel * 4
        let embeddings = cfg.vocabSize * cfg.dModel
        return cfg.nLayers * block + embeddings * 2 + cfg.dModel
    }

    private static func formatParams(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.2fB", Double(n) / 1e9) }
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1e6) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1e3) }
        return "\(n)"
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt score-bench <model.tinygpt> [options]

        --benchmarks <path>   Benchmark descriptor JSON (default: bench/benchmarks.json)
        --manifest <path>     Gallery manifest to patch in place
                              (default: browser/public/gallery/manifest.json)
        --id <gallery-id>     Override the manifest key (default: model filename stem)
        --dry-run             Print what would be written; don't touch the manifest.

        Scores the model against each benchmark in the descriptor and writes
        the result back into the manifest. Vocab-incompatible benchmarks are
        recorded as `null` (consistent with the existing browser scorer).
        """)
        exit(2)
    }
}

// TODO(score-bench-merge): wire `tinygpt score-bench` into
// `native-mac/Sources/TinyGPT/TinyGPT.swift`'s `switch cmd` near the
// other `case "eval": …` lines. Suggested:
//
//     case "score-bench":
//         Score.run(args: Array(args.dropFirst()))
//
// Left as a TODO per the worktree spec to keep dispatch changes
// reviewable separately.
