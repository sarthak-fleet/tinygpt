import Foundation
import MLX
import TinyGPTModel

enum RunBench {
    private struct TaskSpec {
        let name: String
        let trials: Int
        let seed: UInt32
    }

    static func run(args: [String]) {
        var modelPath: String?
        var corpusPath: String?
        var tasks = "sort-6,reverse-16"
        var outPath: String?
        var ctxOverride: Int?
        var batchSize = 8
        var batches = 32
        var modelName: String?
        var modelStep: Int?
        var limit: Int?

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--model": modelPath = args[i + 1]; i += 2
            case "--perplexity", "--corpus": corpusPath = args[i + 1]; i += 2
            case "--tasks": tasks = args[i + 1]; i += 2
            case "--out": outPath = args[i + 1]; i += 2
            case "--ctx": ctxOverride = Int(args[i + 1]); i += 2
            case "--batch": batchSize = Int(args[i + 1]) ?? batchSize; i += 2
            case "--batches": batches = Int(args[i + 1]) ?? batches; i += 2
            case "--model-name": modelName = args[i + 1]; i += 2
            case "--model-step": modelStep = Int(args[i + 1]); i += 2
            case "--limit": limit = Int(args[i + 1]); i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") {
                    fputs("run-bench: unknown flag \(args[i])\n", stderr); exitUsage()
                }
                modelPath = args[i]; i += 1
            }
        }

        guard let modelPath else { fputs("run-bench: --model <path> required\n", stderr); exitUsage() }
        guard let outPath else { fputs("run-bench: --out <jsonl> required\n", stderr); exitUsage() }

        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch { fputs("run-bench: model load failed: \(error)\n", stderr); exit(1) }
        let model = load.model
        var cfg = load.config
        if let ctxOverride { cfg.contextLength = min(ctxOverride, cfg.contextLength) }

        let tokenizer: HFTokenizer?
        if let dir = load.hfTokenizerDir {
            do { tokenizer = try HFTokenizer.loadBlocking(from: dir) }
            catch { fputs("run-bench: tokenizer load failed: \(error)\n", stderr); exit(1) }
        } else {
            tokenizer = nil
        }

        let displayName = modelName ?? URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent
        let outURL = URL(fileURLWithPath: outPath)
        let started = Date()
        var emitted = 0

        if let corpusPath {
            let r = scorePerplexity(
                model: model,
                cfg: cfg,
                tokenizer: tokenizer,
                corpusPath: corpusPath,
                batches: batches,
                batchSize: batchSize
            )
            append(outURL, modelPath: modelPath, modelName: displayName, modelStep: modelStep,
                   task: "perplexity", subtask: URL(fileURLWithPath: corpusPath).lastPathComponent,
                   metric: "perplexity", score: r.perplexity, n: r.tokens, wall: -started.timeIntervalSinceNow)
            append(outURL, modelPath: modelPath, modelName: displayName, modelStep: modelStep,
                   task: "perplexity", subtask: URL(fileURLWithPath: corpusPath).lastPathComponent,
                   metric: "loss", score: r.loss, n: r.tokens, wall: -started.timeIntervalSinceNow)
            emitted += 2
        }

        for task in parseTasks(tasks, limit: limit) {
            let r = scoreTask(task, model: model, cfg: cfg, tokenizer: tokenizer)
            append(outURL, modelPath: modelPath, modelName: displayName, modelStep: modelStep,
                   task: task.name, subtask: nil, metric: "acc", score: r.accuracy,
                   n: r.total, wall: -started.timeIntervalSinceNow)
            emitted += 1
        }

        print("✓ wrote \(emitted) E0 rows to \(outPath)")
        print("  view: tinygpt eval-compare \(outPath) --by task")
    }

    private static func scorePerplexity(
        model: AnyModel,
        cfg: ModelConfig,
        tokenizer: HFTokenizer?,
        corpusPath: String,
        batches: Int,
        batchSize: Int
    ) -> (loss: Double, perplexity: Double, tokens: Int) {
        let text: String
        do { text = try loadCorpusText(corpusPath) }
        catch { fputs("run-bench: corpus load failed: \(error)\n", stderr); exit(1) }

        let sampleBatch: (Int, Int) -> (MLXArray, MLXArray)
        if let tokenizer {
            let ids = ((try? tokenizer.encode(text)) ?? []).map { Int32($0) }
            guard ids.count > cfg.contextLength + 1 else {
                fputs("run-bench: tokenized corpus too small for ctx=\(cfg.contextLength)\n", stderr); exit(1)
            }
            let corpus = TokenizedCorpus(tokens: ids, vocabSize: cfg.vocabSize)
            sampleBatch = { b, t in corpus.sampleBatch(batchSize: b, contextLength: t) }
        } else {
            let data = Data(text.utf8)
            guard data.count > cfg.contextLength + 1 else {
                fputs("run-bench: byte corpus too small for ctx=\(cfg.contextLength)\n", stderr); exit(1)
            }
            let corpus = ByteCorpus(data)
            sampleBatch = { b, t in corpus.sampleBatch(batchSize: b, contextLength: t) }
        }

        var lossSum: Float = 0
        for k in 0..<batches {
            let (x, y) = sampleBatch(batchSize, cfg.contextLength)
            let loss = model.loss(x, y)
            eval(loss)
            lossSum += loss.item(Float.self)
            if k == 0 || (k + 1) % 8 == 0 || k == batches - 1 {
                fputs(String(format: "  ppl batch %3d/%3d avg %.4f\n", k + 1, batches, lossSum / Float(k + 1)), stderr)
            }
        }
        let avg = Double(lossSum / Float(max(1, batches)))
        return (avg, exp(avg), batches * batchSize * cfg.contextLength)
    }

    private static func loadCorpusText(_ path: String) throws -> String {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        if path.hasSuffix(".json"), let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let stories = obj["stories"] as? [String] {
            return stories.joined(separator: "\n\n")
        }
        if path.hasSuffix(".jsonl") {
            let text = String(decoding: data, as: UTF8.self)
            var parts: [String] = []
            for line in text.split(separator: "\n") {
                guard let d = String(line).data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
                else { continue }
                for key in ["text", "content", "prompt", "instruction", "output"] {
                    if let s = obj[key] as? String, !s.isEmpty { parts.append(s); break }
                }
            }
            if !parts.isEmpty { return parts.joined(separator: "\n\n") }
        }
        return String(decoding: data, as: UTF8.self)
    }

    private static func scoreTask(
        _ task: TaskSpec,
        model: AnyModel,
        cfg: ModelConfig,
        tokenizer: HFTokenizer?
    ) -> (accuracy: Double, total: Int) {
        let trials: [(prompt: String, expected: String)]
        switch task.name {
        case "sort-6": trials = buildSort6(trials: task.trials, seed: task.seed)
        case "reverse-16": trials = buildReverse16(trials: task.trials, seed: task.seed)
        default:
            fputs("run-bench: unknown task \(task.name), skipping\n", stderr)
            return (0, 0)
        }
        var correct = 0
        for (idx, trial) in trials.enumerated() {
            let got = GenerationUtils.greedyGenerate(
                model: model,
                cfg: cfg,
                tokenizer: tokenizer,
                prompt: trial.prompt,
                maxNewTokens: trial.expected.count + 2
            )
            let trimmed = got.drop(while: { $0.isWhitespace })
            let prefix = String(trimmed.prefix(trial.expected.count))
            if prefix == trial.expected { correct += 1 }
            if (idx + 1) % 50 == 0 {
                fputs("  \(task.name) \(idx + 1)/\(trials.count) correct=\(correct)\n", stderr)
            }
        }
        let acc = trials.isEmpty ? 0 : Double(correct) / Double(trials.count)
        return (acc, trials.count)
    }

    private static func parseTasks(_ csv: String, limit: Int?) -> [TaskSpec] {
        csv.split(separator: ",").map { raw in
            let name = raw.trimmingCharacters(in: .whitespaces)
            let trials = limit ?? 200
            switch name {
            case "sort-6": return TaskSpec(name: name, trials: trials, seed: 0x517)
            case "reverse-16": return TaskSpec(name: name, trials: trials, seed: 0x3EE5)
            default: return TaskSpec(name: name, trials: trials, seed: 1)
            }
        }
    }

    private struct Mulberry32 {
        var s: UInt32
        init(_ seed: UInt32) { self.s = seed }
        mutating func next() -> Double {
            s &+= 0x6D2B79F5
            var t = s
            t = (t ^ (t >> 15)) &* (t | 1)
            t ^= t &+ ((t ^ (t >> 7)) &* (t | 61))
            return Double((t ^ (t >> 14)) & 0xFFFF_FFFF) / 4_294_967_296.0
        }
    }

    private static func buildSort6(trials: Int, seed: UInt32) -> [(String, String)] {
        var rng = Mulberry32(seed)
        return (0..<trials).map { _ in
            let digits = (0..<6).map { _ in Int(rng.next() * 10) }
            return (
                "sort: \(digits.map(String.init).joined(separator: " ")) = ",
                digits.sorted().map(String.init).joined(separator: " ")
            )
        }
    }

    private static func buildReverse16(trials: Int, seed: UInt32) -> [(String, String)] {
        var rng = Mulberry32(seed)
        let alphabet = Array("abcdefghijklmnopqrstuvwxyz")
        return (0..<trials).map { _ in
            let len = 4 + Int(rng.next() * 13)
            let s = String((0..<len).map { _ in alphabet[Int(rng.next() * 26)] })
            return ("reverse: \(s) = ", String(s.reversed()))
        }
    }

    private static func append(
        _ outURL: URL,
        modelPath: String,
        modelName: String,
        modelStep: Int?,
        task: String,
        subtask: String?,
        metric: String,
        score: Double,
        n: Int,
        wall: Double
    ) {
        let row = EvalCompare.Row(
            run_id: UUID().uuidString,
            model_path: modelPath,
            model_name: modelName,
            model_step: modelStep,
            task: task,
            subtask: subtask,
            metric: metric,
            score: score,
            n_examples: n,
            wall_seconds: wall,
            harness_version: "tinygpt-run-bench"
        )
        do { try EvalCompare.Row.append(row, to: outURL) }
        catch { fputs("run-bench: could not append row: \(error)\n", stderr); exit(1) }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt run-bench --model <path> --out <jsonl> [options]

        Modes:
          --perplexity <txt|json|jsonl>   corpus loss/perplexity rows
          --tasks sort-6,reverse-16       exact-match synthetic tasks

        Options:
          --ctx N                         cap context below model max
          --batch N                       perplexity batch size (default: 8)
          --batches N                     perplexity batches (default: 32)
          --limit N                       synthetic task trials per task
          --model-name NAME               eval-compare display name
          --model-step N                  checkpoint step
        """)
        exit(code)
    }
}
