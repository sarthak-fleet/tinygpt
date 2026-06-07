import Foundation

enum JudgeShim {
    struct Item {
        let prompt: String
        let responseA: String
        let responseB: String?
    }

    static func run(args: [String]) {
        var input: String?
        var mode = "pairwise"
        var judgeModel: String?
        var out: String?
        var servePort = 8102
        var modelName = "judge-run"
        var limit = 0

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--mode": mode = args[i + 1]; i += 2
            case "--judge-model": judgeModel = args[i + 1]; i += 2
            case "--out": out = args[i + 1]; i += 2
            case "--serve-port": servePort = Int(args[i + 1]) ?? servePort; i += 2
            case "--model-name": modelName = args[i + 1]; i += 2
            case "--limit": limit = Int(args[i + 1]) ?? limit; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                input = args[i]; i += 1
            }
        }
        guard let input, let judgeModel, let out else {
            fputs("input JSONL, --judge-model, and --out are required\n", stderr); exitUsage()
        }

        let serve = EvalHarnessSupport.startServe(modelPath: judgeModel, port: servePort)
        defer { if serve.isRunning { serve.terminate() } }
        let base = "http://127.0.0.1:\(servePort)/v1"
        let items = load(input).prefix(limit > 0 ? limit : Int.max)
        let common = EvalHarnessSupport.Common(
            modelPath: judgeModel,
            tokenizer: nil,
            outJsonl: out,
            modelName: modelName,
            modelStep: nil,
            baseline: false,
            limit: limit,
            servePort: servePort
        )
        let start = Date()
        var scores: [Double] = []
        for item in items {
            if mode == "rate" {
                let prompt = ratePrompt(item)
                let text = EvalHarnessSupport.completion(baseURL: base, prompt: prompt, maxTokens: 96) ?? ""
                let rating = parseRating(text)
                scores.append(rating / 10.0)
            } else {
                let prompt = pairwisePrompt(item)
                let text = EvalHarnessSupport.completion(baseURL: base, prompt: prompt, maxTokens: 96) ?? ""
                scores.append(parseWinner(text) == "A" ? 1.0 : 0.0)
            }
        }
        let wall = -start.timeIntervalSinceNow
        let avg = scores.isEmpty ? 0 : scores.reduce(0, +) / Double(scores.count)
        EvalHarnessSupport.appendRow(
            common: common,
            task: "judge",
            subtask: mode,
            metric: mode == "rate" ? "judge_rating" : "judge_win_rate",
            score: avg,
            n: scores.count,
            wall: wall,
            harness: "local-judge"
        )
        print("✓ wrote judge row to \(out)")
    }

    static func load(_ path: String) -> [Item] {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            guard let data = String(line).data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            let prompt = obj["prompt"] as? String ?? ""
            let a = (obj["response_a"] as? String) ?? (obj["response"] as? String) ?? ""
            let b = obj["response_b"] as? String
            return Item(prompt: prompt, responseA: a, responseB: b)
        }
    }

    static func ratePrompt(_ item: Item) -> String {
        """
        You are grading an assistant response. Rate it from 1 to 10 for correctness, helpfulness, and directness.
        Return exactly: Rating: <number>

        Prompt:
        \(item.prompt)

        Response:
        \(item.responseA)

        Rating:
        """
    }

    static func pairwisePrompt(_ item: Item) -> String {
        """
        Choose the better assistant response. Consider correctness, helpfulness, and directness.
        Return exactly one letter: A or B.

        Prompt:
        \(item.prompt)

        Response A:
        \(item.responseA)

        Response B:
        \(item.responseB ?? "")

        Winner:
        """
    }

    static func parseRating(_ text: String) -> Double {
        let regex = try? NSRegularExpression(pattern: #"([0-9]+(?:\.[0-9]+)?)"#)
        let ns = text as NSString
        guard let m = regex?.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)),
              m.range.location != NSNotFound
        else { return 0 }
        return min(10, max(0, Double(ns.substring(with: m.range)) ?? 0))
    }

    static func parseWinner(_ text: String) -> String {
        let upper = text.uppercased()
        if upper.contains("A") && !upper.contains("B") { return "A" }
        if upper.contains("B") { return "B" }
        return "B"
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt judge <input.jsonl> --judge-model <model.tinygpt|hf-dir> --out <jsonl> [options]

        input rate rows:     {"prompt":"...","response":"..."}
        input pairwise rows: {"prompt":"...","response_a":"...","response_b":"..."}

        --mode rate|pairwise   default: pairwise
        --serve-port N         default: 8102
        --limit N              cap rows
        --model-name NAME      display name in eval-compare
        """)
        exit(code)
    }
}
