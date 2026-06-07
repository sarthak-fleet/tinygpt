import Foundation

enum EvalHumanEval {
    struct Problem {
        let suite: String
        let id: String
        let prompt: String
        let tests: String
        let entryPoint: String
    }

    static func run(args: [String]) {
        var suites = "humaneval,mbpp"
        var sandboxPath = "scripts/humaneval-sandbox/target/release/humaneval-sandbox"
        let parsed = EvalHarnessSupport.parseCommon(args, usage: { exitUsage() })
        var common = parsed.0
        var rest = parsed.1
        var i = 0
        while i < rest.count {
            switch rest[i] {
            case "--suites": suites = rest[i + 1]; i += 2
            case "--sandbox": sandboxPath = rest[i + 1]; i += 2
            default: fputs("unknown flag: \(rest[i])\n", stderr); exitUsage()
            }
        }
        common = EvalHarnessSupport.require(common, usage: { exitUsage() })
        guard let model = common.modelPath else { exitUsage() }
        let sandbox = URL(fileURLWithPath: sandboxPath)
        guard FileManager.default.isExecutableFile(atPath: sandbox.path) else {
            fputs("sandbox binary missing; run `cd scripts/humaneval-sandbox && cargo build --release`\n", stderr); exit(1)
        }

        let serve = EvalHarnessSupport.startServe(modelPath: model, port: common.servePort)
        defer { if serve.isRunning { serve.terminate() } }
        let base = "http://127.0.0.1:\(common.servePort)/v1"
        let selected = suites.split(separator: ",").map(String.init)
        let problems = selected.flatMap { loadSuite($0) }
            .prefix(common.limit > 0 ? common.limit : Int.max)
        let start = Date()
        var bySuite: [String: (pass: Int, total: Int)] = [:]
        for p in problems {
            let completion = EvalHarnessSupport.completion(baseURL: base, prompt: p.prompt, maxTokens: 256) ?? ""
            let code = p.prompt + "\n" + completion
            let work = URL(fileURLWithPath: "/tmp/tinygpt-humaneval-\(UUID().uuidString.prefix(8))")
            try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
            let codeURL = work.appendingPathComponent("code.py")
            let testURL = work.appendingPathComponent("test.py")
            try? code.write(to: codeURL, atomically: true, encoding: String.Encoding.utf8)
            try? p.tests.write(to: testURL, atomically: true, encoding: String.Encoding.utf8)
            let pipe = Pipe()
            let proc = Process()
            proc.executableURL = sandbox
            proc.arguments = ["--code", codeURL.path, "--test", testURL.path, "--timeout", "10", "--memory-mb", "256"]
            proc.standardOutput = pipe
            proc.standardError = Pipe()
            try? proc.run()
            proc.waitUntilExit()
            let data = pipe.fileHandleForReading.availableData
            let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let passed = obj?["passed"] as? Bool ?? false
            bySuite[p.suite, default: (0, 0)].pass += passed ? 1 : 0
            bySuite[p.suite, default: (0, 0)].total += 1
            EvalHarnessSupport.appendRow(common: common, task: p.suite, subtask: p.id,
                                         metric: "pass@1", score: passed ? 1 : 0, n: 1,
                                         wall: EvalHarnessSupport.doubleValue(obj?["wall_seconds"]) ?? 0,
                                         harness: "humaneval-sandbox")
            try? FileManager.default.removeItem(at: work)
        }
        let wall = -start.timeIntervalSinceNow
        for (suite, counts) in bySuite {
            let score = counts.total == 0 ? 0 : Double(counts.pass) / Double(counts.total)
            EvalHarnessSupport.appendRow(common: common, task: suite, subtask: "aggregate",
                                         metric: "pass@1", score: score, n: counts.total,
                                         wall: wall, harness: "humaneval-sandbox")
        }
        print("✓ wrote HumanEval/MBPP rows to \(common.outJsonl!)")
    }

    static func loadSuite(_ suite: String) -> [Problem] {
        let root = "\(FileManager.default.homeDirectoryForCurrentUser.path)/.cache/tinygpt/datasets"
        let path = suite == "mbpp" ? "\(root)/mbpp-test.jsonl" : "\(root)/humaneval-test.jsonl"
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            guard let data = String(line).data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            if suite == "mbpp" {
                let id = "\(obj["task_id"] ?? obj["id"] ?? UUID().uuidString)"
                let prompt = (obj["prompt"] as? String) ?? (obj["text"] as? String) ?? ""
                let tests = ((obj["test_list"] as? [String]) ?? []).joined(separator: "\n")
                return Problem(suite: "mbpp", id: id, prompt: prompt + "\n", tests: "def check(candidate):\n    \(tests.replacingOccurrences(of: "\n", with: "\n    "))", entryPoint: "candidate")
            }
            let id = obj["task_id"] as? String ?? UUID().uuidString
            let prompt = obj["prompt"] as? String ?? ""
            let test = obj["test"] as? String ?? ""
            let entry = obj["entry_point"] as? String ?? "candidate"
            return Problem(suite: "humaneval", id: id, prompt: prompt, tests: "\(test)\n\ncandidate = \(entry)", entryPoint: entry)
        }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt eval-humaneval <model.tinygpt|hf-dir> --out <jsonl> [options]

        --suites humaneval,mbpp    suites to run
        --limit N                  cap total problems across selected suites
        --serve-port N             local tinygpt serve port
        --sandbox <path>           humaneval-sandbox binary
        """)
        exit(code)
    }
}
