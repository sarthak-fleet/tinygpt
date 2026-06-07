import Foundation

enum EvalTauBench {
    static func run(args: [String]) {
        var envName = "retail"
        var root = "\(FileManager.default.homeDirectoryForCurrentUser.path)/.cache/tinygpt/datasets/_external/tau-bench"
        var userModel = "gpt-4o"
        var userProvider = "openai"
        let parsed = EvalHarnessSupport.parseCommon(args, usage: { exitUsage() })
        var common = parsed.0
        var rest = parsed.1
        var i = 0
        while i < rest.count {
            switch rest[i] {
            case "--env": envName = rest[i + 1]; i += 2
            case "--tau-root": root = rest[i + 1]; i += 2
            case "--user-model": userModel = rest[i + 1]; i += 2
            case "--user-provider": userProvider = rest[i + 1]; i += 2
            default: fputs("unknown flag: \(rest[i])\n", stderr); exitUsage()
            }
        }
        common = EvalHarnessSupport.require(common, usage: { exitUsage() })
        guard let model = common.modelPath else { exitUsage() }

        if ProcessInfo.processInfo.environment["OPENAI_API_KEY"] == nil && userProvider == "openai" {
            fputs("tau-bench requires OPENAI_API_KEY for the user simulator in v1\n", stderr)
        }

        let serve = EvalHarnessSupport.startServe(modelPath: model, port: common.servePort)
        defer { if serve.isRunning { serve.terminate() } }

        let work = URL(fileURLWithPath: "/tmp/tinygpt-tau-\(UUID().uuidString.prefix(8))")
        try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        let py = EvalHarnessSupport.resolveExecutable("python3") ?? URL(fileURLWithPath: "/usr/bin/python3")
        let base = "http://127.0.0.1:\(common.servePort)/v1"
        var env = ProcessInfo.processInfo.environment
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        env["OPENAI_API_KEY"] = env["OPENAI_API_KEY"] ?? "tinygpt"
        let end = common.limit > 0 ? "\(common.limit)" : "-1"
        let start = Date()
        let status = EvalHarnessSupport.runProcess(py, [
            "run.py",
            "--agent-strategy", "tool-calling",
            "--env", envName,
            "--model", "tinygpt",
            "--model-provider", "openai",
            "--user-model", userModel,
            "--user-model-provider", userProvider,
            "--user-strategy", "llm",
            "--max-concurrency", "1",
            "--start-index", "0",
            "--end-index", end,
            "--log-dir", work.path
        ], cwd: URL(fileURLWithPath: root), env: env)
        guard status == 0 else { fputs("tau-bench failed with exit \(status)\n", stderr); exit(status) }
        let wall = -start.timeIntervalSinceNow

        guard let resultURL = EvalHarnessSupport.latestJSON(under: work),
              let results = EvalHarnessSupport.jsonObject(resultURL) as? [[String: Any]]
        else { fputs("could not find tau-bench result JSON under \(work.path)\n", stderr); exit(1) }
        let rewards = results.compactMap { EvalHarnessSupport.doubleValue($0["reward"]) }
        let score = rewards.isEmpty ? 0 : rewards.reduce(0, +) / Double(rewards.count)
        EvalHarnessSupport.appendRow(common: common, task: "tau-bench", subtask: envName,
                                     metric: "pass_at_1", score: score, n: rewards.count,
                                     wall: wall, harness: "tau-bench")
        print("✓ wrote tau-bench row to \(common.outJsonl!)")
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt eval-tau-bench <model.tinygpt|hf-dir> --out <jsonl> [options]

        --env retail|airline      tau-bench env (default: retail)
        --limit N                 first N tasks (default: full)
        --serve-port N            local tinygpt serve port (default: 8097)
        --tau-root <dir>          local tau-bench checkout
        --user-model NAME         user simulator model (default: gpt-4o)
        --user-provider NAME      user simulator provider (default: openai)
        """)
        exit(code)
    }
}
