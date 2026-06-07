import Foundation

enum EvalBFCL {
    static func run(args: [String]) {
        var categories = "simple,multiple,parallel,parallel_multiple,relevance,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple"
        var root = "\(FileManager.default.homeDirectoryForCurrentUser.path)/.cache/tinygpt/datasets/_external/gorilla-bfcl/berkeley-function-call-leaderboard"
        var bfclModel = "openbmb/MiniCPM-SALA-FC"
        let parsed = EvalHarnessSupport.parseCommon(args, usage: { exitUsage() })
        var common = parsed.0
        var rest = parsed.1
        var i = 0
        while i < rest.count {
            switch rest[i] {
            case "--tasks", "--categories": categories = rest[i + 1]; i += 2
            case "--bfcl-root": root = rest[i + 1]; i += 2
            case "--bfcl-model": bfclModel = rest[i + 1]; i += 2
            default: fputs("unknown flag: \(rest[i])\n", stderr); exitUsage()
            }
        }
        common = EvalHarnessSupport.require(common, usage: { exitUsage() })
        guard let model = common.modelPath else { exitUsage() }

        let serve = EvalHarnessSupport.startServe(modelPath: model, port: common.servePort)
        defer { if serve.isRunning { serve.terminate() } }

        let work = URL(fileURLWithPath: "/tmp/tinygpt-bfcl-\(UUID().uuidString.prefix(8))")
        let resultDir = work.appendingPathComponent("result")
        let scoreDir = work.appendingPathComponent("score")
        try? FileManager.default.createDirectory(at: resultDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: scoreDir, withIntermediateDirectories: true)

        let py = EvalHarnessSupport.resolveExecutable("python3") ?? URL(fileURLWithPath: "/usr/bin/python3")
        let base = "http://127.0.0.1:\(common.servePort)/v1"
        let env = [
            "OPENAI_API_KEY": "tinygpt",
            "OPENAI_BASE_URL": base,
            "BFCL_PROJECT_ROOT": root
        ]
        let start = Date()
        let genArgs = [
            "-m", "bfcl_eval._llm_response_generation",
            "--model", bfclModel,
            "--test-category"
        ] + categories.split(separator: ",").map(String.init) + [
            "--result-dir", resultDir.path,
            "--skip-server-setup",
            "--allow-overwrite"
        ]
        let genStatus = EvalHarnessSupport.runProcess(py, genArgs, cwd: URL(fileURLWithPath: root), env: env)
        guard genStatus == 0 else { fputs("BFCL generation failed with exit \(genStatus)\n", stderr); exit(genStatus) }

        let evalArgs = [
            "-m", "bfcl_eval.eval_checker.eval_runner",
            "--model", bfclModel,
            "--test-category"
        ] + categories.split(separator: ",").map(String.init) + [
            "--result-dir", resultDir.path,
            "--score-dir", scoreDir.path,
            "--partial-eval"
        ]
        let evalStatus = EvalHarnessSupport.runProcess(py, evalArgs, cwd: URL(fileURLWithPath: root), env: env)
        guard evalStatus == 0 else { fputs("BFCL scoring failed with exit \(evalStatus)\n", stderr); exit(evalStatus) }

        let wall = -start.timeIntervalSinceNow
        guard let scoreURL = EvalHarnessSupport.latestJSON(under: scoreDir),
              let json = EvalHarnessSupport.jsonObject(scoreURL)
        else { fputs("could not find BFCL score JSON under \(scoreDir.path)\n", stderr); exit(1) }

        var emitted = 0
        for (name, score, n) in EvalHarnessSupport.numericScores(json) {
            let metric = name.split(separator: "/").last.map(String.init) ?? "accuracy"
            let subtask = name.split(separator: "/").dropLast().last.map(String.init)
            EvalHarnessSupport.appendRow(common: common, task: "bfcl", subtask: subtask,
                                         metric: metric, score: score, n: n, wall: wall,
                                         harness: "bfcl")
            emitted += 1
        }
        print("✓ wrote \(emitted) BFCL rows to \(common.outJsonl!)")
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt eval-bfcl <model.tinygpt|hf-dir> --out <jsonl> [options]

        --tokenizer <dir>       accepted for symmetry; serve reads model config
        --tasks <csv>           BFCL categories (default: core non-exec set)
        --limit N               reserved for future BFCL run-id sampling
        --serve-port N          local tinygpt serve port (default: 8097)
        --bfcl-root <dir>       local BFCL checkout
        --bfcl-model NAME       BFCL registry model id (default: openbmb/MiniCPM-SALA-FC)
        --model-name NAME       display name in eval-compare
        --model-step N          checkpoint step
        """)
        exit(code)
    }
}
