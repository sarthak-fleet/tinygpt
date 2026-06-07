import Foundation

/// `tinygpt eval-mteb` — score an embedder against MTEB/BEIR subtasks.
/// Wraps the Python `mteb` library via `scripts/eval-mteb-adapter.py` and
/// emits E0-conformant JSONL rows.
enum EvalMTEB {
    static func run(args: [String]) {
        var hfModel: String? = nil
        var embedModel: String? = nil
        var tasks = "BEIR/scifact,BEIR/nfcorpus,MTEB/StackOverflowDupQuestions"
        var limit = 500
        var outJsonl: String? = nil
        var modelName: String? = nil
        var modelStep: Int? = nil
        var baseline = false
        var workDir = "/tmp/tinygpt-mteb"
        var batchSize = 32

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--hf-model":   hfModel = args[i + 1]; i += 2
            case "--model":      embedModel = args[i + 1]; i += 2
            case "--tasks":      tasks = args[i + 1]; i += 2
            case "--limit":      limit = Int(args[i + 1]) ?? limit; i += 2
            case "--out":        outJsonl = args[i + 1]; i += 2
            case "--model-name": modelName = args[i + 1]; i += 2
            case "--model-step": modelStep = Int(args[i + 1]); i += 2
            case "--baseline":   baseline = true; i += 1
            case "--work-dir":   workDir = args[i + 1]; i += 2
            case "--batch":      batchSize = Int(args[i + 1]) ?? batchSize; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                } else if embedModel == nil && hfModel == nil {
                    embedModel = args[i]; i += 1
                } else {
                    fputs("unknown arg: \(args[i])\n", stderr); exitUsage()
                }
            }
        }

        guard hfModel != nil || embedModel != nil else {
            fputs("either --hf-model <dir|hub-id> or --model <embedder> is required\n", stderr)
            exitUsage()
        }
        if hfModel != nil && embedModel != nil {
            fputs("--hf-model and --model are mutually exclusive\n", stderr); exitUsage()
        }
        guard let outJsonl else { fputs("--out <path.jsonl> required\n", stderr); exitUsage() }

        let modelPath = hfModel ?? embedModel!
        let displayName = modelName ?? URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent
        let adapter = resolveAdapterScript()
        guard let adapter else {
            fputs("scripts/eval-mteb-adapter.py not found — run from repo root\n", stderr)
            exit(1)
        }
        let py = EvalHarnessSupport.resolveExecutable("python3")
            ?? URL(fileURLWithPath: "/usr/bin/python3")

        let workURL = URL(fileURLWithPath: workDir)
            .appendingPathComponent(UUID().uuidString.prefix(8).description)
        try? FileManager.default.createDirectory(at: workURL, withIntermediateDirectories: true)

        var subprocessArgs = [
            adapter.path,
            "--model", modelPath,
            "--tasks", tasks,
            "--limit", "\(limit)",
            "--batch-size", "\(batchSize)",
            "--work-dir", workURL.path,
            "--results-json", workURL.appendingPathComponent("mteb_results.json").path,
        ]
        if hfModel != nil { subprocessArgs.append(contentsOf: ["--hf"]) }

        let start = Date()
        print("""

        tinygpt eval-mteb
        -----------------
        model:  \(modelPath)
        name:   \(displayName)\(baseline ? " (baseline)" : "")\(modelStep.map { " · step \($0)" } ?? "")
        tasks:  \(tasks)
        limit:  \(limit) docs/query cap per task
        out:    \(outJsonl)

        """)

        let status = EvalHarnessSupport.runProcess(py, subprocessArgs)
        guard status == 0 else {
            fputs("mteb adapter failed with exit \(status)\n", stderr); exit(status)
        }

        let resultsURL = workURL.appendingPathComponent("mteb_results.json")
        guard let json = EvalHarnessSupport.jsonObject(resultsURL) as? [String: Any] else {
            fputs("could not read MTEB results at \(resultsURL.path)\n", stderr); exit(1)
        }

        let wall = -start.timeIntervalSinceNow
        var emitted = 0
        if let rows = json["rows"] as? [[String: Any]] {
            for row in rows {
                guard let task = row["task"] as? String,
                      let metric = row["metric"] as? String,
                      let score = row["score"] as? Double else { continue }
                let n = row["n_examples"] as? Int ?? 0
                let subtask = row["subtask"] as? String
                let common = EvalHarnessSupport.Common(
                    modelPath: modelPath,
                    outJsonl: outJsonl,
                    modelName: displayName,
                    modelStep: modelStep,
                    baseline: baseline
                )
                EvalHarnessSupport.appendRow(
                    common: common, task: task, subtask: subtask,
                    metric: metric, score: score, n: n, wall: wall / Double(max(rows.count, 1)),
                    harness: "mteb"
                )
                emitted += 1
            }
        }
        print("✓ wrote \(emitted) MTEB rows to \(outJsonl)")
    }

    private static func resolveAdapterScript() -> URL? {
        let cwd = FileManager.default.currentDirectoryPath
        let candidates = [
            URL(fileURLWithPath: cwd).appendingPathComponent("scripts/eval-mteb-adapter.py"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("scripts/eval-mteb-adapter.py"),
        ]
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        return nil
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt eval-mteb --hf-model <hub-id|dir> --out <jsonl> [options]
               tinygpt eval-mteb --model <embedder.tinygpt-embed> --out <jsonl> [options]

          --tasks CSV             MTEB/BEIR task names (default: scifact,nfcorpus,StackOverflowDupQ)
          --limit N               Cap docs per task (default: 500)
          --batch N               Encode batch size (default: 32)
          --model-name NAME       Display name in eval-compare
          --model-step N          Checkpoint step metadata
          --baseline              Mark rows as baseline reference
          --work-dir PATH         Scratch dir (default: /tmp/tinygpt-mteb)
        """)
        exit(code)
    }
}
