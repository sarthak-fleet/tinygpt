import Foundation

/// `tinygpt run-lm-eval` — wrap EleutherAI's lm-evaluation-harness as a
/// subprocess and emit results to the shared `EvalCompare.Row` schema
/// (E0). One wrapper unlocks MMLU + ARC + HellaSwag + PIQA + BoolQ +
/// WinoGrande + GSM8K + dozens of other tasks.
///
/// v1 takes any HuggingFace-loadable model directory (HF safetensors +
/// config.json + tokenizer files). Scoring a `.tinygpt` checkpoint is
/// v2 — needs a synthetic HF dir built around our weights so AutoModel
/// can load it. The architecture overlap is real, so v2 is the natural
/// follow-up. For now, this unlocks BASELINE comparison (SmolLM2,
/// Qwen3, Phi-mini all become scorable side-by-side).
///
/// Output: one JSONL row per (task, metric) pair to `--out <path>`. Use
/// `tinygpt eval-compare <jsonl>+ --by model` to render the comparison.
enum RunLmEval {
    static func run(args: [String]) {
        var hfModel: String? = nil
        var tasks: String = "arc_easy"
        var limit: Int = 0                         // 0 = full
        var batchSize: Int = 4
        var outJsonl: String? = nil
        var modelName: String? = nil
        var modelStep: Int? = nil
        var baseline: Bool = false
        var lmEvalBin: String = "lm_eval"          // PATH lookup
        var workDir: String = "/tmp/tinygpt-lm-eval"
        var dtype: String = "float32"
        var device: String = "mps"                 // Metal on Apple Silicon

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--hf-model":   hfModel = args[i+1]; i += 2
            case "--tasks":      tasks = args[i+1]; i += 2
            case "--limit":      limit = Int(args[i+1]) ?? limit; i += 2
            case "--batch":      batchSize = Int(args[i+1]) ?? batchSize; i += 2
            case "--out":        outJsonl = args[i+1]; i += 2
            case "--model-name": modelName = args[i+1]; i += 2
            case "--model-step": modelStep = Int(args[i+1]); i += 2
            case "--baseline":   baseline = true; i += 1
            case "--lm-eval":    lmEvalBin = args[i+1]; i += 2
            case "--work-dir":   workDir = args[i+1]; i += 2
            case "--dtype":      dtype = args[i+1]; i += 2
            case "--device":     device = args[i+1]; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        guard let hfModel = hfModel else { fputs("--hf-model <dir> required (v1 takes HF-loadable models only; .tinygpt support is v2)\n", stderr); exitUsage() }
        guard let outJsonl = outJsonl else { fputs("--out <path.jsonl> required (E0 schema target)\n", stderr); exitUsage() }

        // Default the friendly model name to the dir's last path component
        // — usually the snapshot SHA, which is informative enough for v1.
        let displayName = modelName ?? URL(fileURLWithPath: hfModel).lastPathComponent

        // Find lm_eval on PATH if a bare name was supplied. Mise + user-
        // local pip installs land under ~/.local/share/mise/.../bin/ which
        // doesn't always make it into bash's PATH.
        let lmEvalURL = Self.resolveExecutable(lmEvalBin)
        guard let lmEvalURL = lmEvalURL else {
            fputs("'\(lmEvalBin)' not found. Install with `pip install lm-eval`, then either ensure it's on PATH or pass --lm-eval /full/path/to/lm_eval\n", stderr)
            exit(1)
        }

        // Fresh work dir for this run; lm_eval writes a results JSON
        // under <work-dir>/<sanitized-pretrained>/results_*.json.
        let workURL = URL(fileURLWithPath: workDir)
            .appendingPathComponent(UUID().uuidString.prefix(8).description)
        let fm = FileManager.default
        try? fm.createDirectory(at: workURL, withIntermediateDirectories: true)

        let modelArgs = "pretrained=\(hfModel),dtype=\(dtype)"
        var subprocessArgs: [String] = [
            "--model", "hf",
            "--model_args", modelArgs,
            "--tasks", tasks,
            "--batch_size", "\(batchSize)",
            "--output_path", workURL.path,
            "--device", device,
        ]
        if limit > 0 { subprocessArgs.append(contentsOf: ["--limit", "\(limit)"]) }

        let limitTag = limit > 0 ? "limit \(limit)" : "FULL splits"
        print("""

        tinygpt run-lm-eval
        -------------------
        model:    \(hfModel)
        name:     \(displayName)\(baseline ? " (baseline)" : "")\(modelStep.map { " · step \($0)" } ?? "")
        tasks:    \(tasks)
        batch:    \(batchSize) · device \(device) · dtype \(dtype) · \(limitTag)
        out:      \(outJsonl)
        work:     \(workURL.path)
        lm_eval:  \(lmEvalURL.path)
        """)

        // Run the harness inline (no streaming UI; lm_eval prints its
        // own progress bars to stderr which the user sees naturally).
        let start = Date()
        let p = Process()
        p.executableURL = lmEvalURL
        p.arguments = subprocessArgs
        do { try p.run() }
        catch { fputs("couldn't launch lm_eval: \(error)\n", stderr); exit(1) }
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            fputs("lm_eval exited with code \(p.terminationStatus)\n", stderr); exit(p.terminationStatus)
        }
        let wall = -start.timeIntervalSinceNow

        // Locate the produced results JSON. lm_eval writes one file
        // per run under a deeply-nested sanitized-name subdir.
        guard let resultsURL = Self.findLatestResultsJSON(under: workURL) else {
            fputs("could not find results JSON under \(workURL.path)\n", stderr); exit(1)
        }
        print("  → results: \(resultsURL.path)")

        // Parse + emit one EvalCompare.Row per (task, primary-metric).
        // lm_eval emits "acc,none" and "acc_norm,none" — we pick the
        // primary metric per task by convention (acc for most, em for
        // generation tasks like gsm8k).
        let runId = UUID().uuidString
        let rowsEmitted = try? Self.emitRows(
            from: resultsURL,
            outJsonl: URL(fileURLWithPath: outJsonl),
            runId: runId,
            modelPath: hfModel,
            modelName: displayName,
            modelStep: modelStep,
            baseline: baseline,
            wallSeconds: wall
        )

        print("")
        print("✓ wrote \(rowsEmitted ?? 0) rows to \(outJsonl)")
        print("  → view: tinygpt eval-compare \(outJsonl) --by model")
        print("")
    }

    // MARK: - results → schema

    private static func emitRows(
        from resultsURL: URL, outJsonl: URL,
        runId: String, modelPath: String, modelName: String,
        modelStep: Int?, baseline: Bool, wallSeconds: Double
    ) throws -> Int {
        let data = try Data(contentsOf: resultsURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return 0
        }
        let results = (root["results"] as? [String: Any]) ?? [:]
        let nSamples = (root["n-samples"] as? [String: Any]) ?? [:]
        let harnessVersion = root["lm_eval_version"] as? String
        var emitted = 0

        for (taskName, raw) in results {
            guard let taskDict = raw as? [String: Any] else { continue }
            // n-samples shape: {"task": {"original": N, "effective": M}}
            let nDict = nSamples[taskName] as? [String: Any]
            let nExamples = (nDict?["effective"] as? Int)
                ?? (nDict?["original"] as? Int)
                ?? (taskDict["sample_len"] as? Int) ?? 0

            // lm_eval keys look like "acc,none" / "acc_norm,none" / "em,strict-match"
            // — split on `,` to pull the metric name. The "stderr" siblings are
            // skipped explicitly. Score must be Double-coercible.
            for (key, value) in taskDict {
                guard key.contains(","),
                      !key.contains("stderr"),
                      !key.contains("alias"),
                      let score = value as? Double
                else { continue }
                let metric = String(key.split(separator: ",")[0])
                let row = EvalCompare.Row(
                    run_id: runId,
                    model_path: modelPath,
                    model_name: modelName,
                    model_step: modelStep,
                    baseline: baseline,
                    task: taskName,
                    subtask: nil,
                    metric: metric,
                    score: score,
                    n_examples: nExamples,
                    wall_seconds: wallSeconds,
                    timestamp: nil,
                    harness_version: harnessVersion
                )
                try EvalCompare.Row.append(row, to: outJsonl)
                emitted += 1
            }
        }
        return emitted
    }

    private static func findLatestResultsJSON(under root: URL) -> URL? {
        let fm = FileManager.default
        guard let it = fm.enumerator(at: root, includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        var newest: (URL, Date)? = nil
        for case let url as URL in it {
            guard url.lastPathComponent.hasPrefix("results_"), url.pathExtension == "json" else { continue }
            let mod = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? Date.distantPast
            if newest == nil || mod > newest!.1 { newest = (url, mod) }
        }
        return newest?.0
    }

    private static func resolveExecutable(_ name: String) -> URL? {
        // Already a path?
        if name.contains("/") {
            let url = URL(fileURLWithPath: name)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
            return nil
        }
        // Try `which`.
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = ["which", name]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do { try proc.run() } catch { return nil }
        proc.waitUntilExit()
        let data = pipe.fileHandleForReading.availableData
        let path = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !path.isEmpty, FileManager.default.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        // Fall back to common locations.
        for candidate in [
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/.local/share/mise/installs/python/3.12.13/bin/\(name)",
            "/usr/local/bin/\(name)",
            "/opt/homebrew/bin/\(name)",
        ] {
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return URL(fileURLWithPath: candidate)
            }
        }
        return nil
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt run-lm-eval --hf-model <dir> --tasks <csv> --out <jsonl> [options]

        Wraps EleutherAI's lm-evaluation-harness (PyPI: `lm-eval`) and emits
        results to the shared E0 schema so they compare cleanly via
        `tinygpt eval-compare`.

        --hf-model <dir>      HuggingFace model directory (config.json +
                              model.safetensors + tokenizer files). REQUIRED.
                              .tinygpt support is v2 (needs a synthetic HF
                              dir built around our weights).
        --tasks <csv>         lm-eval task names (default: arc_easy).
                              Common: mmlu,arc_easy,hellaswag,piqa,
                              winogrande,boolq,gsm8k,gsm8k_cot
        --out <jsonl>         where to append EvalCompare.Row JSONL.
                              Re-run safe — rows append; eval-compare sorts.
        --limit N             cap eval examples per task (default 0 = full).
                              Useful for smoke runs.
        --batch N             batch size (default 4).
        --device mps|cpu|cuda (default: mps — Metal on Apple Silicon)
        --dtype float32|bfloat16|float16 (default: float32)
        --model-name NAME     display name in eval-compare. Default: dir's
                              last path component (snapshot SHA).
        --model-step N        training step (for our multi-checkpoint runs)
        --baseline            mark this row as a reference model
        --lm-eval <path>      override lm_eval CLI lookup
        --work-dir <path>     where lm_eval drops its results JSON
                              (default: /tmp/tinygpt-lm-eval/<uuid>)

        Examples:
          # Score SmolLM2 on ARC-Easy + HellaSwag, mark as baseline:
          tinygpt run-lm-eval \\
              --hf-model ~/.cache/huggingface/hub/.../SmolLM2-135M/snapshots/.../ \\
              --tasks arc_easy,hellaswag \\
              --baseline --model-name SmolLM2-135M \\
              --out ~/eval/baselines.jsonl

          # Score our trained huge-base at step 50000:
          tinygpt run-lm-eval \\
              --hf-model /tmp/huge-base-v1.step-50000.hf/ \\
              --tasks arc_easy --model-name tinygpt-huge-base-v1 --model-step 50000 \\
              --out ~/eval/huge-base-timeline.jsonl

          # Compare:
          tinygpt eval-compare ~/eval/*.jsonl --by model
          tinygpt eval-compare ~/eval/huge-base-timeline.jsonl --by step
        """)
        exit(code)
    }
}
