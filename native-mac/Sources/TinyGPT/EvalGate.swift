import Foundation
import TinyGPTModel

/// `tinygpt eval-gate` (B32) — run a project's declared eval suites against
/// a baseline and exit non-zero when any suite regresses past its threshold.
///
/// The judgement lives in `TinyGPTModel.EvalGate` (pure, unit-tested). This
/// command is the orchestration shell: resolve the spec, obtain candidate
/// rows (from a `--candidate` JSONL or by running the suites), compare,
/// print, write `gate-result.json`, and set the exit code.
enum EvalGateCommand {
    static func run(args: [String]) {
        var specPath: String? = nil
        var candidatePath: String? = nil
        var baselineOverride: String? = nil
        var outPath: String? = nil
        var thresholdOverride: Double? = nil
        var passes = 1
        var updateBaseline = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--spec": specPath = value(args, &i)
            case "--candidate": candidatePath = value(args, &i)
            case "--baseline": baselineOverride = value(args, &i)
            case "--out": outPath = value(args, &i)
            case "--threshold": thresholdOverride = Double(value(args, &i) ?? "")
            case "--passes": passes = Int(value(args, &i) ?? "") ?? 1
            case "--update-baseline": updateBaseline = true; i += 1
            case "-h", "--help": exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }

        // 1. Resolve the spec.
        let baseSpec = resolveSpec(explicitPath: specPath)
        let baselinePath = baselineOverride ?? baseSpec.baseline
        let spec = EvalGate.Spec(
            baseline: baselinePath,
            defaultThreshold: thresholdOverride ?? baseSpec.defaultThreshold,
            suites: baseSpec.suites)

        // 2. Obtain candidate rows + the JSONL file they came from (so
        //    --update-baseline can copy full EvalCompare.Row fidelity).
        let (candidateRows, candidateFile) = resolveCandidate(
            candidatePath: candidatePath, spec: spec, passes: passes)

        // 3. --update-baseline: re-stamp the baseline from the candidate run.
        if updateBaseline {
            guard let src = candidateFile else {
                fputs("--update-baseline needs candidate rows from a file or a suite run\n", stderr)
                exit(2)
            }
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: src))
                try data.write(to: URL(fileURLWithPath: baselinePath), options: .atomic)
                print("✓ baseline re-stamped: \(baselinePath) (\(candidateRows.count) rows)")
                exit(0)
            } catch {
                fputs("could not write baseline \(baselinePath): \(error)\n", stderr); exit(1)
            }
        }

        // 4. Load baseline + evaluate.
        let baselineRows: [EvalGate.Row]
        do {
            baselineRows = try EvalGate.loadRows(fromJSONLAt: baselinePath)
        } catch {
            fputs("could not read baseline \(baselinePath): \(error)\n", stderr)
            fputs("hint: run with --update-baseline once to stamp the first baseline\n", stderr)
            exit(2)
        }

        let averaged = passes > 1 ? EvalGate.averagedByKey(candidateRows) : candidateRows
        let report = EvalGate.evaluate(baseline: baselineRows, candidate: averaged, spec: spec)

        // 5. Print + persist + exit code.
        printReport(report, spec: spec, passes: passes)
        writeResult(report, to: outPath ?? "gate-result.json")
        exit(report.passed ? 0 : 1)
    }

    // MARK: - spec resolution

    private static func resolveSpec(explicitPath: String?) -> EvalGate.Spec {
        if let p = explicitPath {
            guard let s = loadStandaloneSpec(p) else {
                fputs("could not parse eval-gate spec at \(p)\n", stderr); exit(2)
            }
            return s
        }
        // Default search order: ./eval-gate.json, then the project file.
        if FileManager.default.fileExists(atPath: "eval-gate.json"),
           let s = loadStandaloneSpec("eval-gate.json") {
            return s
        }
        if FileManager.default.fileExists(atPath: "tinygpt.project.json") {
            if let manifest = try? ProjectManifest.load(path: "tinygpt.project.json"),
               let s = manifest.eval {
                return s
            }
            fputs("tinygpt.project.json has no \"eval\" block; add one or pass --spec\n", stderr)
            exit(2)
        }
        fputs("no eval-gate spec found (looked for --spec, ./eval-gate.json, ./tinygpt.project.json)\n", stderr)
        exit(2)
    }

    private static func loadStandaloneSpec(_ path: String) -> EvalGate.Spec? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        return try? EvalGate.Spec.jsonDecoder.decode(EvalGate.Spec.self, from: data)
    }

    // MARK: - candidate resolution

    /// Returns (rows, sourceJSONLPath). When `--candidate` is given we read
    /// it directly (the GPU-free CI path). Otherwise we run each declared
    /// suite's command, appending to one temp JSONL, K times for K passes.
    private static func resolveCandidate(candidatePath: String?,
                                         spec: EvalGate.Spec,
                                         passes: Int) -> ([EvalGate.Row], String?) {
        if let p = candidatePath {
            guard let rows = try? EvalGate.loadRows(fromJSONLAt: p) else {
                fputs("could not read candidate \(p)\n", stderr); exit(2)
            }
            return (rows, p)
        }

        // Run the suites. Each suite command is expected to write
        // EvalCompare.Row JSONL to the path we pass as TINYGPT_EVAL_OUT.
        guard !spec.suites.isEmpty else {
            fputs("no --candidate and spec declares no suites to run\n", stderr); exit(2)
        }
        let temp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tinygpt-gate-\(UUID().uuidString.prefix(8)).jsonl")
        for pass in 0..<max(1, passes) {
            for suite in spec.suites {
                guard let cmd = suite.command, let exe = cmd.first else {
                    fputs("suite '\(suite.name)' has no command; cannot run (pass --candidate instead)\n", stderr)
                    exit(2)
                }
                let exeURL = EvalHarnessSupport.resolveExecutable(exe)
                    ?? URL(fileURLWithPath: exe)
                // Process() does no shell expansion, so substitute the
                // $TINYGPT_EVAL_OUT token in the argv ourselves (and still
                // export it as env for harnesses that read it directly).
                let expanded = cmd.dropFirst().map {
                    $0.replacingOccurrences(of: "${TINYGPT_EVAL_OUT}", with: temp.path)
                        .replacingOccurrences(of: "$TINYGPT_EVAL_OUT", with: temp.path)
                }
                let status = EvalHarnessSupport.runProcess(
                    exeURL, expanded,
                    env: ["TINYGPT_EVAL_OUT": temp.path])
                if status != 0 {
                    fputs("suite '\(suite.name)' exited \(status) on pass \(pass + 1)\n", stderr)
                    exit(status)
                }
            }
        }
        let rows = (try? EvalGate.loadRows(fromJSONLAt: temp.path)) ?? []
        return (rows, temp.path)
    }

    // MARK: - render

    private static func printReport(_ report: EvalGate.Report, spec: EvalGate.Spec, passes: Int) {
        let nameW = max(12, report.suites.map { ($0.name + "/" + ($0.subtask ?? $0.metric)).count }.max() ?? 12)
        print("")
        print("eval-gate — baseline \(spec.baseline)" + (passes > 1 ? "  (K=\(passes) passes, mean)" : ""))
        let header = "suite".padding(toLength: nameW, withPad: " ", startingAt: 0)
            + "  base    cand    Δpp     thr    verdict"
        print(header)
        print(String(repeating: "─", count: header.count))
        for s in report.suites {
            let label = (s.name + "/" + (s.subtask ?? s.metric)).padding(toLength: nameW, withPad: " ", startingAt: 0)
            let base = s.baselineScore.map { String(format: "%.3f", $0) } ?? "  —  "
            let cand = s.candidateScore.map { String(format: "%.3f", $0) } ?? "  —  "
            let delta = s.deltaPP.map { String(format: "%+.1f", $0) } ?? "  —  "
            let thr = String(format: "%.1f", s.thresholdPP)
            let mark: String
            switch s.verdict {
            case .pass: mark = "PASS"
            case .fail: mark = "FAIL ✗"
            case .missing: mark = "MISSING"
            }
            print("\(label)  \(base)   \(cand)   \(delta.padding(toLength: 6, withPad: " ", startingAt: 0)) \(thr.padding(toLength: 5, withPad: " ", startingAt: 0))  \(mark)")
        }
        print(String(repeating: "─", count: header.count))
        if report.passed {
            print("✓ gate PASSED — \(report.suites.count) suites, 0 regressions"
                + (report.missingCount > 0 ? " (\(report.missingCount) missing baseline)" : ""))
        } else {
            print("✗ gate FAILED — \(report.failedCount) suite(s) regressed past threshold")
        }
        print("")
    }

    private static func writeResult(_ report: EvalGate.Report, to path: String) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? enc.encode(report) else { return }
        try? data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    // MARK: - arg helper

    private static func value(_ args: [String], _ i: inout Int) -> String? {
        guard i + 1 < args.count else { i += 1; return nil }
        let v = args[i + 1]; i += 2; return v
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt eval-gate [options]

        Run a project's declared eval suites against a baseline and exit
        non-zero when any suite regresses past its threshold. Designed to run
        on a self-hosted Mac runner so the model never leaves the device.

        Spec resolution (first found wins):
          --spec <path>            explicit eval-gate.json
          ./eval-gate.json         standalone spec in the cwd
          ./tinygpt.project.json   the optional "eval" block (B31)

        Options:
          --candidate <jsonl>      compare these rows instead of running the
                                   suites (CI/test path; no GPU needed)
          --baseline <jsonl>       override the spec's baseline path
          --threshold <pp>         override the default regression tolerance
                                   (percentage points; per-suite values win)
          --passes <K>             run each suite K times, gate on the mean
          --update-baseline        re-stamp the baseline from this run, exit 0
          --out <path>             gate-result.json location (default: ./)

        Exit code: 0 if all suites pass, 1 if any regresses past threshold.
        """)
        exit(code)
    }
}
