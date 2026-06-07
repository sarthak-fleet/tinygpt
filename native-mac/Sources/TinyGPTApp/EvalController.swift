import Foundation
import AppKit

enum EvalMode: String, CaseIterable, Identifiable {
    case quick = "Score"
    case emergence = "Sweep"
    case custom = "Custom"
    var id: String { rawValue }

    /// One-line subtitle that explains what the mode actually does.
    var subtitle: String {
        switch self {
        case .quick:     return "Score one model on a set of tasks. Minutes."
        case .emergence: return "Score every checkpoint of a training run."
        case .custom:    return "Score on YOUR JSONL of (prompt, expected)."
        }
    }
}

enum CustomEvalMetric: String, CaseIterable, Identifiable {
    case exact = "exact"
    case contains = "contains"
    case regex = "regex"
    var id: String { rawValue }
}

struct AppEvalRow: Identifiable, Codable, Hashable {
    var id = UUID()
    let run_id: String
    let model_path: String
    let model_name: String
    let model_step: Int?
    let baseline: Bool
    let task: String
    let subtask: String?
    let metric: String
    let score: Double
    let n_examples: Int
    let wall_seconds: Double
    let timestamp: String
    let harness_version: String?
}

struct CustomEvalCase: Identifiable, Hashable {
    var id = UUID()
    let prompt: String
    let expected: String
    let output: String
    let passed: Bool
}

struct EvalHistoryItem: Identifiable, Codable, Hashable {
    var id = UUID()
    let timestamp: Date
    let mode: String
    let modelPath: String
    let resultsPath: String
    let rowCount: Int
}

@MainActor
final class EvalController: ObservableObject {
    @Published var mode: EvalMode = .quick
    @Published var modelPath: String = ""
    @Published var runStemPath: String = ""
    @Published var customPath: String = ""
    @Published var selectedTasks: Set<String> = ["arc_easy"]
    @Published var limitIndex: Int = 1
    @Published var includeBaseline: Bool = false
    @Published var customMetric: CustomEvalMetric = .exact
    @Published var log: String = ""
    @Published var rows: [AppEvalRow] = []
    @Published var customRows: [CustomEvalCase] = []
    @Published var isRunning: Bool = false
    @Published var resultsPath: String = ""
    @Published var history: [EvalHistoryItem] = []
    @Published var lastError: String? = nil

    let commonTasks = ["arc_easy", "hellaswag", "piqa", "gsm8k", "mmlu", "truthfulqa", "bfcl", "humaneval", "mbpp", "sort-6", "reverse-16"]
    let limits = [10, 30, 50, 100, 500, 0]

    private var process: Process?
    private var pollTask: Task<Void, Never>?
    private var sweepTask: Task<Void, Never>?
    private var customTask: Task<Void, Never>?
    private static let historyKey = "tinygpt.eval.history.v1"
    private static let historyMax = 20

    init() {
        loadHistory()
        resultsPath = defaultResultsURL().path
    }

    var selectedLimit: Int { limits[min(max(0, limitIndex), limits.count - 1)] }
    var tasksCSV: String { selectedTasks.sorted().joined(separator: ",") }

    func pickModel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [.data, .directory]
        panel.message = "Pick a .tinygpt checkpoint or HuggingFace model directory."
        if panel.runModal() == .OK, let url = panel.url {
            modelPath = url.path
        }
    }

    func pickRunStem() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.data]
        panel.message = "Pick a .tinygpt checkpoint with .step-N siblings."
        if panel.runModal() == .OK, let url = panel.url {
            runStemPath = url.path
            modelPath = url.path
        }
    }

    func pickCustomJSONL() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, .plainText, .utf8PlainText, .data]
        panel.message = "Pick a JSONL file with {prompt, expected} rows."
        if panel.runModal() == .OK, let url = panel.url {
            customPath = url.path
        }
    }

    func run() {
        switch mode {
        case .quick: runQuick()
        case .emergence: runSweep()
        case .custom: runCustom()
        }
    }

    func cancel() {
        process?.terminate()
        process = nil
        pollTask?.cancel()
        sweepTask?.cancel()
        customTask?.cancel()
        isRunning = false
        appendLog("\n[cancelled]\n")
    }

    func revealResults() {
        guard !resultsPath.isEmpty else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: resultsPath)])
    }

    func loadHistoryItem(_ item: EvalHistoryItem) {
        resultsPath = item.resultsPath
        parseRows(from: URL(fileURLWithPath: item.resultsPath))
        mode = EvalMode(rawValue: item.mode) ?? .quick
        modelPath = item.modelPath
    }

    /// Load a saved E0 JSONL artifact (e.g. one of docs/artifacts/*.jsonl)
    /// directly into the table. Replaces what the dropped Compare workspace
    /// used to surface — folds it into Eval cleanly.
    func loadArtifact(path: String) {
        resultsPath = path
        parseRows(from: URL(fileURLWithPath: path))
    }

    private func runQuick() {
        guard !modelPath.isEmpty else { lastError = "pick a model first"; return }
        guard !selectedTasks.isEmpty else { lastError = "select at least one task"; return }
        resetForRun()
        let out = defaultResultsURL()
        resultsPath = out.path
        let tasks = selectedTasks
        if tasks.allSatisfy({ $0 == "sort-6" || $0 == "reverse-16" }) {
            var args = ["run-bench", "--model", modelPath, "--tasks", tasks.sorted().joined(separator: ","), "--out", out.path]
            if selectedLimit > 0 { args += ["--limit", "\(selectedLimit)"] }
            launch(args: args, outURL: out, modeLabel: mode.rawValue)
        } else {
            var lmTasks = tasks.filter { $0 != "bfcl" && $0 != "humaneval" && $0 != "mbpp" }
            if lmTasks.isEmpty { lmTasks = ["arc_easy"] }
            var args = [
                "run-lm-eval",
                "--tinygpt-model", modelPath,
                "--tasks", lmTasks.sorted().joined(separator: ","),
                "--out", out.path
            ]
            if selectedLimit > 0 { args += ["--limit", "\(selectedLimit)"] }
            launch(args: args, outURL: out, modeLabel: mode.rawValue)
        }
    }

    private func runSweep() {
        let root = runStemPath.isEmpty ? modelPath : runStemPath
        guard !root.isEmpty else { lastError = "pick a run checkpoint first"; return }
        resetForRun()
        let out = defaultResultsURL()
        resultsPath = out.path
        isRunning = true
        let checkpoints = checkpointHistory(for: URL(fileURLWithPath: root))
        appendLog("sweep: \(checkpoints.count) checkpoint(s)\n")
        sweepTask = Task { @MainActor in
            for checkpoint in checkpoints {
                if Task.isCancelled { break }
                appendLog("\n==> \(checkpoint.lastPathComponent)\n")
                let step = Self.step(from: checkpoint)
                var args = ["run-bench", "--model", checkpoint.path, "--tasks", selectedTasks.sorted().joined(separator: ","), "--out", out.path]
                if let step { args += ["--model-step", "\(step)"] }
                if selectedLimit > 0 { args += ["--limit", "\(selectedLimit)"] }
                await runProcessSync(args: args)
                parseRows(from: out)
            }
            isRunning = false
            saveHistory(modeLabel: mode.rawValue, outURL: out)
        }
    }

    private func runCustom() {
        guard !modelPath.isEmpty else { lastError = "pick a model first"; return }
        guard !customPath.isEmpty else { lastError = "pick a custom JSONL first"; return }
        resetForRun()
        let out = defaultResultsURL()
        resultsPath = out.path
        isRunning = true
        customTask = Task { @MainActor in
            let started = Date()
            let cases = loadCustomCases(path: customPath)
            var passed = 0
            var rendered: [CustomEvalCase] = []
            for item in cases {
                if Task.isCancelled { break }
                let output = runSample(prompt: item.prompt)
                let ok = score(output: output, expected: item.expected, metric: customMetric)
                if ok { passed += 1 }
                rendered.append(CustomEvalCase(prompt: item.prompt, expected: item.expected, output: output, passed: ok))
                customRows = rendered
            }
            let score = cases.isEmpty ? 0 : Double(passed) / Double(cases.count)
            let row = AppEvalRow(
                run_id: UUID().uuidString,
                model_path: modelPath,
                model_name: URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent,
                model_step: nil,
                baseline: false,
                task: "custom",
                subtask: URL(fileURLWithPath: customPath).lastPathComponent,
                metric: customMetric.rawValue,
                score: score,
                n_examples: cases.count,
                wall_seconds: -started.timeIntervalSinceNow,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                harness_version: "tinygpt-app-custom"
            )
            rows = [row]
            writeRows([row], to: out)
            appendLog(String(format: "custom score %.3f (%d/%d)\n", score, passed, cases.count))
            isRunning = false
            saveHistory(modeLabel: mode.rawValue, outURL: out)
        }
    }

    private func launch(args: [String], outURL: URL, modeLabel: String) {
        guard let cli = locateCLI() else {
            lastError = "tinygpt CLI not found; build native-mac first"
            return
        }
        isRunning = true
        let p = Process()
        p.executableURL = cli
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(chunk) }
        }
        do {
            try p.run()
            process = p
            pollTask = Task { @MainActor in
                while p.isRunning {
                    parseRows(from: outURL)
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
                pipe.fileHandleForReading.readabilityHandler = nil
                parseRows(from: outURL)
                isRunning = false
                process = nil
                saveHistory(modeLabel: modeLabel, outURL: outURL)
            }
        } catch {
            isRunning = false
            lastError = "couldn't launch eval: \(error)"
        }
    }

    private func runProcessSync(args: [String]) async {
        guard let cli = locateCLI() else {
            appendLog("tinygpt CLI not found\n")
            return
        }
        let p = Process()
        p.executableURL = cli
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(chunk) }
        }
        do {
            try p.run()
            process = p
            p.waitUntilExit()
            pipe.fileHandleForReading.readabilityHandler = nil
            process = nil
        } catch {
            appendLog("spawn failed: \(error)\n")
        }
    }

    private func locateCLI() -> URL? {
        let fm = FileManager.default
        if let exec = Bundle.main.executableURL {
            var dir = exec.deletingLastPathComponent()
            for _ in 0..<8 {
                let candidate = dir.appendingPathComponent(".build/arm64-apple-macosx/release/tinygpt")
                if fm.fileExists(atPath: candidate.path) { return candidate }
                let nested = dir.appendingPathComponent("native-mac/.build/arm64-apple-macosx/release/tinygpt")
                if fm.fileExists(atPath: nested.path) { return nested }
                dir = dir.deletingLastPathComponent()
            }
        }
        let local = URL(fileURLWithPath: "native-mac/.build/arm64-apple-macosx/release/tinygpt")
        if fm.fileExists(atPath: local.path) { return local }
        let usr = URL(fileURLWithPath: "/usr/local/bin/tinygpt")
        return fm.fileExists(atPath: usr.path) ? usr : nil
    }

    private func parseRows(from url: URL) {
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8)
        else { return }
        let decoder = JSONDecoder()
        var parsed: [AppEvalRow] = []
        for line in text.split(separator: "\n") {
            guard let d = String(line).data(using: .utf8),
                  let row = try? decoder.decode(AppEvalRow.self, from: d)
            else { continue }
            parsed.append(row)
        }
        rows = parsed
    }

    private func appendLog(_ chunk: String) {
        log += chunk
        if log.count > 80_000 { log = String(log.suffix(80_000)) }
    }

    private func resetForRun() {
        cancel()
        log = ""
        rows = []
        customRows = []
        lastError = nil
        resultsPath = defaultResultsURL().path
    }

    private func defaultResultsURL() -> URL {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
            .appendingPathComponent("tinygpt")
            .appendingPathComponent("evals")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("app-eval-\(Int(Date().timeIntervalSince1970)).jsonl")
    }

    private func checkpointHistory(for url: URL) -> [URL] {
        let dir = url.deletingLastPathComponent()
        let stem = url.deletingPathExtension().lastPathComponent
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        let history = files.filter {
            $0.pathExtension == "tinygpt" &&
            $0.deletingPathExtension().lastPathComponent.hasPrefix("\(stem).step-")
        }
        let sorted = history.sorted { (Self.step(from: $0) ?? 0) < (Self.step(from: $1) ?? 0) }
        return sorted.isEmpty ? [url] : sorted + [url]
    }

    private static func step(from url: URL) -> Int? {
        let name = url.deletingPathExtension().lastPathComponent
        guard let range = name.range(of: ".step-") else { return nil }
        return Int(name[range.upperBound...])
    }

    private func loadCustomCases(path: String) -> [(prompt: String, expected: String)] {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { line in
            guard let data = String(line).data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let prompt = obj["prompt"] as? String,
                  let expected = obj["expected"] as? String
            else { return nil }
            return (prompt, expected)
        }
    }

    private func runSample(prompt: String) -> String {
        guard let cli = locateCLI() else { return "" }
        let p = Process()
        p.executableURL = cli
        p.arguments = ["sample", modelPath, "--prompt", prompt, "--max-tokens", "64", "--temperature", "0"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        do {
            try p.run()
            p.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return ""
        }
    }

    private func score(output: String, expected: String, metric: CustomEvalMetric) -> Bool {
        switch metric {
        case .exact: return output.trimmingCharacters(in: .whitespacesAndNewlines) == expected
        case .contains: return output.localizedCaseInsensitiveContains(expected)
        case .regex: return (try? NSRegularExpression(pattern: expected)).map {
            !$0.matches(in: output, range: NSRange(output.startIndex..., in: output)).isEmpty
        } ?? false
        }
    }

    private func writeRows(_ rows: [AppEvalRow], to url: URL) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let payload = rows.compactMap { try? encoder.encode($0) }
            .map { String(decoding: $0, as: UTF8.self) }
            .joined(separator: "\n") + "\n"
        try? payload.write(to: url, atomically: true, encoding: .utf8)
    }

    private func loadHistory() {
        guard let data = UserDefaults.standard.data(forKey: Self.historyKey),
              let decoded = try? JSONDecoder().decode([EvalHistoryItem].self, from: data)
        else { return }
        history = decoded
    }

    private func saveHistory(modeLabel: String, outURL: URL) {
        let item = EvalHistoryItem(
            timestamp: Date(),
            mode: modeLabel,
            modelPath: modelPath,
            resultsPath: outURL.path,
            rowCount: rows.count
        )
        history.insert(item, at: 0)
        history = Array(history.prefix(Self.historyMax))
        if let data = try? JSONEncoder().encode(history) {
            UserDefaults.standard.set(data, forKey: Self.historyKey)
        }
    }
}

