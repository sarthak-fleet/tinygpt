import Foundation

enum EvalHarnessSupport {
    struct Common {
        var modelPath: String?
        var tokenizer: String?
        var outJsonl: String?
        var modelName: String?
        var modelStep: Int?
        var baseline = false
        var limit = 0
        var servePort = 8097
    }

    static func parseCommon(_ args: [String], usage: () -> Never) -> (Common, [String]) {
        var common = Common()
        var rest: [String] = []
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--tokenizer": common.tokenizer = args[i + 1]; i += 2
            case "--out": common.outJsonl = args[i + 1]; i += 2
            case "--model-name": common.modelName = args[i + 1]; i += 2
            case "--model-step": common.modelStep = Int(args[i + 1]); i += 2
            case "--baseline": common.baseline = true; i += 1
            case "--limit": common.limit = Int(args[i + 1]) ?? common.limit; i += 2
            case "--serve-port": common.servePort = Int(args[i + 1]) ?? common.servePort; i += 2
            case "-h", "--help": usage()
            default:
                if args[i].hasPrefix("-") {
                    rest.append(args[i])
                    if i + 1 < args.count, !args[i + 1].hasPrefix("-") {
                        rest.append(args[i + 1]); i += 2
                    } else {
                        i += 1
                    }
                } else if common.modelPath == nil {
                    common.modelPath = args[i]; i += 1
                } else {
                    rest.append(args[i]); i += 1
                }
            }
        }
        return (common, rest)
    }

    static func require(_ common: Common, usage: () -> Never) -> Common {
        guard common.modelPath != nil else { fputs("missing <model.tinygpt | hf-dir>\n", stderr); usage() }
        guard common.outJsonl != nil else { fputs("--out <path.jsonl> required\n", stderr); usage() }
        return common
    }

    static func displayName(_ common: Common) -> String {
        common.modelName ?? URL(fileURLWithPath: common.modelPath ?? "model").deletingPathExtension().lastPathComponent
    }

    static func startServe(modelPath: String, port: Int, maxContext: Int = 4096) -> Process {
        let cli = resolveExecutable(CommandLine.arguments.first ?? "tinygpt")
            ?? resolveExecutable("tinygpt")
            ?? resolveExecutable("tinygpt-cli")
        guard let cli else {
            fputs("tinygpt CLI not found for serve. Build with `swift build -c release`.\n", stderr); exit(1)
        }
        let p = Process()
        p.executableURL = cli
        p.arguments = ["serve", modelPath, "--host", "127.0.0.1", "--port", "\(port)", "--max-context", "\(maxContext)"]
        p.standardOutput = Pipe()
        p.standardError = Pipe()
        do { try p.run() }
        catch { fputs("could not start tinygpt serve: \(error)\n", stderr); exit(1) }
        waitForServe(port: port, process: p)
        return p
    }

    static func waitForServe(port: Int, process: Process) {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        for _ in 0..<60 {
            Thread.sleep(forTimeInterval: 1)
            var req = URLRequest(url: url)
            req.timeoutInterval = 2
            let sem = DispatchSemaphore(value: 0)
            final class Box { var ok = false }
            let box = Box()
            URLSession.shared.dataTask(with: req) { _, resp, _ in
                if let http = resp as? HTTPURLResponse, http.statusCode < 500 { box.ok = true }
                sem.signal()
            }.resume()
            _ = sem.wait(timeout: .now() + 3)
            if box.ok { return }
            if !process.isRunning { break }
        }
        fputs("tinygpt serve did not become ready on port \(port)\n", stderr); exit(1)
    }

    @discardableResult
    static func runProcess(_ exe: URL, _ args: [String], cwd: URL? = nil, env: [String: String] = [:]) -> Int32 {
        let p = Process()
        p.executableURL = exe
        p.arguments = args
        p.currentDirectoryURL = cwd
        var e = ProcessInfo.processInfo.environment
        for (k, v) in env { e[k] = v }
        p.environment = e
        do { try p.run() }
        catch { fputs("could not launch \(exe.path): \(error)\n", stderr); return 127 }
        p.waitUntilExit()
        return p.terminationStatus
    }

    static func resolveExecutable(_ name: String) -> URL? {
        if name.contains("/") {
            let url = URL(fileURLWithPath: name)
            return FileManager.default.isExecutableFile(atPath: url.path) ? url : nil
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = ["which", name]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do { try p.run() } catch { return nil }
        p.waitUntilExit()
        let path = String(data: pipe.fileHandleForReading.availableData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return path.isEmpty ? nil : URL(fileURLWithPath: path)
    }

    static func latestJSON(under root: URL) -> URL? {
        guard let it = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        var newest: (URL, Date)? = nil
        for case let url as URL in it where url.pathExtension == "json" {
            let mod = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            if newest == nil || mod > newest!.1 { newest = (url, mod) }
        }
        return newest?.0
    }

    static func jsonObject(_ url: URL) -> Any? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }

    static func appendRow(common: Common, task: String, subtask: String?, metric: String,
                          score: Double, n: Int, wall: Double, harness: String?) {
        guard let out = common.outJsonl, let model = common.modelPath else { return }
        let row = EvalCompare.Row(
            run_id: UUID().uuidString,
            model_path: model,
            model_name: displayName(common),
            model_step: common.modelStep,
            baseline: common.baseline,
            task: task,
            subtask: subtask,
            metric: metric,
            score: score,
            n_examples: n,
            wall_seconds: wall,
            harness_version: harness
        )
        do { try EvalCompare.Row.append(row, to: URL(fileURLWithPath: out)) }
        catch { fputs("could not append eval row: \(error)\n", stderr) }
    }

    static func numericScores(_ value: Any, path: [String] = []) -> [(String, Double, Int)] {
        var out: [(String, Double, Int)] = []
        if let dict = value as? [String: Any] {
            let n = intValue(dict["n"] ?? dict["num"] ?? dict["count"] ?? dict["total"])
            for (k, v) in dict {
                if let d = doubleValue(v), looksLikeMetric(k) {
                    out.append(((path + [k]).joined(separator: "/"), d, n ?? 0))
                } else {
                    out.append(contentsOf: numericScores(v, path: path + [k]))
                }
            }
        } else if let arr = value as? [Any] {
            for (idx, v) in arr.enumerated() {
                out.append(contentsOf: numericScores(v, path: path + ["\(idx)"]))
            }
        }
        return out
    }

    static func doubleValue(_ value: Any?) -> Double? {
        if let d = value as? Double { return d }
        if let i = value as? Int { return Double(i) }
        if let s = value as? String { return Double(s) }
        return nil
    }

    static func intValue(_ value: Any?) -> Int? {
        if let i = value as? Int { return i }
        if let d = value as? Double { return Int(d) }
        if let s = value as? String { return Int(s) }
        return nil
    }

    static func looksLikeMetric(_ key: String) -> Bool {
        let k = key.lowercased()
        return k.contains("acc") || k.contains("pass") || k == "reward" || k == "score" || k == "avg_reward"
    }

    static func completion(baseURL: String, prompt: String, maxTokens: Int = 128, temperature: Double = 0) -> String? {
        guard let url = URL(string: "\(baseURL)/completions") else { return nil }
        let body: [String: Any] = [
            "model": "tinygpt",
            "prompt": prompt,
            "max_tokens": maxTokens,
            "temperature": temperature
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = data
        req.timeoutInterval = 120
        let sem = DispatchSemaphore(value: 0)
        final class Box { var text: String? }
        let box = Box()
        URLSession.shared.dataTask(with: req) { data, _, _ in
            defer { sem.signal() }
            guard let data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = obj["choices"] as? [[String: Any]]
            else { return }
            box.text = choices.first?["text"] as? String
        }.resume()
        _ = sem.wait(timeout: .now() + 180)
        return box.text
    }
}
