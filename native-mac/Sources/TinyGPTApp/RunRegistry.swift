import Foundation

/// Scans `~/.cache/tinygpt/runs/` for training-run directories, surfacing
/// each as a `RunSummary` so the Train tab can show history alongside
/// the currently-active run.
///
/// A run directory pattern (from `docs/prds/persistent-training-output.md`):
///
///     ~/.cache/tinygpt/runs/<run-name>/
///       <run-name>.tinygpt        ← canonical
///       <run-name>.best.tinygpt   ← lowest-val checkpoint
///       <run-name>.jsonl          ← training log
///       <run-name>.step-N.tinygpt ← history checkpoints
///
/// Pre-PRD runs that wrote directly into `/tmp/` are not surfaced
/// (volatile path → assumed lost on reboot anyway). This is intentional;
/// the registry models "what's durable" not "what existed."
public struct RunSummary: Identifiable, Hashable {
    public let id: String          // run name (also dir basename)
    public let directory: URL
    public let canonicalPath: URL
    public let logPath: URL?
    public let bestPath: URL?
    public let lastStep: Int?
    public let lastLoss: Float?
    public let lastValLoss: Float?
    public let totalSteps: Int?
    public let isActive: Bool      // a process is currently running for this dir
    public let pid: Int32?
    public let isStopped: Bool     // SIGSTOP'd (`T` state)
    public let updatedAt: Date
}

// MARK: - Inference processes (running `tinygpt serve` instances)

public struct ServeProcess: Identifiable, Hashable {
    public let id: Int32          // pid
    public let pid: Int32
    public let modelPath: String
    public let port: Int
    public let host: String
}

public enum ServeRegistry {
    /// Scan running `tinygpt serve` processes; map each to its
    /// `--port` + model. Backs an "Inference" sidebar section showing
    /// what models are currently answering OpenAI-compat requests.
    public static func discover() -> [ServeProcess] {
        let pidsOut = pgrep(["-f", "tinygpt serve"])
        let pids = pidsOut.split(separator: "\n").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
        var out: [ServeProcess] = []
        for pid in pids {
            let cmd = psCmd(pid).trimmingCharacters(in: .whitespacesAndNewlines)
            // Skip our own grepping + caffeinate wrappers + lookalikes
            guard cmd.contains("tinygpt serve") && !cmd.hasPrefix("caffeinate") else { continue }
            let parts = cmd.split(separator: " ").map(String.init)
            guard let serveIdx = parts.firstIndex(of: "serve") else { continue }
            // Model path is the first positional arg after "serve"
            let model = serveIdx + 1 < parts.count ? parts[serveIdx + 1] : "?"
            let port = Self.flagInt(parts, "--port") ?? 8080
            let host = Self.flagStr(parts, "--host") ?? "127.0.0.1"
            out.append(ServeProcess(id: pid, pid: pid, modelPath: model, port: port, host: host))
        }
        return out
    }

    private static func flagInt(_ parts: [String], _ flag: String) -> Int? {
        if let i = parts.firstIndex(of: flag), i + 1 < parts.count { return Int(parts[i + 1]) }
        return nil
    }
    private static func flagStr(_ parts: [String], _ flag: String) -> String? {
        if let i = parts.firstIndex(of: flag), i + 1 < parts.count { return parts[i + 1] }
        return nil
    }
    private static func pgrep(_ args: [String]) -> String {
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep"); p.arguments = args
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = Pipe()
        do { try p.run(); p.waitUntilExit() } catch { return "" }
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }
    private static func psCmd(_ pid: Int32) -> String {
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/ps"); p.arguments = ["-p", "\(pid)", "-o", "command="]
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = Pipe()
        do { try p.run(); p.waitUntilExit() } catch { return "" }
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }
}

public enum RunRegistry {

    public static var runsRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/tinygpt/runs", isDirectory: true)
    }

    /// Walk `~/.cache/tinygpt/runs/`, return one `RunSummary` per
    /// directory. Sorted most-recently-modified first.
    public static func discover() -> [RunSummary] {
        let fm = FileManager.default
        let root = runsRoot
        guard let entries = try? fm.contentsOfDirectory(at: root,
                                                       includingPropertiesForKeys: [.contentModificationDateKey])
        else { return [] }

        // Map PID-by-canonical-path so we can mark active runs.
        let active = activeTrainingByOut()

        var out: [RunSummary] = []
        for dir in entries where (try? dir.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            let name = dir.lastPathComponent
            // canonical path convention: <dir>/<run-name>.tinygpt
            var canonical = dir.appendingPathComponent("\(name).tinygpt")
            if !fm.fileExists(atPath: canonical.path) {
                // Some runs auto-named with different stem (e.g. huge-base-v1 inside n02-…/)
                if let first = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
                    .first(where: { $0.pathExtension == "tinygpt" && !$0.lastPathComponent.contains(".step-") && !$0.lastPathComponent.contains(".best") })
                {
                    canonical = first
                } else {
                    continue
                }
            }
            let stem = canonical.deletingPathExtension().lastPathComponent
            let log = dir.appendingPathComponent("\(stem).jsonl")
            let best = dir.appendingPathComponent("\(stem).best.tinygpt")

            let (lastStep, lastLoss, lastVal, totalSteps) = parseLog(log)
            let pid = active[canonical.path]
            let stopped = pid.flatMap(processIsStopped) ?? false
            let mtime = (try? canonical.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? Date.distantPast

            out.append(RunSummary(
                id: name, directory: dir, canonicalPath: canonical,
                logPath: fm.fileExists(atPath: log.path) ? log : nil,
                bestPath: fm.fileExists(atPath: best.path) ? best : nil,
                lastStep: lastStep, lastLoss: lastLoss, lastValLoss: lastVal,
                totalSteps: totalSteps,
                isActive: pid != nil, pid: pid, isStopped: stopped,
                updatedAt: mtime
            ))
        }
        return out.sorted { $0.updatedAt > $1.updatedAt }
    }

    // MARK: Helpers

    private static func parseLog(_ log: URL) -> (Int?, Float?, Float?, Int?) {
        guard let data = try? Data(contentsOf: log),
              let text = String(data: data, encoding: .utf8) else { return (nil, nil, nil, nil) }
        var step: Int? = nil; var loss: Float? = nil; var val: Float? = nil; var total: Int? = nil
        // Read last ~200 lines for speed.
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        let tail = lines.suffix(200)
        for raw in tail {
            guard let d = raw.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { continue }
            if let t = json["total_steps"] as? Int { total = t }
            if let v = json["val"] as? Double { val = Float(v) }
            if let v = json["val_loss"] as? Double { val = Float(v) }
            if (json["type"] as? String) == "step", let s = json["step"] as? Int {
                step = s
                if let l = json["loss"] as? Double { loss = Float(l) }
            }
        }
        return (step, loss, val, total)
    }

    /// Map of `--out` path → PID for any currently-alive `tinygpt train`
    /// process. Lets us mark `isActive` on summaries.
    private static func activeTrainingByOut() -> [String: Int32] {
        let pidsOut = runShort("/usr/bin/pgrep", ["-f", "tinygpt train"])
        let pids = pidsOut.split(separator: "\n").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
        var map: [String: Int32] = [:]
        for pid in pids {
            let cmd = runShort("/bin/ps", ["-p", "\(pid)", "-o", "command="])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard cmd.contains("tinygpt train") && !cmd.hasPrefix("caffeinate") else { continue }
            if let path = argValue(cmd: cmd, flag: "--out") {
                map[path] = pid
            }
        }
        return map
    }

    private static func processIsStopped(_ pid: Int32) -> Bool {
        let stat = runShort("/bin/ps", ["-p", "\(pid)", "-o", "stat="])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return stat.contains("T")
    }

    private static func runShort(_ path: String, _ args: [String]) -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let outPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = Pipe()
        do { try p.run(); p.waitUntilExit() } catch { return "" }
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8) ?? ""
    }

    private static func argValue(cmd: String, flag: String) -> String? {
        let toks = cmd.split(separator: " ").map(String.init)
        for (i, t) in toks.enumerated() {
            if t == flag, i + 1 < toks.count { return toks[i + 1] }
            if t.hasPrefix("\(flag)=") { return String(t.dropFirst(flag.count + 1)) }
        }
        return nil
    }
}
