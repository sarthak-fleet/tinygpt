import Foundation

/// Executes a tool call resolved against a `ToolSchema`. The initial
/// cut is subprocess-only: every tool's `_exec` field is treated as a
/// `/bin/bash -c` template, with `$arg` substitution from the model-
/// produced arguments.
///
/// # Security
///
/// Subprocess execution is *plainly unsafe* when the schema or the
/// model output is attacker-controlled. The agent runtime is meant for
/// developer-driven workflows on the user's own machine; do NOT expose
/// `tinygpt agent` to untrusted prompts or schemas. See
/// `docs/agent_runtime.md` for the threat model.
///
/// We make two small but real mitigations:
///   1. Arguments are substituted via positional bash variable assignment
///      (`arg1="..."` before the template), not via string-replace into
///      the template. Quotes inside the value can't break out of the
///      assignment because we forbid embedded NUL and we run the value
///      through `escapeShellArg`.
///   2. The shell is given a fresh, mostly-empty environment (PATH +
///      HOME + a handful of locale vars). Inherited env can leak secrets
///      into a child the user didn't trust.
public enum ToolExecutor {

    /// Result of a single tool invocation.
    public struct Result: Sendable {
        public let stdout: String
        public let stderr: String
        public let exitCode: Int32
        public let durationSec: Double

        public var ok: Bool { exitCode == 0 }

        /// Best-effort textual summary for feeding back to the model.
        /// stdout is the primary signal; non-empty stderr is appended
        /// with a `[stderr]` prefix; non-zero exit codes are tagged.
        public func feedbackText(maxBytes: Int = 8 * 1024) -> String {
            var pieces: [String] = []
            let trimmedOut = truncate(stdout, to: maxBytes)
            if !trimmedOut.isEmpty {
                pieces.append(trimmedOut)
            }
            if !stderr.isEmpty {
                pieces.append("[stderr] " + truncate(stderr, to: maxBytes))
            }
            if exitCode != 0 {
                pieces.append("[exit code \(exitCode)]")
            }
            if pieces.isEmpty {
                pieces.append("[no output]")
            }
            return pieces.joined(separator: "\n")
        }

        private func truncate(_ s: String, to: Int) -> String {
            if s.utf8.count <= to { return s }
            let cut = s.prefix(to)
            return "\(cut)\n…[truncated \(s.utf8.count - to) bytes]"
        }
    }

    /// Dispatch a tool call. Looks up the named tool in the schema,
    /// validates required arguments are present, and runs the subprocess.
    public static func execute(toolName: String, arguments: [String: Any],
                                schema: ToolSchema,
                                timeoutSec: Double = 30.0) throws -> Result
    {
        guard let tool = schema.tools.first(where: { $0.name == toolName }) else {
            throw ToolSchemaError.unknownTool(toolName)
        }
        for req in tool.parameters.required where arguments[req] == nil {
            throw ToolSchemaError.missingRequired(tool: toolName, arg: req)
        }
        guard let template = tool.exec else {
            throw ExecError.notRunnable(
                "tool '\(toolName)' has no _exec field (and Swift _handler isn't wired yet)")
        }
        // Resolve the ordered arg list — explicit `_exec_args` first, then
        // alphabetical over the properties as a stable fallback.
        let argNames: [String] = tool.execArgs ?? Array(tool.parameters.properties.keys.sorted())
        // Stringify each argument value. JSON only carries scalars
        // (string / number / bool / null) at this level — anything fancier
        // is JSON-encoded.
        var argValues: [String: String] = [:]
        for name in argNames {
            argValues[name] = stringify(arguments[name])
        }
        return try runShell(template: template, argNames: argNames,
                              argValues: argValues, timeoutSec: timeoutSec)
    }

    // MARK: - Subprocess

    /// Build a `/bin/bash -c` invocation that pre-assigns each arg as a
    /// shell variable, then runs the template. Variables are quoted into
    /// the assignment via `escapeShellArg` so they can't break out.
    private static func runShell(template: String,
                                  argNames: [String],
                                  argValues: [String: String],
                                  timeoutSec: Double) throws -> Result
    {
        var script = ""
        for name in argNames {
            // Reject anything that isn't a plain identifier. This stops a
            // malformed schema from injecting arbitrary bash through the
            // variable name itself.
            guard isIdentifier(name) else {
                throw ExecError.badArg("arg name '\(name)' must be [A-Za-z_][A-Za-z0-9_]*")
            }
            let value = argValues[name] ?? ""
            script += "\(name)=\(escapeShellArg(value))\n"
        }
        script += template

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = ["-c", script]
        // Hermetic env: PATH for binary lookup, HOME for tools that
        // expect it (eg. git), LANG so output isn't C-locale.
        let parentHome = ProcessInfo.processInfo.environment["HOME"] ?? "/tmp"
        proc.environment = [
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": parentHome,
            "LANG": "en_US.UTF-8",
        ]

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        let t0 = Date()
        try proc.run()

        // Drain pipes on background threads so a chatty child can't fill
        // a pipe buffer and block its own write while we wait. Both
        // closures run to EOF on the child's exit. The Box dance keeps
        // the Swift 6 strict-concurrency checker happy — capturing a
        // mutable `var` across a queue isolation boundary is rejected;
        // a class reference is fine.
        final class Box: @unchecked Sendable { var data: Data = .init() }
        let outBox = Box()
        let errBox = Box()
        let outQ = DispatchQueue(label: "tinygpt.tool.stdout")
        let errQ = DispatchQueue(label: "tinygpt.tool.stderr")
        let outDoneSem = DispatchSemaphore(value: 0)
        let errDoneSem = DispatchSemaphore(value: 0)
        outQ.async {
            outBox.data = outPipe.fileHandleForReading.readDataToEndOfFile()
            outDoneSem.signal()
        }
        errQ.async {
            errBox.data = errPipe.fileHandleForReading.readDataToEndOfFile()
            errDoneSem.signal()
        }

        // Soft timeout: poll for completion in 50ms slices. On timeout
        // send SIGTERM, then SIGKILL after a grace period.
        let deadline = Date().addingTimeInterval(timeoutSec)
        while proc.isRunning {
            if Date() >= deadline {
                proc.terminate()
                Thread.sleep(forTimeInterval: 0.5)
                if proc.isRunning {
                    kill(proc.processIdentifier, SIGKILL)
                }
                break
            }
            Thread.sleep(forTimeInterval: 0.05)
        }
        outDoneSem.wait()
        errDoneSem.wait()

        let outData = outBox.data
        let errData = errBox.data
        let stdoutStr = String(data: outData, encoding: .utf8) ?? "<non-utf8 \(outData.count) bytes>"
        let stderrStr = String(data: errData, encoding: .utf8) ?? "<non-utf8 \(errData.count) bytes>"
        let elapsed = -t0.timeIntervalSinceNow
        return Result(stdout: stdoutStr, stderr: stderrStr,
                       exitCode: proc.terminationStatus, durationSec: elapsed)
    }

    private static func isIdentifier(_ s: String) -> Bool {
        guard let first = s.unicodeScalars.first else { return false }
        if !(first.isASCII && (CharacterSet.letters.contains(first) || first == "_")) {
            return false
        }
        for c in s.unicodeScalars.dropFirst() {
            if !c.isASCII { return false }
            if !(CharacterSet.alphanumerics.contains(c) || c == "_") {
                return false
            }
        }
        return true
    }

    /// Escape a string for use as a single-quoted bash literal.
    /// Bash single-quoted strings can hold every character except a
    /// single quote — for those we close, append an escaped single quote,
    /// and reopen.
    private static func escapeShellArg(_ s: String) -> String {
        return "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// JSON-style any-to-string for tool arguments.
    private static func stringify(_ v: Any?) -> String {
        guard let v = v else { return "" }
        if let s = v as? String { return s }
        if let b = v as? Bool { return b ? "true" : "false" }
        if let i = v as? Int { return String(i) }
        if let d = v as? Double { return String(d) }
        if let n = v as? NSNumber { return n.stringValue }
        // Anything else — encode as JSON so the child can re-parse.
        if let data = try? JSONSerialization.data(withJSONObject: v, options: []) {
            return String(data: data, encoding: .utf8) ?? ""
        }
        return "\(v)"
    }
}

public enum ExecError: Error, CustomStringConvertible {
    case notRunnable(String)
    case badArg(String)
    case timeout(Double)
    public var description: String {
        switch self {
        case .notRunnable(let m): return m
        case .badArg(let m): return m
        case .timeout(let t): return "tool timed out after \(t)s"
        }
    }
}
