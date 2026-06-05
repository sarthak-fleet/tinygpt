import Foundation

/// Drives the Server tab. Spawns `tinygpt-cli serve <model> --port N`
/// as a subprocess and exposes Start/Stop + a live log + the current
/// endpoint to the view. Same pattern as InterpController — the CLI is
/// the authoritative server, the app is the orchestrator.
///
/// The server speaks OpenAI's chat-completions shape (per
/// TinyGPTServe/Serve.swift), so anything that talks to OpenAI can be
/// repointed at `http://127.0.0.1:<port>/v1`. That makes TinyGPT a
/// drop-in local backend for Cursor / Continue / Cline / your own
/// scripts — the LM Studio / Ollama play.
@MainActor
final class ServerController: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var port: Int = 8080
    @Published var host: String = "127.0.0.1"
    @Published var modelPath: String = ""
    @Published var log: String = ""
    @Published var lastError: String? = nil
    @Published var requestCount: Int = 0

    private var process: Process? = nil

    /// Endpoint a client would hit. Updated on Start.
    @Published var endpoint: String = ""

    /// Same CLI lookup as InterpController — bundled binary first,
    /// SwiftPM dev paths next, /usr/local/bin last.
    private func locateCLI() -> URL? {
        let fm = FileManager.default
        if let exec = Bundle.main.executableURL {
            let sibling = exec.deletingLastPathComponent().appendingPathComponent("tinygpt-cli")
            if fm.fileExists(atPath: sibling.path) { return sibling }
            var dir = exec.deletingLastPathComponent()
            for _ in 0..<8 {
                let c1 = dir.appendingPathComponent(".build/arm64-apple-macosx/release/tinygpt")
                if fm.fileExists(atPath: c1.path) { return c1 }
                let c2 = dir.appendingPathComponent("native-mac/.build/arm64-apple-macosx/release/tinygpt")
                if fm.fileExists(atPath: c2.path) { return c2 }
                dir = dir.deletingLastPathComponent()
            }
        }
        let p = URL(fileURLWithPath: "/usr/local/bin/tinygpt")
        return fm.fileExists(atPath: p.path) ? p : nil
    }

    func start() {
        guard !isRunning else { return }
        guard !modelPath.isEmpty else {
            lastError = "pick a model first"; return
        }
        guard let cli = locateCLI() else {
            lastError = "tinygpt CLI not found — build with `swift build -c release`"; return
        }
        lastError = nil
        log = ""
        requestCount = 0
        endpoint = "http://\(host):\(port)"
        isRunning = true

        let p = Process()
        p.executableURL = cli
        p.arguments = ["serve", modelPath, "--host", host, "--port", "\(port)"]

        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                self?.log += chunk
                // Coarse "request happened" counter — the CLI's access-log
                // line includes `POST /v1/`; count those.
                self?.requestCount += Self.countMatches(of: "POST /v1/", in: chunk)
            }
        }

        do {
            try p.run()
            self.process = p
            // Off-main wait for the exit so the actor stays free for the UI.
            Task.detached { [weak self] in
                p.waitUntilExit()
                pipe.fileHandleForReading.readabilityHandler = nil
                let code = p.terminationStatus
                await MainActor.run {
                    self?.process = nil
                    self?.isRunning = false
                    if code != 0 && code != 15 && code != -15 {
                        self?.lastError = "server exited with code \(code)"
                    }
                }
            }
        } catch {
            isRunning = false
            lastError = "couldn't launch: \(error)"
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        isRunning = false
    }

    /// Ensure the server isn't orphaned when the app quits — called from
    /// the NSApplicationDelegate's applicationWillTerminate.
    nonisolated func terminate() {
        // Reach into the process pointer; @MainActor isolation is fine
        // here because terminate() on Process is signal-safe and we're
        // in the app-quit phase.
        Task { @MainActor in
            self.stop()
        }
    }

    /// Quick helper for the per-chunk POST counter. Tiny, no regex needed.
    private static func countMatches(of needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var search = haystack.startIndex
        while let r = haystack.range(of: needle, range: search..<haystack.endIndex) {
            count += 1
            search = r.upperBound
        }
        return count
    }
}
