import Foundation

/// Drives the Interp tab. Today: launches `tinygpt sae` as a subprocess
/// and streams stdout/stderr into a Published string the view renders.
/// Tomorrow: same pattern for memit, patch, sae-to-saelens — the CLI is
/// the authoritative implementation, the app is the orchestrator + viewer.
@MainActor
final class InterpController: ObservableObject {
    @Published var output: String = ""
    @Published var isRunning: Bool = false
    @Published var status: String = "ready"
    /// Diagnostic stats parsed from the final lines of the SAE run.
    /// Populated when the run completes successfully.
    @Published var lastMSE: Double? = nil
    @Published var lastL0Pct: Double? = nil
    @Published var lastSAEPath: String? = nil

    private var process: Process? = nil

    /// Find the bundled `tinygpt-cli` next to the app binary, then fall
    /// back to common dev paths so this works equally well when running
    /// from `swift run TinyGPTApp` and from an installed .app.
    private func locateCLI() -> URL? {
        let fm = FileManager.default
        // 1. Same MacOS dir as us (production .app layout).
        if let exec = Bundle.main.executableURL {
            let sibling = exec.deletingLastPathComponent().appendingPathComponent("tinygpt-cli")
            if fm.fileExists(atPath: sibling.path) { return sibling }
            // 2. Walk up looking for a SwiftPM build.
            var dir = exec.deletingLastPathComponent()
            for _ in 0..<8 {
                let candidate = dir.appendingPathComponent(".build/arm64-apple-macosx/release/tinygpt")
                if fm.fileExists(atPath: candidate.path) { return candidate }
                let candidate2 = dir.appendingPathComponent("native-mac/.build/arm64-apple-macosx/release/tinygpt")
                if fm.fileExists(atPath: candidate2.path) { return candidate2 }
                dir = dir.deletingLastPathComponent()
            }
        }
        // 3. /usr/local/bin/tinygpt for users who `cp`'d it there.
        let systemPath = URL(fileURLWithPath: "/usr/local/bin/tinygpt")
        if fm.fileExists(atPath: systemPath.path) { return systemPath }
        return nil
    }

    /// Run `tinygpt sae <model> --corpus <text> --layer N --features F
    ///                          --steps S --batch B --ctx T --out <out>`.
    /// All paths/numbers come from the view; this stays a thin wrapper.
    func trainSAE(modelPath: String, corpusPath: String,
                  layer: Int, features: Int, steps: Int,
                  batch: Int, ctx: Int, outPath: String) {
        cancel()
        guard let cli = locateCLI() else {
            status = "tinygpt CLI not found — build with `swift build -c release`"
            return
        }
        output = ""
        lastMSE = nil
        lastL0Pct = nil
        lastSAEPath = nil
        isRunning = true
        status = "starting SAE training…"

        let args: [String] = [
            "sae", modelPath,
            "--corpus", corpusPath,
            "--layer", "\(layer)",
            "--features", "\(features)",
            "--steps", "\(steps)",
            "--batch", "\(batch)",
            "--ctx", "\(ctx)",
            "--out", outPath,
        ]
        runProcess(cli: cli, args: args, finishTag: outPath)
    }

    func cancel() {
        process?.terminate()
        process = nil
        if isRunning {
            isRunning = false
            status = "cancelled"
        }
    }

    private func runProcess(cli: URL, args: [String], finishTag: String) {
        let p = Process()
        p.executableURL = cli
        p.arguments = args

        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe

        // Stream output to the @Published string as chunks arrive. The
        // readabilityHandler fires on a background queue; hop back to
        // the main actor before publishing so SwiftUI sees one source
        // of truth.
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                self?.output += chunk
                // Parse "reconstruction MSE:   X.XXXXe-XX" and the L0 line
                // when they fly past, so the view can show a small stats
                // pill while a longer run is still mid-stream.
                if let mse = Self.extract(from: chunk, key: "reconstruction MSE:") {
                    self?.lastMSE = mse
                }
                if let l0 = Self.extractPercent(from: chunk) {
                    self?.lastL0Pct = l0
                }
            }
        }

        do {
            try p.run()
            self.process = p
            // Wait off-thread so the main actor stays responsive.
            Task.detached { [weak self] in
                p.waitUntilExit()
                pipe.fileHandleForReading.readabilityHandler = nil
                let exitCode = p.terminationStatus
                await MainActor.run {
                    self?.isRunning = false
                    self?.process = nil
                    if exitCode == 0 {
                        self?.status = "done — sidecar saved"
                        self?.lastSAEPath = finishTag
                    } else if exitCode == 15 || exitCode == -15 {
                        self?.status = "cancelled"
                    } else {
                        self?.status = "failed (exit \(exitCode))"
                    }
                }
            }
        } catch {
            isRunning = false
            status = "couldn't launch: \(error)"
        }
    }

    /// Extract a floating-point number that immediately follows `key` in
    /// the chunk, tolerant of scientific notation. Returns nil if absent.
    private static func extract(from text: String, key: String) -> Double? {
        guard let r = text.range(of: key) else { return nil }
        let rest = text[r.upperBound...]
        let scanner = Scanner(string: String(rest))
        scanner.charactersToBeSkipped = .whitespacesAndNewlines
        var value: Double = 0
        if scanner.scanDouble(&value) { return value }
        return nil
    }

    /// Parse the "(NN.NN%)" payload from the L0 line.
    private static func extractPercent(from text: String) -> Double? {
        guard text.contains("active features per sample") else { return nil }
        // Locate the `(  XX.XX%)` substring; everything between `(` and `%`.
        if let openParen = text.range(of: "("),
           let percent = text.range(of: "%)") {
            let inside = text[openParen.upperBound..<percent.lowerBound]
                .trimmingCharacters(in: .whitespaces)
            return Double(inside)
        }
        return nil
    }
}
