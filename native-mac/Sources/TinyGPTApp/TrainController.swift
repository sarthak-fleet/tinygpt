import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// Owns a training run and streams progress to the UI. Lives on the main
/// actor so views can subscribe directly; training-step work happens
/// inline (MLX is GPU-async, so the per-step cost on the main thread
/// is mostly orchestration, not compute).
@MainActor
final class TrainController: ObservableObject {
    @Published var lossHistory: [LossPoint] = []
    @Published var status: String = "configure a run and press Start"
    @Published var stepCount: Int = 0
    @Published var targetSteps: Int = 1000
    @Published var stepsPerSec: Double = 0
    @Published var isTraining: Bool = false
    @Published var isPaused: Bool = false
    @Published var isPausing: Bool = false
    @Published var currentLoss: Float = 0
    @Published var sampleText: String = ""
    @Published var presetIdx: Int = 0  // index into Self.presets
    /// Default 50% per 2026-06-07 post-incident lesson — sustained 75%
    /// thermally-pauses on Macs even with clean fans. 50% leaves margin.
    /// Users can opt up to 75/100% explicitly; the auto-throttle hook
    /// still drops back toward 25% under thermal pressure.
    @Published var throttle: Double = 0.50
    @Published var allowAutoThrottle: Bool = true
    @Published var autoThrottleNote: String? = nil

    // Today's CLI flags surfaced in the UI:
    enum LRSchedule: String, CaseIterable, Identifiable {
        case cosine, wsd, constant
        var id: Self { self }
    }
    @Published var lrSchedule: LRSchedule = .cosine
    @Published var maxLR: Float = 3e-4
    @Published var minLR: Float = 3e-5
    @Published var warmupSteps: Int = 100
    @Published var seedText: String = ""              // empty = random
    @Published var spikeDetectEnabled: Bool = true
    @Published var spikeAlert: String? = nil          // most recent spike message

    /// Picks the user can choose between. Each is a (name, config) pair.
    /// Browser-reachable sizes at the top; Mac-only behemoth sizes below.
    static let presets: [(name: String, cfg: ModelConfig)] = [
        ("Tiny ·   842K",  ModelConfig(vocabSize: 256, contextLength: 128, nLayers: 4,
                                       nHeads: 4, dModel: 128, dMlp: 512)),
        ("Small · 2.4M",   ModelConfig(vocabSize: 256, contextLength: 256, nLayers: 6,
                                       nHeads: 6, dModel: 192, dMlp: 768)),
        ("Huge ·  9.6M",   ModelConfig.huge),
        ("Mega ·   76M",   ModelConfig.mega),
        ("Behemoth · 404M (Mac only)",  ModelConfig.behemoth),
        ("Titan · 1.3B (Mac only)",     ModelConfig.titan),
    ]

    private var trainTask: Task<Void, Never>? = nil
    private var pauseRequested = false
    private var manualThrottleCap: Double = 0.50

    private enum DefaultsKey {
        static let paused = "tinygpt.train.paused"
        static let pausedStep = "tinygpt.train.paused.step"
        static let pausedPreset = "tinygpt.train.paused.preset"
        static let pausedTarget = "tinygpt.train.paused.target"
        static let pausedCheckpoint = "tinygpt.train.paused.checkpoint"
    }

    func start(corpus: Data) {
        let resumePath = UserDefaults.standard.string(forKey: DefaultsKey.pausedCheckpoint)
        cancel()
        let cfg = Self.presets[presetIdx].cfg
        let presetName = Self.presets[presetIdx].name
        lossHistory.removeAll()
        stepCount = 0
        currentLoss = 0
        sampleText = ""
        spikeAlert = nil
        isPaused = false
        isPausing = false
        pauseRequested = false
        autoThrottleNote = nil
        isTraining = true
        clearPausedDefaults()

        // Seed RNG before any model construction so init is reproducible.
        // Treat blank or non-numeric seedText as "no seed" (random init).
        let parsedSeed: UInt64? = UInt64(seedText.trimmingCharacters(in: .whitespaces))
        if let s = parsedSeed { MLXRandom.seed(s) }

        let seedTag = parsedSeed.map { "seed \($0)" } ?? "random init"
        status = "building \(presetName) (\(formatParams(estimateParams(cfg))) params) · \(seedTag)…"

        trainTask = Task {
            await runTraining(corpus: corpus, cfg: cfg, presetName: presetName, resumePath: resumePath)
        }
    }

    func cancel() {
        trainTask?.cancel()
        trainTask = nil
        pauseRequested = false
        isPaused = false
        isPausing = false
        clearPausedDefaults()
        if isTraining {
            isTraining = false
            status = "stopped at step \(stepCount)"
        }
    }

    func pause() {
        guard isTraining, !isPaused else { return }
        pauseRequested = true
        isPausing = true
        status = "pausing after current step…"
    }

    func resume() {
        guard isTraining, isPaused || isPausing else { return }
        pauseRequested = false
        isPaused = false
        isPausing = false
        clearPausedDefaults()
        status = "resuming at step \(stepCount + 1)…"
    }

    func setThrottle(_ value: Double, userInitiated: Bool = true) {
        throttle = value
        if userInitiated {
            manualThrottleCap = value
            autoThrottleNote = nil
        }
        writeThrottleControlFile(value)
        if value <= 0 {
            pause()
        } else if isPaused {
            resume()
        }
    }

    func applyThermalThrottle(_ state: ProcessInfo.ThermalState) {
        guard allowAutoThrottle else { return }
        let thermalCap: Double
        switch state {
        case .nominal: thermalCap = manualThrottleCap
        case .fair: thermalCap = min(manualThrottleCap, 0.75)
        case .serious: thermalCap = min(manualThrottleCap, 0.5)
        case .critical: thermalCap = min(manualThrottleCap, 0.25)
        @unknown default: thermalCap = manualThrottleCap
        }
        guard abs(thermalCap - throttle) > 0.001 else { return }
        throttle = thermalCap
        writeThrottleControlFile(thermalCap)
        autoThrottleNote = state == .nominal
            ? nil
            : String(format: "auto-throttled to %.0f%% due to thermal pressure", thermalCap * 100)
    }

    func restorePausedMetadataIfPresent() -> Bool {
        let defaults = UserDefaults.standard
        guard defaults.bool(forKey: DefaultsKey.paused) else { return false }
        let preset = defaults.integer(forKey: DefaultsKey.pausedPreset)
        if preset >= 0 && preset < Self.presets.count { presetIdx = preset }
        let target = defaults.integer(forKey: DefaultsKey.pausedTarget)
        if target > 0 { targetSteps = target }
        stepCount = defaults.integer(forKey: DefaultsKey.pausedStep)
        status = "previous run paused at step \(stepCount); press Start to resume from checkpoint"
        return true
    }

    // MARK: External-run detection (orphan CLI training runs)

    /// Snapshot of an externally-spawned `tinygpt train` process so the
    /// Train tab can surface "you have training already going" without
    /// requiring the user to remember to start it via the app.
    struct ExternalRun: Equatable {
        let pid: Int32
        let logPath: String?
        let outPath: String?
        let totalSteps: Int?
        let lastStep: Int?
        let lastLoss: Float?
        let lastValLoss: Float?
        let isStopped: Bool
        let stepsPerSec: Double?
        let etaSeconds: Double?
    }

    @Published var externalRun: ExternalRun? = nil
    @Published var detectedRuns: [ExternalRun] = []

    /// Detect training runs via lock file, pgrep, and recent JSONL logs.
    func detectExistingRun() {
        var found: [ExternalRun] = []

        if let lock = RunLockFile.read(), !RunLockFile.isStale(lock) {
            found.append(runFromLock(lock))
        }

        found.append(contentsOf: Self.runsFromPgrep())

        if let recent = Self.runFromRecentLog() {
            if !found.contains(where: { $0.logPath == recent.logPath && $0.outPath == recent.outPath }) {
                found.append(recent)
            }
        }

        detectedRuns = found
        externalRun = found.first
        if externalRun == nil {
            status = found.isEmpty ? "configure a run and press Start" : status
        }
    }

    func detectExternalRun() { detectExistingRun() }

    func selectDetectedRun(at index: Int) {
        guard index >= 0 && index < detectedRuns.count else { return }
        externalRun = detectedRuns[index]
        loadLossChartFromExternalRun()
        viewedRun = .live
    }

    private func runFromLock(_ lock: RunLockFile) -> ExternalRun {
        if let run = Self.parseLogProgress(
            logPath: lock.logJsonlPath,
            outPath: lock.canonicalOutPath,
            totalSteps: lock.totalSteps,
            pid: lock.pid
        ) {
            return run
        }
        return ExternalRun(
            pid: lock.pid, logPath: lock.logJsonlPath, outPath: lock.canonicalOutPath,
            totalSteps: lock.totalSteps,
            lastStep: nil, lastLoss: nil, lastValLoss: nil,
            isStopped: false, stepsPerSec: nil, etaSeconds: nil
        )
    }

    private static func runsFromPgrep() -> [ExternalRun] {
        let pidsOut = Self.runShort("/usr/bin/pgrep", ["-f", "tinygpt train.*--steps"])
        let myPid = ProcessInfo.processInfo.processIdentifier
        let pids = pidsOut.split(separator: "\n").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
        var out: [ExternalRun] = []
        for pid in pids where pid != myPid {
            let cmd = Self.runShort("/bin/ps", ["-p", "\(pid)", "-o", "command="])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard cmd.contains("tinygpt train") && !cmd.contains("--help") else { continue }
            if cmd.hasPrefix("caffeinate") { continue }
            let logPath = Self.argValue(cmd: cmd, flag: "--log-jsonl")
                ?? Self.argValue(cmd: cmd, flag: "--out").map { ($0 as NSString).deletingPathExtension + ".jsonl" }
            let outPath = Self.argValue(cmd: cmd, flag: "--out")
            let totalSteps = Self.argValue(cmd: cmd, flag: "--steps").flatMap { Int($0) }
            if let run = parseLogProgress(logPath: logPath, outPath: outPath, totalSteps: totalSteps, pid: pid) {
                out.append(run)
            }
        }
        return out
    }

    private static func parseLogProgress(
        logPath: String?, outPath: String?, totalSteps: Int?, pid: Int32
    ) -> ExternalRun? {
        var lastStep: Int? = nil
        var lastLoss: Float? = nil
        var lastValLoss: Float? = nil
        var stepsPerSec: Double? = nil
        if let logPath {
            let recent = Self.readLastJSONLines(at: logPath, count: 50)
            var firstStep: (Int, Double)? = nil
            var lastStepPair: (Int, Double)? = nil
            for evt in recent {
                if let v = evt["val"] as? Double { lastValLoss = Float(v) }
                if let v = evt["val_loss"] as? Double { lastValLoss = Float(v) }
                if (evt["type"] as? String) == "step",
                   let s = evt["step"] as? Int,
                   let e = evt["elapsed_s"] as? Double {
                    if firstStep == nil { firstStep = (s, e) }
                    lastStepPair = (s, e)
                    lastStep = s
                    if let l = evt["loss"] as? Double { lastLoss = Float(l) }
                }
            }
            if let f = firstStep, let l = lastStepPair, l.1 > f.1, l.0 > f.0 {
                stepsPerSec = Double(l.0 - f.0) / (l.1 - f.1)
            }
        }
        var etaSeconds: Double? = nil
        if let total = totalSteps, let last = lastStep, let rate = stepsPerSec, rate > 0 {
            etaSeconds = Double(total - last) / rate
        }
        let stat = Self.runShort("/bin/ps", ["-p", "\(pid)", "-o", "stat="])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let stopped = stat.contains("T")
        return ExternalRun(
            pid: pid, logPath: logPath, outPath: outPath, totalSteps: totalSteps,
            lastStep: lastStep, lastLoss: lastLoss, lastValLoss: lastValLoss,
            isStopped: stopped,
            stepsPerSec: stepsPerSec, etaSeconds: etaSeconds
        )
    }

    private static func runFromRecentLog() -> ExternalRun? {
        guard let (logPath, outPath) = findRecentTrainingLog() else { return nil }
        var lastStep: Int? = nil
        var lastLoss: Float? = nil
        var lastValLoss: Float? = nil
        var totalSteps: Int? = nil
        for evt in Self.readLastJSONLines(at: logPath, count: 200) {
            if let total = evt["total_steps"] as? Int { totalSteps = total }
            if let v = evt["val"] as? Double { lastValLoss = Float(v) }
            if let v = evt["val_loss"] as? Double { lastValLoss = Float(v) }
            if (evt["type"] as? String) == "step", let s = evt["step"] as? Int {
                lastStep = s
                if let l = evt["loss"] as? Double { lastLoss = Float(l) }
            }
        }
        return ExternalRun(
            pid: 0,
            logPath: logPath, outPath: outPath, totalSteps: totalSteps,
            lastStep: lastStep, lastLoss: lastLoss, lastValLoss: lastValLoss,
            isStopped: true,
            stepsPerSec: nil, etaSeconds: nil
        )
    }

    /// Scan persistent run dirs + /tmp for recent `.jsonl` paired with `.tinygpt`.
    private static func findRecentTrainingLog() -> (log: String, out: String)? {
        var candidates: [(log: String, out: String, mtime: Date)] = []
        let roots = [RunRegistry.runsRoot.path, "/tmp"]
        let dayLimit = 24.0 * 3600.0  // PRD: last 24h for /tmp scan
        for root in roots {
            guard let entries = try? FileManager.default.contentsOfDirectory(atPath: root) else { continue }
            let isTmp = root == "/tmp"
            if isTmp {
                for entry in entries where entry.hasSuffix(".jsonl") && !entry.contains(".history.") {
                    let logPath = "\(root)/\(entry)"
                    let stem = String(entry.dropLast(".jsonl".count))
                    let outPath = "\(root)/\(stem).tinygpt"
                    guard FileManager.default.fileExists(atPath: outPath) else { continue }
                    let attrs = try? FileManager.default.attributesOfItem(atPath: logPath)
                    let mtime = (attrs?[.modificationDate] as? Date) ?? Date.distantPast
                    if Date().timeIntervalSince(mtime) < dayLimit {
                        candidates.append((logPath, outPath, mtime))
                    }
                }
            } else {
                for entry in entries {
                    let dir = "\(root)/\(entry)"
                    var isDir: ObjCBool = false
                    guard FileManager.default.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue else { continue }
                    guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir) else { continue }
                    for f in files where f.hasSuffix(".jsonl") && !f.contains(".history.") {
                        let logPath = "\(dir)/\(f)"
                        let stem = String(f.dropLast(".jsonl".count))
                        let outPath = "\(dir)/\(stem).tinygpt"
                        guard FileManager.default.fileExists(atPath: outPath) else { continue }
                        let attrs = try? FileManager.default.attributesOfItem(atPath: logPath)
                        let mtime = (attrs?[.modificationDate] as? Date) ?? Date.distantPast
                        candidates.append((logPath, outPath, mtime))
                    }
                }
            }
        }
        return candidates.sorted { $0.mtime > $1.mtime }.first.map { ($0.log, $0.out) }
    }

    /// SIGCONT / SIGSTOP the detected external run.
    func resumeExternalRun() {
        guard let run = externalRun else { return }
        _ = kill(run.pid, SIGCONT)
        detectExternalRun()
    }
    func pauseExternalRun() {
        guard let run = externalRun else { return }
        _ = kill(run.pid, SIGSTOP)
        detectExternalRun()
    }

    /// Snapshot identifier — what's currently rendered in the loss chart.
    /// `nil` = nothing loaded; live = the auto-detected external run;
    /// historical(name) = a past run loaded by user click.
    enum ViewedRun: Equatable {
        case none, live, historical(String)
    }
    @Published var viewedRun: ViewedRun = .none

    /// Load a specific run (live or historical) into the chart + stats by
    /// parsing its JSONL log. Same shape as `loadLossChartFromExternalRun`
    /// but works on any run, not just the currently-detected one.
    func loadLossChartFromRunSummary(_ run: RunSummary) {
        let logPathURL = run.logPath ?? run.directory
            .appendingPathComponent("\(run.canonicalPath.deletingPathExtension().lastPathComponent).jsonl")

        var byStep: [Int: Float] = [:]
        var lastValLoss: Float? = nil
        let historyPath = logPathURL.deletingPathExtension().path + ".history.jsonl"
        for source in [historyPath, logPathURL.path] {
            guard FileManager.default.fileExists(atPath: source),
                  let data = try? Data(contentsOf: URL(fileURLWithPath: source)),
                  let text = String(data: data, encoding: .utf8) else { continue }
            for raw in text.split(separator: "\n", omittingEmptySubsequences: true) {
                guard let line = raw.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: line) as? [String: Any]
                else { continue }
                if let v = json["val"] as? Double { lastValLoss = Float(v) }
                if let v = json["val_loss"] as? Double { lastValLoss = Float(v) }
                if let type = json["type"] as? String, type != "step" { continue }
                guard let step = json["step"] as? Int else { continue }
                let lossDbl = (json["loss"] as? Double) ?? (json["loss"] as? Float).map(Double.init) ?? 0
                byStep[step] = Float(lossDbl)
            }
        }
        var points = byStep.map { LossPoint(step: $0.key, loss: $0.value) }
            .sorted { $0.step < $1.step }
        let lastStep = points.last?.step ?? 0
        let lastLoss = points.last?.loss ?? 0

        // Downsample to keep canvas responsive on long runs.
        let maxPoints = 500
        if points.count > maxPoints {
            let stride = max(1, points.count / maxPoints)
            points = points.enumerated().compactMap { (idx, p) in idx % stride == 0 ? p : nil }
        }

        lossHistory = points
        stepCount = lastStep
        currentLoss = lastLoss
        if let total = run.totalSteps { targetSteps = total }
        viewedRun = run.isActive ? .live : .historical(run.id)
        if let v = lastValLoss {
            status = String(format: "%@ · step %d · loss %.3f · val %.3f",
                            run.isActive ? "live run" : "historical run",
                            lastStep, lastLoss, v)
        } else {
            status = String(format: "%@ · step %d · loss %.3f",
                            run.isActive ? "live run" : "historical run",
                            lastStep, lastLoss)
        }
    }

    /// Clear the historical-view state, jump back to live tracking.
    func returnToLive() {
        viewedRun = .none
        lossHistory = []
        stepCount = 0
        currentLoss = 0
        detectExistingRun()
        if externalRun != nil {
            loadLossChartFromExternalRun()
            viewedRun = .live
        }
    }

    /// Load the loss chart from a detected external run's JSONL log.
    /// Downsamples to ~500 points so very long runs (e.g. 200K-step N02)
    /// don't choke the Canvas. Also updates `stepCount`, `targetSteps`,
    /// and `currentLoss` so the stats row reflects the real training state.
    func loadLossChartFromExternalRun() {
        guard let run = externalRun, let logPath = run.logPath else { return }

        // Merge a `.history.jsonl` sidecar (reconstructed from prior runs,
        // e.g. when a `--resume` truncated the active log) with the live
        // log. Both keyed by step; live log wins on conflict.
        var byStep: [Int: Float] = [:]
        let historyPath = (logPath as NSString).deletingPathExtension + ".history.jsonl"
        for source in [historyPath, logPath] {
            guard FileManager.default.fileExists(atPath: source),
                  let data = try? Data(contentsOf: URL(fileURLWithPath: source)),
                  let text = String(data: data, encoding: .utf8) else { continue }
            for raw in text.split(separator: "\n", omittingEmptySubsequences: true) {
                guard let line = raw.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: line) as? [String: Any]
                else { continue }
                if let type = json["type"] as? String, type != "step" { continue }
                guard let step = json["step"] as? Int else { continue }
                let lossDbl = (json["loss"] as? Double) ?? (json["loss"] as? Float).map(Double.init) ?? 0
                byStep[step] = Float(lossDbl)
            }
        }
        var points = byStep.map { LossPoint(step: $0.key, loss: $0.value) }
            .sorted { $0.step < $1.step }
        let lastStep = points.last?.step ?? 0
        let lastLoss = points.last?.loss ?? 0

        // Downsample if very large (Canvas slows past ~5K points).
        let maxPoints = 500
        if points.count > maxPoints {
            let stride = max(1, points.count / maxPoints)
            points = points.enumerated().compactMap { (idx, p) in idx % stride == 0 ? p : nil }
        }

        lossHistory = points
        stepCount = lastStep
        currentLoss = lastLoss
        if let total = run.totalSteps { targetSteps = total }
        if let val = run.lastValLoss {
            status = String(format: "external run · step %d · loss %.3f · val %.3f", lastStep, lastLoss, val)
        } else {
            status = String(format: "external run · step %d · loss %.3f", lastStep, lastLoss)
        }
    }

    private static func runShort(_ path: String, _ args: [String]) -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let outPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
        } catch {
            return ""
        }
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8) ?? ""
    }

    private static func argValue(cmd: String, flag: String) -> String? {
        // Tokenise on whitespace; treat `flag value` and `flag=value` both.
        let toks = cmd.split(separator: " ").map(String.init)
        for (i, t) in toks.enumerated() {
            if t == flag, i + 1 < toks.count { return toks[i + 1] }
            if t.hasPrefix("\(flag)=") { return String(t.dropFirst(flag.count + 1)) }
        }
        return nil
    }

    private static func readLastNonEmptyLine(at path: String) -> String? {
        guard FileManager.default.fileExists(atPath: path),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let s = String(data: data, encoding: .utf8) else { return nil }
        let lines = s.split(separator: "\n", omittingEmptySubsequences: true)
        return lines.last.map(String.init)
    }

    /// Read the last N parseable JSON-lines from a file, oldest-first.
    private static func readLastJSONLines(at path: String, count: Int) -> [[String: Any]] {
        guard FileManager.default.fileExists(atPath: path),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let text = String(data: data, encoding: .utf8) else { return [] }
        let allLines = text.split(separator: "\n", omittingEmptySubsequences: true)
        let tailLines = allLines.suffix(count)
        return tailLines.compactMap { line -> [String: Any]? in
            guard let lineData = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any]
            else { return nil }
            return json
        }
    }

    /// Format an ETA in seconds → human string (e.g. "5h 23m", "47m", "12s").
    static func formatETA(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s >= 3600 {
            return String(format: "%dh %02dm", s / 3600, (s % 3600) / 60)
        } else if s >= 60 {
            return "\(s / 60)m"
        } else {
            return "\(s)s"
        }
    }

    private func runTraining(corpus: Data, cfg requestedCfg: ModelConfig, presetName: String, resumePath: String?) async {
        let model: TinyGPTModel
        let cfg: ModelConfig
        let resumeFile: TinyGPTFile?
        let startStep: Int
        if let resumePath,
           FileManager.default.fileExists(atPath: resumePath),
           let loaded = try? AppTrainingCheckpoint.load(URL(fileURLWithPath: resumePath)) {
            model = loaded.model
            cfg = loaded.cfg
            resumeFile = loaded.file
            startStep = Int(loaded.file.step)
            stepCount = startStep
            status = "resumed checkpoint at step \(startStep)"
        } else {
            model = TinyGPTModel(requestedCfg)
            cfg = requestedCfg
            resumeFile = nil
            startStep = 0
        }
        // Snapshot the LR config — `@Published` vars can mutate from the
        // UI mid-run but the trainer needs a stable view.
        let schedule = self.lrSchedule
        let maxLRSnap = self.maxLR
        let minLRSnap = self.minLR
        let warmupSnap = self.warmupSteps
        let total = self.targetSteps
        // WSD decay window — 10% of total steps by default, matches the
        // CLI's --decay-steps auto. The stable middle phase makes WSD
        // friendly to mid-run resume; the decay is fast at the end.
        let decaySteps = max(1, total / 10)

        // Start the optimizer at the LR the schedule says step 0 should
        // be at (typically the first warmup tick, not maxLR).
        let initialLR = lrAt(step: 0, schedule: schedule, maxLR: maxLRSnap,
                              minLR: minLRSnap, warmup: warmupSnap,
                              decaySteps: decaySteps, total: total)
        let trainer = Trainer(model: model, learningRate: initialLR, compileStep: true,
                              optimizer: .adamw, useCompiledLR: schedule != .constant)
        if let resumeFile {
            _ = AppTrainingCheckpoint.restoreOptimizerState(from: resumeFile, into: trainer)
        }
        let byteCorpus = ByteCorpus(corpus)
        let batchSize = batchSizeFor(cfg)

        let scheduleTag = schedule == .constant ? "constant" :
                          schedule == .wsd ? "wsd(warmup \(warmupSnap), decay \(decaySteps))" :
                          "cosine(warmup \(warmupSnap))"
        status = "training \(presetName) · batch \(batchSize) · ctx \(cfg.contextLength) · \(scheduleTag)"
        let t0 = Date()

        // Spike detector reuses TinyGPTModel.LossSpikeDetector — same
        // primitive the CLI uses, so the UI and CLI agree on what
        // counts as a spike.
        var spikeDetector = LossSpikeDetector(window: 50, factor: 3.0)

        for step in startStep..<total {
            if Task.isCancelled { break }
            while pauseRequested && !Task.isCancelled {
                if !isPaused {
                    var checkpointPath: String? = nil
                    do {
                        try AppTrainingCheckpoint.save(
                            model: model,
                            cfg: cfg,
                            trainer: trainer,
                            step: stepCount,
                            loss: currentLoss,
                            to: AppTrainingCheckpoint.defaultURL
                        )
                        checkpointPath = AppTrainingCheckpoint.defaultURL.path
                    } catch {
                        spikeAlert = "pause checkpoint failed: \(error)"
                    }
                    isPaused = true
                    isPausing = false
                    if let checkpointPath {
                        persistPausedDefaults(checkpoint: checkpointPath)
                    }
                    status = "paused at step \(stepCount)"
                }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            if Task.isCancelled { break }

            let stepStarted = Date()

            // Schedule the LR for this step (skip when constant — the
            // initial value already matches).
            if schedule != .constant {
                trainer.optimizer.learningRate = lrAt(
                    step: step, schedule: schedule,
                    maxLR: maxLRSnap, minLR: minLRSnap,
                    warmup: warmupSnap, decaySteps: decaySteps, total: total)
            }

            let (x, y) = byteCorpus.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            let loss = trainer.step(inputs: x, targets: y)

            self.stepCount = step + 1
            self.currentLoss = loss
            if step % 5 == 0 || step == total - 1 {
                self.lossHistory.append(LossPoint(step: step + 1, loss: loss))
            }
            let elapsed = -t0.timeIntervalSinceNow
            if elapsed > 0 { self.stepsPerSec = Double(step + 1) / elapsed }

            // Loss-spike check. The detector silently warms up for its
            // window (50 steps); a triggered spike publishes a banner
            // the UI can render. v1 is observe-only — the operator
            // decides whether to Stop and resume with a lower LR.
            if spikeDetectEnabled {
                let (spike, ma) = spikeDetector.observe(loss: loss, step: step)
                if spike {
                    self.spikeAlert = String(
                        format: "spike at step %d: loss %.3f vs moving-avg %.3f (>%.1f×)",
                        step + 1, loss, ma, 3.0)
                }
            }

            let throttleSnapshot = max(0.0, min(1.0, throttle))
            if throttleSnapshot > 0 && throttleSnapshot < 1 {
                let stepWall = -stepStarted.timeIntervalSinceNow
                let sleepSeconds = stepWall * (1.0 / throttleSnapshot - 1.0)
                if sleepSeconds > 0 {
                    try? await Task.sleep(nanoseconds: UInt64(sleepSeconds * 1_000_000_000))
                }
            }

            await Task.yield()
        }
        self.isTraining = false
        self.isPaused = false
        self.isPausing = false
        self.pauseRequested = false
        clearPausedDefaults()
        self.status = "done — \(stepCount) steps in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s, final loss \(String(format: "%.3f", currentLoss))"
    }

    private func persistPausedDefaults(checkpoint: String) {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: DefaultsKey.paused)
        defaults.set(stepCount, forKey: DefaultsKey.pausedStep)
        defaults.set(presetIdx, forKey: DefaultsKey.pausedPreset)
        defaults.set(targetSteps, forKey: DefaultsKey.pausedTarget)
        defaults.set(checkpoint, forKey: DefaultsKey.pausedCheckpoint)
    }

    private func clearPausedDefaults() {
        let defaults = UserDefaults.standard
        defaults.removeObject(forKey: DefaultsKey.paused)
        defaults.removeObject(forKey: DefaultsKey.pausedStep)
        defaults.removeObject(forKey: DefaultsKey.pausedPreset)
        defaults.removeObject(forKey: DefaultsKey.pausedTarget)
        defaults.removeObject(forKey: DefaultsKey.pausedCheckpoint)
    }

    private func writeThrottleControlFile(_ value: Double) {
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
            .appendingPathComponent("tinygpt")
            .appendingPathComponent("app-training.throttle")
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let tmp = URL(fileURLWithPath: url.path + ".tmp")
            try String(format: "%.2f\n", value).write(to: tmp, atomically: true, encoding: .utf8)
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            try FileManager.default.moveItem(at: tmp, to: url)
        } catch {
            autoThrottleNote = "throttle control file write failed: \(error)"
        }
    }

    /// Dispatch the LR for a step across the three schedule modes. WSD
    /// reuses the same `lrAtWSD` helper the CLI uses so behavior matches
    /// `tinygpt train --lr-schedule wsd`. Cosine is inlined here (small
    /// + the equivalent CLI helper lives in `TrainSupport` which is in
    /// the CLI executable target, not reachable from the app).
    private func lrAt(step: Int, schedule: LRSchedule, maxLR: Float, minLR: Float,
                       warmup: Int, decaySteps: Int, total: Int) -> Float {
        switch schedule {
        case .constant:
            return maxLR
        case .cosine:
            if step < warmup {
                return maxLR * Float(step + 1) / Float(max(1, warmup))
            }
            if step >= total { return minLR }
            let progress = Double(step - warmup) / Double(max(1, total - warmup))
            let cos = 0.5 * (1.0 + Foundation.cos(.pi * progress))
            return minLR + (maxLR - minLR) * Float(cos)
        case .wsd:
            return lrAtWSD(step: step, total: total, warmup: warmup,
                            decaySteps: decaySteps, maxLR: maxLR, minLR: minLR)
        }
    }

    /// Conservative batch sizes — Mega at B=8 is tight on a 16 GB box; smaller
    /// presets can go larger but the perf difference is small at this scale.
    private func batchSizeFor(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 512 { return 4 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }

    private func estimateParams(_ cfg: ModelConfig) -> Int {
        let v = cfg.vocabSize, c = cfg.dModel, ctx = cfg.contextLength
        let m = cfg.dMlp, l = cfg.nLayers
        return v*c + ctx*c + 2*c + l*(4*(c*c + c) + 4*c + 2*(c*m + m + c))
    }

    private func formatParams(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n)/1_000_000) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n)/1_000) }
        return "\(n)"
    }
}
