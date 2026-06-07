import SwiftUI
import UniformTypeIdentifiers

struct TrainView: View {
    @StateObject private var controller = TrainController()
    @State private var corpusText: String = "(no corpus loaded — drop a UTF-8 text file or pick one below)"
    @State private var hasRealCorpus: Bool = false
    @State private var corpusBytes: Int = 0
    @State private var availableCorpora: [CorpusItem] = []
    @State private var selectedCorpus: CorpusItem? = nil
    @StateObject private var thermalMonitor = ThermalMonitor()
    @State private var showLongRunConfirm = false
    @State private var showPausedRestorePrompt = false
    @State private var trainingRuns: [RunSummary] = []
    @State private var selectedDetectedRunIdx: Int = 0

    var body: some View {
        VStack(spacing: 0) {
            // Header — current run summary
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("Train")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("watch a model learn from scratch")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                Spacer()
                if controller.isTraining {
                    HStack(spacing: 6) {
                        Circle().fill(Theme.accent).frame(width: 6, height: 6)
                        Text(String(format: "step %d / %d · %.1f step/s", controller.stepCount, controller.targetSteps, controller.stepsPerSec))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                    }
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 18)
            Divider().background(Theme.line)
            ThermalSafetyBanner()

            // Controls row
            HStack(spacing: 14) {
                // Preset picker
                HStack(spacing: 6) {
                    Text("preset")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    Picker("", selection: $controller.presetIdx) {
                        ForEach(0..<TrainController.presets.count, id: \.self) { i in
                            Text(TrainController.presets[i].name).tag(i)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 100)
                }

                HStack(spacing: 6) {
                    Text("steps")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    TextField(
                        "",
                        value: $controller.targetSteps,
                        format: IntegerFormatStyle<Int>.number
                            .locale(Locale(identifier: "en_US_POSIX"))
                            .grouping(.automatic)
                    )
                        .textFieldStyle(.plain)
                        .frame(width: 80)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(Theme.panel)
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                        .font(.system(size: 11, design: .monospaced))
                }

                // LR schedule picker — exposes today's WSD + cosine + constant.
                HStack(spacing: 6) {
                    Text("schedule")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    Picker("", selection: $controller.lrSchedule) {
                        ForEach(TrainController.LRSchedule.allCases) { s in
                            Text(s.rawValue).tag(s)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 95)
                }

                // Seed — empty = random at runtime (placeholder, not literal text).
                VStack(alignment: .leading, spacing: 4) {
                    Text("seed")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    ZStack(alignment: .leading) {
                        if controller.seedText.isEmpty {
                            Text("random — auto-pick")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.faint.opacity(0.7))
                                .italic()
                                .padding(.horizontal, 8)
                                .allowsHitTesting(false)
                        }
                        TextField("", text: $controller.seedText)
                            .textFieldStyle(.plain)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .font(.system(size: 11, design: .monospaced))
                    }
                    .frame(width: 120)
                    .frame(minHeight: 28)
                    .background(Theme.panel)
                    .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .help("Leave blank for random init; any UInt64 makes init reproducible.")
                }

                // Spike-detector toggle — same primitive the CLI uses.
                Toggle(isOn: $controller.spikeDetectEnabled) {
                    Text("spike")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                }
                .toggleStyle(.checkbox)
                .frame(minHeight: 28)
                .contentShape(Rectangle())
                .help("Logs a warning when loss > 3× moving avg over the last 50 steps.")

                // Throttle — label stacked above controls for readable layout.
                VStack(alignment: .leading, spacing: 6) {
                    Text("Throttle:")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    HStack(spacing: 10) {
                        throttleChip("100%", 1.0, icon: "bolt.fill")
                        throttleChip("75%", 0.75, icon: "hare.fill")
                        throttleChip("50%", 0.50, icon: "tortoise.fill")
                        throttleChip("25%", 0.25, icon: "snowflake")
                        Spacer(minLength: 8)
                        Toggle(isOn: $controller.allowAutoThrottle) {
                            Text("auto")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Theme.faint)
                        }
                        .toggleStyle(.checkbox)
                        .frame(minHeight: 28)
                        .contentShape(Rectangle())
                        .help("Thermal pressure can lower the throttle below your cap.")
                    }
                }
                .help("Lower sustained load without changing training results.")

                // Starter corpora menu — one click loads any of the
                // fetched Project Gutenberg classics or browser gallery
                // corpora. Falls through to "Other..." for arbitrary files.
                Menu {
                    if availableCorpora.isEmpty {
                        Text("no corpora found — run scripts/fetch_corpora.sh").disabled(true)
                    }
                    ForEach(availableCorpora) { c in
                        Button {
                            loadCorpus(c)
                        } label: {
                            HStack {
                                Text(c.icon)
                                Text(c.displayName)
                                Spacer()
                                Text(formattedBytes(c.size))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    Divider()
                    Button("Other file…") { pickCorpus() }
                } label: {
                    HStack(spacing: 6) {
                        if let sel = selectedCorpus {
                            Text(sel.icon)
                            Text(sel.displayName)
                            Text("(\(formattedBytes(corpusBytes)))")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Theme.faint)
                        } else if hasRealCorpus {
                            Text("Corpus (\(formattedBytes(corpusBytes)))")
                        } else {
                            Text("Pick corpus…")
                        }
                    }
                }
                .menuStyle(.borderlessButton)
                .frame(maxWidth: 200)

                Spacer()

                if controller.isTraining {
                    Button(controller.isPaused ? "Resume" : "Pause") {
                        controller.isPaused ? controller.resume() : controller.pause()
                    }
                    .disabled(controller.isPausing)
                    .frame(minHeight: 44)
                    .buttonStyle(PrimaryButtonStyle(color: Theme.warn))
                    Button("Stop") { controller.cancel() }
                        .keyboardShortcut(.cancelAction)
                        .frame(minHeight: 44)
                        .buttonStyle(PrimaryButtonStyle(color: Theme.danger))
                } else if let run = controller.externalRun, run.pid != 0 {
                    // External (CLI-spawned) training is in flight — the
                    // banner has Resume/Pause for it. Don't show our own
                    // Start; would let the user fire a second simultaneous
                    // run that fights for the GPU.
                    Text(run.isStopped ? "external run paused" : "external run active")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                } else {
                    Button(controller.externalRun != nil ? "Attached to existing run" : "Start") {
                        if controller.externalRun == nil { startOrConfirm() }
                    }
                    .disabled(controller.externalRun != nil)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .frame(minHeight: 44)
                    .buttonStyle(PrimaryButtonStyle(color: Theme.accent))
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Theme.panel)
            Divider().background(Theme.line)

            if let note = controller.autoThrottleNote {
                HStack(spacing: 10) {
                    Image(systemName: "speedometer")
                        .foregroundStyle(Theme.warn)
                    Text(note)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                    Spacer()
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
                .background(Theme.warn.opacity(0.10))
            }

            // Spike alert banner — sticky between the controls and the
            // chart so it sits in the operator's peripheral vision but
            // doesn't crowd the chart itself.
            if let alert = controller.spikeAlert {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(Theme.warn)
                    Text(alert)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                    Spacer()
                    Button {
                        controller.spikeAlert = nil
                    } label: {
                        Image(systemName: "xmark")
                            .foregroundStyle(Theme.faint)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
                .background(Theme.warn.opacity(0.12))
                .overlay(Rectangle().fill(Theme.warn).frame(height: 1), alignment: .bottom)
            }

            // Training history strip — all runs found in ~/.cache/tinygpt/runs/.
            // Active runs marked with a green dot; SIGSTOP'd with orange;
            // exited with grey. Click a row to load its loss curve.
            if !trainingRuns.isEmpty {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 12) {
                        Text("RUNS")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Theme.faint)
                            .tracking(1)
                        if case .historical = controller.viewedRun {
                            HStack(spacing: 4) {
                                Image(systemName: "clock.arrow.circlepath")
                                    .font(.system(size: 10))
                                Text("viewing past run — back to live")
                                    .font(.system(size: 10, design: .monospaced))
                            }
                            .foregroundStyle(Theme.accent)
                            .onTapGesture { controller.returnToLive() }
                            .contentShape(Rectangle())
                            .help("Click to return to the live training run.")
                        }
                        Spacer()
                        Button {
                            trainingRuns = RunRegistry.discover()
                        } label: {
                            Text("Refresh")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Theme.muted)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 8)
                    .padding(.bottom, 4)

                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(trainingRuns) { run in
                                runChip(run)
                            }
                        }
                        .padding(.horizontal, 24)
                        .padding(.bottom, 10)
                    }
                }
                .background(Theme.panel.opacity(0.6))
                .overlay(Rectangle().fill(Theme.line).frame(height: 1), alignment: .bottom)
            }

            // External-run banner — surfaces a `tinygpt train` process
            // that wasn't started by this app instance (CLI-spawned nohup,
            // previous app session, etc.). User can SIGCONT/SIGSTOP it
            // without losing in-memory state.
            if controller.detectedRuns.count > 1 {
                HStack(spacing: 10) {
                    Text("\(controller.detectedRuns.count) training runs detected")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                    Picker("Attach to", selection: $selectedDetectedRunIdx) {
                        ForEach(Array(controller.detectedRuns.enumerated()), id: \.offset) { idx, run in
                            Text(runLabel(run, index: idx)).tag(idx)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(minWidth: 200)
                    .onChange(of: selectedDetectedRunIdx) { _, idx in
                        controller.selectDetectedRun(at: idx)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
                .background(Theme.panel.opacity(0.5))
            }

            if let run = controller.externalRun {
                HStack(spacing: 12) {
                    Image(systemName: "antenna.radiowaves.left.and.right")
                        .foregroundStyle(Theme.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(run.pid == 0
                             ? "Last training run — exited cleanly (no live process)"
                             : "Attached to existing run — PID \(run.pid)\(run.isStopped ? " (paused)" : "")")
                            .font(.system(size: 13, weight: .semibold))
                        HStack(spacing: 16) {
                            if let s = run.lastStep {
                                Text("step \(s)\(run.totalSteps.map { " / \($0)" } ?? "")")
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(Theme.muted)
                            }
                            if let l = run.lastLoss {
                                Text(String(format: "loss %.3f", l))
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(Theme.muted)
                            }
                            if let v = run.lastValLoss {
                                Text(String(format: "val %.3f", v))
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(Theme.muted)
                            }
                            if let r = run.stepsPerSec {
                                Text(String(format: "%.1f step/s", r))
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(Theme.muted)
                            }
                            if let eta = run.etaSeconds {
                                Text("ETA \(TrainController.formatETA(eta))")
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(Theme.accent.opacity(0.8))
                            }
                            if let log = run.logPath {
                                Text(log)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(Theme.faint)
                                    .truncationMode(.middle)
                                    .lineLimit(1)
                            }
                        }
                    }
                    Spacer()
                    Button(run.isStopped ? "Resume" : "Pause") {
                        if run.isStopped {
                            controller.resumeExternalRun()
                        } else {
                            controller.pauseExternalRun()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    Button("Refresh") {
                        controller.detectExternalRun()
                    }
                    .buttonStyle(.bordered)
                    if let log = run.logPath {
                        Button("Reveal log") {
                            NSWorkspace.shared.activateFileViewerSelecting(
                                [URL(fileURLWithPath: log)]
                            )
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 10)
                .background(Theme.accent.opacity(0.10))
                .overlay(Rectangle().fill(Theme.accent.opacity(0.4)).frame(height: 1), alignment: .bottom)
            }

            // Chart
            LossChart(points: controller.lossHistory, targetSteps: controller.targetSteps)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.horizontal, 24)
                .padding(.top, 16)

            // Stats row
            HStack(spacing: 24) {
                statBlock(label: "STEP", value: "\(controller.stepCount)")
                statBlock(label: "LOSS",
                          value: controller.currentLoss == 0
                            ? "—"
                            : String(format: "%.3f", controller.currentLoss))
                statBlock(label: "STEP/S",
                          value: controller.stepsPerSec == 0
                            ? "—"
                            : String(format: "%.1f", controller.stepsPerSec))
                statBlock(label: "LOAD",
                          value: controller.isPaused
                            ? "paused"
                            : String(format: "%.0f%%", controller.throttle * 100))
                Spacer()
                ThermalStatusChip(state: thermalMonitor.state,
                                  autoThrottleNote: controller.autoThrottleNote)
                Text(controller.status)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
                    .truncationMode(.tail)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
        }
        .background(Theme.base)
        .onAppear {
            availableCorpora = CorpusDiscovery.discover()
            showPausedRestorePrompt = controller.restorePausedMetadataIfPresent()
            controller.applyThermalThrottle(thermalMonitor.state)
            controller.detectExistingRun()
            if controller.externalRun != nil {
                controller.loadLossChartFromExternalRun()
            }
            trainingRuns = RunRegistry.discover()
        }
        .onReceive(Timer.publish(every: 3, on: .main, in: .common).autoconnect()) { _ in
            // Live-poll external-run state every 3s. Only when we have
            // a detected external run + the user isn't running an
            // in-process training session that'd conflict.
            guard !controller.isTraining else { return }
            controller.detectExistingRun()
            trainingRuns = RunRegistry.discover()
            if controller.externalRun != nil {
                controller.loadLossChartFromExternalRun()
            }
        }
        .onChange(of: thermalMonitor.state) { _, state in
            controller.applyThermalThrottle(state)
        }
        .onReceive(NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification
        )) { _ in
            thermalMonitor.refresh()
        }
        .alert("Long Training Run", isPresented: $showLongRunConfirm) {
            Button("Start") { startNow() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This run is expected to keep the GPU busy for a long time. Use a hard surface with clear airflow before starting.")
        }
        .alert("Previous Paused Run", isPresented: $showPausedRestorePrompt) {
            Button("OK") {}
        } message: {
            Text("The previous app session recorded a paused checkpoint. Load the same corpus and press Start to continue from it.")
        }
    }

    private func startOrConfirm() {
        if controller.targetSteps > 1500 {
            showLongRunConfirm = true
        } else {
            startNow()
        }
    }

    private func startNow() {
        let corpus: Data
        if hasRealCorpus {
            corpus = Data(corpusText.utf8)
        } else {
            // Random bytes — perf demo even without a corpus.
            // Loss will land at ln(256), not below.
            corpus = Data((0..<200_000).map { _ in UInt8.random(in: 0...255) })
        }
        controller.start(corpus: corpus)
    }

    /// One chip per run discovered under ~/.cache/tinygpt/runs/.
    /// Status dot: green = active, orange = paused (SIGSTOP'd),
    /// grey = exited. Click to reveal the run dir in Finder.
    private func runChip(_ run: RunSummary) -> some View {
        let dotColor: Color = run.isActive
            ? (run.isStopped ? .orange : .green)
            : Theme.faint
        let status: String = run.isActive
            ? (run.isStopped ? "paused" : "running")
            : "exited"
        let isViewed: Bool = {
            switch controller.viewedRun {
            case .historical(let id): return id == run.id
            case .live: return run.isActive
            case .none: return false
            }
        }()
        return Button {
            controller.loadLossChartFromRunSummary(run)
        } label: {
            HStack(spacing: 8) {
                Circle()
                    .fill(dotColor)
                    .frame(width: 6, height: 6)
                VStack(alignment: .leading, spacing: 1) {
                    Text(run.id)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(isViewed ? Theme.accent : Theme.fg)
                        .lineLimit(1)
                    HStack(spacing: 6) {
                        Text(status)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(Theme.faint)
                        if let s = run.lastStep {
                            Text("step \(s)\(run.totalSteps.map { "/\($0)" } ?? "")")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(Theme.muted)
                        }
                        if let l = run.lastLoss {
                            Text(String(format: "loss %.2f", l))
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(Theme.muted)
                        }
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .frame(minHeight: 32)
            .background(Theme.panel)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .help("\(run.directory.path)")
    }

    private func throttleChip(_ label: String, _ value: Double, icon: String) -> some View {
        let active = abs(controller.throttle - value) < 0.001
        return Button {
            controller.setThrottle(value)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 9))
                Text(label)
                    .font(.system(size: 11, weight: active ? .semibold : .regular, design: .monospaced))
            }
            .foregroundStyle(active ? Theme.accent : Theme.muted)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(minWidth: 52, minHeight: 28)
            .background(active ? Theme.accent.opacity(0.12) : Theme.panel)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(active ? Theme.accent.opacity(0.5) : Theme.line))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }

    private func runLabel(_ run: TrainController.ExternalRun, index: Int) -> String {
        let step = run.lastStep.map { "step \($0)" } ?? "step ?"
        let pid = run.pid == 0 ? "exited" : "pid \(run.pid)"
        return "Run \(index + 1) · \(pid) · \(step)"
    }

    private func statBlock(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            Text(value)
                .font(.system(size: 18, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.fg)
        }
        .frame(minWidth: 60, alignment: .leading)
    }

    private func loadCorpus(_ c: CorpusItem) {
        do {
            let text = try String(contentsOf: c.url, encoding: .utf8)
            corpusText = text
            corpusBytes = text.utf8.count
            hasRealCorpus = true
            selectedCorpus = c
        } catch {
            corpusText = "(couldn't load \(c.url.lastPathComponent): \(error))"
            hasRealCorpus = false
            selectedCorpus = nil
        }
    }

    private func pickCorpus() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.plainText, .utf8PlainText, .text]
        panel.allowsMultipleSelection = false
        panel.message = "Pick a UTF-8 text file to train on."
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let text = try String(contentsOf: url, encoding: .utf8)
                corpusText = text
                corpusBytes = text.utf8.count
                hasRealCorpus = true
            } catch {
                corpusText = "(couldn't load \(url.lastPathComponent): \(error))"
                hasRealCorpus = false
            }
        }
    }

    private func formattedBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fMB", Double(n)/1_000_000) }
        if n >= 1_000 { return String(format: "%.0fKB", Double(n)/1_000) }
        return "\(n)B"
    }
}

private struct PrimaryButtonStyle: ButtonStyle {
    let color: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .contentShape(Rectangle())
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(color.opacity(configuration.isPressed ? 0.25 : 0.15))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(color.opacity(0.5), lineWidth: 1)
            )
    }
}
