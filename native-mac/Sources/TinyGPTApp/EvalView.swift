import SwiftUI
import AppKit

struct EvalView: View {
    @StateObject private var controller = EvalController()

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.line)
            HStack(spacing: 0) {
                controls
                    .frame(width: 360)
                    .background(Theme.panel)
                Divider().background(Theme.line)
                results
            }
        }
        .background(Theme.base)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Eval")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("score specialists without leaving the app")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
            Spacer()
            if controller.isRunning {
                HStack(spacing: 6) {
                    Circle().fill(Theme.accent).frame(width: 7, height: 7)
                    Text("RUNNING")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.accent)
                }
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 18)
    }

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("", selection: $controller.mode) {
                        ForEach(EvalMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text(controller.mode.subtitle)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }

                section("Model") {
                    pathRow(text: $controller.modelPath, placeholder: "checkpoint or HF dir") {
                        controller.pickModel()
                    }
                }

                if controller.mode == .emergence {
                    section("Run History") {
                        pathRow(text: $controller.runStemPath, placeholder: "run stem .tinygpt") {
                            controller.pickRunStem()
                        }
                    }
                }

                if controller.mode == .custom {
                    section("Custom JSONL") {
                        pathRow(text: $controller.customPath, placeholder: "{prompt, expected}.jsonl") {
                            controller.pickCustomJSONL()
                        }
                        Picker("metric", selection: $controller.customMetric) {
                            ForEach(CustomEvalMetric.allCases) { metric in
                                Text(metric.rawValue).tag(metric)
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                } else {
                    section("Tasks") {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 92), spacing: 6)], spacing: 6) {
                            ForEach(controller.commonTasks, id: \.self) { task in
                                taskChip(task)
                            }
                        }
                    }
                }

                section("Limit") {
                    Picker("", selection: $controller.limitIndex) {
                        Text("10").tag(0)
                        Text("30").tag(1)
                        Text("50").tag(2)
                        Text("100").tag(3)
                        Text("500").tag(4)
                        Text("full").tag(5)
                    }
                    .pickerStyle(.segmented)
                    Toggle("baseline", isOn: $controller.includeBaseline)
                        .toggleStyle(.checkbox)
                        .disabled(true)
                        .help("Baseline orchestration is reserved for the next pass; current runs write model rows.")
                }

                if controller.isRunning {
                    Button { controller.cancel() } label: {
                        Text("Cancel").frame(maxWidth: .infinity)
                    }
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.danger)
                } else {
                    Button { controller.run() } label: {
                        Text(controller.mode == .emergence ? "Sweep" : "Run").frame(maxWidth: .infinity)
                    }
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                }

                // Only show Output path AFTER a run has produced rows.
                // Showing the pre-allocated path before any eval just looks
                // like noise + clutters the panel.
                if !controller.resultsPath.isEmpty && !controller.rows.isEmpty {
                    section("Output") {
                        Text(controller.resultsPath)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                            .textSelection(.enabled)
                        Button {
                            controller.revealResults()
                        } label: {
                            Label("Reveal in Finder", systemImage: "folder")
                        }
                    }
                }

                if !controller.history.isEmpty {
                    section("History") {
                        ForEach(controller.history.prefix(8)) { item in
                            Button {
                                controller.loadHistoryItem(item)
                            } label: {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(item.mode)
                                        .font(.system(size: 11, weight: .semibold))
                                    Text("\(item.rowCount) rows · \(URL(fileURLWithPath: item.resultsPath).lastPathComponent)")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(Theme.faint)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .padding(20)
        }
    }

    private var results: some View {
        VStack(spacing: 0) {
            if controller.mode == .emergence {
                VStack(alignment: .leading, spacing: 8) {
                    Text("EMERGENCE")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    EvalChart(rows: controller.rows)
                        .frame(height: 160)
                        .background(Theme.panel)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
                }
                .padding(20)
                Divider().background(Theme.line)
            }

            table
                .frame(maxHeight: .infinity)
            Divider().background(Theme.line)
            logPane
                .frame(height: 180)
        }
    }

    /// "Past results" surface — replaces the dropped Compare workspace.
    /// Lists every `.jsonl` under `docs/artifacts/` and offers to load it
    /// into the table view. Empty by design when no artifacts exist.
    private var pastResultsList: some View {
        let artifacts = pastArtifacts()
        return VStack(alignment: .leading, spacing: 6) {
            if !artifacts.isEmpty {
                Text("PAST RESULTS")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.faint)
                    .tracking(1)
                ForEach(artifacts, id: \.path) { artifact in
                    Button {
                        controller.loadArtifact(path: artifact.path)
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "doc.text")
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.faint)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(artifact.name)
                                    .font(.system(size: 12, design: .monospaced))
                                    .foregroundStyle(Theme.fg)
                                Text("\(artifact.rows) rows · \(artifact.size)")
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(Theme.muted)
                            }
                            Spacer(minLength: 0)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Theme.panel)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func pastArtifacts() -> [(name: String, path: String, rows: Int, size: String)] {
        guard let repo = repoRoot() else { return [] }
        let dir = repo.appendingPathComponent("docs/artifacts")
        guard let entries = try? FileManager.default.contentsOfDirectory(at: dir,
                                                                         includingPropertiesForKeys: nil)
        else { return [] }
        return entries.compactMap { url -> (String, String, Int, String)? in
            guard url.pathExtension == "jsonl" else { return nil }
            let text = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
            let rows = text.split(separator: "\n", omittingEmptySubsequences: true).count
            let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            let size = bytes > 1_000_000 ? String(format: "%.1f MB", Double(bytes) / 1_000_000)
                : bytes > 1_000 ? String(format: "%.0f KB", Double(bytes) / 1_000)
                : "\(bytes) B"
            return (url.lastPathComponent, url.path, rows, size)
        }.sorted { $0.0 > $1.0 }
            .map { (name: $0.0, path: $0.1, rows: $0.2, size: $0.3) }
    }

    private func repoRoot() -> URL? {
        let fm = FileManager.default
        guard let exec = Bundle.main.executableURL else { return nil }
        var dir = exec.deletingLastPathComponent()
        for _ in 0..<8 {
            if fm.fileExists(atPath: dir.appendingPathComponent("docs").path) { return dir }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }

    private var table: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                if controller.mode == .custom && !controller.customRows.isEmpty {
                    ForEach(controller.customRows) { row in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: row.passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(row.passed ? Theme.accent : Theme.danger)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(row.prompt)
                                    .foregroundStyle(Theme.fg)
                                Text("expected: \(row.expected)")
                                    .foregroundStyle(Theme.muted)
                                Text(row.output.trimmingCharacters(in: .whitespacesAndNewlines))
                                    .foregroundStyle(Theme.faint)
                                    .lineLimit(3)
                            }
                            .font(.system(size: 11, design: .monospaced))
                        }
                        .padding(10)
                        .background(Theme.panel)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                } else if controller.rows.isEmpty {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("No eval rows yet — pick a model + run, or browse past results below:")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(Theme.faint)

                        pastResultsList
                    }
                    .padding(20)
                } else {
                    headerRow
                    ForEach(controller.rows) { row in
                        HStack(spacing: 10) {
                            cell(row.task, width: 100)
                            cell(row.model_step.map(String.init) ?? "-", width: 70)
                            cell(row.metric, width: 90)
                            cell(String(format: "%.3f", row.score), width: 70)
                            cell("n=\(row.n_examples)", width: 70)
                            Text(row.model_name)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.muted)
                                .lineLimit(1)
                            Spacer()
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(Theme.panel.opacity(0.7))
                    }
                }
            }
            .padding(20)
        }
    }

    private var headerRow: some View {
        HStack(spacing: 10) {
            cell("task", width: 100, header: true)
            cell("step", width: 70, header: true)
            cell("metric", width: 90, header: true)
            cell("score", width: 70, header: true)
            cell("n", width: 70, header: true)
            Text("model")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
            Spacer()
        }
        .padding(.horizontal, 14)
    }

    private var logPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("LOG")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.faint)
                Spacer()
                if let error = controller.lastError {
                    Text(error)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.warn)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            ScrollView {
                Text(controller.log.isEmpty ? "eval subprocess output appears here" : controller.log)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(controller.log.isEmpty ? Theme.faint : Theme.muted)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(14)
            }
            .background(Theme.panel)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            content()
        }
    }

    private func pathRow(text: Binding<String>, placeholder: String, action: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            TextField(placeholder, text: text)
                .textFieldStyle(.plain)
                .font(.system(size: 11, design: .monospaced))
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(Theme.panel2)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            Button(action: action) {
                Image(systemName: "folder")
            }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.muted)
        }
    }

    private func taskChip(_ task: String) -> some View {
        let active = controller.selectedTasks.contains(task)
        return Button {
            if active { controller.selectedTasks.remove(task) }
            else { controller.selectedTasks.insert(task) }
        } label: {
            Text(task)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(active ? Theme.accent : Theme.muted)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
                .background(active ? Theme.accentGlow : Theme.panel2)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(active ? Theme.accentDim : Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }

    private func cell(_ text: String, width: CGFloat, header: Bool = false) -> some View {
        Text(text)
            .font(.system(size: header ? 10 : 11, weight: header ? .semibold : .regular, design: .monospaced))
            .foregroundStyle(header ? Theme.faint : Theme.fg)
            .frame(width: width, alignment: .leading)
            .lineLimit(1)
    }
}

