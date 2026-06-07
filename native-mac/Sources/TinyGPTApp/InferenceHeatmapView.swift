import SwiftUI
import AppKit
import TinyGPTModel

struct InferenceHeatmapView: View {
    @State private var trace: InferenceTraceRecord?
    @State private var tracePath: String = ""
    @State private var errorText: String = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.line)
            HStack(spacing: 0) {
                controls
                    .frame(width: 340)
                    .background(Theme.panel)
                Divider().background(Theme.line)
                detail
            }
        }
        .background(Theme.base)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Trace")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("inference heatmap")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
            Spacer()
            if let trace {
                Text(String(format: "%.1f ms", trace.totalMs))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(trace.totalMs <= 100 ? Theme.accent : Theme.warn)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 18)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 18) {
            Button {
                openTrace()
            } label: {
                Label("Open Trace JSON", systemImage: "doc.badge.magnifyingglass")
                    .frame(maxWidth: .infinity)
            }
            .controlSize(.large)
            .buttonStyle(.borderedProminent)
            .tint(Theme.accent)

            if !tracePath.isEmpty {
                section("File") {
                    Text(tracePath)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .lineLimit(4)
                        .textSelection(.enabled)
                }
            }

            if let trace {
                section("Request") {
                    stat("Route", trace.route)
                    stat("Model", trace.model)
                    stat("Prompt", "\(trace.promptTokens) tok")
                    stat("Generated", "\(trace.generatedTokens) tok")
                    stat("Cache", trace.cache.hit ? "hit" : "miss")
                }

                section("Budget") {
                    let over = max(0, trace.totalMs - 100)
                    stat("Target", "100 ms")
                    stat("Over", String(format: "%.1f ms", over))
                }
            }

            if !errorText.isEmpty {
                Text(errorText)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(20)
    }

    private var detail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let trace {
                    summary(trace)
                    bars(trace)
                    tokenTable(trace)
                } else {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Open a trace JSON")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundStyle(Theme.fg)
                        Text("Generate one with `tinygpt serve --trace-infer --trace-dir <dir>`.")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                    }
                    .padding(28)
                }
            }
            .padding(24)
        }
    }

    private func summary(_ trace: InferenceTraceRecord) -> some View {
        HStack(spacing: 10) {
            metric("Total", String(format: "%.1f ms", trace.totalMs),
                   color: trace.totalMs <= 100 ? Theme.accent : Theme.warn)
            metric("Prompt", "\(trace.promptTokens) tok", color: Theme.fg)
            metric("Generated", "\(trace.generatedTokens) tok", color: Theme.fg)
            metric("Cache", trace.cache.hit ? "hit" : "miss",
                   color: trace.cache.hit ? Theme.accent : Theme.warn)
        }
    }

    private func bars(_ trace: InferenceTraceRecord) -> some View {
        let rows = aggregateRows(trace)
        let maxMs = max(rows.map(\.durationMs).max() ?? 1, 1)
        return VStack(alignment: .leading, spacing: 8) {
            Text("HEATMAP")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            ForEach(rows) { row in
                HStack(spacing: 10) {
                    Text(row.name)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .frame(width: 190, alignment: .leading)
                        .lineLimit(1)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Rectangle().fill(Theme.panel2)
                            Rectangle()
                                .fill(color(for: row.name, ms: row.durationMs))
                                .frame(width: max(2, geo.size.width * row.durationMs / maxMs))
                        }
                    }
                    .frame(height: 18)
                    Text(String(format: "%.1f ms", row.durationMs))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                        .frame(width: 88, alignment: .trailing)
                }
            }
        }
        .padding(16)
        .background(Theme.panel)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func tokenTable(_ trace: InferenceTraceRecord) -> some View {
        let tokens = trace.tokens.sorted {
            ($0.modelMs + $0.constraintMs + $0.decodeMs) > ($1.modelMs + $1.constraintMs + $1.decodeMs)
        }
        .prefix(12)
        return VStack(alignment: .leading, spacing: 8) {
            Text("SLOWEST TOKENS")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            ForEach(Array(tokens), id: \.index) { token in
                HStack {
                    Text("#\(token.index)")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .frame(width: 46, alignment: .leading)
                    Text(String(format: "model %.1f", token.modelMs))
                    Text(String(format: "constraint %.1f", token.constraintMs))
                    Text(String(format: "decode %.1f", token.decodeMs))
                    Spacer()
                }
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.fg)
            }
        }
        .padding(16)
        .background(Theme.panel)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func metric(_ label: String, _ value: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
            Text(value)
                .font(.system(size: 16, weight: .semibold, design: .monospaced))
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Theme.panel)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
        .clipShape(RoundedRectangle(cornerRadius: 6))
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

    private func stat(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(Theme.muted)
            Spacer()
            Text(value)
                .foregroundStyle(Theme.fg)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .font(.system(size: 11, design: .monospaced))
    }

    private func openTrace() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let data = try Data(contentsOf: url)
            trace = try JSONDecoder().decode(InferenceTraceRecord.self, from: data)
            tracePath = url.path
            errorText = ""
        } catch {
            errorText = "could not load trace: \(error)"
        }
    }

    private func aggregateRows(_ trace: InferenceTraceRecord) -> [HeatmapRow] {
        var rows: [HeatmapRow] = []
        let grouped = Dictionary(grouping: trace.spans, by: \.name)
        for name in grouped.keys.sorted() {
            rows.append(HeatmapRow(
                name: name,
                durationMs: (grouped[name] ?? []).map(\.durationMs).reduce(0, +)
            ))
        }
        let model = trace.tokens.map(\.modelMs).reduce(0, +)
        let constraint = trace.tokens.map(\.constraintMs).reduce(0, +)
        let decode = trace.tokens.map(\.decodeMs).reduce(0, +)
        if model > 0 { rows.append(HeatmapRow(name: "tokens.model", durationMs: model)) }
        if constraint > 0 { rows.append(HeatmapRow(name: "tokens.constraint", durationMs: constraint)) }
        if decode > 0 { rows.append(HeatmapRow(name: "tokens.decode", durationMs: decode)) }
        return rows.sorted { $0.durationMs > $1.durationMs }
    }

    private func color(for name: String, ms: Double) -> Color {
        if name.contains("constraint") { return Theme.danger }
        if ms > 100 { return Theme.warn }
        return Theme.accent
    }
}

private struct HeatmapRow: Identifiable {
    let id = UUID()
    let name: String
    let durationMs: Double
}
