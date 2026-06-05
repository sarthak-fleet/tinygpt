import SwiftUI
import UniformTypeIdentifiers

/// The Interp tab — TinyGPT's unique-vs-LM-Studio surface. v1 launches
/// a SAE training run via the bundled CLI (`tinygpt sae`) and streams
/// the output here. Future iterations layer MEMIT, activation patching,
/// and the SAE timeline viewer on top of the same primitive.
struct InterpView: View {
    @StateObject private var controller = InterpController()

    @State private var modelPath: String = ""
    @State private var corpusPath: String = ""
    @State private var layer: Int = 3
    @State private var features: Int = 1024
    @State private var steps: Int = 1000
    @State private var batch: Int = 32
    @State private var ctx: Int = 128
    @State private var outPath: String = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.line)
            HStack(alignment: .top, spacing: 0) {
                form
                    .frame(width: 320)
                    .background(Theme.panel)
                Divider().background(Theme.line)
                outputArea
            }
        }
        .background(Theme.base)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Interp")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("decompose what the model learned")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
            Spacer()
            if let mse = controller.lastMSE {
                Text(String(format: "MSE %.4e", mse))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.accent)
            }
            if let l0 = controller.lastL0Pct {
                Text(String(format: "L0 %.2f%%", l0))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.accent)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 18)
    }

    private var form: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                section("Inputs") {
                    pathPicker(label: "Model",
                               value: $modelPath,
                               placeholder: "pick a .tinygpt or .bin",
                               types: [.data])
                    pathPicker(label: "Corpus",
                               value: $corpusPath,
                               placeholder: "pick a UTF-8 .txt",
                               types: [.plainText, .utf8PlainText, .text])
                }

                section("Hyperparameters") {
                    numericRow(label: "Layer", value: $layer, range: 0...64)
                    numericRow(label: "d_features", value: $features, range: 64...8192)
                    numericRow(label: "Steps", value: $steps, range: 1...100000)
                    numericRow(label: "Batch", value: $batch, range: 1...256)
                    numericRow(label: "Context", value: $ctx, range: 16...2048)
                }

                section("Output") {
                    pathSaver(label: "Sidecar",
                              value: $outPath,
                              suggested: "probe.sae")
                }

                if controller.isRunning {
                    Button {
                        controller.cancel()
                    } label: { Text("Cancel").frame(maxWidth: .infinity) }
                        .controlSize(.large)
                        .buttonStyle(.borderedProminent)
                        .tint(Theme.danger)
                } else {
                    Button {
                        launch()
                    } label: { Text("Train SAE").frame(maxWidth: .infinity) }
                        .controlSize(.large)
                        .buttonStyle(.borderedProminent)
                        .tint(Theme.accent)
                        .disabled(modelPath.isEmpty || corpusPath.isEmpty || outPath.isEmpty)
                }

                Text(controller.status)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
            }
            .padding(20)
        }
    }

    private var outputArea: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("OUTPUT")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.faint)
                    .tracking(1)
                Spacer()
                if let path = controller.lastSAEPath, !controller.isRunning {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "doc.text.magnifyingglass")
                            Text("Reveal").font(.system(size: 11))
                        }
                    }
                    .buttonStyle(.borderless)
                }
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 10)
            Divider().background(Theme.line)
            ScrollView {
                ScrollViewReader { proxy in
                    Text(controller.output.isEmpty
                         ? "Output from `tinygpt sae` will stream here once a run starts.\n\nThis trains a sparse autoencoder on the residual stream of the picked layer — the decoder columns become interpretable feature directions, the encoder sparsifies the model's activations into a feature pattern per token. Bricken et al. 2023."
                         : controller.output)
                        .font(.tgMono)
                        .foregroundStyle(controller.output.isEmpty ? Theme.faint : Theme.fg)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(18)
                        .id("interp-out-end")
                        .onChange(of: controller.output) { _, _ in
                            withAnimation(.linear(duration: 0.05)) {
                                proxy.scrollTo("interp-out-end", anchor: .bottom)
                            }
                        }
                }
            }
        }
    }

    // MARK: - small UI bits

    @ViewBuilder
    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            content()
        }
    }

    private func pathPicker(label: String, value: Binding<String>,
                            placeholder: String,
                            types: [UTType]) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
                .frame(width: 60, alignment: .leading)
            TextField(placeholder, text: value)
                .textFieldStyle(.plain)
                .font(.system(size: 11, design: .monospaced))
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(Theme.panel2)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            Button {
                let panel = NSOpenPanel()
                panel.allowedContentTypes = types
                panel.allowsMultipleSelection = false
                if panel.runModal() == .OK, let url = panel.url {
                    value.wrappedValue = url.path
                }
            } label: { Image(systemName: "folder") }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.muted)
        }
    }

    private func pathSaver(label: String, value: Binding<String>,
                           suggested: String) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
                .frame(width: 60, alignment: .leading)
            TextField(suggested, text: value)
                .textFieldStyle(.plain)
                .font(.system(size: 11, design: .monospaced))
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(Theme.panel2)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            Button {
                let panel = NSSavePanel()
                panel.nameFieldStringValue = suggested
                if panel.runModal() == .OK, let url = panel.url {
                    value.wrappedValue = url.path
                }
            } label: { Image(systemName: "folder.badge.plus") }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.muted)
        }
    }

    private func numericRow(label: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
                .frame(width: 90, alignment: .leading)
            TextField("", value: value, format: .number)
                .textFieldStyle(.plain)
                .multilineTextAlignment(.trailing)
                .font(.system(size: 11, design: .monospaced))
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(Theme.panel2)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            Stepper("", value: value, in: range)
                .labelsHidden()
                .controlSize(.small)
        }
    }

    private func launch() {
        if outPath.isEmpty {
            // Auto-derive an output path next to the model if the user
            // didn't pick one — common "happy path" so the flow doesn't
            // block on a Save panel.
            let url = URL(fileURLWithPath: modelPath)
            outPath = url.deletingLastPathComponent()
                .appendingPathComponent("\(url.deletingPathExtension().lastPathComponent).layer\(layer).sae")
                .path
        }
        controller.trainSAE(modelPath: modelPath, corpusPath: corpusPath,
                            layer: layer, features: features,
                            steps: steps, batch: batch, ctx: ctx,
                            outPath: outPath)
    }
}
