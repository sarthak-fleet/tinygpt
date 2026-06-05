import SwiftUI

/// The Server tab — wraps `ServerController`, lets the user point the
/// server at any model on disk + start/stop the OpenAI-compatible HTTP
/// endpoint with one click. Closes the last table-stakes gap vs LM
/// Studio's "Local Server" page.
struct ServerView: View {
    @StateObject private var controller = ServerController()
    @State private var portText: String = "8080"

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.line)
            HStack(alignment: .top, spacing: 0) {
                form
                    .frame(width: 340)
                    .background(Theme.panel)
                Divider().background(Theme.line)
                logArea
            }
        }
        .background(Theme.base)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Server")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("OpenAI-compatible HTTP endpoint")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
            Spacer()
            if controller.isRunning {
                HStack(spacing: 6) {
                    Circle().fill(Theme.accent).frame(width: 8, height: 8)
                    Text("LISTENING")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.accent)
                        .tracking(1)
                }
            } else {
                HStack(spacing: 6) {
                    Circle().stroke(Theme.faint, lineWidth: 1).frame(width: 8, height: 8)
                    Text("STOPPED")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                        .tracking(1)
                }
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 18)
    }

    private var form: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                section("Model") {
                    HStack(spacing: 8) {
                        TextField("path to .tinygpt or HF dir", text: $controller.modelPath)
                            .textFieldStyle(.plain)
                            .font(.system(size: 11, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(Theme.panel2)
                            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                            .disabled(controller.isRunning)
                        Button {
                            pickModel()
                        } label: { Image(systemName: "folder") }
                        .buttonStyle(.plain)
                        .foregroundStyle(Theme.muted)
                        .disabled(controller.isRunning)
                    }
                }

                section("Binding") {
                    HStack(spacing: 8) {
                        Text("Port")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                            .frame(width: 60, alignment: .leading)
                        TextField("8080", text: $portText)
                            .textFieldStyle(.plain)
                            .multilineTextAlignment(.trailing)
                            .font(.system(size: 12, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(Theme.panel2)
                            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                            .disabled(controller.isRunning)
                            .onChange(of: portText) { _, new in
                                if let p = Int(new), p > 0, p < 65536 {
                                    controller.port = p
                                }
                            }
                    }
                    Text("Bound to 127.0.0.1 (localhost only) — change in code for LAN exposure.")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.faint)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if controller.isRunning {
                    Button { controller.stop() } label: {
                        Text("Stop server").frame(maxWidth: .infinity)
                    }
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.danger)
                } else {
                    Button { controller.start() } label: {
                        Text("Start server").frame(maxWidth: .infinity)
                    }
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                    .disabled(controller.modelPath.isEmpty)
                }

                if controller.isRunning {
                    section("Endpoint") {
                        HStack(spacing: 8) {
                            Text(controller.endpoint)
                                .font(.system(size: 12, weight: .medium, design: .monospaced))
                                .foregroundStyle(Theme.accent)
                                .textSelection(.enabled)
                            Spacer()
                            Button {
                                let pb = NSPasteboard.general
                                pb.clearContents()
                                pb.setString(controller.endpoint, forType: .string)
                            } label: { Image(systemName: "doc.on.doc") }
                            .buttonStyle(.plain)
                            .foregroundStyle(Theme.muted)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("POST /v1/chat/completions")
                            Text("POST /v1/completions")
                            Text("GET  /v1/models")
                        }
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.faint)

                        Text("Requests served: \(controller.requestCount)")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                            .padding(.top, 4)
                    }
                }

                if let err = controller.lastError {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(Theme.warn)
                        Text(err)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.fg)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(10)
                    .background(Theme.warn.opacity(0.10))
                    .overlay(Rectangle().fill(Theme.warn).frame(width: 2), alignment: .leading)
                }
            }
            .padding(20)
        }
    }

    private var logArea: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("LOG")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.faint)
                    .tracking(1)
                Spacer()
                if !controller.log.isEmpty {
                    Button("Clear") { controller.log = "" }
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                }
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 10)
            Divider().background(Theme.line)
            ScrollView {
                ScrollViewReader { proxy in
                    Text(controller.log.isEmpty
                         ? "Start the server to stream log lines here.\n\nThe endpoint speaks OpenAI Chat Completions. Point Cursor / Cline / Continue / a Python script at it and it sees TinyGPT as a GPT-style backend. No network calls leave your machine."
                         : controller.log)
                        .font(.tgMono)
                        .foregroundStyle(controller.log.isEmpty ? Theme.faint : Theme.fg)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(18)
                        .id("server-log-end")
                        .onChange(of: controller.log) { _, _ in
                            withAnimation(.linear(duration: 0.05)) {
                                proxy.scrollTo("server-log-end", anchor: .bottom)
                            }
                        }
                }
            }
        }
    }

    @ViewBuilder
    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            content()
        }
    }

    private func pickModel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data, .directory]
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Pick a .tinygpt file OR an HF model directory."
        if panel.runModal() == .OK, let url = panel.url {
            controller.modelPath = url.path
        }
    }
}
