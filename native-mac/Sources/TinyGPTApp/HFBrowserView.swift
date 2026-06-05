import SwiftUI

/// Sheet UI for the HF model browser. Wraps `HFBrowserController` —
/// downloads models from huggingface.co/<owner>/<repo> into the app's
/// Application Support cache and lists what's already local.
///
/// v1 takes an explicit `owner/repo` text input — search/browse with
/// filters is a v2 follow-up. The download flow handles auth via
/// `HF_TOKEN` env var (gated and private models surface a clean error).
struct HFBrowserView: View {
    @ObservedObject var controller: HFBrowserController
    @Binding var isPresented: Bool
    @State private var repoInput: String = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().background(Theme.line)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    promptSection
                    if controller.isDownloading { progressSection }
                    if let err = controller.lastError { errorSection(err) }
                    if !controller.downloadedModels.isEmpty { localSection }
                }
                .padding(20)
            }
        }
        .frame(width: 520, height: 460)
        .background(Theme.base)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("HuggingFace models")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("download to ~/Library/Application Support/TinyGPT/hf/")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Theme.faint)
            }
            Spacer()
            Button { isPresented = false } label: {
                Image(systemName: "xmark")
                    .foregroundStyle(Theme.muted)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("DOWNLOAD")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            HStack(spacing: 10) {
                TextField("HuggingFaceTB/SmolLM2-135M", text: $repoInput)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(Theme.panel)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .onSubmit { startDownload() }
                Button("Download") { startDownload() }
                    .controlSize(.regular)
                    .buttonStyle(.borderedProminent)
                    .tint(Theme.accent)
                    .disabled(controller.isDownloading || repoInput.isEmpty)
            }
            Text("`owner/repo` format. Public models download immediately; gated/private models need `HF_TOKEN` in your environment.")
                .font(.system(size: 10))
                .foregroundStyle(Theme.faint)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(controller.status)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(2)
                    .truncationMode(.middle)
                Spacer()
                Button { controller.cancel() } label: {
                    Text("Cancel").font(.system(size: 11))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            ProgressView(value: controller.progress)
                .progressViewStyle(.linear)
                .tint(Theme.accent)
        }
        .padding(14)
        .background(Theme.panel)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.line))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func errorSection(_ err: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.warn)
            Text(err)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.fg)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
        }
        .padding(12)
        .background(Theme.warn.opacity(0.10))
        .overlay(Rectangle().fill(Theme.warn).frame(width: 2), alignment: .leading)
    }

    private var localSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("DOWNLOADED")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            VStack(spacing: 6) {
                ForEach(controller.downloadedModels) { model in
                    downloadedRow(model)
                }
            }
        }
    }

    private func downloadedRow(_ model: HFBrowserController.DownloadedModel) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "shippingbox")
                .foregroundStyle(Theme.muted)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fg)
                Text(HFBrowserController.formatBytes(model.sizeBytes))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Theme.faint)
            }
            Spacer()
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([model.url])
            } label: { Image(systemName: "folder").font(.system(size: 12)) }
            .buttonStyle(.plain)
            .help("Reveal in Finder")

            Button {
                let pb = NSPasteboard.general
                pb.clearContents()
                pb.setString("./native-mac/.build/release/tinygpt sample \(model.url.path) --prompt \"Hello\" --tokens 100", forType: .string)
            } label: { Image(systemName: "terminal").font(.system(size: 12)) }
            .buttonStyle(.plain)
            .help("Copy CLI sample command to clipboard")

            Button { controller.delete(model) } label: {
                Image(systemName: "trash").font(.system(size: 12))
                    .foregroundStyle(Theme.faint)
            }
            .buttonStyle(.plain)
            .help("Delete from disk")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Theme.panel)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
    }

    private func startDownload() {
        let repo = repoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repo.isEmpty else { return }
        controller.download(repo: repo)
    }
}
