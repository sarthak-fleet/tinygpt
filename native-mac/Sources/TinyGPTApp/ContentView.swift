import SwiftUI

enum AppTab: Hashable { case sample, train, finetune, interp, server }

struct ContentView: View {
    @StateObject private var controller = ModelController()
    @StateObject private var stats = MachineStats()
    @StateObject private var hfBrowser = HFBrowserController()
    @State private var galleryItems: [GalleryItem] = []
    @State private var selectedItem: GalleryItem? = nil
    @State private var showHFBrowser: Bool = false

    // Sampler params — persisted across launches so a tuned recipe sticks.
    @AppStorage("tg.prompt")        private var prompt: String = "ROMEO:"
    @AppStorage("tg.maxTokens")     private var maxTokens: Int = 200
    @AppStorage("tg.temperature")   private var temperature: Double = 0.8
    @AppStorage("tg.topK")          private var topK: Int = 0
    @AppStorage("tg.repPenalty")    private var repPenalty: Double = 1.0
    @AppStorage("tg.showInspector") private var showInspector: Bool = true

    @State private var tab: AppTab = .sample

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 220)
                    .background(Theme.panel)

                Divider().background(Theme.line)

                VStack(spacing: 0) {
                    tabBar
                    Divider().background(Theme.line)
                    Group {
                        switch tab {
                        case .sample:   mainPane
                        case .train:    TrainView()
                        case .finetune: FinetuneView()
                        case .interp:   InterpView()
                        case .server:   ServerView()
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .background(Theme.base)
            }

            // Machine-stats strip — sticky bottom, mono+compact
            Divider().background(Theme.line)
            machineStatsBar
        }
        .onAppear {
            galleryItems = GalleryDiscovery.discover()
        }
        .sheet(isPresented: $showHFBrowser) {
            HFBrowserView(controller: hfBrowser, isPresented: $showHFBrowser)
        }
    }

    private var machineStatsBar: some View {
        HStack(spacing: 16) {
            statsBlock("CHIP", stats.cpuModel.replacingOccurrences(of: "Apple ", with: ""))
            statsBlock("CORES", "\(stats.cpuCores)")
            statsBlock("GPU", stats.gpuName.isEmpty ? "—" : stats.gpuName)
            Divider().frame(height: 18).background(Theme.line)
            statsBlock("APP RAM", FormatBytes.compact(stats.processRSSBytes))
            statsBlock("FREE RAM", FormatBytes.compact(stats.freeRAMBytes))
            statsBlock("TOTAL", FormatBytes.compact(stats.totalRAMBytes))
            Spacer()
            statsBlock("GPU MAX SET", "\(stats.gpuRegistryMB) MB")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Theme.panel2)
    }

    private func statsBlock(_ label: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Theme.fg)
        }
    }

    private var tabBar: some View {
        HStack(spacing: 0) {
            tabButton(.sample, label: "Sample")
            tabButton(.train, label: "Train")
            tabButton(.finetune, label: "Fine-tune")
            tabButton(.interp, label: "Interp")
            tabButton(.server, label: "Server")
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.top, 12)
        .padding(.bottom, 4)
        .background(Theme.panel)
    }

    private func tabButton(_ which: AppTab, label: String) -> some View {
        let active = tab == which
        return Button {
            tab = which
        } label: {
            VStack(spacing: 6) {
                Text(label)
                    .font(.system(size: 13, weight: active ? .semibold : .regular))
                    .foregroundStyle(active ? Theme.accent : Theme.muted)
                Rectangle()
                    .fill(active ? Theme.accent : Color.clear)
                    .frame(height: 2)
            }
            .padding(.horizontal, 12)
        }
        .buttonStyle(.plain)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header — app brand
            VStack(alignment: .leading, spacing: 4) {
                Text("TinyGPT")
                    .font(.tgDisplay)
                    .foregroundStyle(Theme.fg)
                Text("native macOS")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
            }
            .padding(.horizontal, 20)
            .padding(.top, 24)
            .padding(.bottom, 24)

            // Gallery list + open arbitrary file
            HStack {
                Text("GALLERY")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.faint)
                    .tracking(1)
                Spacer()
                Button {
                    showHFBrowser = true
                } label: {
                    Image(systemName: "cloud.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.muted)
                }
                .buttonStyle(.plain)
                .help("Browse + download HuggingFace models.")
                Button {
                    openModelFile()
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.muted)
                }
                .buttonStyle(.plain)
                .help("Open a .tinygpt file from anywhere on disk.")
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 8)

            ScrollView {
                LazyVStack(spacing: 4) {
                    if galleryItems.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("No models found")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(Theme.muted)
                            Text("Drop .tinygpt checkpoints into one of:")
                                .font(.system(size: 10))
                                .foregroundStyle(Theme.faint)
                            Text("data/gallery/")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Theme.faint)
                            Text("browser/public/gallery/")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Theme.faint)
                            Text("~/Library/Application Support/TinyGPT/gallery/")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Theme.faint)
                            Button("Reload gallery") {
                                galleryItems = GalleryDiscovery.discover()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .padding(.top, 4)
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                    } else {
                        ForEach(galleryItems) { item in
                            galleryRow(item)
                        }
                    }
                }
                .padding(.horizontal, 12)
            }

            Spacer()

            // Status bar
            VStack(alignment: .leading, spacing: 6) {
                Divider().background(Theme.line)
                Text(controller.status)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.muted)
                    .lineLimit(2)
                    .truncationMode(.tail)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
            }
        }
    }

    private func galleryRow(_ item: GalleryItem) -> some View {
        let isSelected = controller.loadedItem?.id == item.id
        return Button {
            selectedItem = item
            prompt = item.prompt
            Task { await controller.load(item) }
        } label: {
            HStack(spacing: 10) {
                Text(item.icon)
                    .font(.system(size: 18))
                Text(item.displayName)
                    .font(.system(size: 13, weight: isSelected ? .medium : .regular))
                    .foregroundStyle(isSelected ? Theme.accent : Theme.fg)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Theme.accentGlow : Color.clear)
            )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var mainPane: some View {
        if controller.loadedItem == nil {
            placeholderPane
        } else {
            generationPane
        }
    }

    private var placeholderPane: some View {
        VStack(spacing: 24) {
            VStack(spacing: 14) {
                Text("TinyGPT")
                    .font(.system(size: 36, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("native macOS ML lab")
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .tracking(2)
            }
            .padding(.top, 60)

            VStack(alignment: .leading, spacing: 16) {
                welcomeRow(
                    icon: "play.fill",
                    title: "Sample",
                    description: "Pick a model from the gallery (left sidebar) and prompt it."
                )
                welcomeRow(
                    icon: "waveform.path.ecg",
                    title: "Train",
                    description: "Watch a transformer learn from scratch with live loss + step rate."
                )
                welcomeRow(
                    icon: "slider.horizontal.3",
                    title: "Fine-tune",
                    description: "LoRA SFT or DPO against any base model, with composable adapters."
                )
            }
            .padding(.horizontal, 40)
            .frame(maxWidth: 540)

            Spacer()

            VStack(spacing: 4) {
                Text("Each gallery model is a 9.6M-parameter byte-level transformer")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.faint)
                Text("trained on a different corpus. Same architecture, different mind.")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.faint)
            }
            .padding(.bottom, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// One past completion as a card. Prompt is highlighted in the
    /// accent colour to set it off from the model's output, with a
    /// monospaced footer line for the sampler recipe so re-running the
    /// same prompt at the same settings stays easy.
    private func historyCard(_ item: ModelController.HistoryItem) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(item.prompt)
                    .font(.tgMono)
                    .foregroundStyle(Theme.accent)
                Spacer()
                Text(item.timestamp, format: .dateTime.hour().minute().second())
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Theme.faint)
            }
            Text(item.output)
                .font(.tgMono)
                .foregroundStyle(Theme.fg)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
            HStack(spacing: 10) {
                Text("T=\(String(format: "%.2f", item.temperature))")
                if item.topK > 0 { Text("top-k=\(item.topK)") }
                if item.repetitionPenalty > 1.001 { Text("rp=\(String(format: "%.2f", item.repetitionPenalty))") }
                Text("\(item.tokensGenerated) tok")
                Text(String(format: "%.0f tok/s", item.tokensPerSec))
                Spacer()
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(item.output, forType: .string)
                } label: { Text("Copy") }
                .buttonStyle(.borderless)
                .controlSize(.mini)
            }
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(Theme.faint)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Theme.panel)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Theme.line, lineWidth: 1)
        )
    }

    private func welcomeRow(icon: String, title: String, description: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 16))
                .foregroundStyle(Theme.accent)
                .frame(width: 24, height: 24, alignment: .center)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text(description)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }

    private var generationPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Model header — also hosts the inspector toggle.
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(controller.loadedItem?.icon ?? "•")
                    .font(.system(size: 24))
                Text(controller.loadedItem?.displayName ?? "")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("\(formattedInt(controller.paramCount)) params")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                Spacer()
                if controller.isGenerating || controller.tokensPerSec > 0 {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(controller.isGenerating ? Theme.accent : Theme.muted)
                            .frame(width: 6, height: 6)
                        Text(String(format: "%.0f tok/s", controller.tokensPerSec))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                    }
                }
                Button {
                    showInspector.toggle()
                } label: {
                    Image(systemName: showInspector ? "sidebar.right" : "sidebar.squares.right")
                        .foregroundStyle(showInspector ? Theme.accent : Theme.muted)
                }
                .buttonStyle(.plain)
                .help(showInspector ? "Hide sampler inspector" : "Show sampler inspector")
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            Divider().background(Theme.line)

            // Output + inspector. Output reads top-down; inspector is a
            // fixed-width column with all sampler controls so the prompt
            // box at the bottom can stay focused on prompt-not-knobs.
            HStack(spacing: 0) {
                ScrollView {
                    ScrollViewReader { proxy in
                        VStack(alignment: .leading, spacing: 16) {
                            // Past completions (this session). Each is a
                            // card with the prompt + output + the sampler
                            // recipe that produced it.
                            ForEach(controller.history) { item in
                                historyCard(item)
                            }

                            // Live in-flight buffer.
                            Text(controller.generated.isEmpty && controller.history.isEmpty
                                 ? "Output will appear here as the model generates token-by-token."
                                 : controller.generated)
                                .font(.tgMono)
                                .foregroundStyle((controller.generated.isEmpty && controller.history.isEmpty) ? Theme.faint : Theme.fg)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        }
                        .padding(24)
                        .id("output-end")
                        .onChange(of: controller.generated) { _, _ in
                            withAnimation(.linear(duration: 0.1)) {
                                proxy.scrollTo("output-end", anchor: .bottom)
                            }
                        }
                        .onChange(of: controller.history.count) { _, _ in
                            withAnimation(.linear(duration: 0.1)) {
                                proxy.scrollTo("output-end", anchor: .bottom)
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                if showInspector {
                    Divider().background(Theme.line)
                    samplerInspector
                        .frame(width: 240)
                        .frame(maxHeight: .infinity)
                        .background(Theme.panel)
                }
            }

            Divider().background(Theme.line)

            // Controls — prompt + generate. Sampler knobs (temp/topK/penalty)
            // live in the inspector panel above so this row stays focused
            // on the actual prompt + action, like Cursor's chat box.
            HStack(spacing: 16) {
                TextField("Prompt", text: $prompt, axis: .horizontal)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Theme.panel)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .font(.tgMono)

                if controller.isGenerating {
                    Button("Stop") {
                        controller.cancelGeneration()
                    }
                    .keyboardShortcut(.cancelAction)
                    .buttonStyle(PrimaryButtonStyle(color: Theme.danger))
                } else {
                    Button("Generate") {
                        controller.generate(prompt: prompt, maxTokens: maxTokens,
                                          temperature: Float(temperature),
                                          topK: topK,
                                          repetitionPenalty: Float(repPenalty))
                    }
                    .keyboardShortcut(.return, modifiers: [.command])
                    .buttonStyle(PrimaryButtonStyle(color: Theme.accent))
                    .disabled(controller.loadedItem == nil)
                }

                Button(controller.isEvaluating ? "Scoring…" : "Score") {
                    runEval()
                }
                .buttonStyle(.bordered)
                .disabled(controller.loadedItem == nil || controller.isEvaluating)
                .help("Pick a text file; the model's cross-entropy loss + BPB + perplexity print to the status line.")

                if !controller.history.isEmpty {
                    Button {
                        controller.clearHistory()
                    } label: {
                        Image(systemName: "trash")
                            .foregroundStyle(Theme.faint)
                    }
                    .buttonStyle(.plain)
                    .help("Clear completion history for this session.")
                }
            }
            .padding(20)
            .background(Theme.panel)

            if let result = controller.evalResult {
                HStack {
                    Text("EVAL")
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.accent)
                        .tracking(1)
                    Text(result)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                    Spacer()
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
                .background(Theme.accentGlow)
            }
        }
    }

    // MARK: - Sampler inspector

    /// Right-hand inspector panel. Mirrors what LM Studio/Ollama users
    /// expect from a "decent" local-AI app: temperature, top-K,
    /// repetition penalty, max tokens. Persisted via @AppStorage so the
    /// tuned recipe survives a restart.
    private var samplerInspector: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("SAMPLING")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.faint)
                    .tracking(1)
                    .padding(.top, 18)

                inspectorRow(
                    label: "Temperature",
                    hint: "0 = greedy · 1 = sample raw · >1 = more random",
                    value: temperature,
                    range: 0...2,
                    format: "%.2f"
                ) { temperature = $0 }

                inspectorRow(
                    label: "Top-K",
                    hint: topK == 0 ? "0 = off — sample over full vocab" :
                                       "keep only the \(topK) highest-prob tokens",
                    value: Double(topK),
                    range: 0...256,
                    format: "%.0f"
                ) { topK = Int($0) }

                inspectorRow(
                    label: "Rep. penalty",
                    hint: repPenalty <= 1.001 ? "1.0 = off — Keskar et al. 2019" :
                                                 "divides logits of recent tokens",
                    value: repPenalty,
                    range: 1.0...2.0,
                    format: "%.2f"
                ) { repPenalty = $0 }

                Divider().background(Theme.line).padding(.vertical, 4)

                Text("LENGTH")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.faint)
                    .tracking(1)

                HStack {
                    Text("max tokens")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                    Spacer()
                    TextField("", value: $maxTokens, format: .number)
                        .textFieldStyle(.plain)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(Theme.panel2)
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.line))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                        .font(.system(size: 12, design: .monospaced))
                }

                Spacer(minLength: 16)

                Button {
                    temperature = 0.8
                    topK = 0
                    repPenalty = 1.0
                    maxTokens = 200
                } label: {
                    HStack {
                        Image(systemName: "arrow.counterclockwise")
                        Text("Reset to defaults")
                    }
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.muted)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 18)
            .padding(.bottom, 20)
        }
    }

    /// One labeled slider row with current value + hint text. Generic over
    /// the slider type so int / float fields share one layout.
    private func inspectorRow(label: String, hint: String, value: Double,
                              range: ClosedRange<Double>, format: String,
                              setter: @escaping (Double) -> Void) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.fg)
                Spacer()
                Text(String(format: format, value))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
            }
            Slider(
                value: Binding(get: { value }, set: setter),
                in: range
            )
            .controlSize(.small)
            .tint(Theme.accent)
            Text(hint)
                .font(.system(size: 10))
                .foregroundStyle(Theme.faint)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func runEval() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.plainText, .utf8PlainText, .text]
        panel.allowsMultipleSelection = false
        panel.message = "Pick a UTF-8 text file to score the model on."
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let data = try Data(contentsOf: url)
                controller.evaluate(corpus: data)
            } catch {
                controller.evalResult = "couldn't read \(url.lastPathComponent): \(error)"
            }
        }
    }

    /// File-picker entry to the sidebar "+" button. Any .tinygpt file
    /// becomes a one-off GalleryItem with the filename as display name.
    /// The item isn't added to the persistent gallery list — close +
    /// reopen the app to re-pick — but it loads + samples identically
    /// to a gallery entry.
    private func openModelFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        // Allow any extension type — .tinygpt is custom, .bin is the
        // browser-shipping format, and macOS would otherwise hide both.
        panel.allowedContentTypes = [.data]
        panel.message = "Pick a .tinygpt or .bin model checkpoint."
        if panel.runModal() == .OK, let url = panel.url {
            let stem = url.deletingPathExtension().lastPathComponent
            let item = GalleryItem(
                id: "user-\(stem)-\(UUID().uuidString.prefix(6))",
                displayName: stem.replacingOccurrences(of: "-", with: " ").capitalized,
                icon: "📦",
                url: url,
                prompt: "Hello"
            )
            // Append to the sidebar list so it's selectable for this session.
            if !galleryItems.contains(where: { $0.url == item.url }) {
                galleryItems.append(item)
            }
            Task { await controller.load(item) }
        }
    }

    private func formattedInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}

private struct PrimaryButtonStyle: ButtonStyle {
    let color: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
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
