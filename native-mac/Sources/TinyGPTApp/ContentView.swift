import SwiftUI

/// Seven workspaces. Train internally has sub-modes
/// (pretrain / fine-tune / DPO / distill). Modalities surfaces
/// what we ship today + the queued multi-modal arcs.
enum AppTab: Hashable {
    case sample      // chat + A/B compare for currently loaded model
    case gallery     // browse loadable models — click → opens chat
    case train       // pretrain / fine-tune / DPO / distill (sub-modes)
    case eval        // score + compare
    case trace       // inference heatmap
    case interp      // mech-interp power tools
    case serve       // HTTP endpoint
    case modalities  // text / code / tool-calls (shipped) + vision/voice/image (queued)
    case learn       // markdown viewer
}

struct ContentView: View {
    @StateObject private var controller = ModelController()
    @StateObject private var controllerB = ModelController()  // A/B compare slot
    @State private var compareMode: Bool = false
    @StateObject private var stats = MachineStats()
    @StateObject private var hfBrowser = HFBrowserController()
    @State private var galleryItems: [GalleryItem] = []
    @State private var selectedItem: GalleryItem? = nil
    @State private var showHFBrowser: Bool = false
    @AppStorage("tinygpt.gallery.expanded") private var galleryExpanded: Bool = false

    // Sampler params — persisted across launches so a tuned recipe sticks.
    @AppStorage("tg.prompt")        private var prompt: String = "ROMEO:"
    @AppStorage("tg.maxTokens")     private var maxTokens: Int = 200
    @AppStorage("tg.temperature")   private var temperature: Double = 0.8
    @AppStorage("tg.topK")          private var topK: Int = 0
    @AppStorage("tg.repPenalty")    private var repPenalty: Double = 1.0
    @AppStorage("tg.showInspector") private var showInspector: Bool = true

    @State private var tab: AppTab = .sample
    @State private var liveServes: [ServeProcess] = []
    // Sidebar nav default lands on Sample — most common "use a model" entry.

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 220)
                    .background(Theme.panel)

                Divider().background(Theme.line)

                VStack(spacing: 0) {
                    Group {
                        switch tab {
                        case .sample:     mainPane
                        case .gallery:    galleryPane
                        case .train:      TrainHubView()
                        case .eval:       EvalView()
                        case .trace:      InferenceHeatmapView()
                        case .interp:     InterpView()
                        case .serve:      ServerView()
                        case .modalities: ModalitiesView()
                        case .learn:      LearnView()
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
            liveServes = ServeRegistry.discover()
        }
        .onReceive(Timer.publish(every: 5, on: .main, in: .common).autoconnect()) { _ in
            liveServes = ServeRegistry.discover()
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

    // Old top tabBar + grouped sections — replaced 2026-06-07 PM
    // after user audit consolidated to 6 flat workspaces.

    /// Sidebar workspace nav row — left-aligned icon + label + active highlight.
    /// Replaces the old top tabBar (2026-06-07). Hit area = full row width
    /// × ~32pt height (macOS HIG-compliant for compact controls).
    private func navRow(_ which: AppTab, icon: String, label: String) -> some View {
        let active = tab == which
        return Button {
            tab = which
        } label: {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(active ? Theme.accent : Theme.muted)
                    .frame(width: 18)
                Text(label)
                    .font(.system(size: 13, weight: active ? .semibold : .regular))
                    .foregroundStyle(active ? Theme.fg : Theme.muted)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
            .contentShape(Rectangle())
            .background(active ? Theme.accent.opacity(0.10) : Color.clear)
            .overlay(
                Rectangle()
                    .fill(active ? Theme.accent : Color.clear)
                    .frame(width: 2),
                alignment: .leading
            )
        }
        .buttonStyle(.plain)
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
            // Header — app brand + global actions (HF browser + open file).
            // Actions moved here when the bottom-sidebar Gallery section
            // was retired in favor of the Gallery workspace tab.
            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("TinyGPT")
                        .font(.tgDisplay)
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                        .minimumScaleFactor(0.7)
                    Text("native macOS")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .lineLimit(1)
                }
                Spacer(minLength: 4)
                Button {
                    showHFBrowser = true
                } label: {
                    Image(systemName: "cloud.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(Theme.muted)
                        .frame(width: 28, height: 28)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Browse + download HuggingFace models.")
                Button {
                    openModelFile()
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(Theme.muted)
                        .frame(width: 28, height: 28)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Open a .tinygpt file from anywhere on disk.")
            }
            .padding(.horizontal, 20)
            .padding(.top, 24)
            .padding(.bottom, 16)

            // Workspace navigation — six consolidated workspaces
            // (2026-06-07 PM, after user audit found 12 tabs overengineered).
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 1) {
                    navRow(.sample,     icon: "text.bubble",                       label: "Sample")
                    navRow(.gallery,    icon: "rectangle.grid.2x2",                label: "Gallery")
                    navRow(.train,      icon: "waveform.path.ecg",                 label: "Train")
                    navRow(.eval,       icon: "checkmark.gobackward",              label: "Eval")
                    navRow(.trace,      icon: "chart.bar.xaxis",                   label: "Trace")
                    navRow(.interp,     icon: "scope",                             label: "Interp")
                    navRow(.serve,      icon: "antenna.radiowaves.left.and.right", label: "Serve")
                    navRow(.modalities, icon: "square.stack.3d.up",                label: "Modalities")
                    navRow(.learn,      icon: "graduationcap",                     label: "Learn")

                    // Inference section — live `tinygpt serve` processes
                    // detected via pgrep. Click a row to jump to Serve tab.
                    if !liveServes.isEmpty {
                        Text("INFERENCE")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Theme.faint)
                            .tracking(1)
                            .padding(.horizontal, 20)
                            .padding(.top, 16)
                            .padding(.bottom, 4)
                        ForEach(liveServes) { p in
                            Button {
                                tab = .serve
                            } label: {
                                HStack(spacing: 8) {
                                    Circle().fill(Color.green).frame(width: 6, height: 6)
                                    VStack(alignment: .leading, spacing: 1) {
                                        Text((p.modelPath as NSString).lastPathComponent)
                                            .font(.system(size: 11, design: .monospaced))
                                            .foregroundStyle(Theme.fg)
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                        Text(":\(p.port) · pid \(p.pid)")
                                            .font(.system(size: 9, design: .monospaced))
                                            .foregroundStyle(Theme.faint)
                                    }
                                    Spacer(minLength: 0)
                                }
                                .padding(.horizontal, 20)
                                .padding(.vertical, 6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(.bottom, 12)
            }

            // The bottom Gallery section was removed (2026-06-07) — Gallery
            // is now a dedicated workspace in the sidebar nav, so a
            // duplicate quick-list at the bottom was redundant. Global
            // actions (HF browser + open arbitrary file) live in the
            // brand header above.

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

    /// Gallery workspace — a grid of model cards. Each card has 3
    /// actions: Chat (Sample), Eval, Interp. Replaces the old
    /// collapsible-in-sidebar approach which buried the models.
    private var galleryPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Gallery")
                        .font(.tgDisplay)
                        .foregroundStyle(Theme.fg)
                    Text("models loadable from data/gallery/ + ~/.cache/tinygpt/runs/ · click an action below each model")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                }
                if galleryItems.isEmpty {
                    Text("no models found. drop .tinygpt files into data/gallery/ or train via Train tab.")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                } else {
                    LazyVGrid(columns: [
                        GridItem(.flexible(), spacing: 16),
                        GridItem(.flexible(), spacing: 16),
                    ], spacing: 16) {
                        ForEach(galleryItems) { item in
                            galleryCardWithActions(item)
                        }
                    }
                }
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Theme.base)
    }

    /// Gallery card with model info + 3 action buttons (Chat / Eval / Interp).
    private func galleryCardWithActions(_ item: GalleryItem) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Text(item.icon).font(.system(size: 28))
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.displayName)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    Text(item.url.deletingLastPathComponent().lastPathComponent + "/")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
            }
            Divider().background(Theme.line)
            HStack(spacing: 8) {
                galleryActionButton(label: "Chat", icon: "text.bubble") {
                    selectedItem = item
                    prompt = item.prompt
                    Task { await controller.load(item) }
                    tab = .sample
                }
                galleryActionButton(label: "Eval", icon: "checkmark.gobackward") {
                    selectedItem = item
                    Task { await controller.load(item) }
                    tab = .eval
                }
                galleryActionButton(label: "Interp", icon: "scope") {
                    selectedItem = item
                    Task { await controller.load(item) }
                    tab = .interp
                }
            }
        }
        .padding(16)
        .background(Theme.panel)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.line))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func galleryActionButton(label: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 10))
                Text(label)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
            }
            .foregroundStyle(Theme.accent)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Theme.accent.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 5))
        }
        .buttonStyle(.plain)
    }

    /// Inline grid-card for the Sample placeholder. Click loads the model
    /// + jumps to the generation pane.
    private func modelCard(_ item: GalleryItem) -> some View {
        Button {
            selectedItem = item
            prompt = item.prompt
            Task { await controller.load(item) }
        } label: {
            HStack(spacing: 12) {
                Text(item.icon)
                    .font(.system(size: 24))
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.displayName)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    Text(item.url.deletingLastPathComponent().lastPathComponent + "/")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, minHeight: 56, alignment: .leading)
            .background(Theme.panel)
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.line))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    /// Collapsible gallery row — single line when collapsed; click to expand.
    private var gallerySidebarSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                galleryExpanded.toggle()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: galleryExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Theme.faint)
                        .frame(width: 12)
                    Text("Gallery")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.muted)
                    Spacer(minLength: 0)
                    Text("\(galleryItems.count)")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Theme.panel2)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if galleryExpanded {
                if galleryItems.isEmpty {
                    Text("no models in data/gallery/")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                        .padding(.horizontal, 32)
                        .padding(.bottom, 6)
                } else {
                    ForEach(galleryItems) { item in
                        galleryRow(item)
                            .padding(.leading, 12)
                    }
                }
            }
        }
        .padding(.top, 8)
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
            .frame(minHeight: 32)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Theme.accentGlow : Color.clear)
            )
            .contentShape(Rectangle())
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
            VStack(spacing: 12) {
                Text("Chat with a model")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("pick from the gallery — base models, your trained specialists, or HF downloads")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Theme.muted)
            }
            .padding(.top, 40)

            // Model picker — inline grid replaces the prior sidebar Gallery.
            if galleryItems.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "tray")
                        .font(.system(size: 28))
                        .foregroundStyle(Theme.faint)
                    Text("No models found")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.muted)
                    Text("Drop .tinygpt files into data/gallery/ or click the + above")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                }
                .frame(maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 220, maximum: 320), spacing: 12)],
                              alignment: .leading, spacing: 12) {
                        ForEach(galleryItems) { item in
                            modelCard(item)
                        }
                    }
                    .padding(.horizontal, 40)
                }
            }
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
    /// One output column for a given ModelController. Used by both
    /// single-model view (A only) and compare mode (A and B side-by-side).
    private func outputColumn(controller: ModelController, label: String) -> some View {
        ScrollView {
            ScrollViewReader { proxy in
                VStack(alignment: .leading, spacing: 16) {
                    if compareMode {
                        HStack(spacing: 6) {
                            Text(label)
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .foregroundStyle(Theme.accent)
                            Text(controller.loadedItem?.displayName ?? "—")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.muted)
                            if controller.tokensPerSec > 0 {
                                Spacer()
                                Text(String(format: "%.0f tok/s", controller.tokensPerSec))
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(Theme.faint)
                            }
                        }
                        .padding(.bottom, 4)
                    }

                    ForEach(controller.historyForCurrentModel) { item in
                        historyCard(item)
                    }

                    Text(controller.generated.isEmpty && controller.historyForCurrentModel.isEmpty
                         ? "Output will appear here as the model generates token-by-token."
                         : controller.generated)
                        .font(.tgMono)
                        .foregroundStyle((controller.generated.isEmpty && controller.historyForCurrentModel.isEmpty) ? Theme.faint : Theme.fg)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .padding(24)
                .id("output-end-\(label)")
                .onChange(of: controller.generated) { _, _ in
                    withAnimation(.linear(duration: 0.1)) {
                        proxy.scrollTo("output-end-\(label)", anchor: .bottom)
                    }
                }
                .onChange(of: controller.historyForCurrentModel.count) { _, _ in
                    withAnimation(.linear(duration: 0.1)) {
                        proxy.scrollTo("output-end-\(label)", anchor: .bottom)
                    }
                }
            }
        }
    }

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
                    compareMode.toggle()
                    if !compareMode {
                        controllerB.cancelGeneration()
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: compareMode ? "rectangle.split.2x1.fill" : "rectangle.split.2x1")
                        Text(compareMode ? "compare" : "+ compare")
                            .font(.system(size: 11, design: .monospaced))
                    }
                    .foregroundStyle(compareMode ? Theme.accent : Theme.muted)
                }
                .buttonStyle(.plain)
                .help("Run a second model side-by-side on the same prompt.")
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

            // Compare mode: second model picker (model B). Renders as a
            // pill strip above the output panes — visible only in compare
            // mode so the single-model view stays minimal.
            if compareMode {
                HStack(spacing: 12) {
                    Text("B:")
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                    if let b = controllerB.loadedItem {
                        Text(b.icon + " " + b.displayName)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(Theme.fg)
                    } else {
                        Text("pick a second model")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.faint)
                    }
                    Spacer()
                    Menu("change") {
                        ForEach(galleryItems) { item in
                            Button(item.displayName) {
                                Task { await controllerB.load(item) }
                            }
                        }
                    }
                    .menuStyle(.borderlessButton)
                    .frame(width: 90)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
                .background(Theme.panel.opacity(0.5))
                .overlay(Rectangle().fill(Theme.line).frame(height: 1), alignment: .bottom)
            }

            // Output + inspector. Compare mode splits the output area
            // into two columns (A | B) that both run on the same prompt.
            HStack(spacing: 0) {
                outputColumn(controller: controller, label: "A")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                if compareMode {
                    Divider().background(Theme.line)
                    outputColumn(controller: controllerB, label: "B")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }

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

                if controller.isGenerating || (compareMode && controllerB.isGenerating) {
                    Button("Stop") {
                        controller.cancelGeneration()
                        if compareMode { controllerB.cancelGeneration() }
                    }
                    .keyboardShortcut(.cancelAction)
                    .buttonStyle(PrimaryButtonStyle(color: Theme.danger))
                } else {
                    Button("Generate") {
                        controller.generate(prompt: prompt, maxTokens: maxTokens,
                                          temperature: Float(temperature),
                                          topK: topK,
                                          repetitionPenalty: Float(repPenalty))
                        if compareMode && controllerB.loadedItem != nil {
                            controllerB.generate(prompt: prompt, maxTokens: maxTokens,
                                              temperature: Float(temperature),
                                              topK: topK,
                                              repetitionPenalty: Float(repPenalty))
                        }
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

                if !controller.historyForCurrentModel.isEmpty {
                    Button {
                        controller.clearHistory()
                    } label: {
                        Image(systemName: "trash")
                            .foregroundStyle(Theme.faint)
                    }
                    .buttonStyle(.plain)
                    .help("Clear completion history for the current model only.")
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
