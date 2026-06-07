import SwiftUI

/// Modalities workspace — surfaces the platform's full modality scope.
/// Shipped today: text, code, tool-calls. Queued: vision (VLM), voice
/// (Whisper/TTS), image gen (SDXL Turbo / Flux), embeddings, music.
/// Lets users see WHAT TinyGPT is for + what's coming, not just what's
/// in the binary right now.
struct ModalitiesView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Modalities")
                        .font(.tgDisplay)
                        .foregroundStyle(Theme.fg)
                    Text("what TinyGPT specialists handle — shipped vs queued")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                }

                modalitySection(title: "SHIPPED", items: [
                    .init(icon: "text.alignleft",  name: "Text",
                          desc: "next-token prediction; SmolLM2 BPE or byte tokenizer.",
                          status: .shipped),
                    .init(icon: "chevron.left.forwardslash.chevron.right", name: "Code",
                          desc: "same architecture, code corpora (the-stack-smol, repo scrapes).",
                          status: .shipped),
                    .init(icon: "wrench.adjustable", name: "Tool calls",
                          desc: "function-call structured output; hermes-fc / xlam datasets.",
                          status: .shipped),
                    .init(icon: "scope", name: "Mechanistic interp",
                          desc: "SAE, MEMIT, activation patching — read the model's internals.",
                          status: .shipped),
                ])

                modalitySection(title: "QUEUED (TIER 2)", items: [
                    .init(icon: "eye", name: "Vision (VLM)",
                          desc: "LLaVA-style: CLIP encoder → projection → LLM body. ~2-3 weeks.",
                          status: .queued),
                    .init(icon: "waveform", name: "Speech-in (Whisper)",
                          desc: "Apple's CoreML Whisper → text token stream → existing LLM.",
                          status: .queued),
                    .init(icon: "speaker.wave.2", name: "TTS / voice cloning",
                          desc: "F5-TTS or Kokoro-82M — open weights, voice-cloning support.",
                          status: .queued),
                    .init(icon: "photo.artframe", name: "Image gen",
                          desc: "SDXL Turbo / Flux Schnell + LoRA via Apple's ml-stable-diffusion.",
                          status: .queued),
                    .init(icon: "magnifyingglass", name: "Embeddings",
                          desc: "Distill BGE-M3 → 22M-110M domain embedder. Wedge into RAG.",
                          status: .queued),
                ])

                modalitySection(title: "DEFERRED (TIER 4)", items: [
                    .init(icon: "video", name: "Video understanding",
                          desc: "frame-sample → VLM. Falls out free after VLM lands.",
                          status: .deferred),
                    .init(icon: "music.note", name: "Music gen",
                          desc: "MusicGen-small (Meta). Real, just not v1 priority.",
                          status: .deferred),
                    .init(icon: "play.rectangle", name: "Video gen",
                          desc: "Local Mac quality is far below Sora/Veo/Kling. Skip until that changes.",
                          status: .skipped),
                ])
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Theme.base)
    }

    private func modalitySection(title: String, items: [Item]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.faint)
                .tracking(1)
            ForEach(items) { item in
                HStack(alignment: .top, spacing: 14) {
                    Image(systemName: item.icon)
                        .font(.system(size: 16))
                        .foregroundStyle(item.status.color)
                        .frame(width: 24)
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 8) {
                            Text(item.name)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(Theme.fg)
                            Text(item.status.label)
                                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                                .foregroundStyle(item.status.color)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(item.status.color.opacity(0.15))
                                .cornerRadius(3)
                        }
                        Text(item.desc)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.muted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.panel)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    struct Item: Identifiable {
        let id = UUID()
        let icon: String
        let name: String
        let desc: String
        let status: Status
    }
    enum Status {
        case shipped, queued, deferred, skipped
        var label: String {
            switch self {
            case .shipped:  return "SHIPPED"
            case .queued:   return "QUEUED"
            case .deferred: return "DEFERRED"
            case .skipped:  return "SKIP"
            }
        }
        var color: Color {
            switch self {
            case .shipped:  return .green
            case .queued:   return .blue
            case .deferred: return .orange
            case .skipped:  return .red
            }
        }
    }
}
