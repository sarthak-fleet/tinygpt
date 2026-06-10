import SwiftUI

/// Roadmap workspace — what tinygpt has shipped, what's actively in flight,
/// what's queued next. Anchored to the (speed × accuracy) / cost formula
/// per the 2026-06-09 North Star — every queued item must justify itself
/// against a measurable formula delta.
///
/// Replaces the old "Modalities" view: this is the canonical "what's the
/// state of the project" surface.
struct RoadmapView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Roadmap")
                        .font(.tgDisplay)
                        .foregroundStyle(Theme.fg)
                    Text("North Star: (speed × accuracy) / cost. ≥5% per 2-week investment to make the queue.")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                }

                roadmapSection(title: "SHIPPED — PLANNER + INFRA", items: [
                    .init(icon: "checkmark.seal", name: "Pace planner v9",
                          desc: "Qwen3-0.6B + LoRA, 33.3% non-compose (matches v8) + 70% compose. Ships.",
                          status: .shipped),
                    .init(icon: "bolt.fill", name: "TTFW 330→119 ms (warm)",
                          desc: "token-bytes cache shipped (Serve.swift). 5-run variance 118-123 ms.",
                          status: .shipped),
                    .init(icon: "doc.badge.gearshape", name: "DoRA serialization v2",
                          desc: "TGLA format bump; magnitudes persist; backward compat verified.",
                          status: .shipped),
                    .init(icon: "ruler", name: "Formula score CLI",
                          desc: "scripts/score_formula.py — canonical (speed × accuracy / cost) per model.",
                          status: .shipped),
                ])

                roadmapSection(title: "SHIPPED — QUALIFIED MODELS", items: [
                    .init(icon: "waveform", name: "WhisperKit large-v3-turbo",
                          desc: "perfect accuracy on Pace-vocab w/o biasing; 1.5 GB, 9× realtime ANE.",
                          status: .shipped),
                    .init(icon: "magnifyingglass", name: "Qwen3-Embedding-0.6B",
                          desc: "swapped from mxbai 2026-06-09; faster on batches, same family as planner.",
                          status: .shipped),
                ])

                roadmapSection(title: "IN FLIGHT — TINYGPT", items: [
                    .init(icon: "arrow.triangle.2.circlepath", name: "Pace planner v10",
                          desc: "parameterized actions schema; v10 cascade ready (scripts/v10_pipeline.sh).",
                          status: .inFlight),
                    .init(icon: "eye", name: "VLM A/B + port (#266, #308)",
                          desc: "UI-Venus-1.5-2B vs Qwen3-VL-2B for Pace vision pillar; both on disk.",
                          status: .inFlight),
                ])

                roadmapSection(title: "QUEUED — NEXT 5 MODALITIES", items: [
                    .init(icon: "scribble.variable", name: "Voice-edit specialist (#294)",
                          desc: "AX selection → 'make this concise' / 'delete last sentence' transforms.",
                          status: .queued),
                    .init(icon: "text.cursor", name: "Dictation post-processor (#295)",
                          desc: "punctuation + capitalization + code-mode on Whisper output.",
                          status: .queued),
                    .init(icon: "tray.and.arrow.down", name: "RAG layer (#293)",
                          desc: "Qwen3-Embedding + SQLite-vec over Mail/Notes/files/past sessions.",
                          status: .queued),
                    .init(icon: "wrench.adjustable", name: "Tool-call specialist (#289)",
                          desc: "Pace Skills routing — folded into v10 unless BFCL gap remains.",
                          status: .queued),
                    .init(icon: "gauge.with.dots.needle.bottom.50percent", name: "Executor surface (Pace)",
                          desc: "AX dispatcher + EventKit + Shortcuts CLI; per pace-executor-surface.md PRD.",
                          status: .queued),
                ])

                roadmapSection(title: "QUEUED — PERF + COST", items: [
                    .init(icon: "memorychip", name: "Swift QuantizedLinear (#305)",
                          desc: "load mlx_lm convert -q output; 4× cost cut, ~1.5× speed.",
                          status: .queued),
                    .init(icon: "cpu", name: "macOS 26 int8 ANE handoff (#306)",
                          desc: "port direct int8 array binding into M8 chain; ~1.8× ANE speed.",
                          status: .queued),
                    .init(icon: "speedometer", name: "MLX compile + spec decode (#262)",
                          desc: "graph-level dev-side optimization for serve loop.",
                          status: .queued),
                ])

                roadmapSection(title: "REJECTED — RESEARCH SAID NO", items: [
                    .init(icon: "xmark.octagon", name: "anemll migration",
                          desc: "no LoRA path + open Qwen3 bugs on macOS 26. Port int8 in-house instead.",
                          status: .rejected),
                    .init(icon: "xmark.octagon", name: "tinygrad",
                          desc: "Apple-silicon island converged on MLX (Ollama switched Mar 2026).",
                          status: .rejected),
                    .init(icon: "xmark.octagon", name: "Apple FM-adapter pathway",
                          desc: "restricted to Apple's system model; can't wrap Qwen3-0.6B.",
                          status: .rejected),
                    .init(icon: "xmark.octagon", name: "FSM trie mask optimization",
                          desc: "bit-exact correct but 47% slower than legacy. Code behind useTrie=false flag.",
                          status: .rejected),
                ])
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Theme.base)
    }

    private func roadmapSection(title: String, items: [Item]) -> some View {
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
        case shipped, inFlight, queued, rejected
        var label: String {
            switch self {
            case .shipped:  return "SHIPPED"
            case .inFlight: return "IN FLIGHT"
            case .queued:   return "QUEUED"
            case .rejected: return "REJECTED"
            }
        }
        var color: Color {
            switch self {
            case .shipped:  return .green
            case .inFlight: return .blue
            case .queued:   return .orange
            case .rejected: return .red
            }
        }
    }
}
