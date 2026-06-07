import SwiftUI

/// Train workspace = top-level segmented picker between training modes
/// (pretrain / fine-tune / DPO / distill), each routes to its own view.
/// Restored 2026-06-07 PM after the consolidation pass dropped fine-tune
/// from the sidebar — it was always shipped in the CLI, just hidden in
/// the app.
struct TrainHubView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case pretrain = "Pretrain"
        case finetune = "Fine-tune"
        case dpo      = "DPO"
        case distill  = "Distill"
        var id: String { rawValue }

        var subtitle: String {
            switch self {
            case .pretrain:
                return "Train a transformer from scratch on a text corpus. Watch loss drop live."
            case .finetune:
                return "LoRA / SFT a base model on instruction-response pairs. Save adapters."
            case .dpo:
                return "Preference tuning — chosen vs rejected responses. Shapes style + helpfulness."
            case .distill:
                return "Distill a smaller specialist from a larger teacher. Local — no API spend."
            }
        }
    }

    @State private var mode: Mode = .pretrain

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Picker("", selection: $mode) {
                    ForEach(Mode.allCases) { m in Text(m.rawValue).tag(m) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 540)
                Text(mode.subtitle)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 14)
            .background(Theme.panel.opacity(0.5))
            .overlay(Rectangle().fill(Theme.line).frame(height: 1), alignment: .bottom)

            Group {
                switch mode {
                case .pretrain: TrainView()
                case .finetune: FinetuneView()
                case .dpo:      DPOStubView()
                case .distill:  DistillStubView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Theme.base)
    }
}

// Light stubs for DPO + Distill. They have shipped CLI surface but
// no dedicated app view yet; these point users at the CLI + recipe docs.

struct DPOStubView: View {
    var body: some View {
        WorkspaceShelf(title: "DPO — preference tuning",
                       tagline: "Shape model style + helpfulness via chosen-vs-rejected pairs.",
                       items: [
                        .init("CLI shipped",
                              "tinygpt dpo <base> --data prefs.jsonl --out model.lora",
                              .ok),
                        .init("Recipe",
                              "docs/recipes/distillation-fc.md describes the broader specialist arc",
                              .info),
                        .init("App UI",
                              "queued — same shape as Fine-tune tab; sub-PRD when prioritized",
                              .pending),
                       ])
    }
}

struct DistillStubView: View {
    var body: some View {
        WorkspaceShelf(title: "Distill — teacher → student",
                       tagline: "Distill a smaller specialist from a local teacher. Zero API spend.",
                       items: [
                        .init("CLI shipped",
                              "tinygpt distill --teacher <model> --student <preset> --data <jsonl>",
                              .ok),
                        .init("Recipe",
                              "docs/recipes/distillation-fc.md — full Phi-3-mini → 22M function-calling spec",
                              .info),
                        .init("App UI",
                              "queued — pick teacher (HF / local), pick student, pick data → train",
                              .pending),
                       ])
    }
}
