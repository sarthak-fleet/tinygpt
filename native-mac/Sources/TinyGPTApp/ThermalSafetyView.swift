import SwiftUI

@MainActor
final class ThermalMonitor: ObservableObject {
    @Published private(set) var state: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState

    func refresh() {
        state = ProcessInfo.processInfo.thermalState
    }
}

struct ThermalSafetyBanner: View {
    @AppStorage("tinygpt.train.thermalBannerDismissed")
    private var dismissed = false

    var body: some View {
        if !dismissed {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "fan.fill")
                    .foregroundStyle(Theme.warn)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Training runs the CPU and GPU at sustained load.")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text("Use a hard surface with clear airflow. A laptop stand or clamshell setup is better for long runs. Do not cover the keyboard or bottom vents.")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Button {
                    dismissed = true
                } label: {
                    Image(systemName: "xmark")
                        .foregroundStyle(Theme.faint)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 10)
            .background(Theme.warn.opacity(0.10))
            .overlay(Rectangle().fill(Theme.warn.opacity(0.45)).frame(height: 1), alignment: .bottom)
        }
    }
}

struct ThermalStatusChip: View {
    let state: ProcessInfo.ThermalState
    let autoThrottleNote: String?

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text("thermal \(label)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.muted)
            Image(systemName: "info.circle")
                .foregroundStyle(Theme.faint)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Theme.panel2)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .help("""
        Training needs sustained airflow. Use a hard surface, keep vents clear, avoid covering the keyboard, and prefer a stand or clamshell setup for long runs.
        \(autoThrottleNote ?? "")
        """)
     }

    private var label: String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private var color: Color {
        switch state {
        case .nominal: return Theme.accent
        case .fair: return Theme.warn
        case .serious: return Color.orange
        case .critical: return Theme.danger
        @unknown default: return Theme.faint
        }
    }
}
