import SwiftUI

struct EvalChart: View {
    let rows: [AppEvalRow]

    var body: some View {
        GeometryReader { geo in
            let stepped = rows.filter { $0.model_step != nil }
            ZStack {
                Path { path in
                    let w = geo.size.width
                    let h = geo.size.height
                    for y in stride(from: 0.25, through: 0.75, by: 0.25) {
                        path.move(to: CGPoint(x: 0, y: h * (1 - y)))
                        path.addLine(to: CGPoint(x: w, y: h * (1 - y)))
                    }
                }
                .stroke(Theme.line, lineWidth: 1)

                ForEach(Array(Dictionary(grouping: stepped, by: { $0.task }).keys.sorted().enumerated()), id: \.element) { idx, task in
                    let series = stepped.filter { $0.task == task }.sorted { ($0.model_step ?? 0) < ($1.model_step ?? 0) }
                    line(series: series, in: geo.size)
                        .stroke(color(idx), lineWidth: 2)
                }
            }
        }
    }

    private func line(series: [AppEvalRow], in size: CGSize) -> Path {
        let steps = series.compactMap(\.model_step)
        let minStep = Double(steps.min() ?? 0)
        let maxStep = Double(steps.max() ?? 1)
        let span = max(1, maxStep - minStep)
        var path = Path()
        for (idx, row) in series.enumerated() {
            let x = ((Double(row.model_step ?? 0) - minStep) / span) * size.width
            let y = (1 - min(1, max(0, row.score))) * size.height
            if idx == 0 { path.move(to: CGPoint(x: x, y: y)) }
            else { path.addLine(to: CGPoint(x: x, y: y)) }
        }
        return path
    }

    private func color(_ idx: Int) -> Color {
        let palette = [Theme.accent, Theme.warn, Color.cyan, Color.pink, Color.orange, Color.green]
        return palette[idx % palette.count]
    }
}

