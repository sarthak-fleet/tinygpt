import Foundation
import TinyGPTModel

enum InferHeatmap {
    static func run(args: [String]) {
        var input: String?
        var htmlOut: String?
        var summary = false
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--html":
                guard i + 1 < args.count else { exitUsage() }
                htmlOut = args[i + 1]; i += 2
            case "--summary":
                summary = true; i += 1
            case "-h", "--help":
                exitUsage(0)
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                input = args[i]; i += 1
            }
        }
        guard let input else {
            fputs("infer-heatmap: missing <trace.json>\n", stderr); exitUsage()
        }

        do {
            let traces = try loadTraces(pattern: input)
            if summary {
                printSummary(traces)
            } else if let first = traces.first {
                printHeatmap(first)
                if let htmlOut {
                    try renderHTML(first).write(
                        to: URL(fileURLWithPath: htmlOut),
                        atomically: true,
                        encoding: .utf8
                    )
                    print("\nwrote \(htmlOut)")
                }
            }
        } catch {
            fputs("infer-heatmap: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func loadTraces(pattern: String) throws -> [InferenceTraceRecord] {
        let urls: [URL]
        if pattern.contains("*") {
            let nsPattern = NSString(string: pattern)
            let dir = nsPattern.deletingLastPathComponent.isEmpty ? "." : nsPattern.deletingLastPathComponent
            let filePattern = nsPattern.lastPathComponent
            let re = "^" + NSRegularExpression.escapedPattern(for: filePattern)
                .replacingOccurrences(of: "\\*", with: ".*") + "$"
            let regex = try NSRegularExpression(pattern: re)
            urls = try FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: dir), includingPropertiesForKeys: nil)
                .filter { url in
                    let name = url.lastPathComponent
                    let range = NSRange(name.startIndex..<name.endIndex, in: name)
                    return regex.firstMatch(in: name, range: range) != nil
                }
                .sorted { $0.path < $1.path }
        } else {
            urls = [URL(fileURLWithPath: pattern)]
        }
        let decoder = JSONDecoder()
        return try urls.map { try decoder.decode(InferenceTraceRecord.self, from: Data(contentsOf: $0)) }
    }

    private static func printSummary(_ traces: [InferenceTraceRecord]) {
        guard !traces.isEmpty else { print("no traces"); return }
        let totals = traces.map(\.totalMs).sorted()
        print("traces: \(traces.count)")
        print(String(format: "total p50 %.1fms · p95 %.1fms · max %.1fms",
                     percentile(totals, 0.50), percentile(totals, 0.95), totals.last ?? 0))
        let grouped = Dictionary(grouping: traces.flatMap(\.spans), by: \.name)
        for name in grouped.keys.sorted() {
            let vals = (grouped[name] ?? []).map(\.durationMs).sorted()
            print(String(format: "%@ p50 %8.1fms  p95 %8.1fms",
                         pad(name, 24), percentile(vals, 0.50), percentile(vals, 0.95)))
        }
    }

    private static func printHeatmap(_ t: InferenceTraceRecord) {
        print(String(format: "total %.1fms · prompt %d tok · generated %d tok · cache %@",
                     t.totalMs, t.promptTokens, t.generatedTokens, t.cache.hit ? "hit" : "miss"))
        print("")
        let aggregate = aggregateRows(t)
        let maxMs = max(aggregate.map(\.1).max() ?? 1, 1)
        for (name, ms) in aggregate {
            print(String(format: "%@ %8.1fms   %@", pad(name, 24), ms, bar(ms, max: maxMs)))
        }
        if !t.tokens.isEmpty {
            print("\nslowest token spans:")
            let slow = t.tokens.sorted {
                ($0.modelMs + $0.constraintMs + $0.decodeMs) > ($1.modelMs + $1.constraintMs + $1.decodeMs)
            }.prefix(5)
            for tok in slow {
                print(String(format: "  tok %02d  model %.1fms  constraint %.1fms  decode %.1fms",
                             tok.index, tok.modelMs, tok.constraintMs, tok.decodeMs))
            }
        }
    }

    private static func aggregateRows(_ t: InferenceTraceRecord) -> [(String, Double)] {
        var rows: [(String, Double)] = []
        let spanGroups = Dictionary(grouping: t.spans, by: \.name)
        for name in spanGroups.keys.sorted() {
            rows.append((name, (spanGroups[name] ?? []).map(\.durationMs).reduce(0, +)))
        }
        let model = t.tokens.map(\.modelMs).reduce(0, +)
        let constraint = t.tokens.map(\.constraintMs).reduce(0, +)
        let decode = t.tokens.map(\.decodeMs).reduce(0, +)
        if model > 0 { rows.append(("tokens.model", model)) }
        if constraint > 0 { rows.append(("tokens.constraint", constraint)) }
        if decode > 0 { rows.append(("tokens.decode", decode)) }
        return rows.sorted { $0.1 > $1.1 }
    }

    private static func renderHTML(_ t: InferenceTraceRecord) -> String {
        let rows = aggregateRows(t)
        let maxMs = max(rows.map(\.1).max() ?? 1, 1)
        let body = rows.map { name, ms in
            let width = max(1, Int((ms / maxMs) * 100))
            return """
            <div class="row"><div class="name">\(escape(name))</div><div class="bar"><span style="width:\(width)%"></span></div><div class="ms">\(String(format: "%.1f", ms))ms</div></div>
            """
        }.joined(separator: "\n")
        return """
        <!doctype html>
        <meta charset="utf-8">
        <title>TinyGPT Inference Heatmap</title>
        <style>
        body{font:14px -apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#1f2937}
        h1{font-size:20px}.meta{color:#4b5563;margin-bottom:18px}
        .row{display:grid;grid-template-columns:220px 1fr 90px;gap:12px;align-items:center;margin:8px 0}
        .name{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.bar{background:#f3f4f6;height:18px}
        .bar span{display:block;height:18px;background:#ef4444}.ms{text-align:right}
        </style>
        <h1>TinyGPT Inference Heatmap</h1>
        <div class="meta">total \(String(format: "%.1f", t.totalMs))ms · prompt \(t.promptTokens) tok · generated \(t.generatedTokens) tok · cache \(t.cache.hit ? "hit" : "miss")</div>
        \(body)
        """
    }

    private static func bar(_ value: Double, max: Double) -> String {
        let n = max == 0 ? 1 : Swift.max(1, Int((value / max) * 44))
        return String(repeating: "█", count: n)
    }

    private static func percentile(_ vals: [Double], _ p: Double) -> Double {
        guard !vals.isEmpty else { return 0 }
        let idx = min(vals.count - 1, max(0, Int(Double(vals.count - 1) * p)))
        return vals[idx]
    }

    private static func escape(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        if s.count >= width { return s }
        return s + String(repeating: " ", count: width - s.count)
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt infer-heatmap <trace.json|glob> [--summary] [--html out.html]
        """)
        exit(code)
    }
}
