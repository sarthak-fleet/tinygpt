import Foundation
import TinyGPTModel

enum RerankEval {
    static func run(args: [String]) {
        var modelPath: String?
        var dataPath: String?
        var outPath: String?
        var modelName: String?

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--model": modelPath = args[i + 1]; i += 2
            case "--data", "--task": dataPath = args[i + 1]; i += 2
            case "--out": outPath = args[i + 1]; i += 2
            case "--model-name": modelName = args[i + 1]; i += 2
            case "--base-retriever", "--top-k":
                i += 2
            case "-h", "--help": exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        guard let modelPath else { fputs("--model required\n", stderr); exitUsage() }
        guard let dataPath else { fputs("--data <eval.jsonl> required\n", stderr); exitUsage() }
        guard let outPath else { fputs("--out required\n", stderr); exitUsage() }

        let model: LexicalRerankerModel
        do { model = try LexicalRerankerModel.load(URL(fileURLWithPath: modelPath)) }
        catch { fputs("load failed: \(error)\n", stderr); exit(1) }
        let triples = readTriples(dataPath)
        guard !triples.isEmpty else { fputs("no usable eval triples\n", stderr); exit(1) }

        var correct = 0
        var mrr: Double = 0
        for t in triples {
            let pos = model.score(query: t.query, doc: t.pos)
            let neg = model.score(query: t.query, doc: t.neg)
            if pos >= neg {
                correct += 1
                mrr += 1.0
            } else {
                mrr += 0.5
            }
        }
        let acc = Double(correct) / Double(triples.count)
        let common = EvalHarnessSupport.Common(
            modelPath: modelPath,
            outJsonl: outPath,
            modelName: modelName ?? URL(fileURLWithPath: modelPath).lastPathComponent,
            modelStep: nil,
            baseline: false
        )
        EvalHarnessSupport.appendRow(common: common, task: "rerank/custom", subtask: nil,
                                     metric: "accuracy", score: acc, n: triples.count,
                                     wall: 0, harness: "tinygpt-rerank")
        EvalHarnessSupport.appendRow(common: common, task: "rerank/custom", subtask: nil,
                                     metric: "mrr", score: mrr / Double(triples.count),
                                     n: triples.count, wall: 0, harness: "tinygpt-rerank")
        print("✓ wrote rerank eval rows to \(outPath)")
    }

    private struct Triple { let query: String; let pos: String; let neg: String }

    private static func readTriples(_ path: String) -> [Triple] {
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        return raw.split(separator: "\n").compactMap { line in
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let q = obj["query"] as? String,
                  let p = (obj["pos_doc"] as? String) ?? (obj["positive"] as? String),
                  let n = (obj["neg_doc"] as? String) ?? (obj["negative"] as? String)
            else { return nil }
            return Triple(query: q, pos: p, neg: n)
        }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt rerank-eval --model model.tinygpt-rerank --data triples.jsonl --out results.jsonl
        """)
        exit(code)
    }
}
