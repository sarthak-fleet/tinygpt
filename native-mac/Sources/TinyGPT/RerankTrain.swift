import Foundation
import TinyGPTModel

/// `tinygpt rerank-train` — lightweight lexical cross-encoder baseline.
///
/// This v1 trains a pairwise logistic model over query+document overlap
/// features from `{query,pos_doc,neg_doc}` JSONL triples. It gives the
/// factory a real reranker artifact without pulling a BERT stack into the
/// Swift package.
enum RerankTrain {
    static func run(args: [String]) {
        var triplesPath: String?
        var outPath: String?
        var steps = 5
        var lr: Float = 0.05

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--triples": triplesPath = args[i + 1]; i += 2
            case "--out": outPath = args[i + 1]; i += 2
            case "--steps": steps = Int(args[i + 1]) ?? steps; i += 2
            case "--lr": lr = Float(args[i + 1]) ?? lr; i += 2
            case "--teacher", "--student-preset":
                i += 2
            case "-h", "--help": exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        guard let triplesPath else { fputs("--triples required\n", stderr); exitUsage() }
        guard let outPath else { fputs("--out required\n", stderr); exitUsage() }
        let triples = readTriples(triplesPath)
        guard !triples.isEmpty else { fputs("no usable triples\n", stderr); exit(1) }

        var model = LexicalRerankerModel()
        for epoch in 0..<max(1, steps) {
            var loss: Double = 0
            for t in triples {
                let pos = model.features(query: t.query, doc: t.pos)
                let neg = model.features(query: t.query, doc: t.neg)
                let margin = model.score(pos) - model.score(neg)
                let p = 1.0 / (1.0 + exp(-Double(margin)))
                let grad = Float(p - 1.0)
                model.apply(diff: zip(pos, neg).map { $0 - $1 }, grad: grad, lr: lr)
                loss += -log(max(p, 1e-9))
            }
            print(String(format: "epoch %d/%d loss %.4f", epoch + 1, max(1, steps), loss / Double(triples.count)))
        }
        do {
            try model.save(to: URL(fileURLWithPath: outPath))
            print("✓ wrote reranker → \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
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
        usage: tinygpt rerank-train --triples triples.jsonl --out model.tinygpt-rerank [options]

        triples.jsonl rows: {"query": "...", "pos_doc": "...", "neg_doc": "..."}
        --steps N   Epochs over triples (default 5)
        --lr F      Learning rate (default 0.05)
        """)
        exit(code)
    }
}
