import Foundation

public struct LexicalRerankerModel: Codable, Sendable {
    public var weights: [Float]
    public var bias: Float

    public init(weights: [Float] = [0.5, 0.5, 0.25, 0.1], bias: Float = 0) {
        self.weights = weights
        self.bias = bias
    }

    public func score(query: String, doc: String) -> Float {
        score(features(query: query, doc: doc))
    }

    public func score(_ features: [Float]) -> Float {
        zip(weights, features).reduce(bias) { $0 + $1.0 * $1.1 }
    }

    public mutating func apply(diff: [Float], grad: Float, lr: Float) {
        for i in 0..<min(weights.count, diff.count) {
            weights[i] -= lr * grad * diff[i]
        }
        bias -= lr * grad
    }

    public func features(query: String, doc: String) -> [Float] {
        let q = tokens(query)
        let d = tokens(doc)
        let qs = Set(q)
        let ds = Set(d)
        let overlap = Float(qs.intersection(ds).count)
        let recall = overlap / Float(max(qs.count, 1))
        let precision = overlap / Float(max(ds.count, 1))
        let jaccard = overlap / Float(max(qs.union(ds).count, 1))
        let phrase = doc.lowercased().contains(query.lowercased()) ? Float(1) : Float(0)
        return [recall, precision, jaccard, phrase]
    }

    public func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    public static func load(_ url: URL) throws -> LexicalRerankerModel {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(LexicalRerankerModel.self, from: data)
    }

    private func tokens(_ s: String) -> [String] {
        s.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
    }
}
