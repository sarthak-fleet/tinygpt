import Foundation

/// One named scalar dimension of a composite reward.
///
/// Castform-inspired (`docs/learn/castform-rl-finetune.md` §1): a real
/// specialist's reward is rarely "one thing" — it's "tool-call is right"
/// + "spoken_text is concise" + "didn't hallucinate" + …, each scored
/// independently and combined.
///
/// Conventions:
/// - `name` is the dimension's identifier (snake_case; lowercase ASCII).
/// - `score` is the raw scalar, no normalization. Conventionally in
///   [0, 1] or [-1, 1] but the abstraction doesn't enforce — different
///   recipes use different ranges (e.g. corpus-PPL deltas are unbounded
///   below zero).
/// - `weight` is the contribution multiplier in the total. Weights are
///   NOT normalized — the user knows what their composite means; the
///   total is plotted as-is.
public struct RewardDimension: Codable, Hashable, Sendable {
    public let name: String
    public let score: Double
    public let weight: Double

    public init(name: String, score: Double, weight: Double = 1.0) {
        self.name = name
        self.score = score
        self.weight = weight
    }
}

/// A reward that is *a bag of named dimensions* aggregated to a total.
///
/// Consumed by DPO (`tinygpt dpo`), ES (`tinygpt es`), and GRPO
/// (`tinygpt grpo` — see PRD 5.1) once the per-recipe integrations
/// land. Until then, this struct ships standalone so other call sites
/// (eval scorers, judge shims) can already emit composite rewards
/// that the training loops will consume in B28's V2.
///
/// The wire format (Codable JSON) is the per-rollout dashboard input —
/// train-viewer (C10) renders one line per dimension when present.
public struct CompositeReward: Codable, Hashable, Sendable {
    public let dimensions: [RewardDimension]

    public init(dimensions: [RewardDimension]) {
        self.dimensions = dimensions
    }

    /// Weighted sum across all dimensions. No normalization; if the
    /// caller wanted a normalized total they should pass weights that
    /// sum to 1.
    public var total: Double {
        dimensions.reduce(0.0) { acc, dim in acc + dim.score * dim.weight }
    }

    /// Quick lookup by name. Returns nil for unknown — by design we
    /// don't synthesize zero values, since the absence of a dimension
    /// in a recipe is meaningful (it never ran), distinct from a
    /// scored-zero value (it ran and got 0).
    public subscript(name: String) -> RewardDimension? {
        dimensions.first { $0.name == name }
    }

    /// All names in declaration order. Stable iteration for plotting.
    public var names: [String] { dimensions.map(\.name) }

    /// JSON encoder pre-configured for our train-viewer + leaderboard
    /// shipping format: snake_case keys, no pretty-printing.
    public static let jsonEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    public static let jsonDecoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    public func encoded() throws -> Data { try Self.jsonEncoder.encode(self) }

    public static func decoded(from data: Data) throws -> CompositeReward {
        try Self.jsonDecoder.decode(CompositeReward.self, from: data)
    }
}

/// A user-pluggable composition rule: the caller provides per-
/// dimension scorers (each returning a `RewardDimension`); the
/// builder collects them into a `CompositeReward`. Adapter shape
/// for the recipe layer — DPO/ES/GRPO recipes register their
/// scoring closures here.
public struct CompositeRewardBuilder: Sendable {
    public typealias Scorer = @Sendable () -> RewardDimension

    private var scorers: [Scorer]

    public init() { self.scorers = [] }

    public mutating func add(name: String, weight: Double = 1.0,
                              score: @escaping @Sendable () -> Double) {
        scorers.append { RewardDimension(name: name, score: score(), weight: weight) }
    }

    public mutating func add(scorer: @escaping Scorer) {
        scorers.append(scorer)
    }

    public func build() -> CompositeReward {
        CompositeReward(dimensions: scorers.map { $0() })
    }
}
