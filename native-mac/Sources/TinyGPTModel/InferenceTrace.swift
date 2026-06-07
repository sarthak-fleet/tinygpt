import Foundation

public struct InferenceCacheTrace: Codable, Sendable {
    public var enabled: Bool
    public var hit: Bool
    public var prefixTokens: Int

    public init(enabled: Bool = false, hit: Bool = false, prefixTokens: Int = 0) {
        self.enabled = enabled
        self.hit = hit
        self.prefixTokens = prefixTokens
    }
}

public struct InferenceSpan: Codable, Sendable {
    public var name: String
    public var startMs: Double
    public var durationMs: Double
}

public struct InferenceTokenTrace: Codable, Sendable {
    public var index: Int
    public var modelMs: Double
    public var constraintMs: Double
    public var decodeMs: Double
}

public struct InferenceTraceRecord: Codable, Sendable {
    public var requestId: String
    public var route: String
    public var model: String
    public var totalMs: Double
    public var promptTokens: Int
    public var generatedTokens: Int
    public var cache: InferenceCacheTrace
    public var spans: [InferenceSpan]
    public var tokens: [InferenceTokenTrace]
}

public final class InferenceTracer {
    private let requestId: String
    private let route: String
    private let model: String
    private let startNs: UInt64
    private var spans: [InferenceSpan] = []
    private var tokens: [InferenceTokenTrace] = []
    private var promptTokens = 0
    private var generatedTokens = 0
    private var cache = InferenceCacheTrace()

    public init(route: String, model: String) {
        self.requestId = UUID().uuidString
        self.route = route
        self.model = model
        self.startNs = Self.nowNs()
    }

    public func setTokenCounts(prompt: Int, generated: Int) {
        promptTokens = prompt
        generatedTokens = generated
    }

    public func setCache(enabled: Bool, hit: Bool, prefixTokens: Int) {
        cache = InferenceCacheTrace(enabled: enabled, hit: hit, prefixTokens: prefixTokens)
    }

    public func addToken(index: Int, modelMs: Double, constraintMs: Double, decodeMs: Double) {
        tokens.append(InferenceTokenTrace(
            index: index,
            modelMs: modelMs,
            constraintMs: constraintMs,
            decodeMs: decodeMs
        ))
    }

    @discardableResult
    public func span<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        let start = Self.nowNs()
        defer {
            let end = Self.nowNs()
            spans.append(InferenceSpan(
                name: name,
                startMs: Self.ms(from: startNs, to: start),
                durationMs: Self.ms(from: start, to: end)
            ))
        }
        return try body()
    }

    public func record(_ name: String, startNs: UInt64, endNs: UInt64) {
        spans.append(InferenceSpan(
            name: name,
            startMs: Self.ms(from: self.startNs, to: startNs),
            durationMs: Self.ms(from: startNs, to: endNs)
        ))
    }

    public func record(_ name: String, durationMs: Double) {
        spans.append(InferenceSpan(name: name, startMs: 0, durationMs: durationMs))
    }

    public func finish() -> InferenceTraceRecord {
        InferenceTraceRecord(
            requestId: requestId,
            route: route,
            model: model,
            totalMs: Self.ms(from: startNs, to: Self.nowNs()),
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            cache: cache,
            spans: spans,
            tokens: tokens
        )
    }

    public func write(to directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let record = finish()
        let url = directory.appendingPathComponent("\(record.requestId).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(record)
        try data.write(to: url, options: [.atomic])
        return url
    }

    public static func nowNs() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

    public static func ms(from start: UInt64, to end: UInt64) -> Double {
        Double(end > start ? end - start : 0) / 1_000_000.0
    }
}
