import Foundation

struct TrainingProgressSnapshot: Equatable {
    var step: Int?
    var totalSteps: Int?
    var loss: Float?
    var learningRate: Float?
    var stepsPerSec: Double?
}

enum TrainingProgressParser {
    static func parseLine(_ line: String) -> TrainingProgressSnapshot? {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        var snapshot = TrainingProgressSnapshot()
        snapshot.step = obj["step"] as? Int
        snapshot.totalSteps = obj["total_steps"] as? Int
        snapshot.loss = floatValue(obj["loss"])
        snapshot.learningRate = floatValue(obj["lr"])
        snapshot.stepsPerSec = doubleValue(obj["step_per_s"])
        return snapshot
    }

    private static func floatValue(_ value: Any?) -> Float? {
        if let n = value as? NSNumber { return n.floatValue }
        if let s = value as? String { return Float(s) }
        return nil
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        if let n = value as? NSNumber { return n.doubleValue }
        if let s = value as? String { return Double(s) }
        return nil
    }
}

