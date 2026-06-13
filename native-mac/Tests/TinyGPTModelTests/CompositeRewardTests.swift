import Foundation
import XCTest
@testable import TinyGPTModel

/// Coverage for the B28 composite-reward scaffolding shipped in
/// `Sources/TinyGPTModel/CompositeReward.swift`. The training-loop
/// integrations (DPO/ES/GRPO) come in the rest of B28; this suite
/// covers the data structure only.
final class CompositeRewardTests: XCTestCase {

    private func sampleReward() -> CompositeReward {
        CompositeReward(dimensions: [
            RewardDimension(name: "correctness", score: 0.9, weight: 1.0),
            RewardDimension(name: "conciseness", score: 0.6, weight: 0.5),
            RewardDimension(name: "tool_call_efficiency", score: 0.8, weight: 0.25),
        ])
    }

    func test_total_isWeightedSumAcrossAllDimensions() {
        let r = sampleReward()
        // 0.9*1.0 + 0.6*0.5 + 0.8*0.25 = 0.9 + 0.3 + 0.2 = 1.4
        XCTAssertEqual(r.total, 1.4, accuracy: 1e-9)
    }

    func test_subscript_returnsDimensionOrNilForUnknown() {
        let r = sampleReward()
        XCTAssertEqual(r["correctness"]?.score, 0.9)
        XCTAssertNil(r["does_not_exist"])
    }

    func test_names_preservesDeclarationOrder() {
        let r = sampleReward()
        XCTAssertEqual(r.names, ["correctness", "conciseness", "tool_call_efficiency"])
    }

    func test_jsonRoundtrip_preservesEveryField() throws {
        let r = sampleReward()
        let data = try r.encoded()
        let back = try CompositeReward.decoded(from: data)
        XCTAssertEqual(back, r)
        // Spot check: total survives the round-trip exactly.
        XCTAssertEqual(back.total, r.total)
    }

    func test_builder_collectsScorersInRegistrationOrder() {
        var b = CompositeRewardBuilder()
        b.add(name: "correctness", weight: 1.0) { 0.9 }
        b.add(name: "conciseness", weight: 0.5) { 0.6 }
        let r = b.build()
        XCTAssertEqual(r.names, ["correctness", "conciseness"])
        XCTAssertEqual(r.total, 0.9 + 0.3, accuracy: 1e-9)
    }

    func test_zeroDimensions_totalIsZero() {
        // Edge case: an empty composite is a legal artifact (e.g. a
        // recipe that disables all reward sources for a smoke test).
        // total = 0 is the unambiguous answer; nil would force every
        // consumer to handle "no rewards yet" specially.
        let empty = CompositeReward(dimensions: [])
        XCTAssertEqual(empty.total, 0.0)
        XCTAssertEqual(empty.names, [])
    }
}
