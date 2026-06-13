import Foundation
import XCTest
@testable import TinyGPTModel

/// B32 coverage for the pure gate logic: direction heuristic, pass/fail
/// thresholding, per-suite overrides, missing-baseline handling, the
/// lower-is-better inversion, and K-pass averaging. No GPU, no server.
final class EvalGateTests: XCTestCase {

    private func row(_ task: String, _ metric: String, _ score: Double,
                     subtask: String? = nil) -> EvalGate.Row {
        EvalGate.Row(task: task, subtask: subtask, metric: metric,
                     score: score, n_examples: 100)
    }

    private func spec(_ suites: [EvalGate.SuiteSpec], threshold: Double? = nil) -> EvalGate.Spec {
        EvalGate.Spec(baseline: "baseline.jsonl", defaultThreshold: threshold, suites: suites)
    }

    func test_direction_heuristic() {
        XCTAssertEqual(EvalGate.direction(for: "accuracy"), .higherIsBetter)
        XCTAssertEqual(EvalGate.direction(for: "pass@1"), .higherIsBetter)
        XCTAssertEqual(EvalGate.direction(for: "f1"), .higherIsBetter)
        XCTAssertEqual(EvalGate.direction(for: "ppl"), .lowerIsBetter)
        XCTAssertEqual(EvalGate.direction(for: "val_loss"), .lowerIsBetter)
        XCTAssertEqual(EvalGate.direction(for: "latency_ms"), .lowerIsBetter)
    }

    func test_passes_when_candidate_matches_baseline() {
        let base = [row("bfcl", "accuracy", 0.80)]
        let cand = [row("bfcl", "accuracy", 0.80)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "bfcl")]))
        XCTAssertTrue(report.passed)
        XCTAssertEqual(report.suites.first?.verdict, .pass)
    }

    func test_small_regression_within_threshold_passes() {
        // -1.5pp with a 2pp default tolerance → pass.
        let base = [row("bfcl", "accuracy", 0.800)]
        let cand = [row("bfcl", "accuracy", 0.785)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "bfcl")]))
        XCTAssertTrue(report.passed)
        XCTAssertEqual(report.suites.first?.deltaPP ?? 0, -1.5, accuracy: 0.001)
    }

    func test_regression_past_threshold_fails() {
        // -3pp with a 2pp default tolerance → fail.
        let base = [row("bfcl", "accuracy", 0.80)]
        let cand = [row("bfcl", "accuracy", 0.77)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "bfcl")]))
        XCTAssertFalse(report.passed)
        XCTAssertEqual(report.failedCount, 1)
        XCTAssertEqual(report.suites.first?.verdict, .fail)
    }

    func test_improvement_always_passes() {
        let base = [row("tau", "pass@1", 0.50)]
        let cand = [row("tau", "pass@1", 0.62)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "tau", task: "tau")]))
        XCTAssertTrue(report.passed)
        XCTAssertGreaterThan(report.suites.first?.deltaPP ?? 0, 0)
    }

    func test_per_suite_threshold_override() {
        // bfcl tolerates 5pp; a -4pp drop passes under that override even
        // though the 2pp default would fail it.
        let base = [row("bfcl", "accuracy", 0.80)]
        let cand = [row("bfcl", "accuracy", 0.76)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "bfcl", threshold: 5.0)]))
        XCTAssertTrue(report.passed)
        XCTAssertEqual(report.suites.first?.thresholdPP, 5.0)
    }

    func test_lower_is_better_inversion() {
        // Perplexity going UP is a regression for a lower-is-better metric.
        let base = [row("lm", "ppl", 0.20)]
        let cand = [row("lm", "ppl", 0.25)]  // worse
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "lm", task: "lm")]))
        XCTAssertFalse(report.passed)
        // Direction-adjusted delta is negative (worse) even though raw rose.
        XCTAssertLessThan(report.suites.first?.deltaPP ?? 0, 0)
    }

    func test_missing_baseline_is_not_a_failure() {
        let base: [EvalGate.Row] = []
        let cand = [row("bfcl", "accuracy", 0.80)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "bfcl")]))
        XCTAssertTrue(report.passed)
        XCTAssertEqual(report.missingCount, 1)
        XCTAssertEqual(report.suites.first?.verdict, .missing)
    }

    func test_only_declared_suites_are_gated() {
        // Candidate has a humaneval regression, but the spec only gates bfcl.
        let base = [row("bfcl", "accuracy", 0.80), row("humaneval", "pass@1", 0.50)]
        let cand = [row("bfcl", "accuracy", 0.80), row("humaneval", "pass@1", 0.30)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand,
                                       spec: spec([.init(name: "bfcl")]))
        XCTAssertTrue(report.passed)
        XCTAssertEqual(report.suites.count, 1)
    }

    func test_empty_suites_gates_everything() {
        let base = [row("bfcl", "accuracy", 0.80), row("humaneval", "pass@1", 0.50)]
        let cand = [row("bfcl", "accuracy", 0.80), row("humaneval", "pass@1", 0.30)]
        let report = EvalGate.evaluate(baseline: base, candidate: cand, spec: spec([]))
        XCTAssertFalse(report.passed)
        XCTAssertEqual(report.suites.count, 2)
    }

    func test_averaging_collapses_repeated_keys() {
        let rows = [row("bfcl", "accuracy", 0.70), row("bfcl", "accuracy", 0.80),
                    row("bfcl", "accuracy", 0.90)]
        let avg = EvalGate.averagedByKey(rows)
        XCTAssertEqual(avg.count, 1)
        XCTAssertEqual(avg.first?.score ?? 0, 0.80, accuracy: 0.0001)
    }

    func test_loadRows_parses_evalcompare_jsonl_and_skips_garbage() throws {
        let jsonl = """
        {"task":"bfcl","subtask":"simple","metric":"accuracy","score":0.8,"n_examples":40,"baseline":true,"run_id":"x","model_path":"/m","model_name":"m","timestamp":"t","wall_seconds":1.0}
        not-json
        {"task":"tau","metric":"pass@1","score":0.5,"n_examples":20}
        """
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("gate-rows-\(UUID().uuidString.prefix(6)).jsonl")
        try jsonl.data(using: .utf8)!.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let rows = try EvalGate.loadRows(fromJSONLAt: tmp.path)
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows.first?.subtask, "simple")
        XCTAssertTrue(rows.first?.baseline ?? false)
    }

    func test_spec_decodes_from_project_eval_block() throws {
        // The optional "eval" block on tinygpt.project.json (B31 schema add).
        let json = #"""
        {
          "name": "proj",
          "models": [{"id": "qwen3-4b", "role": "base"}],
          "eval": {
            "baseline": "evals/baseline.jsonl",
            "default_threshold": 3.0,
            "suites": [
              {"name": "bfcl", "threshold": 5.0},
              {"name": "tau", "task": "tau-bench"}
            ]
          }
        }
        """#
        let manifest = try ProjectManifest.decode(from: json.data(using: .utf8)!)
        let eval = try XCTUnwrap(manifest.eval)
        XCTAssertEqual(eval.baseline, "evals/baseline.jsonl")
        XCTAssertEqual(eval.resolvedDefaultThreshold, 3.0)
        XCTAssertEqual(eval.suites.count, 2)
        XCTAssertEqual(eval.suites.first?.threshold, 5.0)
        XCTAssertEqual(eval.suites.last?.resolvedTask, "tau-bench")
    }
}
