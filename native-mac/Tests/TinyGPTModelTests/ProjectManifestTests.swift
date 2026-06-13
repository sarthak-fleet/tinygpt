import Foundation
import XCTest
@testable import TinyGPTModel

/// B31 coverage for `tinygpt.project.json` parsing + the schema-level
/// validation pass (the cross-check against an actual GalleryManifest
/// lives in the CLI's `tinygpt validate`, not here).
final class ProjectManifestTests: XCTestCase {

    func test_parses_fixture_example() throws {
        // The shipped fixture under examples/ — round-trip parse +
        // basic field check.
        let candidates = [
            "examples/tinygpt.project.json",
            "../examples/tinygpt.project.json",
        ]
        var data: Data?
        for p in candidates {
            if let d = try? Data(contentsOf: URL(fileURLWithPath: p)) {
                data = d; break
            }
        }
        guard let data else {
            throw XCTSkip("fixture project file not found at expected paths")
        }
        let manifest = try ProjectManifest.decode(from: data)
        XCTAssertEqual(manifest.name, "example-pace-project")
        XCTAssertGreaterThanOrEqual(manifest.models.count, 2)
        XCTAssertNoThrow(try manifest.validate())
    }

    func test_validates_well_formed_pair() throws {
        let json = #"""
        {
          "name": "ok",
          "models": [
            {"id": "qwen3-4b-instruct-2507", "role": "base"},
            {"id": "a1-tool-caller", "role": "adapter",
             "applies_to": "qwen3-4b-instruct-2507"}
          ]
        }
        """#
        let m = try ProjectManifest.decode(from: Data(json.utf8))
        XCTAssertNoThrow(try m.validate())
    }

    func test_rejects_adapter_without_appliesTo() throws {
        let json = #"""
        {
          "name": "bad",
          "models": [
            {"id": "stray-adapter", "role": "adapter"}
          ]
        }
        """#
        let m = try ProjectManifest.decode(from: Data(json.utf8))
        XCTAssertThrowsError(try m.validate()) { err in
            guard case ProjectManifest.ValidationError
                .adapterMissingAppliesTo(let id) = err else {
                return XCTFail("expected adapterMissingAppliesTo, got \(err)")
            }
            XCTAssertEqual(id, "stray-adapter")
        }
    }

    func test_rejects_adapter_whose_base_isnt_pinned() throws {
        let json = #"""
        {
          "name": "bad",
          "models": [
            {"id": "a1-tool-caller", "role": "adapter",
             "applies_to": "qwen3-4b-instruct-2507"}
          ]
        }
        """#
        let m = try ProjectManifest.decode(from: Data(json.utf8))
        XCTAssertThrowsError(try m.validate())
    }

    func test_rejects_duplicate_id() throws {
        let json = #"""
        {
          "name": "dupe",
          "models": [
            {"id": "x", "role": "base"},
            {"id": "x", "role": "judge"}
          ]
        }
        """#
        let m = try ProjectManifest.decode(from: Data(json.utf8))
        XCTAssertThrowsError(try m.validate()) { err in
            guard case ProjectManifest.ValidationError
                .duplicateModelId(let id) = err else {
                return XCTFail("expected duplicateModelId, got \(err)")
            }
            XCTAssertEqual(id, "x")
        }
    }

    func test_datasets_optional() throws {
        // A pure-inference project may carry no datasets at all.
        let json = #"""
        {
          "name": "inference-only",
          "models": [{"id": "qwen3-4b-instruct-2507", "role": "base"}]
        }
        """#
        let m = try ProjectManifest.decode(from: Data(json.utf8))
        XCTAssertNil(m.datasets)
        XCTAssertNoThrow(try m.validate())
    }
}
