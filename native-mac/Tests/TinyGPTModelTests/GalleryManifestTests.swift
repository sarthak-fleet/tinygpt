import Foundation
import XCTest
@testable import TinyGPTModel

/// B31 coverage for the gallery-manifest reader. Two-way invariant:
/// (a) the shipped browser manifest parses cleanly (V1-shape rows
/// with no `kind` discriminator default to .browserBin), and
/// (b) the new B31 fields round-trip.
final class GalleryManifestTests: XCTestCase {

    func test_parses_V1_browser_manifest_with_implicit_kind() throws {
        let v1JSON = #"""
        {
          "version": 1,
          "note": "All five models share the same architecture.",
          "models": [
            {
              "id": "shakespeare",
              "name": "Shakespeare",
              "file": "shakespeare.bin",
              "params": "9.6M",
              "paramCount": 9608704
            }
          ]
        }
        """#
        let manifest = try GalleryManifest.decode(from: Data(v1JSON.utf8))
        XCTAssertEqual(manifest.version, 1)
        XCTAssertEqual(manifest.models.count, 1)
        let m = manifest.models[0]
        XCTAssertEqual(m.id, "shakespeare")
        // V1 rows have no kind set; resolvedKind must default to
        // .browserBin so existing manifests keep working.
        XCTAssertNil(m.kind)
        XCTAssertEqual(m.resolvedKind, .browserBin)
    }

    func test_parses_B31_adapter_entry_with_kind_and_parent() throws {
        let json = #"""
        {
          "version": 1,
          "models": [
            {
              "id": "a1-tool-caller",
              "name": "A1 Tool-Caller",
              "file": "a1-tool-caller.lora",
              "kind": "mac-adapter",
              "parent": "qwen3-4b-instruct-2507",
              "r2_path": "adapters/a1-tool-caller.lora",
              "tags": ["tool-call", "english"]
            }
          ]
        }
        """#
        let manifest = try GalleryManifest.decode(from: Data(json.utf8))
        let m = manifest.models[0]
        XCTAssertEqual(m.resolvedKind, .macAdapter)
        XCTAssertEqual(m.parent, "qwen3-4b-instruct-2507")
        XCTAssertEqual(m.r2Path, "adapters/a1-tool-caller.lora")
        XCTAssertEqual(m.tags, ["tool-call", "english"])
    }

    func test_model_lookup_byId() throws {
        let json = #"""
        {"version": 1, "models": [
          {"id": "a", "name": "A", "file": "a.bin"},
          {"id": "b", "name": "B", "file": "b.bin"}
        ]}
        """#
        let manifest = try GalleryManifest.decode(from: Data(json.utf8))
        XCTAssertEqual(manifest.model(id: "a")?.name, "A")
        XCTAssertEqual(manifest.model(id: "b")?.name, "B")
        XCTAssertNil(manifest.model(id: "nope"))
    }

    func test_rejects_unknown_kind() {
        let json = #"""
        {"version": 1, "models": [
          {"id": "x", "name": "X", "file": "x.bin", "kind": "alien-format"}
        ]}
        """#
        XCTAssertThrowsError(
            try GalleryManifest.decode(from: Data(json.utf8))
        ) { error in
            // The DecodingError will mention the kind enum's available
            // cases — we don't pin the exact message string, just that
            // the throw happens.
            XCTAssertTrue("\(error)".contains("kind") || "\(error)".contains("alien-format"))
        }
    }

    func test_parses_shipped_browser_manifest_if_present() throws {
        // When run from the repo root (CI / local dev), the shipped
        // manifest should parse cleanly. Skip if the file isn't where
        // we expect (e.g. running from a fresh checkout that hasn't
        // built the browser bundle yet).
        let candidates = [
            "browser/public/gallery/manifest.json",
            "../browser/public/gallery/manifest.json",
        ]
        var data: Data?
        for p in candidates {
            if let d = try? Data(contentsOf: URL(fileURLWithPath: p)) {
                data = d; break
            }
        }
        guard let data else {
            throw XCTSkip("shipped gallery manifest not found at expected paths")
        }
        let manifest = try GalleryManifest.decode(from: data)
        XCTAssertEqual(manifest.version, 1)
        XCTAssertFalse(manifest.models.isEmpty)
    }
}
