import Foundation
import XCTest
@testable import TinyGPTServe

/// B26 (docs/prds/B26-deferred-tools.md): coverage for the pieces of
/// deferred-tool mode that don't need a live model — the compact prompt
/// rendering, the extended grammar verb enum, the `toolInfo(name:)`
/// schema lookup, and the get_tool_info parser used by the
/// /v1/chat/completions interception loop.
///
/// The end-to-end gate (BFCL parity between --tool-mode full and
/// --tool-mode deferred) lives in the PRD — it's run by the user
/// against a loaded model, not by this suite.
final class DeferredToolsTests: XCTestCase {
    // Fixture: two tools with non-trivial schemas so we can confirm the
    // compact prompt strips schema detail while the full mode keeps it.
    private static let toolsJSON: String = """
    {"tools": [
      {"type": "function", "function": {
        "name": "AX.press",
        "description": "click a UI element by label.\\nargs: target",
        "parameters": {"type": "object",
                       "properties": {"target": {"type": "string"}},
                       "required": ["target"]}
      }},
      {"type": "function", "function": {
        "name": "Cal.event",
        "description": "create a calendar event",
        "parameters": {"type": "object",
                       "properties": {"title": {"type": "string"},
                                      "start": {"type": "string"}},
                       "required": ["title", "start"]}
      }}
    ]}
    """

    private func loadFixtureSpec() throws -> ServeToolsSpec {
        try ServeToolsSpec.parse(data: Self.toolsJSON.data(using: .utf8)!)
    }

    func test_compactSystemPrompt_omitsToolSchemas() throws {
        let spec = try loadFixtureSpec()
        let compact = spec.compactSystemPrompt()

        // The full-mode prompt embeds every "parameters" block; the
        // compact prompt must not — that's the whole point of B26.
        XCTAssertFalse(compact.contains("\"parameters\""),
                       "compact prompt leaks tool parameters")
        XCTAssertFalse(compact.contains("\"properties\""),
                       "compact prompt leaks parameter properties")

        // It MUST include each tool's name + one-line summary plus the
        // get_tool_info meta-tool.
        XCTAssertTrue(compact.contains("AX.press"))
        XCTAssertTrue(compact.contains("Cal.event"))
        XCTAssertTrue(compact.contains("get_tool_info"))

        // First-line-only summary discipline: the AX.press description
        // has a second line ("args: target") that must be dropped.
        XCTAssertFalse(compact.contains("args: target"),
                       "compact prompt should keep only the first line of each description")
    }

    func test_compactGrammarSpec_verbEnumIncludesGetToolInfo() throws {
        let spec = try loadFixtureSpec()
        let full = try spec.outputSchemaJSON()
        XCTAssertFalse(full.contains("\"get_tool_info\""),
                       "full grammar must not advertise the meta-tool")

        let compactJSON = try spec.compactOutputSchemaJSON()
        XCTAssertTrue(compactJSON.contains("\"get_tool_info\""),
                      "compact grammar must include the get_tool_info verb")
        XCTAssertTrue(compactJSON.contains("\"AX.press\""))
        XCTAssertTrue(compactJSON.contains("\"Cal.event\""))
        // Sanity: the parser accepts the schema.
        XCTAssertNoThrow(try spec.compactGrammarSpec())
    }

    func test_toolInfo_returnsSchemaForKnownTool() throws {
        let spec = try loadFixtureSpec()
        let schema = spec.toolInfo(name: "AX.press")
        XCTAssertNotNil(schema)
        guard let schema else { return }
        // The lookup must carry the parameters back — that's the whole
        // payload of the meta-tool interception.
        XCTAssertTrue(schema.contains("\"target\""))
        XCTAssertTrue(schema.contains("\"required\""))
    }

    func test_toolInfo_returnsNilForUnknownTool() throws {
        let spec = try loadFixtureSpec()
        XCTAssertNil(spec.toolInfo(name: "does.not.exist"))
    }

    func test_parseGetToolInfoCall_recognizesMetaToolJSON() {
        // The interception loop in handleChatCompletions calls this on
        // the model's raw output — must accept the canonical shape.
        let yes = #"{"verb":"get_tool_info","args":{"name":"AX.press"},"spoken_text":"checking"}"#
        XCTAssertEqual(Serve.Server.parseGetToolInfoCall(yes), "AX.press")

        // Non-meta-tool calls must return nil so the interception loop
        // exits and the response is sent to the client unchanged.
        let action = #"{"verb":"AX.press","args":{"target":"Save"}}"#
        XCTAssertNil(Serve.Server.parseGetToolInfoCall(action))

        // Defense in depth: malformed args.
        let badArgs = #"{"verb":"get_tool_info","args":{}}"#
        XCTAssertNil(Serve.Server.parseGetToolInfoCall(badArgs))

        // And totally non-JSON garbage shouldn't crash — the model
        // can drop out of the grammar under some serve configs.
        XCTAssertNil(Serve.Server.parseGetToolInfoCall("hello world"))
    }
}
