import Foundation
import XCTest
@testable import TinyGPTModel

/// Tests for the JSON-Schema FSM that drives constrained generation.
/// Pure-Swift (no MLX), so these run under `swift test` even when the
/// Metal library isn't available.
final class ConstrainedGenTests: XCTestCase {

    // MARK: - Schema parse

    func test_parsesObjectWithRequiredAndProperties() throws {
        let json = """
        {
          "type": "object",
          "required": ["name"],
          "properties": {
            "name": { "type": "string" }
          }
        }
        """
        let data = json.data(using: .utf8)!
        let node = try JSONSchemaNode.from(data: data)
        if case .object(let props, let req) = node {
            XCTAssertEqual(props.count, 1)
            XCTAssertEqual(props[0].0, "name")
            XCTAssertTrue(req.contains("name"))
        } else {
            XCTFail("expected .object, got \(node)")
        }
    }

    func test_parsesEnumString() throws {
        let json = """
        { "type": "string", "enum": ["a", "b", "c"] }
        """
        let data = json.data(using: .utf8)!
        let node = try JSONSchemaNode.from(data: data)
        if case .string(let e, _, _) = node {
            XCTAssertEqual(e, ["a", "b", "c"])
        } else {
            XCTFail("expected enum string, got \(node)")
        }
    }

    // MARK: - FSM accepts well-formed JSON

    func test_fsmAcceptsSimpleObject() {
        let schemaJSON = #"""
        {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
          },
          "required": ["name", "age"]
        }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        // Minified — the FSM rejects insignificant whitespace by design;
        // see comment on `allowsWhitespaceHere`. Real-world tool output
        // matches `JSON.stringify(x)` (no spaces, no newlines).
        let s = #"{"name":"Alice","age":42}"#
        XCTAssertTrue(fsm.acceptString(s), "FSM should accept conformant JSON")
        XCTAssertTrue(fsm.isComplete, "FSM should be complete after closing brace")
    }

    func test_fsmRejectsInterTokenWhitespace() {
        // Insignificant whitespace between structural tokens is rejected.
        // Inside string content, whitespace is fine. See test_fsmStringAllowsWhitespaceInContent.
        let schemaJSON = #"""
        { "type": "object", "properties": { "x": { "type": "integer" } }, "required": ["x"] }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        XCTAssertFalse(fsm.acceptString("  {\"x\":1}"))
    }

    func test_fsmStringAllowsWhitespaceInContent() {
        let node: JSONSchemaNode = .string(enumValues: nil, minLength: nil, maxLength: nil)
        let fsm = JSONSchemaFSM(rootSchema: node)
        XCTAssertTrue(fsm.acceptString("\"hello world\""))
        XCTAssertTrue(fsm.isComplete)
    }

    func test_fsmRejectsInvalidJSON() {
        let schemaJSON = #"""
        { "type": "object", "properties": { "x": { "type": "integer" } }, "required": ["x"] }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        // Garbage byte right at the start
        XCTAssertFalse(fsm.acceptString("garbage"))
        XCTAssertFalse(fsm.isComplete)
    }

    func test_fsmRejectsUndeclaredKey() {
        let schemaJSON = #"""
        { "type": "object", "properties": { "x": { "type": "integer" } }, "required": [] }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        // Key "y" is not declared in the schema → should be rejected when
        // we get to the closing quote (or earlier if y conflicts with
        // declared keys' prefixes).
        // Open object + first key char that doesn't start any declared key.
        // For schema with only "x", a key starting with `y` should fail.
        XCTAssertTrue(fsm.acceptString("{\""))
        XCTAssertFalse(fsm.acceptByte(UInt8(ascii: "y")))
    }

    func test_fsmEnumRestrictsContent() {
        let schemaJSON = #"""
        { "type": "string", "enum": ["foo", "bar"] }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        // "foo" should fully match.
        XCTAssertTrue(fsm.acceptString("\"foo\""))
        XCTAssertTrue(fsm.isComplete)
    }

    func test_fsmEnumRejectsNonMember() {
        let schemaJSON = #"""
        { "type": "string", "enum": ["foo", "bar"] }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        XCTAssertTrue(fsm.acceptString("\""))
        XCTAssertTrue(fsm.acceptString("f"))
        // Only "foo" extends "f" in the enum; "x" should be rejected.
        XCTAssertFalse(fsm.acceptByte(UInt8(ascii: "x")))
        XCTAssertTrue(fsm.acceptString("oo\""))
        XCTAssertTrue(fsm.isComplete)
    }

    func test_fsmAcceptsArray() {
        let schemaJSON = #"""
        { "type": "array", "items": { "type": "integer" } }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        // Minified — no whitespace between structural tokens.
        XCTAssertTrue(fsm.acceptString("[1,2,3]"))
        XCTAssertTrue(fsm.isComplete)
    }

    func test_fsmAcceptsNestedObject() {
        let schemaJSON = #"""
        {
          "type": "object",
          "properties": {
            "tool_name": { "type": "string", "enum": ["read_file", "run_test"] },
            "arguments": {
              "type": "object",
              "properties": { "path": { "type": "string" } }
            }
          },
          "required": ["tool_name", "arguments"]
        }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        let s = #"{"tool_name":"read_file","arguments":{"path":"/etc/passwd"}}"#
        XCTAssertTrue(fsm.acceptString(s))
        XCTAssertTrue(fsm.isComplete)
    }

    func test_fsmAcceptsBooleansAndNull() {
        var f1 = JSONSchemaFSM(rootSchema: .boolean)
        XCTAssertTrue(f1.acceptString("true"))
        XCTAssertTrue(f1.isComplete)
        f1 = JSONSchemaFSM(rootSchema: .boolean)
        XCTAssertTrue(f1.acceptString("false"))
        XCTAssertTrue(f1.isComplete)
        var f2 = JSONSchemaFSM(rootSchema: .null)
        XCTAssertTrue(f2.acceptString("null"))
        XCTAssertTrue(f2.isComplete)
    }

    func test_fsmRejectsTrailingCommaObject() {
        let schemaJSON = #"""
        { "type": "object", "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } }, "required": [] }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        // `{"a":1,}` — trailing comma must be rejected.
        XCTAssertTrue(fsm.acceptString(#"{"a":1"#))
        XCTAssertTrue(fsm.acceptByte(UInt8(ascii: ",")))
        XCTAssertFalse(fsm.acceptByte(UInt8(ascii: "}")), "trailing comma must be rejected")
    }

    func test_fsmRejectsTrailingCommaArray() {
        let schemaJSON = #"""
        { "type": "array", "items": { "type": "integer" } }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        XCTAssertTrue(fsm.acceptString("[1,2"))
        XCTAssertTrue(fsm.acceptByte(UInt8(ascii: ",")))
        XCTAssertFalse(fsm.acceptByte(UInt8(ascii: "]")), "trailing comma must be rejected")
    }

    func test_fsmRequiresAllRequiredKeys() {
        let schemaJSON = #"""
        {
          "type": "object",
          "properties": {
            "a": { "type": "integer" },
            "b": { "type": "integer" }
          },
          "required": ["a", "b"]
        }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        // Closing the object with only `a` emitted must fail.
        XCTAssertTrue(fsm.acceptString(#"{"a":1"#))
        XCTAssertFalse(fsm.acceptByte(UInt8(ascii: "}")))
        // But adding `b` and closing is fine.
        XCTAssertTrue(fsm.acceptString(#","b":2}"#))
        XCTAssertTrue(fsm.isComplete)
    }

    // MARK: - LogitsMasker (byte-level vocab)

    func test_byteLevelMasker_onlyOpenBraceValidAtStart() {
        let schemaJSON = #"""
        { "type": "object", "properties": {}, "required": [] }
        """#
        let node = try! JSONSchemaNode.from(data: schemaJSON.data(using: .utf8)!)
        let fsm = JSONSchemaFSM(rootSchema: node)
        let masker = LogitsMasker(vocabSize: 256, eosTokenId: nil) { id in
            if id < 0 || id >= 256 { return "" }
            return String(decoding: [UInt8(id)], as: UTF8.self)
        }
        let mask = masker.mask(forFSM: fsm)
        // `{` is the only legal byte. Whitespace is REJECTED — see the
        // comment on `allowsWhitespaceHere` for why we minify.
        XCTAssertEqual(mask[Int(UInt8(ascii: "{"))], 0)
        XCTAssertEqual(mask[Int(UInt8(ascii: " "))], -.infinity)
        XCTAssertEqual(mask[Int(UInt8(ascii: "a"))], -.infinity)
        XCTAssertEqual(mask[Int(UInt8(ascii: "9"))], -.infinity)
    }

    func test_byteLevelMasker_completeFSMmasksAll() {
        let fsm = JSONSchemaFSM(rootSchema: .boolean)
        XCTAssertTrue(fsm.acceptString("true"))
        XCTAssertTrue(fsm.isComplete)
        let masker = LogitsMasker(vocabSize: 256, eosTokenId: nil) { id in
            return String(decoding: [UInt8(id)], as: UTF8.self)
        }
        let mask = masker.mask(forFSM: fsm)
        // No EOS provided; complete state should mask everything to -inf.
        for v in mask {
            XCTAssertEqual(v, -.infinity)
        }
    }

    func test_byteLevelMasker_eosAllowedWhenComplete() {
        let fsm = JSONSchemaFSM(rootSchema: .boolean)
        XCTAssertTrue(fsm.acceptString("true"))
        let masker = LogitsMasker(vocabSize: 256, eosTokenId: 42) { id in
            return String(decoding: [UInt8(id)], as: UTF8.self)
        }
        let mask = masker.mask(forFSM: fsm)
        XCTAssertEqual(mask[42], 0)
        XCTAssertEqual(mask[41], -.infinity)
    }
}
