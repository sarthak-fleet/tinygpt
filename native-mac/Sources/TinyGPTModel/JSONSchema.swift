import Foundation

/// Minimal JSON Schema (Draft 7 subset) parser, just enough to drive a
/// character-level state machine for constrained generation.
///
/// SUPPORTED:
///   - `{"type": "object", "properties": {...}, "required": [...]}`
///   - `{"type": "string"}` (any string)
///   - `{"type": "string", "enum": ["a", "b"]}` (closed enum)
///   - `{"type": "number"}` / `{"type": "integer"}`
///   - `{"type": "boolean"}`
///   - `{"type": "null"}`
///   - `{"type": "array", "items": {...}}` (recursive items)
///   - top-level may be any of these
///
/// NOT supported (silently ignored, treated as no constraint):
///   - `$ref`, `oneOf`, `anyOf`, `allOf`, `not`
///   - string `pattern`, `format`, `minLength`, `maxLength`
///   - number `minimum`, `maximum`, `multipleOf`
///   - object `additionalProperties: <schema>`, `patternProperties`
///   - tuple-form `items: [s1, s2, ...]`
///   - `const`
///
/// These can be added incrementally — the state machine is the hard part.
public enum JSONSchemaError: Error, CustomStringConvertible {
    case parse(String)
    case file(String, Error)

    public var description: String {
        switch self {
        case .parse(let m): return "JSON Schema parse error: \(m)"
        case .file(let p, let e): return "could not read schema at \(p): \(e)"
        }
    }
}

/// In-memory representation of the supported schema subset. Use `from(url:)`
/// or `from(data:)` to parse one out of a JSON Schema file.
public indirect enum JSONSchemaNode: Sendable {
    case object(properties: [(String, JSONSchemaNode)], required: Set<String>)
    case string(enumValues: [String]?)
    case number(integer: Bool)
    case boolean
    case null
    case array(items: JSONSchemaNode)
    /// Unrecognised — treated as "any JSON value". Used as a fallback so that
    /// unsupported schema features degrade to "no constraint at this point"
    /// rather than failing the run.
    case any

    /// Parse from JSON Schema bytes. `Draft-07` is the assumed dialect but we
    /// don't actually check `$schema`; we just walk the structure.
    public static func from(data: Data) throws -> JSONSchemaNode {
        let any: Any
        do { any = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) }
        catch { throw JSONSchemaError.parse("not valid JSON: \(error)") }
        return parse(any)
    }

    /// Parse from a JSON Schema file on disk.
    public static func from(url: URL) throws -> JSONSchemaNode {
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw JSONSchemaError.file(url.path, error) }
        return try from(data: data)
    }

    private static func parse(_ any: Any) -> JSONSchemaNode {
        guard let dict = any as? [String: Any] else { return .any }
        // Handle `enum` even when `type` isn't set — most enum schemas pin
        // it to string anyway, and we don't yet support typed-enum mixes.
        if let enumArr = dict["enum"] as? [Any] {
            let strings = enumArr.compactMap { $0 as? String }
            if !strings.isEmpty { return .string(enumValues: strings) }
        }
        guard let type = dict["type"] as? String else { return .any }
        switch type {
        case "object":
            // JSONSerialization gives us an unordered dictionary; that's
            // fine for properties since JSON object key order is itself
            // unordered. We serialise back in the order properties were
            // declared if the user passed an array-of-pairs, but the
            // standard JSON form here is a dict — we just sort by key
            // for stable output across runs.
            let propsRaw = (dict["properties"] as? [String: Any]) ?? [:]
            let props = propsRaw.keys.sorted().map { ($0, parse(propsRaw[$0] as Any)) }
            let req = Set((dict["required"] as? [String]) ?? [])
            return .object(properties: props, required: req)
        case "string":
            return .string(enumValues: nil)
        case "number":
            return .number(integer: false)
        case "integer":
            return .number(integer: true)
        case "boolean":
            return .boolean
        case "null":
            return .null
        case "array":
            let items = parse(dict["items"] as Any)
            return .array(items: items)
        default:
            return .any
        }
    }
}
