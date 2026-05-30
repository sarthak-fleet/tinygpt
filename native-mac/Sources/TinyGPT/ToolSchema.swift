import Foundation

/// OpenAI-compatible tool schema for the `tinygpt agent` runtime.
///
/// # Format
///
/// The agent reads a JSON file shaped like the OpenAI ChatCompletions
/// `tools` array. Every tool has a `function` block with a `name`,
/// `description`, and JSON-Schema `parameters`. tinygpt extends this
/// with two sibling fields that are otherwise ignored by OpenAI:
///
///   - `_exec`        — bash command template. `$arg` substitution maps
///                       to argument names (typed string, coerced if the
///                       JSON-Schema says so). Example: `cat "$path"`.
///   - `_exec_args`   — explicit ordered list of argument names that
///                       `_exec` references. If absent, derived from the
///                       parameters object's `properties` keys.
///
/// ```json
/// {
///   "tools": [
///     {
///       "type": "function",
///       "function": {
///         "name": "read_file",
///         "description": "Read file contents",
///         "parameters": {
///           "type": "object",
///           "properties": { "path": { "type": "string" } },
///           "required": ["path"]
///         },
///         "_exec": "cat \"$path\"",
///         "_exec_args": ["path"]
///       }
///     }
///   ]
/// }
/// ```
///
/// Future: `_handler` can name a Swift callback registered at startup so
/// host applications can plug tools in without going through subprocess.
/// The struct already carries it as a string; the executor will inspect
/// the registry when the field is present.
public struct ToolSchema {

    /// A single tool definition.
    public struct Tool {
        public let name: String
        public let description: String
        public let parameters: ParameterSpec
        /// Bash command template, when this tool is a subprocess. The
        /// executor substitutes `$arg` tokens at runtime.
        public let exec: String?
        /// Explicit ordered list of argument names the `_exec` template
        /// references. When `nil` the executor falls back to
        /// `parameters.properties` key order.
        public let execArgs: [String]?
        /// Name of a Swift-side handler that overrides `_exec`. Reserved
        /// for future use — the executor errors out today if no `_exec`
        /// is also present.
        public let handler: String?
    }

    /// Subset of JSON Schema we model. `tinygpt agent` only needs a
    /// flat object-of-scalars; nested object schemas pass through
    /// untouched so the schema is still emittable in the system prompt.
    ///
    /// `raw` is the verbatim JSON object; carrying it as `[String: Any]`
    /// would force the type non-`Sendable`, and we don't need Sendable
    /// for the agent runtime (single-threaded loop) — so the struct is
    /// not declared Sendable.
    public struct ParameterSpec {
        public let type: String                       // usually "object"
        public let properties: [String: PropertySpec] // arg-name → spec
        public let required: [String]
        public let raw: [String: Any]                 // verbatim JSON, for prompting
    }

    public struct PropertySpec: Sendable {
        public let type: String        // "string" / "integer" / "number" / "boolean"
        public let description: String?
    }

    public let tools: [Tool]

    /// Parse a JSON file into a `ToolSchema`. Accepts either the OpenAI
    /// shape `{ "tools": [...] }` or a raw `[...]` array at the top
    /// level — both common in the wild.
    public static func load(from url: URL) throws -> ToolSchema {
        let data = try Data(contentsOf: url)
        return try parse(data: data)
    }

    public static func parse(data: Data) throws -> ToolSchema {
        let any = try JSONSerialization.jsonObject(with: data, options: [])
        let rawTools: [Any]
        if let obj = any as? [String: Any], let arr = obj["tools"] as? [Any] {
            rawTools = arr
        } else if let arr = any as? [Any] {
            rawTools = arr
        } else {
            throw ToolSchemaError.malformed("top-level must be {\"tools\": [...]} or [...]")
        }
        var tools: [Tool] = []
        for (i, item) in rawTools.enumerated() {
            guard let entry = item as? [String: Any] else {
                throw ToolSchemaError.malformed("tool #\(i) is not an object")
            }
            tools.append(try parseTool(entry, index: i))
        }
        if tools.isEmpty {
            throw ToolSchemaError.malformed("no tools defined in schema")
        }
        return ToolSchema(tools: tools)
    }

    private static func parseTool(_ entry: [String: Any], index: Int) throws -> Tool {
        // Tolerate both `{ "function": {...} }` (OpenAI shape) and a flat
        // form where the function fields live at the top level (some
        // hand-written schemas do this).
        let f: [String: Any]
        if let inner = entry["function"] as? [String: Any] {
            f = inner
        } else if entry["name"] != nil {
            f = entry
        } else {
            throw ToolSchemaError.malformed("tool #\(index) missing 'function' or 'name'")
        }
        guard let name = f["name"] as? String, !name.isEmpty else {
            throw ToolSchemaError.malformed("tool #\(index) missing 'name'")
        }
        let description = (f["description"] as? String) ?? ""
        let params = try parseParameters(f["parameters"], toolName: name)
        let exec = f["_exec"] as? String
        let execArgs = f["_exec_args"] as? [String]
        let handler = f["_handler"] as? String
        if exec == nil && handler == nil {
            // Permit it — the executor will refuse to run the tool with a
            // clear error message. Some users may want to load a schema
            // that the model can SEE but the host fulfills outside the
            // tinygpt loop.
        }
        return Tool(
            name: name,
            description: description,
            parameters: params,
            exec: exec,
            execArgs: execArgs,
            handler: handler
        )
    }

    private static func parseParameters(_ raw: Any?, toolName: String) throws -> ParameterSpec {
        guard let obj = raw as? [String: Any] else {
            // Tool with no parameters — accept it.
            return ParameterSpec(
                type: "object", properties: [:], required: [], raw: [:])
        }
        let type = (obj["type"] as? String) ?? "object"
        let propsRaw = (obj["properties"] as? [String: Any]) ?? [:]
        var props: [String: PropertySpec] = [:]
        for (k, v) in propsRaw {
            guard let pv = v as? [String: Any] else {
                throw ToolSchemaError.malformed(
                    "tool '\(toolName)' property '\(k)' is not an object")
            }
            props[k] = PropertySpec(
                type: (pv["type"] as? String) ?? "string",
                description: pv["description"] as? String
            )
        }
        let required = (obj["required"] as? [String]) ?? []
        return ParameterSpec(type: type, properties: props,
                              required: required, raw: obj)
    }

    /// Render the schema as a compact textual description suitable for
    /// inclusion in the system prompt. Output looks like:
    ///
    /// ```
    /// - read_file(path: string)  Read file contents
    /// - run_test(name: string, timeout?: integer)  Run a test
    /// ```
    public func systemPromptDescription() -> String {
        var lines: [String] = []
        for tool in tools {
            var argParts: [String] = []
            let required = Set(tool.parameters.required)
            // Preserve a stable iteration order: required first, then the
            // rest alphabetically. JSON property dictionaries don't
            // guarantee insertion order across launches.
            let sortedKeys =
                tool.parameters.required +
                tool.parameters.properties.keys
                    .filter { !required.contains($0) }
                    .sorted()
            for k in sortedKeys {
                let p = tool.parameters.properties[k]
                let opt = required.contains(k) ? "" : "?"
                argParts.append("\(k)\(opt): \(p?.type ?? "string")")
            }
            let head = "- \(tool.name)(\(argParts.joined(separator: ", ")))"
            if tool.description.isEmpty {
                lines.append(head)
            } else {
                lines.append("\(head)  \(tool.description)")
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Render the schema as the literal JSON the agent's system prompt
    /// embeds. We re-serialise rather than echoing the source text so
    /// the prompt is canonicalised (consistent prefix-cache hashing).
    public func canonicalJSONForPrompt() -> String {
        // Build a minimal representation: name + description + parameters
        // raw object. We deliberately drop `_exec` / `_exec_args` so the
        // model doesn't see them — they're a host concern.
        var arr: [[String: Any]] = []
        for tool in tools {
            arr.append([
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.raw,
                ]
            ])
        }
        let payload: [String: Any] = ["tools": arr]
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: payload,
                options: [.prettyPrinted, .sortedKeys]),
            let s = String(data: data, encoding: .utf8)
        else {
            return "[]"
        }
        return s
    }
}

public enum ToolSchemaError: Error, CustomStringConvertible {
    case malformed(String)
    case unknownTool(String)
    case missingRequired(tool: String, arg: String)
    public var description: String {
        switch self {
        case .malformed(let m): return "tool schema malformed: \(m)"
        case .unknownTool(let n): return "unknown tool: \(n)"
        case .missingRequired(let t, let a):
            return "tool '\(t)' missing required argument '\(a)'"
        }
    }
}
