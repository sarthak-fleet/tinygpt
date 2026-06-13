import Foundation

/// How the tool catalog is presented to the model.
///
/// - `.full`: every tool's full JSON schema is injected into the system
///   prompt at boot (the original behavior; default for back-compat).
/// - `.deferred`: only a one-line-per-tool index is injected; schemas are
///   fetched on demand via the built-in `get_tool_info(name)` meta-tool.
///   Serve intercepts that call (non-streaming chat completions only),
///   appends the schema, and re-prompts (capped at 3 hops).
///   See docs/prds/B26-deferred-tools.md.
enum ServeToolMode: String {
    case full
    case deferred
}

struct ServeToolsSpec {
    struct Tool {
        let name: String
        let description: String
        let parameters: [String: Any]
    }

    let tools: [Tool]

    static func load(path: String) throws -> ServeToolsSpec {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try parse(data: data)
    }

    static func parse(data: Data) throws -> ServeToolsSpec {
        let any = try JSONSerialization.jsonObject(with: data, options: [])
        let rawTools: [Any]
        if let obj = any as? [String: Any], let tools = obj["tools"] as? [Any] {
            rawTools = tools
        } else if let arr = any as? [Any] {
            rawTools = arr
        } else {
            throw NSError(domain: "tinygpt.serve.tools", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "tools must be {\"tools\":[...]} or a raw tools array"])
        }

        var parsed: [Tool] = []
        for (i, raw) in rawTools.enumerated() {
            guard let entry = raw as? [String: Any] else {
                throw toolError("tool #\(i) is not an object")
            }
            let fn: [String: Any]
            if let inner = entry["function"] as? [String: Any] {
                fn = inner
            } else {
                fn = entry
            }
            guard let name = fn["name"] as? String, !name.isEmpty else {
                throw toolError("tool #\(i) missing function.name")
            }
            let description = (fn["description"] as? String) ?? ""
            let parameters = (fn["parameters"] as? [String: Any]) ?? [
                "type": "object",
                "properties": [:],
                "required": []
            ]
            parsed.append(Tool(name: name, description: description, parameters: parameters))
        }
        guard !parsed.isEmpty else { throw toolError("empty tools list") }
        return ServeToolsSpec(tools: parsed)
    }

    func systemPrompt() -> String {
        """
        You have these tools available. Use exactly one tool call to fulfill the user intent.

        Tools:
        \(canonicalToolsJSON())

        Emit exactly one JSON object:
        {
          "verb": "<one tool name from the tools list>",
          "args": { ... arguments for that tool ... },
          "spoken_text": "<brief text to say to the user, optional>"
        }
        Do not include Markdown or explanatory text outside the JSON object.
        """
    }

    /// Deferred-mode system prompt (B26): emits only a one-line-per-tool
    /// index plus the `get_tool_info(name)` meta-tool contract. The full
    /// schema for each tool is fetched on demand — serve intercepts the
    /// `verb=get_tool_info` reply, appends a synthetic tool result with
    /// the schema, and re-prompts the model. See docs/prds/B26-deferred-tools.md.
    func compactSystemPrompt() -> String {
        let index = tools
            .sorted { $0.name < $1.name }
            .map { tool -> String in
                let first = tool.description
                    .split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: true)
                    .first.map(String.init) ?? ""
                let desc = first.isEmpty ? "(no description)" : first
                return "  \(tool.name) — \(desc)"
            }
            .joined(separator: "\n")
        return """
        You have these tools available. Use exactly one tool call to fulfill the user intent.

        Tool index (name — purpose):
        \(index)
          get_tool_info — fetch the full JSON schema for one tool by name. The result is appended to the conversation before you act.

        If the index entry is enough to call the tool correctly, call it directly. Otherwise call get_tool_info first; the next assistant turn carries the schema.

        Emit exactly one JSON object:
        {
          "verb": "<tool name from the index above, or 'get_tool_info'>",
          "args": { ... arguments for that tool ... },
          "spoken_text": "<brief text to say to the user, optional>"
        }
        For get_tool_info: args = {"name": "<tool name from the index>"}.
        Do not include Markdown or explanatory text outside the JSON object.
        """
    }

    func grammarSpec() throws -> ServeGrammarSpec {
        try ServeGrammarSpec.parse(outputSchemaJSON())
    }

    /// Deferred-mode grammar: same envelope as `grammarSpec()` but the
    /// verb enum is extended with `get_tool_info`.
    func compactGrammarSpec() throws -> ServeGrammarSpec {
        try ServeGrammarSpec.parse(compactOutputSchemaJSON())
    }

    /// JSON schema for one tool by name. Returns nil for unknown names —
    /// caller emits an error sentinel back to the model so it retries
    /// with a name from the index.
    func toolInfo(name: String) -> String? {
        guard let tool = tools.first(where: { $0.name == name }) else { return nil }
        let payload: [String: Any] = [
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        ]
        return try? Self.jsonString(payload)
    }

    func outputSchemaJSON() throws -> String {
        try outputSchemaJSON(verbEnum: tools.map(\.name).sorted())
    }

    func compactOutputSchemaJSON() throws -> String {
        var verbs = tools.map(\.name)
        verbs.append("get_tool_info")
        return try outputSchemaJSON(verbEnum: verbs.sorted())
    }

    private func outputSchemaJSON(verbEnum: [String]) throws -> String {
        let schema: [String: Any] = [
            "type": "object",
            "required": ["verb", "args"],
            "properties": [
                "verb": [
                    "type": "string",
                    "enum": verbEnum
                ],
                "args": [
                    "type": "object",
                    "properties": [:],
                    "required": []
                ],
                "spoken_text": [
                    "type": "string"
                ]
            ]
        ]
        return try jsonString(schema)
    }

    private func canonicalToolsJSON() -> String {
        let arr: [[String: Any]] = tools
            .sorted { $0.name < $1.name }
            .map { tool in
                [
                    "type": "function",
                    "function": [
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    ]
                ]
            }
        return (try? Self.jsonString(["tools": arr])) ?? "{\"tools\":[]}"
    }

    private static func jsonString(_ object: Any) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys]
        )
        return String(decoding: data, as: UTF8.self)
    }

    private func jsonString(_ object: Any) throws -> String {
        try Self.jsonString(object)
    }

    private static func toolError(_ message: String) -> NSError {
        NSError(domain: "tinygpt.serve.tools", code: 2,
                userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func loadableJSONObject(_ object: Any) -> Any {
        object
    }
}
