import Foundation

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

    func grammarSpec() throws -> ServeGrammarSpec {
        try ServeGrammarSpec.parse(outputSchemaJSON())
    }

    func outputSchemaJSON() throws -> String {
        let schema: [String: Any] = [
            "type": "object",
            "required": ["verb", "args"],
            "properties": [
                "verb": [
                    "type": "string",
                    "enum": tools.map(\.name).sorted()
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
