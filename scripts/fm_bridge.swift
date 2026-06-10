// fm_bridge.swift — line-protocol bridge to Apple Foundation Models.
//
// Reads one JSON request per stdin line: {"system": "...", "user": "..."}
// Writes one JSON response per stdout line: {"text": "..."} or {"error": "..."}
//
// Uses GUIDED generation (@Generable) shaped like the Pace v11 response
// schema — this mirrors how Pace's production AppleFoundationModelsPlanner-
// Client consumes FM (typed output), so the gate measures judgment, not
// JSON-formatting compliance. First plain-text run scored 0/60 on unhappy
// dims while refusing every OOS prompt CORRECTLY in prose — format failure,
// not capability.
//
// Compile: swiftc -O scripts/fm_bridge.swift -o /tmp/fm_bridge

import Foundation
import FoundationModels

struct Request: Decodable {
    let system: String
    let user: String
}

@Generable
enum PaceIntent: String {
    case action
    case answer
    case dictate
    case edit
    case out_of_scope
    case clarify
    case confirm_destructive
}

@Generable
struct PaceResponse {
    @Guide(description: "What TTS reads aloud. Short; the refusal for out_of_scope; the question for clarify; the confirmation request for confirm_destructive.")
    var spokenText: String

    @Guide(description: "Top-level decision route per the system prompt's intent rules.")
    var intent: PaceIntent

    @Guide(description: "For intent=action or confirm_destructive: the action name from the registry, e.g. AX.press, Mail.draft, App.launch.")
    var actionName: String?

    @Guide(description: "For AX.press: the on-screen element label to click. For other actions: the primary target.")
    var target: String?

    @Guide(description: "JSON object string with any additional action arguments beyond target, e.g. {\"to\":[\"__resolve:john\"],\"body\":\"...\"}.")
    var additionalArgsJSON: String?

    @Guide(description: "For intent=answer or dictate: the text content.")
    var text: String?

    @Guide(description: "For intent=out_of_scope: short machine-readable reason.")
    var reason: String?

    @Guide(description: "For intent=clarify: the one clarifying question.")
    var question: String?
}

setbuf(stdout, nil)

let model = SystemLanguageModel.default
guard case .available = model.availability else {
    print(#"{"error": "Apple Intelligence model unavailable"}"#)
    exit(2)
}
FileHandle.standardError.write("fm_bridge ready (guided)\n".data(using: .utf8)!)

func v11JSON(from r: PaceResponse) -> String {
    var payload: [String: Any] = [:]
    if let name = r.actionName, !name.isEmpty { payload["name"] = name }
    var args: [String: Any] = [:]
    if let target = r.target, !target.isEmpty { args["target"] = target }
    if let extra = r.additionalArgsJSON,
       let data = extra.data(using: .utf8),
       let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
        for (k, v) in parsed { args[k] = v }
    }
    if !args.isEmpty { payload["args"] = args }
    if let t = r.text, !t.isEmpty { payload["text"] = t }
    if let v = r.reason, !v.isEmpty { payload["reason"] = v }
    if let v = r.question, !v.isEmpty { payload["question"] = v }
    if r.intent == .confirm_destructive, let name = r.actionName {
        payload["action"] = name
        if let target = r.target { payload["target"] = target }
    }
    let doc: [String: Any] = [
        "spokenText": r.spokenText,
        "intent": r.intent.rawValue,
        "payload": payload,
    ]
    let data = try! JSONSerialization.data(withJSONObject: doc)
    return String(data: data, encoding: .utf8)!
}

func respond(_ request: Request) async -> String {
    do {
        let session = LanguageModelSession(model: model, instructions: request.system)
        let response = try await session.respond(to: request.user, generating: PaceResponse.self)
        return v11JSON(from: response.content)
    } catch {
        return "__FM_ERROR__: \(error)"
    }
}

while let line = readLine(strippingNewline: true) {
    guard !line.isEmpty else { continue }
    guard let data = line.data(using: .utf8),
          let request = try? JSONDecoder().decode(Request.self, from: data) else {
        print(#"{"error": "bad request line"}"#)
        continue
    }
    let semaphore = DispatchSemaphore(value: 0)
    var output = ""
    Task {
        output = await respond(request)
        semaphore.signal()
    }
    semaphore.wait()
    let payload: [String: String] = output.hasPrefix("__FM_ERROR__")
        ? ["error": output]
        : ["text": output]
    let encoded = try! JSONEncoder().encode(payload)
    print(String(data: encoded, encoding: .utf8)!)
}
