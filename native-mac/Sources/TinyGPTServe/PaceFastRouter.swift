import Foundation

struct PaceRouteElement {
    let id: Int
    let role: String
    let x: Int
    let y: Int
    let label: String
    let text: String
}

struct PaceFastRouteResult {
    let verb: String
    let confidence: Double
    let reason: String
    let targetId: Int?
    let text: String?
    let key: String?
    let direction: String?
    let appName: String?
    let actionTags: [String]

    var fallback: Bool {
        verb == "escalate"
    }

    func payload(elements: [PaceRouteElement], latencyMs: Double, freeText: Bool) -> [String: Any] {
        let byId = Dictionary(uniqueKeysWithValues: elements.map { ($0.id, $0) })
        let target = targetId.flatMap { byId[$0] }
        var out: [String: Any] = [
            "verb": verb,
            "confidence": confidence,
            "reason": reason,
            "fallback": fallback,
            "latency_ms": latencyMs,
            "spoken_text": spokenText(target: target, freeText: freeText),
            "action_tags": actionTags,
        ]
        out["target_id"] = jsonValue(targetId)
        out["target_label"] = jsonValue(target?.label)
        out["x"] = jsonValue(target?.x)
        out["y"] = jsonValue(target?.y)
        out["text"] = jsonValue(text)
        out["key"] = jsonValue(key)
        out["direction"] = jsonValue(direction)
        out["app_name"] = jsonValue(appName)
        return out
    }

    private func jsonValue(_ value: Any?) -> Any {
        value ?? NSNull()
    }

    private func spokenText(target: PaceRouteElement?, freeText: Bool) -> String {
        if freeText && !actionTags.isEmpty {
            return actionTags.joined(separator: " ")
        }
        switch verb {
        case "key":
            return "[KEY:\(key ?? "")]"
        case "scroll":
            return "[SCROLL:\(direction ?? "down")]"
        case "type":
            return "[TYPE:\(text ?? "")]"
        case "open_app":
            return "[OPEN_APP:\(appName ?? "")]"
        case "click":
            guard let target else { return "i can't see that on this screen" }
            if freeText { return "[CLICK:\(target.x),\(target.y)]" }
            return "opening the \(target.label)"
        case "answer":
            return text ?? ""
        default:
            return "i can't see that on this screen"
        }
    }
}

enum PaceFastRouter {
    private static let stopwords: Set<String> = [
        "a", "an", "and", "at", "button", "can", "click", "could", "i",
        "it", "like", "maybe", "menu", "open", "press", "select", "tap",
        "the", "thing", "to", "uh", "yeah", "you",
    ]

    static func route(user: String, elements: [PaceRouteElement]) -> PaceFastRouteResult {
        let normalized = normalize(user)

        if regexMatch(normalized, #"\b(command|cmd)\s*\+?\s*s\b"#) || normalized.contains("save shortcut") {
            return PaceFastRouteResult(verb: "key", confidence: 0.99, reason: "key_shortcut",
                                       targetId: nil, text: nil, key: "cmd+s", direction: nil,
                                       appName: nil, actionTags: [])
        }

        if normalized.hasPrefix("scroll ") || " \(normalized) ".contains(" scroll ") {
            let direction = " \(normalized) ".contains(" up") ? "up" : "down"
            return PaceFastRouteResult(verb: "scroll", confidence: 0.98, reason: "scroll_direction",
                                       targetId: nil, text: nil, key: nil, direction: direction,
                                       appName: nil, actionTags: [])
        }

        if normalized.hasPrefix("open ") && !normalized.contains("menu") && !normalized.contains("button") {
            let appName = user.split(separator: " ", maxSplits: 1).dropFirst().first.map(String.init) ?? ""
            return PaceFastRouteResult(verb: "open_app", confidence: 0.85, reason: "open_app_phrase",
                                       targetId: nil, text: nil, key: nil, direction: nil,
                                       appName: appName.trimmingCharacters(in: .whitespacesAndNewlines),
                                       actionTags: [])
        }

        if normalized.contains("who are you") || normalized.contains("are you siri") {
            return PaceFastRouteResult(verb: "answer", confidence: 0.99, reason: "identity_allowlist",
                                       targetId: nil, text: "i'm pace", key: nil, direction: nil,
                                       appName: nil, actionTags: [])
        }

        if normalized.contains("what is html") {
            return PaceFastRouteResult(
                verb: "answer", confidence: 0.95, reason: "tiny_qa_allowlist",
                targetId: nil, text: "html is the markup language used to structure web pages",
                key: nil, direction: nil, appName: nil, actionTags: [])
        }
        if normalized.contains("what is css") {
            return PaceFastRouteResult(
                verb: "answer", confidence: 0.95, reason: "tiny_qa_allowlist",
                targetId: nil, text: "css is the language used to style web pages",
                key: nil, direction: nil, appName: nil, actionTags: [])
        }

        if normalized.contains("describe") || normalized.contains("what does this screen show") {
            let labels = elements.prefix(4).map(\.label).filter { !$0.isEmpty }
            let text = labels.isEmpty ? "i can't see the screen" : "this screen shows " + labels.joined(separator: ", ")
            return PaceFastRouteResult(verb: "answer", confidence: 0.82, reason: "screen_summary",
                                       targetId: nil, text: text, key: nil, direction: nil,
                                       appName: nil, actionTags: [])
        }

        if normalized.contains("type ") {
            if let typed = extractTypeText(user) {
                if normalized.contains("click") {
                    let chosen = chooseTarget(user: user, elements: elements)
                    var tags: [String] = []
                    if let target = chosen.element {
                        tags.append("[CLICK:\(target.x),\(target.y)]")
                    }
                    tags.append("[TYPE:\(typed)]")
                    return PaceFastRouteResult(
                        verb: "click",
                        confidence: min(chosen.confidence, 0.96),
                        reason: "click_then_type",
                        targetId: chosen.element?.id,
                        text: typed,
                        key: nil,
                        direction: nil,
                        appName: nil,
                        actionTags: tags)
                }
                return PaceFastRouteResult(verb: "type", confidence: 0.98, reason: "type_phrase",
                                           targetId: nil, text: typed, key: nil, direction: nil,
                                           appName: nil, actionTags: [])
            }
        }

        let words = Set(normalized.split(separator: " ").map(String.init))
        if !words.intersection(["click", "tap", "select", "press"]).isEmpty {
            let chosen = chooseTarget(user: user, elements: elements)
            guard let target = chosen.element else {
                return PaceFastRouteResult(verb: "escalate", confidence: 0.2,
                                           reason: "missing_or_ambiguous_target",
                                           targetId: nil, text: nil, key: nil, direction: nil,
                                           appName: nil, actionTags: [])
            }
            return PaceFastRouteResult(verb: "click", confidence: chosen.confidence,
                                       reason: "deterministic_label_match",
                                       targetId: target.id, text: nil, key: nil, direction: nil,
                                       appName: nil, actionTags: [])
        }

        return PaceFastRouteResult(verb: "escalate", confidence: 0.35, reason: "not_obvious_action",
                                   targetId: nil, text: nil, key: nil, direction: nil,
                                   appName: nil, actionTags: [])
    }

    private static func chooseTarget(user: String,
                                     elements: [PaceRouteElement]) -> (element: PaceRouteElement?, confidence: Double)
    {
        guard !elements.isEmpty else { return (nil, 0.0) }
        let normalizedUser = normalize(user)
        if normalizedUser.contains("second tab") {
            if let match = elements.first(where: {
                $0.role == "tab" && (normalize($0.label).contains("second") || $0.id == 1)
            }) {
                return (match, 0.99)
            }
        }

        let target = compactTargetPhrase(user)
        let scored = elements.map { (scoreElement(user: user, targetPhrase: target, element: $0), $0) }
            .sorted { lhs, rhs in
                if lhs.0 == rhs.0 { return lhs.1.id < rhs.1.id }
                return lhs.0 > rhs.0
            }
        guard let best = scored.first else { return (nil, 0.0) }
        if best.0 < 0.58 { return (nil, best.0) }
        return (best.1, min(best.0, 0.99))
    }

    private static func scoreElement(user: String, targetPhrase: String, element: PaceRouteElement) -> Double {
        let label = normalize(element.label)
        let text = normalize(element.text)
        let target = normalize(targetPhrase)
        let userTokens = contentTokens(user)
        let labelTokens = contentTokens(label)
        let textTokens = contentTokens(text)

        var score = 0.0
        if !target.isEmpty && (target == label || target == text) {
            score += 1.0
        }
        if !target.isEmpty && (!label.isEmpty && (target.contains(label) || label.contains(target))) {
            score += 0.55
        }
        if !target.isEmpty && (!text.isEmpty && (target.contains(text) || text.contains(target))) {
            score += 0.25
        }
        if !labelTokens.isEmpty {
            score += 0.7 * Double(userTokens.intersection(labelTokens).count) / Double(labelTokens.count)
        }
        if !textTokens.isEmpty {
            score += 0.25 * Double(userTokens.intersection(textTokens).count) / Double(textTokens.count)
        }
        if !target.isEmpty && !label.isEmpty {
            score += 0.35 * stringSimilarity(target, label)
        }
        if ["button", "tab", "menu_item", "text_field", "text_area"].contains(element.role) {
            score += 0.03
        }
        return score
    }

    private static func normalize(_ value: String) -> String {
        var scalars: [UnicodeScalar] = []
        var lastWasSpace = true
        for scalar in value.lowercased().unicodeScalars {
            let keep = CharacterSet.alphanumerics.contains(scalar) || scalar == "+" || scalar == " "
            if keep {
                if scalar == " " {
                    if !lastWasSpace {
                        scalars.append(" ")
                        lastWasSpace = true
                    }
                } else {
                    scalars.append(scalar)
                    lastWasSpace = false
                }
            } else if !lastWasSpace {
                scalars.append(" ")
                lastWasSpace = true
            }
        }
        return String(String.UnicodeScalarView(scalars)).trimmingCharacters(in: .whitespaces)
    }

    private static func contentTokens(_ value: String) -> Set<String> {
        Set(normalize(value).split(separator: " ").map(String.init).filter {
            !$0.isEmpty && !stopwords.contains($0)
        })
    }

    private static func compactTargetPhrase(_ user: String) -> String {
        var lowered = normalize(user)
        if let range = lowered.range(of: #"\band\s+type\b"#, options: .regularExpression) {
            lowered = String(lowered[..<range.lowerBound])
        }
        lowered = regexReplace(lowered, pattern: #"^(can you |could you |please )+"#, replacement: "")
        lowered = regexReplace(lowered, pattern: #"^(click|tap|select|press|open)\s+"#, replacement: "")
        return lowered.trimmingCharacters(in: .whitespaces)
    }

    private static func extractTypeText(_ user: String) -> String? {
        guard let range = user.range(of: #"\btype\s+(.+)$"#, options: [.regularExpression, .caseInsensitive]) else {
            return nil
        }
        var value = String(user[range]).replacingOccurrences(
            of: #"^\s*type\s+"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        value = value.replacingOccurrences(
            of: #"^(the text|text)\s+"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func stringSimilarity(_ lhs: String, _ rhs: String) -> Double {
        let a = Array(lhs)
        let b = Array(rhs)
        if a.isEmpty && b.isEmpty { return 1.0 }
        if a.isEmpty || b.isEmpty { return 0.0 }
        var previous = Array(0...b.count)
        var current = Array(repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            current[0] = i
            for j in 1...b.count {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            }
            swap(&previous, &current)
        }
        let distance = previous[b.count]
        return 1.0 - (Double(distance) / Double(max(a.count, b.count)))
    }

    private static func regexMatch(_ value: String, _ pattern: String) -> Bool {
        value.range(of: pattern, options: .regularExpression) != nil
    }

    private static func regexReplace(_ value: String, pattern: String, replacement: String) -> String {
        value.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
    }
}
