import Foundation

/// `tinygpt synthesize` labels prompt rows with any OpenAI-compatible teacher
/// endpoint and writes distillation-ready `{input, output, _meta}` JSONL.
enum Synthesize {
    struct Example {
        let index: Int
        let input: String
    }

    struct Config {
        var teacherBaseURL: String?
        var teacherModel: String?
        var inputsPath: String?
        var inputField = "prompt"
        var systemPrompt: String?
        var schemaPath: String?
        var grammarPath: String?
        var maxTokens = 256
        var temperature = 0.0
        var parallel = 1
        var rateLimit: Double = 0
        var retries = 2
        var timeout: TimeInterval = 120
        var outPath: String?
    }

    static func run(args: [String]) {
        var cfg = Config()
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--teacher":
                guard i + 1 < args.count else { exitUsage() }
                cfg.teacherBaseURL = args[i + 1]; i += 2
            case "--teacher-model", "--model":
                guard i + 1 < args.count else { exitUsage() }
                cfg.teacherModel = args[i + 1]; i += 2
            case "--inputs":
                guard i + 1 < args.count else { exitUsage() }
                cfg.inputsPath = args[i + 1]; i += 2
            case "--input-field":
                guard i + 1 < args.count else { exitUsage() }
                cfg.inputField = args[i + 1]; i += 2
            case "--system":
                guard i + 1 < args.count else { exitUsage() }
                cfg.systemPrompt = args[i + 1]; i += 2
            case "--system-file":
                guard i + 1 < args.count else { exitUsage() }
                cfg.systemPrompt = tryReadText(args[i + 1]); i += 2
            case "--schema":
                guard i + 1 < args.count else { exitUsage() }
                cfg.schemaPath = args[i + 1]; i += 2
            case "--grammar":
                guard i + 1 < args.count else { exitUsage() }
                cfg.grammarPath = args[i + 1]; i += 2
            case "--max-tokens":
                guard i + 1 < args.count else { exitUsage() }
                cfg.maxTokens = max(1, Int(args[i + 1]) ?? cfg.maxTokens); i += 2
            case "--temperature", "--temp":
                guard i + 1 < args.count else { exitUsage() }
                cfg.temperature = Double(args[i + 1]) ?? cfg.temperature; i += 2
            case "--parallel":
                guard i + 1 < args.count else { exitUsage() }
                cfg.parallel = max(1, Int(args[i + 1]) ?? cfg.parallel); i += 2
            case "--rate-limit":
                guard i + 1 < args.count else { exitUsage() }
                cfg.rateLimit = max(0, Double(args[i + 1]) ?? cfg.rateLimit); i += 2
            case "--retries":
                guard i + 1 < args.count else { exitUsage() }
                cfg.retries = max(0, Int(args[i + 1]) ?? cfg.retries); i += 2
            case "--timeout":
                guard i + 1 < args.count else { exitUsage() }
                cfg.timeout = max(1, TimeInterval(Double(args[i + 1]) ?? cfg.timeout)); i += 2
            case "--out":
                guard i + 1 < args.count else { exitUsage() }
                cfg.outPath = args[i + 1]; i += 2
            case "-h", "--help":
                exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr)
                exitUsage()
            }
        }

        guard let teacher = cfg.teacherBaseURL else { fputs("--teacher required\n", stderr); exitUsage() }
        guard let model = cfg.teacherModel else { fputs("--teacher-model required\n", stderr); exitUsage() }
        guard let inputs = cfg.inputsPath else { fputs("--inputs required\n", stderr); exitUsage() }
        guard let out = cfg.outPath else { fputs("--out required\n", stderr); exitUsage() }

        let schemaText = cfg.schemaPath.flatMap { tryReadText($0) }
        let schemaObject = schemaText.flatMap { parseJSON($0.data(using: .utf8) ?? Data()) }
        if cfg.schemaPath != nil && schemaObject == nil {
            fputs("schema parse failed: \(cfg.schemaPath!)\n", stderr)
            exit(2)
        }
        let grammarText = cfg.grammarPath.flatMap { tryReadText($0) }
        let examples = loadExamples(path: inputs, field: cfg.inputField)
        let existing = loadCompletedInputs(path: out)
        let pending = examples.filter { !existing.contains($0.input) }

        do {
            let writer = try JSONLAppendWriter(path: out)
            let client = OpenAICompatClient(
                teacherBaseURL: teacher,
                model: model,
                apiKey: ProcessInfo.processInfo.environment["TEACHER_API_KEY"]
                    ?? ProcessInfo.processInfo.environment["OPENAI_API_KEY"],
                timeout: cfg.timeout
            )
            let limiter = RateLimiter(perSecond: cfg.rateLimit)
            let progress = SynthesizeProgress(total: pending.count, skipped: existing.count)
            let startedAt = isoNow()
            let queue = OperationQueue()
            queue.maxConcurrentOperationCount = cfg.parallel

            fputs("[synthesize] loaded \(examples.count) inputs · skipping \(existing.count) already labeled · pending \(pending.count)\n", stderr)
            if pending.isEmpty {
                fputs("[synthesize] nothing to do\n", stderr)
                return
            }

            for ex in pending {
                queue.addOperation {
                    var finalError: String?
                    for attempt in 0...cfg.retries {
                        if attempt > 0 {
                            progress.recordRetry()
                            Thread.sleep(forTimeInterval: min(2.0, 0.25 * Double(attempt)))
                        }
                        limiter.wait()
                        let t0 = Date()
                        do {
                            var userPrompt = ex.input
                            var system = cfg.systemPrompt
                            if let schemaText {
                                let schemaInstruction = """

                                Respond only with valid JSON matching this JSON Schema:
                                \(schemaText)
                                """
                                if let existingSystem = system, !existingSystem.isEmpty {
                                    system = existingSystem + schemaInstruction
                                } else {
                                    system = schemaInstruction.trimmingCharacters(in: .whitespacesAndNewlines)
                                }
                                userPrompt += "\n\nReturn JSON only. Do not wrap it in markdown."
                            }

                            let response = try client.chat(
                                system: system,
                                user: userPrompt,
                                maxTokens: cfg.maxTokens,
                                temperature: cfg.temperature,
                                grammar: grammarText
                            )
                            var output = response.content
                            if let schemaObject {
                                guard let candidate = jsonCandidate(from: output),
                                      let parsed = parseJSON(candidate.data(using: .utf8) ?? Data()),
                                      validateJSON(parsed, against: schemaObject)
                                else {
                                    progress.recordSchemaFail()
                                    finalError = nil
                                    break
                                }
                                output = compactJSONString(parsed) ?? candidate
                            }
                            let latencyMs = Int((-t0.timeIntervalSinceNow) * 1000)
                            let meta: [String: Any] = [
                                "teacher_model": model,
                                "teacher_endpoint": teacher,
                                "timestamp": isoNow(),
                                "run_started_at": startedAt,
                                "input_index": ex.index,
                                "tokens_used": response.tokensUsed ?? NSNull(),
                                "latency_ms": latencyMs
                            ]
                            let row: [String: Any] = [
                                "input": ex.input,
                                "output": output,
                                "_meta": meta
                            ]
                            try writer.write(row)
                            progress.recordSuccess()
                            finalError = nil
                            break
                        } catch {
                            finalError = "\(error)"
                        }
                    }
                    if let finalError {
                        progress.recordError(finalError)
                    }
                }
            }
            queue.waitUntilAllOperationsAreFinished()
            progress.finish()
        } catch {
            fputs("synthesize failed: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func loadExamples(path: String, field: String) -> [Example] {
        let url = URL(fileURLWithPath: path)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            fputs("could not read inputs: \(path)\n", stderr)
            exit(1)
        }
        var rows: [Example] = []
        var missing = 0
        for (idx, line) in text.split(whereSeparator: \.isNewline).enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            guard let data = trimmed.data(using: .utf8),
                  let obj = parseJSON(data) as? [String: Any]
            else {
                fputs("warning: skipping invalid JSONL line \(idx + 1)\n", stderr)
                continue
            }
            if let value = obj[field] as? String, !value.isEmpty {
                rows.append(Example(index: idx, input: value))
            } else {
                missing += 1
            }
        }
        if missing > 0 {
            fputs("warning: skipped \(missing) rows missing string field '\(field)'\n", stderr)
        }
        return rows
    }

    private static func loadCompletedInputs(path: String) -> Set<String> {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path),
              let text = try? String(contentsOf: url, encoding: .utf8)
        else { return [] }
        var completed = Set<String>()
        for line in text.split(whereSeparator: \.isNewline) {
            guard let data = String(line).data(using: .utf8),
                  let obj = parseJSON(data) as? [String: Any],
                  let input = obj["input"] as? String
            else { continue }
            completed.insert(input)
        }
        return completed
    }

    private static func tryReadText(_ path: String) -> String? {
        do { return try String(contentsOfFile: path, encoding: .utf8) }
        catch {
            fputs("could not read \(path): \(error)\n", stderr)
            exit(1)
        }
    }

    private static func parseJSON(_ data: Data) -> Any? {
        try? JSONSerialization.jsonObject(with: data)
    }

    private static func compactJSONString(_ obj: Any) -> String? {
        guard JSONSerialization.isValidJSONObject(obj),
              let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes])
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func jsonCandidate(from text: String) -> String? {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            let lines = s.split(whereSeparator: \.isNewline).map(String.init)
            if lines.count >= 2 {
                s = lines.dropFirst().joined(separator: "\n")
                if let range = s.range(of: "```", options: .backwards) {
                    s = String(s[..<range.lowerBound])
                }
                s = s.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        if let first = s.firstIndex(where: { $0 == "{" || $0 == "[" }),
           let last = s.lastIndex(where: { $0 == "}" || $0 == "]" }),
           first <= last {
            return String(s[first...last])
        }
        return nil
    }

    private static func validateJSON(_ value: Any, against schema: Any) -> Bool {
        guard let schema = schema as? [String: Any] else { return true }
        if let enumValues = schema["enum"] as? [Any] {
            return enumValues.contains { jsonScalarEqual($0, value) }
        }
        if let typeSpec = schema["type"] {
            let allowed: [String]
            if let s = typeSpec as? String {
                allowed = [s]
            } else if let arr = typeSpec as? [String] {
                allowed = arr
            } else {
                allowed = []
            }
            if !allowed.isEmpty && !allowed.contains(where: { jsonTypeMatches(value, type: $0) }) {
                return false
            }
        }
        if let obj = value as? [String: Any] {
            if let required = schema["required"] as? [String] {
                for key in required where obj[key] == nil { return false }
            }
            if let props = schema["properties"] as? [String: Any] {
                for (key, childSchema) in props {
                    if let child = obj[key], !validateJSON(child, against: childSchema) {
                        return false
                    }
                }
            }
        }
        if let arr = value as? [Any], let itemSchema = schema["items"] {
            for item in arr where !validateJSON(item, against: itemSchema) {
                return false
            }
        }
        return true
    }

    private static func jsonTypeMatches(_ value: Any, type: String) -> Bool {
        switch type {
        case "object": return value is [String: Any]
        case "array": return value is [Any]
        case "string": return value is String
        case "number":
            return (value as? NSNumber).map { !isBool($0) } ?? false
        case "integer":
            guard let n = value as? NSNumber, !isBool(n) else { return false }
            return floor(n.doubleValue) == n.doubleValue
        case "boolean":
            return (value as? NSNumber).map(isBool) ?? false
        case "null":
            return value is NSNull
        default:
            return true
        }
    }

    private static func jsonScalarEqual(_ lhs: Any, _ rhs: Any) -> Bool {
        switch (lhs, rhs) {
        case (let a as String, let b as String): return a == b
        case (let a as NSNumber, let b as NSNumber): return a == b
        case (_ as NSNull, _ as NSNull): return true
        default: return false
        }
    }

    private static func isBool(_ value: NSNumber) -> Bool {
        CFGetTypeID(value) == CFBooleanGetTypeID()
    }

    private static func isoNow() -> String {
        ISO8601DateFormatter().string(from: Date())
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt synthesize --teacher <base-url> --teacher-model <id> --inputs <jsonl> --out <jsonl> [options]

        Required:
          --teacher URL              OpenAI-compatible base URL, e.g. http://127.0.0.1:1234/v1
          --teacher-model ID         Model id to send in the request body
          --inputs path.jsonl        Input JSONL
          --out labeled.jsonl        Output JSONL of {input, output, _meta}

        Options:
          --input-field NAME         Input prompt field in each row (default: prompt)
          --system TEXT              Optional system prompt
          --system-file PATH         Read system prompt from file
          --schema schema.json       Validate teacher output as JSON matching this schema
          --grammar grammar.gbnf     Pass llama.cpp/LM Studio-style grammar in request body
          --max-tokens N             Default: 256
          --temperature F            Default: 0.0
          --parallel N               Concurrent requests (default: 1)
          --rate-limit N             Global requests/sec cap, 0 = unlimited
          --retries N                HTTP retry count (default: 2)
          --timeout S                Per-request timeout seconds (default: 120)

        Auth:
          Reads TEACHER_API_KEY first, then OPENAI_API_KEY, and sends
          Authorization: Bearer <key> when present.
        """)
        exit(code)
    }
}

private final class JSONLAppendWriter {
    private let handle: FileHandle
    private let lock = NSLock()

    init(path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    func write(_ obj: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes])
        lock.lock()
        defer { lock.unlock() }
        handle.write(data)
        handle.write(Data([0x0a]))
    }
}

private final class RateLimiter {
    private let interval: TimeInterval
    private var nextAllowed = Date()
    private let lock = NSLock()

    init(perSecond: Double) {
        interval = perSecond > 0 ? 1.0 / perSecond : 0
    }

    func wait() {
        guard interval > 0 else { return }
        let sleepFor: TimeInterval
        lock.lock()
        let now = Date()
        if now < nextAllowed {
            sleepFor = nextAllowed.timeIntervalSince(now)
            nextAllowed = nextAllowed.addingTimeInterval(interval)
        } else {
            sleepFor = 0
            nextAllowed = now.addingTimeInterval(interval)
        }
        lock.unlock()
        if sleepFor > 0 {
            Thread.sleep(forTimeInterval: sleepFor)
        }
    }
}

private final class SynthesizeProgress {
    private let total: Int
    private let skipped: Int
    private let start = Date()
    private var completed = 0
    private var written = 0
    private var retries = 0
    private var schemaFails = 0
    private var errors = 0
    private var lastError: String?
    private let lock = NSLock()

    init(total: Int, skipped: Int) {
        self.total = total
        self.skipped = skipped
    }

    func recordSuccess() {
        lock.lock()
        completed += 1
        written += 1
        emitLocked(force: false)
        lock.unlock()
    }

    func recordRetry() {
        lock.lock(); retries += 1; lock.unlock()
    }

    func recordSchemaFail() {
        lock.lock()
        completed += 1
        schemaFails += 1
        emitLocked(force: false)
        lock.unlock()
    }

    func recordError(_ message: String) {
        lock.lock()
        completed += 1
        errors += 1
        lastError = message
        emitLocked(force: false)
        lock.unlock()
    }

    func finish() {
        lock.lock()
        emitLocked(force: true)
        fputs("\n", stderr)
        lock.unlock()
    }

    private func emitLocked(force: Bool) {
        guard force || completed == total || completed % max(1, min(25, max(1, total / 20))) == 0 else { return }
        let elapsed = max(0.001, -start.timeIntervalSinceNow)
        let rps = Double(completed) / elapsed
        let remaining = max(0, total - completed)
        let eta = rps > 0 ? Double(remaining) / rps : 0
        let pct = total > 0 ? 100.0 * Double(completed) / Double(total) : 100
        var line = String(
            format: "[synthesize] %d / %d (%.1f%%) · %.1f req/s · ETA %.0fs · wrote %d · skipped %d · retries %d · schema-fails %d · errors %d",
            completed, total, pct, rps, eta, written, skipped, retries, schemaFails, errors
        )
        if let lastError, force {
            line += " · last-error \(lastError)"
        }
        fputs(line + "\n", stderr)
    }
}

private final class OpenAICompatClient {
    struct Response {
        let content: String
        let tokensUsed: Int?
    }

    private let endpoint: URL
    private let model: String
    private let apiKey: String?
    private let timeout: TimeInterval

    init(teacherBaseURL: String, model: String, apiKey: String?, timeout: TimeInterval) {
        let trimmed = teacherBaseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let endpointString: String
        if trimmed.hasSuffix("/chat/completions") {
            endpointString = trimmed
        } else {
            endpointString = trimmed + "/chat/completions"
        }
        endpoint = URL(string: endpointString)!
        self.model = model
        self.apiKey = apiKey?.isEmpty == false ? apiKey : nil
        self.timeout = timeout
    }

    func chat(system: String?, user: String, maxTokens: Int, temperature: Double, grammar: String?) throws -> Response {
        var messages: [[String: String]] = []
        if let system, !system.isEmpty {
            messages.append(["role": "system", "content": system])
        }
        messages.append(["role": "user", "content": user])
        var body: [String: Any] = [
            "model": model,
            "messages": messages,
            "max_tokens": maxTokens,
            "temperature": temperature,
            "stream": false
        ]
        if let grammar, !grammar.isEmpty {
            body["grammar"] = grammar
        }
        let payload = try JSONSerialization.data(withJSONObject: body, options: [])
        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.timeoutInterval = timeout
        req.httpBody = payload
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey {
            req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try Self.syncData(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "tinygpt.synthesize", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "missing HTTP response"])
        }
        guard (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? ""
            throw NSError(domain: "tinygpt.synthesize", code: http.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode): \(text.prefix(500))"])
        }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let first = choices.first
        else {
            throw NSError(domain: "tinygpt.synthesize", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "bad OpenAI response"])
        }
        let content: String?
        if let message = first["message"] as? [String: Any] {
            content = message["content"] as? String
        } else {
            content = first["text"] as? String
        }
        guard let content else {
            throw NSError(domain: "tinygpt.synthesize", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "response choice had no content/text"])
        }
        let tokens = (obj["usage"] as? [String: Any])?["total_tokens"] as? Int
        return Response(content: content, tokensUsed: tokens)
    }

    private static func syncData(for req: URLRequest) throws -> (Data, URLResponse) {
        let sem = DispatchSemaphore(value: 0)
        final class Box {
            var data: Data?
            var response: URLResponse?
            var error: Error?
        }
        let box = Box()
        URLSession.shared.dataTask(with: req) { data, response, error in
            box.data = data
            box.response = response
            box.error = error
            sem.signal()
        }.resume()
        sem.wait()
        if let error = box.error { throw error }
        guard let data = box.data, let response = box.response else {
            throw NSError(domain: "tinygpt.synthesize", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "empty URLSession response"])
        }
        return (data, response)
    }
}
