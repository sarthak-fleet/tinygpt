import Foundation
import Darwin
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

// MARK: - lm-eval-harness adapter
//
// `tinygpt serve` exposes an OpenAI-compatible HTTP endpoint over a loaded
// tinygpt / HF model. This unlocks running the canonical `lm-evaluation-harness`
// against any tinygpt model — HellaSwag, MMLU-Pro, GSM8K, IFEval, GPQA-Diamond
// — by pointing the harness at `local-chat-completions` with our base_url.
//
// Wire-up:
//   POST /v1/chat/completions     — chat-style requests (messages: [...])
//   POST /v1/completions          — plain text completion (prompt: "...")
//   GET  /v1/models               — list "tinygpt" so clients can probe readiness
//
// Implementation notes:
//   - Uses POSIX sockets (Darwin) — zero new dependencies, works on macOS 14+.
//   - One thread per connection. Sample throughput dominates anyway; the
//     request rate from a single `lm-eval` driver is sequential.
//   - JSON parse/encode via Foundation. SSE streaming is supported when
//     the request body has `stream: true` — emits one event per token in
//     OpenAI's `chat.completion.chunk` format. lm-evaluation-harness
//     itself doesn't need streaming for loglikelihood / generate-until
//     tasks, but realtime/interactive clients do (the north-star
//     interaction model).
//   - All MLX work is serialised on a single dispatch queue (`inferenceQueue`)
//     because the model + KV cache are not thread-safe.
//
// TODO(merge): wire `case "serve": Serve.run(args: Array(args.dropFirst()))`
// into `TinyGPT.swift`'s subcommand dispatch. Left out per agent-coordination
// rules — another agent is touching that dispatch table.
//
// Tested via `Tests/TinyGPTServeTests/TinyGPTServeTests.swift` which boots a
// server on a random port, fires curl-equivalent NSURLSession requests, and
// asserts the JSON shape matches the OpenAI spec.
public enum Serve {
    public static func run(args: [String]) {
        var modelPath: String? = nil
        var host = "127.0.0.1"
        var port: UInt16 = 8080
        var maxContext: Int? = nil
        var loraPath: String? = nil
        var grammarPath: String? = nil
        var toolsPath: String? = nil
        var promptCacheDir: String? = nil
        var traceDir: String? = nil
        var eosStopEnabled = true
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--port":
                guard i + 1 < args.count, let p = UInt16(args[i + 1]) else { exitUsage() }
                port = p; i += 2
            case "--host":
                guard i + 1 < args.count else { exitUsage() }
                host = args[i + 1]; i += 2
            case "--max-context":
                guard i + 1 < args.count, let n = Int(args[i + 1]) else { exitUsage() }
                maxContext = n; i += 2
            case "--lora":
                guard i + 1 < args.count else { exitUsage() }
                loraPath = args[i + 1]; i += 2
            case "--grammar":
                guard i + 1 < args.count else { exitUsage() }
                grammarPath = args[i + 1]; i += 2
            case "--tools":
                guard i + 1 < args.count else { exitUsage() }
                toolsPath = args[i + 1]; i += 2
            case "--prompt-cache-dir":
                guard i + 1 < args.count else { exitUsage() }
                promptCacheDir = args[i + 1]; i += 2
            case "--trace-infer":
                traceDir = traceDir ?? "/tmp/tinygpt-traces"; i += 1
            case "--trace-dir":
                guard i + 1 < args.count else { exitUsage() }
                traceDir = args[i + 1]; i += 2
            case "--no-eos-stop":
                eosStopEnabled = false; i += 1
            case "-h", "--help":
                exitUsage(0)
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else {
            fputs("serve: missing <model path>\n", stderr); exitUsage()
        }

        // Without this, stdout is block-buffered when piped (e.g. when
        // the Mac app's ServerController spawns us as a subprocess), so
        // the startup banner never reaches the log pane until the buffer
        // fills 4KB later. The Server tab looks broken even though the
        // process is healthy. Make every print() flush immediately.
        setbuf(stdout, nil)
        setbuf(stderr, nil)
        do {
            let grammarText = try grammarPath.map { try String(contentsOfFile: $0, encoding: .utf8) }
            let toolsSpec = try toolsPath.map { try ServeToolsSpec.load(path: $0) }
            let server = try Server.boot(modelPath: modelPath, host: host, port: port,
                                          maxContextOverride: maxContext,
                                          loraPath: loraPath,
                                          grammarText: grammarText,
                                          toolsSpec: toolsSpec,
                                          promptCacheDir: promptCacheDir,
                                          traceDir: traceDir,
                                          eosStopEnabled: eosStopEnabled)
            print("tinygpt serve — listening on http://\(host):\(server.port)")
            print("model: \(modelPath)  ·  ctx=\(server.maxContext)  ·  vocab=\(server.config.vocabSize)")
            if let lp = loraPath { print("lora:  \(lp)") }
            if let gp = grammarPath { print("grammar: \(gp)") }
            if let tp = toolsPath { print("tools: \(tp)") }
            if let dir = promptCacheDir { print("prompt cache: \(dir)") }
            if let dir = traceDir { print("inference traces: \(dir)") }
            // Block forever — the listener thread runs detached.
            dispatchMain()
        } catch {
            fputs("serve: failed to start: \(error)\n", stderr)
            exit(1)
        }
    }

    /// Programmatic entry point used by tests. Returns a `Server` handle that
    /// owns the bound socket + listener thread. Call `stop()` to release the
    /// port.
    public static func start(modelPath: String, host: String = "127.0.0.1",
                              port: UInt16 = 0, maxContext: Int? = nil) throws -> Server
    {
        return try Server.boot(modelPath: modelPath, host: host, port: port,
                                maxContextOverride: maxContext)
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt serve <model.tinygpt | hf-dir> [options]

        --port N              TCP port (default: 8080; 0 = pick any free port)
        --host HOST           bind address (default: 127.0.0.1)
        --max-context N       cap context length below the model's native limit
                              (useful when running lm-eval on long-prompt tasks
                               like MMLU-Pro where the harness sometimes overshoots)
        --lora <path.lora>    Apply a LoRA adapter on top of the base before serving.
                              Works for both .tinygpt and HF-dir bases.
        --grammar <path>      Constrain generated text. Supports JSON Schema files
                              and the Pace tool-tag GBNF subset used by
                              grammars/pace-tool-tags.gbnf.
        --tools <path.json>   Planner-v7 tools-in-prompt mode. Injects the
                              tool catalog into the system prompt and constrains
                              output to {"verb","args","spoken_text"} with
                              verb limited to the provided tool names.
        --prompt-cache-dir <dir>
                              Auto-cache repeated prompt-prefix KV state.
        --trace-infer          Write per-request inference traces to /tmp/tinygpt-traces.
        --trace-dir <dir>      Write per-request inference traces to a custom directory.
        --no-eos-stop         Do not auto-stop on tokenizer EOS/chat-end tokens.

        Endpoints (OpenAI surface):
          POST /v1/chat/completions   OpenAI ChatCompletion (SSE if stream:true)
          POST /v1/completions        OpenAI Completion (SSE if stream:true)
          GET  /v1/models             list loaded model

        Endpoints (Ollama surface — Continue.dev / Cline / Aider):
          POST /api/chat              Ollama chat (NDJSON stream by default)
          POST /api/generate          Ollama completion (NDJSON stream by default)
          GET  /api/tags              Ollama model list
          GET  /api/version           Ollama version probe
          POST /api/show              Ollama model info

        For lm-eval (OpenAI surface):
          lm-eval --model local-chat-completions \\
              --tasks hellaswag,arc_easy --model_args \\
              "base_url=http://127.0.0.1:8080/v1/chat/completions,model=tinygpt"

        For Continue.dev (Ollama surface) — run with --port 11434 and add
        to ~/.continue/config.json:
          { "models": [{ "title": "tinygpt", "provider": "ollama",
                          "model": "tinygpt:latest",
                          "apiBase": "http://127.0.0.1:11434" }] }
        See docs/continue_provider.md for the full walkthrough.
        """)
        exit(code)
    }
}

// MARK: - Server

extension Serve {
    /// Handle to a running HTTP server. Reachable from tests + the CLI entry
    /// point. Owns the listener socket + the inference state.
    public final class Server: @unchecked Sendable {
        public let port: UInt16
        public let host: String
        public let config: ModelConfig
        public let maxContext: Int
        let model: AnyModel
        let tokenizer: TokenizerBox
        let defaultGrammar: ServeGrammarSpec?
        let toolsSystemPrompt: String?
        let eosTokenIds: Set<Int>
        let promptCacheDir: URL?
        let traceDir: URL?
        let modelFingerprint: String
        private let listenFd: Int32
        private let inferenceQueue: DispatchQueue
        private var running: Bool = true

        init(listenFd: Int32, host: String, port: UInt16,
             model: AnyModel, config: ModelConfig, tokenizer: TokenizerBox,
             maxContext: Int, defaultGrammar: ServeGrammarSpec?,
             toolsSystemPrompt: String?,
             eosTokenIds: Set<Int>, promptCacheDir: URL?,
             traceDir: URL?,
             modelFingerprint: String)
        {
            self.listenFd = listenFd
            self.host = host
            self.port = port
            self.model = model
            self.config = config
            self.tokenizer = tokenizer
            self.maxContext = maxContext
            self.defaultGrammar = defaultGrammar
            self.toolsSystemPrompt = toolsSystemPrompt
            self.eosTokenIds = eosTokenIds
            self.promptCacheDir = promptCacheDir
            self.traceDir = traceDir
            self.modelFingerprint = modelFingerprint
            self.inferenceQueue = DispatchQueue(label: "tinygpt.serve.inference")
        }

        static func boot(modelPath: String, host: String, port: UInt16,
                          maxContextOverride: Int?, loraPath: String? = nil,
                          grammarText: String? = nil,
                          toolsSpec: ServeToolsSpec? = nil,
                          promptCacheDir: String? = nil,
                          traceDir: String? = nil,
                          eosStopEnabled: Bool = true) throws -> Server
        {
            // Writes to a socket whose peer has hung up raise SIGPIPE,
            // which by default kills the process. SSE clients (curl
            // --max-time, browser fetch().cancel(), user closing a tab)
            // routinely close mid-stream, so we ignore SIGPIPE
            // process-wide and rely on write()'s EPIPE return + the
            // cancellation path in streamChat / streamCompletion.
            signal(SIGPIPE, SIG_IGN)

            // Load model + (optional) BPE tokenizer up front. Same logic as
            // Sample.swift so behaviour matches between `sample` and `serve`.
            let load = try ModelLoader.load(modelPath)
            let cfg = load.config

            // Optional LoRA adapter — applied AFTER base load. Routes to
            // the right injector based on model class (fromScratch vs
            // huggingFace), using the same adapter file format.
            if let lp = loraPath {
                let adapter = try LoraAdapterReader.read(URL(fileURLWithPath: lp))
                switch load.model {
                case .fromScratch(let m):
                    try LoraAdapterReader.apply(adapter, to: m)
                case .huggingFace(let m):
                    try LoraAdapterHFReader.apply(adapter, to: m)
                }
            }
            let tok: TokenizerBox
            var eosIds = Set<Int>()
            if let dir = load.hfTokenizerDir {
                do {
                    let hf = try HFTokenizer.loadBlocking(from: dir)
                    tok = .hf(hf)
                    if eosStopEnabled {
                        eosIds = Self.detectEOSTokenIds(tokenizerDir: dir, tokenizer: tok)
                    }
                } catch {
                    fputs("warning: tokenizer load failed (\(error)); falling back to byte-level\n", stderr)
                    tok = .byteLevel
                }
            } else {
                tok = .byteLevel
            }
            if eosStopEnabled {
                eosIds.formUnion(Self.defaultChatStopTokenIds(tokenizer: tok))
            }
            let defaultGrammar: ServeGrammarSpec?
            if let grammarText {
                defaultGrammar = try ServeGrammarSpec.parse(grammarText)
            } else if let toolsSpec {
                defaultGrammar = try toolsSpec.grammarSpec()
            } else {
                defaultGrammar = nil
            }
            let promptCacheURL = promptCacheDir.map { URL(fileURLWithPath: $0) }
            if let promptCacheURL {
                try KVCachePersist.ensureDir(promptCacheURL)
            }
            let traceURL = traceDir.map { URL(fileURLWithPath: $0) }
            if let traceURL {
                try FileManager.default.createDirectory(at: traceURL, withIntermediateDirectories: true)
            }
            let fingerprint = [
                "model:\(modelPath):\(KVCachePersist.fingerprint(of: modelPath))",
                loraPath.map { "lora:\($0):\(KVCachePersist.fingerprint(of: $0))" } ?? "lora:none"
            ].joined(separator: "|")

            let (fd, boundPort) = try Self.bindListener(host: host, port: port)
            let maxCtx = min(maxContextOverride ?? cfg.contextLength, cfg.contextLength)
            let server = Server(listenFd: fd, host: host, port: boundPort,
                                 model: load.model, config: cfg, tokenizer: tok,
                                 maxContext: maxCtx, defaultGrammar: defaultGrammar,
                                 toolsSystemPrompt: toolsSpec?.systemPrompt(),
                                 eosTokenIds: eosIds,
                                 promptCacheDir: promptCacheURL,
                                 traceDir: traceURL,
                                 modelFingerprint: fingerprint)
            server.startAcceptLoop()
            return server
        }

        private static func detectEOSTokenIds(tokenizerDir: URL, tokenizer: TokenizerBox) -> Set<Int> {
            var out = Set<Int>()
            let configURL = tokenizerDir.appendingPathComponent("tokenizer_config.json")
            if let data = try? Data(contentsOf: configURL),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                collectTokenStrings(from: obj["eos_token"]).forEach { token in
                    out.formUnion(tokenizer.encode(token))
                }
                collectTokenStrings(from: obj["additional_special_tokens"]).forEach { token in
                    if token.lowercased().contains("end") || token.lowercased().contains("eos") {
                        out.formUnion(tokenizer.encode(token))
                    }
                }
            }

            let tokenizerURL = tokenizerDir.appendingPathComponent("tokenizer.json")
            if let data = try? Data(contentsOf: tokenizerURL),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let added = obj["added_tokens"] as? [[String: Any]] {
                for token in added {
                    guard let content = token["content"] as? String else { continue }
                    let lower = content.lowercased()
                    if lower.contains("eos") || lower.contains("end") || lower.contains("eot") {
                        if let id = token["id"] as? Int {
                            out.insert(id)
                        } else {
                            out.formUnion(tokenizer.encode(content))
                        }
                    }
                }
            }
            return out
        }

        private static func defaultChatStopTokenIds(tokenizer: TokenizerBox) -> Set<Int> {
            var out = Set<Int>()
            for token in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>", "<eos>"] {
                let ids = tokenizer.encode(token)
                if ids.count == 1 {
                    out.insert(ids[0])
                }
            }
            return out
        }

        private static func collectTokenStrings(from raw: Any?) -> [String] {
            guard let raw, !(raw is NSNull) else { return [] }
            if let s = raw as? String { return [s] }
            if let dict = raw as? [String: Any], let content = dict["content"] as? String {
                return [content]
            }
            if let arr = raw as? [Any] {
                return arr.flatMap { collectTokenStrings(from: $0) }
            }
            return []
        }

        /// Stops accepting new connections and closes the listening socket.
        /// In-flight requests on existing connections continue to completion.
        public func stop() {
            running = false
            Darwin.close(listenFd)
        }

        // MARK: BSD socket setup

        private static func bindListener(host: String, port: UInt16) throws -> (Int32, UInt16) {
            let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else {
                throw NSError(domain: "tinygpt.serve", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "socket() failed: \(String(cString: strerror(errno)))"])
            }
            // SO_REUSEADDR — convenient when restarting the server quickly.
            var yes: Int32 = 1
            _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = in_port_t(port).bigEndian
            // inet_pton: parse "127.0.0.1" into the in_addr struct.
            let inetResult = host.withCString { hostC -> Int32 in
                inet_pton(AF_INET, hostC, &addr.sin_addr)
            }
            guard inetResult == 1 else {
                Darwin.close(fd)
                throw NSError(domain: "tinygpt.serve", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "bad host: \(host)"])
            }
            let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    Darwin.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindResult == 0 else {
                let err = String(cString: strerror(errno))
                Darwin.close(fd)
                throw NSError(domain: "tinygpt.serve", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "bind() failed: \(err)"])
            }
            guard listen(fd, 16) == 0 else {
                let err = String(cString: strerror(errno))
                Darwin.close(fd)
                throw NSError(domain: "tinygpt.serve", code: 4,
                              userInfo: [NSLocalizedDescriptionKey: "listen() failed: \(err)"])
            }
            // Resolve the actual bound port (port==0 → kernel-assigned).
            var bound = sockaddr_in()
            var bound_len = socklen_t(MemoryLayout<sockaddr_in>.size)
            let actualPort: UInt16
            let gotName = withUnsafeMutablePointer(to: &bound) { ptr -> Int32 in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    getsockname(fd, sockPtr, &bound_len)
                }
            }
            if gotName == 0 {
                actualPort = UInt16(bigEndian: bound.sin_port)
            } else {
                actualPort = port
            }
            return (fd, actualPort)
        }

        // MARK: Accept loop

        private func startAcceptLoop() {
            Thread.detachNewThread { [weak self] in
                guard let self = self else { return }
                while self.running {
                    var clientAddr = sockaddr_in()
                    var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
                    let clientFd = withUnsafeMutablePointer(to: &clientAddr) { ptr -> Int32 in
                        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                            accept(self.listenFd, sockPtr, &clientLen)
                        }
                    }
                    if clientFd < 0 {
                        if !self.running { return }
                        if errno == EINTR { continue }
                        // Listening socket closed — exit loop cleanly.
                        return
                    }
                    // One thread per connection. The inference is serialised
                    // on `inferenceQueue` regardless of how many connections
                    // there are.
                    Thread.detachNewThread { [weak self] in
                        guard let self = self else {
                            Darwin.close(clientFd); return
                        }
                        self.handleConnection(clientFd: clientFd)
                    }
                }
            }
        }

        // MARK: Per-connection handler

        private func handleConnection(clientFd: Int32) {
            defer { Darwin.close(clientFd) }
            guard let request = HTTPRequest.read(from: clientFd) else {
                respond(clientFd: clientFd, status: 400, body: "bad request")
                return
            }

            // Health check / model listing.
            if request.method == "GET" && request.path == "/v1/models" {
                let payload: [String: Any] = [
                    "object": "list",
                    "data": [
                        ["id": "tinygpt", "object": "model", "owned_by": "tinygpt"]
                    ]
                ]
                respondJSON(clientFd: clientFd, status: 200, payload: payload)
                return
            }

            if request.method == "POST" && request.path == "/v1/pace/route" {
                handlePaceRoute(clientFd: clientFd, body: request.body)
                return
            }

            if request.method == "POST" && request.path == "/v1/chat/completions" {
                handleChatCompletions(clientFd: clientFd, body: request.body)
                return
            }
            if request.method == "POST" && request.path == "/v1/completions" {
                handleCompletions(clientFd: clientFd, body: request.body)
                return
            }

            // Ollama-compatible surface — Continue.dev / Cline / Aider
            // configured with `provider: ollama` talk to tinygpt directly.
            // NDJSON streaming (not SSE); shared generation core with the
            // OpenAI handlers above. See docs/continue_provider.md.
            if request.method == "GET" && request.path == "/api/tags" {
                handleOllamaTags(clientFd: clientFd)
                return
            }
            if request.method == "GET" && request.path == "/api/version" {
                respondJSON(clientFd: clientFd, status: 200,
                             payload: ["version": "0.5.0-tinygpt"])
                return
            }
            if request.method == "POST" && request.path == "/api/show" {
                handleOllamaShow(clientFd: clientFd, body: request.body)
                return
            }
            if request.method == "POST" && request.path == "/api/chat" {
                handleOllamaChat(clientFd: clientFd, body: request.body)
                return
            }
            if request.method == "POST" && request.path == "/api/generate" {
                handleOllamaGenerate(clientFd: clientFd, body: request.body)
                return
            }

            respond(clientFd: clientFd, status: 404, body: "not found: \(request.method) \(request.path)")
        }

        // MARK: /v1/pace/route

        private func handlePaceRoute(clientFd: Int32, body: Data) {
            let start = DispatchTime.now().uptimeNanoseconds
            do {
                guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                    respond(clientFd: clientFd, status: 400, body: "json must be object")
                    return
                }
                guard let user = json["user"] as? String, !user.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    respond(clientFd: clientFd, status: 400, body: "missing non-empty user")
                    return
                }
                let elements = try parsePaceElements(json["elements"])
                let freeText = (json["free_text"] as? Bool) ?? (json["free_text_mode"] as? Bool) ?? false
                let route = PaceFastRouter.route(user: user, elements: elements)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
                respondJSON(clientFd: clientFd, status: 200,
                            payload: route.payload(elements: elements, latencyMs: elapsed, freeText: freeText))
            } catch {
                respond(clientFd: clientFd, status: 400, body: "pace route error: \(error)")
            }
        }

        private func parsePaceElements(_ raw: Any?) throws -> [PaceRouteElement] {
            guard let raw, !(raw is NSNull) else { return [] }
            if let objects = raw as? [[String: Any]] {
                return objects.compactMap(parsePaceElementObject)
            }
            if let strings = raw as? [String] {
                return strings.compactMap(parsePaceElementString)
            }
            throw NSError(domain: "tinygpt.serve.pace", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "elements must be an array of objects or fixture strings"])
        }

        private func parsePaceElementObject(_ object: [String: Any]) -> PaceRouteElement? {
            guard let id = intValue(object["id"]) else { return nil }
            return PaceRouteElement(
                id: id,
                role: stringValue(object["role"]),
                x: intValue(object["x"]) ?? 0,
                y: intValue(object["y"]) ?? 0,
                label: stringValue(object["label"]),
                text: stringValue(object["text"])
            )
        }

        private func parsePaceElementString(_ raw: String) -> PaceRouteElement? {
            let pattern = #"^\[(\d+)\]\s*([^|]+)\|(-?\d+),(-?\d+)\|([^|]*)\|(.*)$"#
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
            let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
            guard let match = regex.firstMatch(in: raw, range: range), match.numberOfRanges == 7 else {
                return nil
            }
            func group(_ index: Int) -> String {
                guard let swiftRange = Range(match.range(at: index), in: raw) else { return "" }
                return String(raw[swiftRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard let id = Int(group(1)), let x = Int(group(3)), let y = Int(group(4)) else { return nil }
            return PaceRouteElement(id: id, role: group(2), x: x, y: y, label: group(5), text: group(6))
        }

        private func intValue(_ value: Any?) -> Int? {
            if let int = value as? Int { return int }
            if let double = value as? Double { return Int(double) }
            if let number = value as? NSNumber { return number.intValue }
            if let string = value as? String { return Int(string) }
            return nil
        }

        private func stringValue(_ value: Any?) -> String {
            guard let value, !(value is NSNull) else { return "" }
            return String(describing: value).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // MARK: /v1/chat/completions

        private func handleChatCompletions(clientFd: Int32, body: Data) {
            let tracer = traceDir.map { _ in
                InferenceTracer(route: "serve.chat.completions", model: config.modelName)
            }
            do {
                let jsonAny = try traceSpan(tracer, "json_parse") {
                    try JSONSerialization.jsonObject(with: body)
                }
                guard let json = jsonAny as? [String: Any] else {
                    respond(clientFd: clientFd, status: 400, body: "json must be object"); return
                }
                let messages = (json["messages"] as? [[String: Any]]) ?? []
                let prompt = traceSpan(tracer, "render_prompt") {
                    renderChatMessages(messages)
                }
                let cachePrefix = traceSpan(tracer, "render_cache_prefix") {
                    renderChatPromptCachePrefix(messages)
                }
                let maxTokens = (json["max_tokens"] as? Int) ?? 128
                let temperature = (json["temperature"] as? Double).map { Float($0) } ?? 0.0
                let stopParam = readStopParam(json["stop"])
                let stopTokenIds = readStopTokenIds(json["stop_token_ids"])
                let grammar = try grammarSpec(from: json["grammar"])
                let stream = (json["stream"] as? Bool) ?? false

                if stream {
                    streamChat(clientFd: clientFd, prompt: prompt,
                                maxTokens: maxTokens, temperature: temperature,
                                stop: stopParam, grammar: grammar,
                                extraStopTokenIds: stopTokenIds,
                                cachePrefix: cachePrefix)
                    return
                }

                let (text, promptTokens, completionTokens) = try inferenceQueue.sync {
                    try self.generate(prompt: prompt, maxTokens: maxTokens,
                                       temperature: temperature, stop: stopParam,
                                       grammar: grammar, extraStopTokenIds: stopTokenIds,
                                       cachePrefix: cachePrefix,
                                       tracer: tracer)
                }
                tracer?.setTokenCounts(prompt: promptTokens, generated: completionTokens)

                let payload: [String: Any] = [
                    "id": "chatcmpl-\(UUID().uuidString)",
                    "object": "chat.completion",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": "tinygpt",
                    "choices": [[
                        "index": 0,
                        "message": ["role": "assistant", "content": text],
                        "finish_reason": "stop"
                    ]],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens
                    ]
                ]
                traceSpan(tracer, "response_write") {
                    respondJSON(clientFd: clientFd, status: 200, payload: payload)
                }
                if let tracer, let traceDir {
                    _ = try? tracer.write(to: traceDir)
                }
            } catch {
                respond(clientFd: clientFd, status: 500, body: "error: \(error)")
                if let tracer, let traceDir {
                    _ = try? tracer.write(to: traceDir)
                }
            }
        }

        // MARK: /v1/completions

        private func handleCompletions(clientFd: Int32, body: Data) {
            do {
                guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                    respond(clientFd: clientFd, status: 400, body: "json must be object"); return
                }
                // OpenAI Completions spec: `prompt` may be String OR [String]
                // OR [[Int]] (pre-tokenized). lm-eval-harness's local-completions
                // adapter sends [String] when batch_size > 1. We expand any
                // shape into a list-of-strings and return one `choice` per
                // input prompt. Pre-tokenized [[Int]] inputs are decoded via
                // the tokenizer before scoring.
                let prompts: [String] = parsePromptField(json["prompt"])
                let prompt = prompts.first ?? ""
                let maxTokens = (json["max_tokens"] as? Int) ?? 128
                let temperature = (json["temperature"] as? Double).map { Float($0) } ?? 0.0
                let stopParam = readStopParam(json["stop"])
                let stopTokenIds = readStopTokenIds(json["stop_token_ids"])
                let grammar = try grammarSpec(from: json["grammar"])
                let stream = (json["stream"] as? Bool) ?? false
                // OpenAI Completions spec: `logprobs` is an int N (top-N
                // alternatives) or null. We treat any non-null value as
                // "yes, return token_logprobs". `echo` includes the prompt
                // in the response — what lm-eval-harness's loglikelihood
                // path uses.
                let logprobsRequested = json["logprobs"] != nil && !(json["logprobs"] is NSNull)
                let echo = (json["echo"] as? Bool) ?? false

                if stream {
                    streamCompletion(clientFd: clientFd, prompt: prompt,
                                      maxTokens: maxTokens, temperature: temperature,
                                      stop: stopParam, grammar: grammar,
                                      extraStopTokenIds: stopTokenIds,
                                      cachePrefix: prompt)
                    return
                }

                // Logprobs + echo is lm-eval's loglikelihood signature.
                // lm-eval-harness sends max_tokens=0 OR max_tokens=1
                // depending on version; we trigger the teacher-forced
                // scoring path whenever both flags are set and skip the
                // generation loop entirely. Lm-eval slices [ctxlen:-1]
                // on the returned token_logprobs so the boundary token
                // doesn't matter.
                if logprobsRequested && echo {
                    // Score each prompt in the batch on a single trip through
                    // the inference queue to keep lm-eval's request-response
                    // count exactly 1:1 with the input list. Without this,
                    // lm-eval's `get_original` zip fails with
                    // "argument 2 is shorter than argument 1".
                    var choices: [[String: Any]] = []
                    var totalTokens = 0
                    try inferenceQueue.sync {
                        for (i, p) in prompts.enumerated() {
                            let (tokens, tokenLogprobs) = try self.scoreLogprobs(prompt: p)
                            totalTokens += tokens.count
                            choices.append([
                                "index": i,
                                "text": p,
                                "finish_reason": "length",
                                "logprobs": [
                                    "tokens": tokens,
                                    // First-token logprob is NSNull (no preceding context).
                                    // lm-eval slices [ctxlen:-1] so the leading None doesn't matter.
                                    "token_logprobs": tokenLogprobs as [Any],
                                    "top_logprobs": [] as [Any],
                                    "text_offset": [] as [Any],
                                ] as [String: Any],
                            ])
                        }
                    }
                    let payload: [String: Any] = [
                        "id": "cmpl-\(UUID().uuidString)",
                        "object": "text_completion",
                        "created": Int(Date().timeIntervalSince1970),
                        "model": "tinygpt",
                        "choices": choices,
                        "usage": [
                            "prompt_tokens": totalTokens,
                            "completion_tokens": 0,
                            "total_tokens": totalTokens
                        ]
                    ]
                    respondJSON(clientFd: clientFd, status: 200, payload: payload)
                    return
                }

                let (text, promptTokens, completionTokens) = try inferenceQueue.sync {
                    try self.generate(prompt: prompt, maxTokens: maxTokens,
                                       temperature: temperature, stop: stopParam,
                                       grammar: grammar, extraStopTokenIds: stopTokenIds,
                                       cachePrefix: prompt)
                }

                let payload: [String: Any] = [
                    "id": "cmpl-\(UUID().uuidString)",
                    "object": "text_completion",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": "tinygpt",
                    "choices": [[
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop"
                    ]],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens
                    ]
                ]
                respondJSON(clientFd: clientFd, status: 200, payload: payload)
            } catch {
                respond(clientFd: clientFd, status: 500, body: "error: \(error)")
            }
        }

        /// Single teacher-forced forward over `prompt`, returning the
        /// per-token logprobs. lm-eval-harness's `local-completions`
        /// backend uses this for loglikelihood scoring on multi-choice
        /// tasks (ARC, MMLU, HellaSwag, PIQA, BoolQ, WinoGrande, ...).
        /// First-position logprob is `null` — no preceding context to
        /// condition on. lm-eval slices `[ctxlen:-1]` so the leading
        /// null is intentional, not a bug.
        func scoreLogprobs(prompt: String) throws -> (tokens: [String], tokenLogprobs: [Any]) {
            let promptIds = tokenizer.encode(prompt)
            guard !promptIds.isEmpty else { return ([], []) }

            let ctxCap = maxContext
            // Drop from the left if the prompt overflows the model's ctx.
            // Same policy as `generate`.
            let head: Int
            if promptIds.count > ctxCap {
                head = promptIds.count - ctxCap
            } else {
                head = 0
            }
            let kept = Array(promptIds[head..<promptIds.count])
            let arr = MLXArray(kept.map { Int32($0) }, [1, kept.count])

            // One forward pass — logits at every position. log_softmax to
            // probability space; gather the logprob of the token AT POSITION
            // i+1 from the logits AT POSITION i (the next-token prediction).
            let logits = model(arr)  // [1, T, vocab]
            let logProbs = MLX.log(MLX.softmax(logits, axis: -1))
            eval(logProbs)

            var lps: [Any] = [NSNull()]   // position 0 has no preceding context
            for i in 0..<(kept.count - 1) {
                let nextTok = kept[i + 1]
                let lp = logProbs[0, i, nextTok]
                eval(lp)
                lps.append(Double(lp.item(Float.self)))
            }

            // Decode each token id individually so the response carries
            // human-readable token strings (lm-eval ignores the field
            // but downstream debugging benefits).
            let tokenStrings = kept.map { tokenizer.decode([$0]) }
            return (tokenStrings, lps)
        }

        // MARK: SSE streaming

        /// Streaming variant of /v1/chat/completions.
        ///
        /// Wire format (OpenAI-compatible):
        ///   data: {"id":"chatcmpl-...","object":"chat.completion.chunk",
        ///           "created":TS,"model":"tinygpt",
        ///           "choices":[{"index":0,"delta":{"role":"assistant"}}]}
        ///
        ///   data: {...,"choices":[{"index":0,"delta":{"content":"hello"}}]}
        ///   data: {...,"choices":[{"index":0,"delta":{"content":" world"}}]}
        ///   ...
        ///   data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
        ///   data: [DONE]
        ///
        /// One generation token may not produce visible output (partial
        /// BPE byte) — we only emit a delta when the running decoded
        /// suffix grows. This matches OpenAI's behavior on multi-byte
        /// tokens (Chinese, emoji, etc.) and is what SSE clients expect.
        private func streamChat(clientFd: Int32, prompt: String,
                                  maxTokens: Int, temperature: Float,
                                  stop: [String], grammar: ServeGrammarSpec?,
                                  extraStopTokenIds: Set<Int>,
                                  cachePrefix: String?) {
            let id = "chatcmpl-\(UUID().uuidString)"
            writeSSEHead(clientFd: clientFd)
            // Opening delta — sets role on the assistant message.
            writeSSEEvent(clientFd: clientFd, payload: chunkPayload(
                id: id, object: "chat.completion.chunk",
                delta: ["role": "assistant"], finishReason: nil))

            var finishReason = "stop"
            var clientGone = false
            do {
                try inferenceQueue.sync {
                    try self.generateStreaming(prompt: prompt, maxTokens: maxTokens,
                                                temperature: temperature, stop: stop,
                                                grammar: grammar,
                                                extraStopTokenIds: extraStopTokenIds,
                                                cachePrefix: cachePrefix)
                    { newText in
                        let ok = self.writeSSEEvent(clientFd: clientFd, payload: self.chunkPayload(
                            id: id, object: "chat.completion.chunk",
                            delta: ["content": newText], finishReason: nil))
                        if !ok { clientGone = true }
                        return ok
                    }
                }
            } catch {
                finishReason = "error"
            }
            // Client disconnected — don't bother sending final chunk + DONE,
            // the socket is dead. Just return so the connection closes.
            if clientGone { return }
            // Final delta with finish_reason — empty delta per OpenAI spec.
            writeSSEEvent(clientFd: clientFd, payload: chunkPayload(
                id: id, object: "chat.completion.chunk",
                delta: [:], finishReason: finishReason))
            writeSSETerminator(clientFd: clientFd)
        }

        /// Streaming variant of /v1/completions. Same wire as streamChat
        /// but uses "text_completion" object type and `text` field in the
        /// choice instead of `delta`.
        private func streamCompletion(clientFd: Int32, prompt: String,
                                        maxTokens: Int, temperature: Float,
                                        stop: [String], grammar: ServeGrammarSpec?,
                                        extraStopTokenIds: Set<Int>,
                                        cachePrefix: String?) {
            let id = "cmpl-\(UUID().uuidString)"
            writeSSEHead(clientFd: clientFd)

            var finishReason = "stop"
            var clientGone = false
            do {
                try inferenceQueue.sync {
                    try self.generateStreaming(prompt: prompt, maxTokens: maxTokens,
                                                temperature: temperature, stop: stop,
                                                grammar: grammar,
                                                extraStopTokenIds: extraStopTokenIds,
                                                cachePrefix: cachePrefix)
                    { newText in
                        let payload: [String: Any] = [
                            "id": id,
                            "object": "text_completion",
                            "created": Int(Date().timeIntervalSince1970),
                            "model": "tinygpt",
                            "choices": [[
                                "index": 0,
                                "text": newText,
                                "finish_reason": NSNull()
                            ]]
                        ]
                        let ok = self.writeSSEEvent(clientFd: clientFd, payload: payload)
                        if !ok { clientGone = true }
                        return ok
                    }
                }
            } catch {
                finishReason = "error"
            }
            if clientGone { return }
            let final: [String: Any] = [
                "id": id,
                "object": "text_completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": "tinygpt",
                "choices": [[
                    "index": 0,
                    "text": "",
                    "finish_reason": finishReason
                ]]
            ]
            writeSSEEvent(clientFd: clientFd, payload: final)
            writeSSETerminator(clientFd: clientFd)
        }

        private func chunkPayload(id: String, object: String,
                                   delta: [String: Any],
                                   finishReason: String?) -> [String: Any] {
            var choice: [String: Any] = [
                "index": 0,
                "delta": delta
            ]
            choice["finish_reason"] = finishReason ?? NSNull()
            return [
                "id": id,
                "object": object,
                "created": Int(Date().timeIntervalSince1970),
                "model": "tinygpt",
                "choices": [choice]
            ]
        }

        /// Streaming generation. Calls `onText` with the newly-decoded
        /// suffix each time a step extends the visible string. The
        /// callback returns `true` to continue, `false` to abort (used
        /// to propagate client-disconnect through to early exit). The
        /// token loop mirrors `generate(...)` — KV-cached decode so long
        /// generations stay O(T) per step instead of O(T²).
        func generateStreaming(prompt: String, maxTokens: Int,
                                temperature: Float, stop: [String],
                                grammar: ServeGrammarSpec? = nil,
                                extraStopTokenIds: Set<Int> = [],
                                cachePrefix: String? = nil,
                                onText: (String) -> Bool) throws
        {
            let promptIds = tokenizer.encode(prompt)
            if promptIds.isEmpty { return }

            // Bound prompt to leave room for at least 1 generated token
            // inside the model's context window. Left-truncate (drop the
            // head) — same policy as the historic uncached loop.
            let ctxCap = maxContext
            let promptCap = max(1, ctxCap - 1)
            let kept: [Int]
            if promptIds.count > promptCap {
                kept = Array(promptIds[(promptIds.count - promptCap)..<promptIds.count])
            } else {
                kept = promptIds
            }

            // KV-cached decode. Fresh cache per request (no cross-call
            // sharing — generate() and streamX() each allocate their own).
            // Uncached decode + LoRA-wrapped projections leaks graph nodes
            // on every step; at ~500 tokens MLX's unified-memory pressure
            // kills the process silently. forwardCached keeps per-step
            // work at [B,1] so the graph stays bounded.
            let prefill = try prefillPromptCache(kept: kept, cachePrefix: cachePrefix)
            let cache = prefill.cache
            var lastLogits = prefill.lastLogits
            let tokenDType = prefill.tokenDType
            let constraint = try makeConstraint(grammar)
            let activeStopTokenIds = eosTokenIds.union(extraStopTokenIds)

            var generated: [Int] = []
            generated.reserveCapacity(maxTokens)
            var lastDecoded: String = ""
            for _ in 0..<maxTokens {
                if constraint?.isComplete == true { return }
                let sampled = sampleToken(from: lastLogits, temperature: temperature,
                                          constraint: constraint)
                let tokenInt = sampled.token
                if activeStopTokenIds.contains(tokenInt) { return }
                generated.append(tokenInt)
                let nextId = MLXArray([Int32(tokenInt)], [1, 1])

                let nowDecoded = tokenizer.decode(generated)
                // Only emit a delta when the visible string grew. BPE /
                // multi-byte UTF-8 can leave us with intermediate bytes
                // that don't yet form a complete character.
                if nowDecoded.count > lastDecoded.count
                    && nowDecoded.hasPrefix(lastDecoded) {
                    let suffix = String(nowDecoded.dropFirst(lastDecoded.count))
                    let keepGoing = onText(suffix)
                    lastDecoded = nowDecoded
                    if !keepGoing { return }  // client disconnected — abort early
                }

                if !stop.isEmpty {
                    if stop.contains(where: { !$0.isEmpty && nowDecoded.contains($0) }) {
                        return
                    }
                }
                if cache.currentLength >= ctxCap { return }
                let logits = model.forwardCached(nextId.asType(tokenDType), cache: cache)
                lastLogits = logits[0..., 0, 0...]
                // Force materialization every step. Without this, lazy MLX
                // ops chain across iterations and the graph grows past a
                // soft limit at ~500 tokens with LoRA-wrapped projections
                // — silent SIGSEGV in MLX kernel. CLI sample (HFLoad)
                // implicitly evals via tokenizer.decode + stdout print
                // every step; serve's path is mostly lazy until the next
                // sample → graph never collapses. Explicit eval fixes it.
                eval(lastLogits)
            }
        }

        @discardableResult
        private func writeSSEHead(clientFd: Int32) -> Bool {
            var head = "HTTP/1.1 200 OK\r\n"
            head += "Content-Type: text/event-stream; charset=utf-8\r\n"
            head += "Cache-Control: no-cache\r\n"
            head += "Connection: close\r\n"
            head += "X-Accel-Buffering: no\r\n"  // tell reverse-proxies not to buffer
            head += "\r\n"
            return writeAll(clientFd: clientFd, data: Data(head.utf8))
        }

        /// Returns true if the event was delivered, false if the client
        /// has disconnected. Streaming endpoints use the return value to
        /// short-circuit the generation loop when the user aborts.
        @discardableResult
        private func writeSSEEvent(clientFd: Int32, payload: [String: Any]) -> Bool {
            guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return false }
            var frame = "data: ".data(using: .utf8)!
            frame.append(data)
            frame.append(Data("\n\n".utf8))
            return writeAll(clientFd: clientFd, data: frame)
        }

        @discardableResult
        private func writeSSETerminator(clientFd: Int32) -> Bool {
            return writeAll(clientFd: clientFd, data: Data("data: [DONE]\n\n".utf8))
        }

        // MARK: Ollama-compatible endpoints

        /// `GET /api/tags` — Ollama's model-list endpoint. Continue.dev /
        /// Cline / Aider use it to discover what models the server hosts.
        /// We always have exactly one ("tinygpt"); the response shape mirrors
        /// Ollama's so the client doesn't need a special case.
        private func handleOllamaTags(clientFd: Int32) {
            let now = ISO8601DateFormatter().string(from: Date())
            let payload: [String: Any] = [
                "models": [[
                    "name": "tinygpt:latest",
                    "model": "tinygpt:latest",
                    "modified_at": now,
                    "size": 0,
                    "digest": "tinygpt-\(config.modelName)",
                    "details": [
                        "format": "tinygpt",
                        "family": "tinygpt",
                        "parameter_size": "\(model.numParameters())",
                        "quantization_level": "F32"
                    ] as [String: Any]
                ] as [String: Any]]
            ]
            respondJSON(clientFd: clientFd, status: 200, payload: payload)
        }

        /// `POST /api/show` — Ollama's model-info endpoint. Continue.dev
        /// pings this to verify a model is loaded.
        private func handleOllamaShow(clientFd: Int32, body: Data) {
            let payload: [String: Any] = [
                "modelfile": "# tinygpt model",
                "parameters": "stop \"<|im_end|>\"",
                "template": "{{ .System }}\n{{ .Prompt }}",
                "details": [
                    "format": "tinygpt",
                    "family": "tinygpt",
                    "parameter_size": "\(model.numParameters())",
                    "quantization_level": "F32"
                ] as [String: Any]
            ]
            respondJSON(clientFd: clientFd, status: 200, payload: payload)
        }

        /// `POST /api/chat` — Ollama chat endpoint. NDJSON streaming by
        /// default; explicit `stream: false` for one-shot.
        private func handleOllamaChat(clientFd: Int32, body: Data) {
            do {
                guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                    respond(clientFd: clientFd, status: 400, body: "json must be object"); return
                }
                let messages = (json["messages"] as? [[String: Any]]) ?? []
                let prompt = renderChatMessages(messages)
                let options = (json["options"] as? [String: Any]) ?? [:]
                let maxTokens = (options["num_predict"] as? Int) ?? 256
                let temperature = (options["temperature"] as? Double).map { Float($0) } ?? 0.0
                let stop = readStopParam(options["stop"])
                let stream = (json["stream"] as? Bool) ?? true

                if stream {
                    streamOllamaChat(clientFd: clientFd, prompt: prompt,
                                      maxTokens: maxTokens, temperature: temperature,
                                      stop: stop)
                    return
                }
                let (text, promptTokens, completionTokens) = try inferenceQueue.sync {
                    try self.generate(prompt: prompt, maxTokens: maxTokens,
                                       temperature: temperature, stop: stop)
                }
                let payload: [String: Any] = [
                    "model": "tinygpt:latest",
                    "created_at": ISO8601DateFormatter().string(from: Date()),
                    "message": ["role": "assistant", "content": text] as [String: Any],
                    "done": true,
                    "done_reason": "stop",
                    "prompt_eval_count": promptTokens,
                    "eval_count": completionTokens
                ]
                respondJSON(clientFd: clientFd, status: 200, payload: payload)
            } catch {
                respond(clientFd: clientFd, status: 500, body: "error: \(error)")
            }
        }

        /// `POST /api/generate` — Ollama completion endpoint.
        private func handleOllamaGenerate(clientFd: Int32, body: Data) {
            do {
                guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                    respond(clientFd: clientFd, status: 400, body: "json must be object"); return
                }
                let prompt = (json["prompt"] as? String) ?? ""
                let options = (json["options"] as? [String: Any]) ?? [:]
                let maxTokens = (options["num_predict"] as? Int) ?? 256
                let temperature = (options["temperature"] as? Double).map { Float($0) } ?? 0.0
                let stop = readStopParam(options["stop"])
                let stream = (json["stream"] as? Bool) ?? true

                if stream {
                    streamOllamaGenerate(clientFd: clientFd, prompt: prompt,
                                          maxTokens: maxTokens, temperature: temperature,
                                          stop: stop)
                    return
                }
                let (text, promptTokens, completionTokens) = try inferenceQueue.sync {
                    try self.generate(prompt: prompt, maxTokens: maxTokens,
                                       temperature: temperature, stop: stop)
                }
                let payload: [String: Any] = [
                    "model": "tinygpt:latest",
                    "created_at": ISO8601DateFormatter().string(from: Date()),
                    "response": text,
                    "done": true,
                    "done_reason": "stop",
                    "prompt_eval_count": promptTokens,
                    "eval_count": completionTokens
                ]
                respondJSON(clientFd: clientFd, status: 200, payload: payload)
            } catch {
                respond(clientFd: clientFd, status: 500, body: "error: \(error)")
            }
        }

        private func streamOllamaChat(clientFd: Int32, prompt: String,
                                        maxTokens: Int, temperature: Float,
                                        stop: [String]) {
            writeNDJSONHead(clientFd: clientFd)
            var clientGone = false
            do {
                try inferenceQueue.sync {
                    try self.generateStreaming(prompt: prompt, maxTokens: maxTokens,
                                                temperature: temperature, stop: stop)
                    { newText in
                        let chunk: [String: Any] = [
                            "model": "tinygpt:latest",
                            "created_at": ISO8601DateFormatter().string(from: Date()),
                            "message": ["role": "assistant", "content": newText] as [String: Any],
                            "done": false
                        ]
                        let ok = self.writeNDJSONLine(clientFd: clientFd, payload: chunk)
                        if !ok { clientGone = true }
                        return ok
                    }
                }
            } catch { /* fall through to terminator */ }
            if clientGone { return }
            writeNDJSONLine(clientFd: clientFd, payload: [
                "model": "tinygpt:latest",
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "message": ["role": "assistant", "content": ""] as [String: Any],
                "done": true,
                "done_reason": "stop"
            ])
        }

        private func streamOllamaGenerate(clientFd: Int32, prompt: String,
                                            maxTokens: Int, temperature: Float,
                                            stop: [String]) {
            writeNDJSONHead(clientFd: clientFd)
            var clientGone = false
            do {
                try inferenceQueue.sync {
                    try self.generateStreaming(prompt: prompt, maxTokens: maxTokens,
                                                temperature: temperature, stop: stop)
                    { newText in
                        let chunk: [String: Any] = [
                            "model": "tinygpt:latest",
                            "created_at": ISO8601DateFormatter().string(from: Date()),
                            "response": newText,
                            "done": false
                        ]
                        let ok = self.writeNDJSONLine(clientFd: clientFd, payload: chunk)
                        if !ok { clientGone = true }
                        return ok
                    }
                }
            } catch { /* fall through to terminator */ }
            if clientGone { return }
            writeNDJSONLine(clientFd: clientFd, payload: [
                "model": "tinygpt:latest",
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "response": "",
                "done": true,
                "done_reason": "stop"
            ])
        }

        @discardableResult
        private func writeNDJSONHead(clientFd: Int32) -> Bool {
            var head = "HTTP/1.1 200 OK\r\n"
            head += "Content-Type: application/x-ndjson; charset=utf-8\r\n"
            head += "Cache-Control: no-cache\r\n"
            head += "Connection: close\r\n"
            head += "X-Accel-Buffering: no\r\n"
            head += "\r\n"
            return writeAll(clientFd: clientFd, data: Data(head.utf8))
        }

        @discardableResult
        private func writeNDJSONLine(clientFd: Int32, payload: [String: Any]) -> Bool {
            guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return false }
            var frame = data
            frame.append(Data("\n".utf8))
            return writeAll(clientFd: clientFd, data: frame)
        }

        private func readStopParam(_ raw: Any?) -> [String] {
            if let s = raw as? String { return [s] }
            if let arr = raw as? [String] { return arr }
            return []
        }

        private func readStopTokenIds(_ raw: Any?) -> Set<Int> {
            guard let raw, !(raw is NSNull) else { return [] }
            if let ids = raw as? [Int] { return Set(ids) }
            if let nums = raw as? [NSNumber] { return Set(nums.map { $0.intValue }) }
            if let one = raw as? Int { return [one] }
            if let one = raw as? NSNumber { return [one.intValue] }
            return []
        }

        private func grammarSpec(from raw: Any?) throws -> ServeGrammarSpec? {
            guard let raw, !(raw is NSNull) else { return defaultGrammar }
            guard let text = raw as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return defaultGrammar
            }
            return try ServeGrammarSpec.parse(text)
        }

        private func makeConstraint(_ spec: ServeGrammarSpec?) throws -> ServeConstraint? {
            guard let spec else { return nil }
            return ServeConstraint(spec: spec, tokenizer: tokenizer, vocabSize: config.vocabSize)
        }

        private func sampleToken(from logits: MLXArray, temperature: Float,
                                 constraint: ServeConstraint?) -> (token: Int, constraintMs: Double) {
            let sampled: MLXArray
            let sourceLogits: MLXArray
            var constraintMs = 0.0
            if let constraint {
                let start = InferenceTracer.nowNs()
                let mask = constraint.mask()
                constraintMs = InferenceTracer.ms(from: start, to: InferenceTracer.nowNs())
                sourceLogits = logits + MLXArray(mask, [mask.count])
            } else {
                sourceLogits = logits
            }
            if temperature <= 0 {
                sampled = argMax(sourceLogits, axis: -1)
            } else {
                let scaled = sourceLogits / MLXArray(temperature)
                sampled = MLXRandom.categorical(scaled)
            }
            eval(sampled)
            let token = Int(sampled.item(Int32.self))
            constraint?.commit(tokenId: token)
            return (token, constraintMs)
        }

        /// Parse the OpenAI Completions `prompt` field into a list of strings.
        /// The spec allows:
        ///   - String           → ["the string"]
        ///   - [String]         → as-is
        ///   - [[Int]]          → decode each token list via the tokenizer
        ///   - [Int]            → single token list (decode once)
        /// Anything unrecognized → empty list (handler turns into [""]).
        private func parsePromptField(_ raw: Any?) -> [String] {
            if raw == nil || raw is NSNull { return [""] }
            if let s = raw as? String { return [s] }
            if let arr = raw as? [String] { return arr.isEmpty ? [""] : arr }
            if let arr = raw as? [[Int]] {
                return arr.map { tokenizer.decode($0) }
            }
            if let arr = raw as? [Int] { return [tokenizer.decode(arr)] }
            // Mixed-type [Any] — try element-by-element.
            if let arr = raw as? [Any] {
                var out: [String] = []
                for item in arr {
                    if let s = item as? String { out.append(s) }
                    else if let ids = item as? [Int] { out.append(tokenizer.decode(ids)) }
                }
                if !out.isEmpty { return out }
            }
            return [""]
        }

        // MARK: Generate

        private struct PromptPrefill {
            let cache: KVCache
            let lastLogits: MLXArray
            let tokenDType: DType
        }

        private func traceSpan<T>(_ tracer: InferenceTracer?, _ name: String,
                                  _ body: () throws -> T) rethrows -> T {
            if let tracer {
                return try tracer.span(name, body)
            }
            return try body()
        }

        private func prefillPromptCache(kept: [Int], cachePrefix: String?,
                                        tracer: InferenceTracer? = nil) throws -> PromptPrefill {
            let uncached: () -> PromptPrefill = {
                let arr = MLXArray(kept.map { Int32($0) }, [1, kept.count])
                let cache = KVCache(nLayers: self.config.nLayers)
                let logits = self.traceSpan(tracer, "prefill_full") {
                    self.model.forwardCached(arr, cache: cache)
                }
                tracer?.setCache(enabled: self.promptCacheDir != nil, hit: false, prefixTokens: 0)
                return PromptPrefill(
                    cache: cache,
                    lastLogits: logits[0..., logits.shape[1] - 1, 0...],
                    tokenDType: arr.dtype
                )
            }
            guard let dir = promptCacheDir,
                  let prefixText = cachePrefix,
                  !prefixText.isEmpty
            else {
                return uncached()
            }
            let prefixIds = tokenizer.encode(prefixText)
            guard !prefixIds.isEmpty,
                  prefixIds.count <= kept.count,
                  Array(kept.prefix(prefixIds.count)) == prefixIds
            else {
                return uncached()
            }

            let key = KVCachePersist.Key(
                modelName: config.modelName,
                modelFileFingerprint: modelFingerprint,
                prompt: prefixText,
                vocabSize: config.vocabSize,
                nLayers: config.nLayers,
                kvTag: .fp32,
                useYOCO: config.useYOCO
            )
            let paths = KVCachePersist.paths(for: key, in: dir)
            if FileManager.default.fileExists(atPath: paths.cache.path) {
                do {
                    let cache = try traceSpan(tracer, "prompt_cache_load") {
                        try KVCache.load(from: paths.cache, nLayers: config.nLayers)
                    }
                    tracer?.setCache(enabled: true, hit: true, prefixTokens: prefixIds.count)
                    guard cache.currentLength == prefixIds.count else {
                        throw NSError(domain: "tinygpt.serve.prompt-cache", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey:
                                        "cache length \(cache.currentLength) != prefix \(prefixIds.count)"])
                    }
                    let remaining = Array(kept.dropFirst(prefixIds.count))
                    if remaining.isEmpty {
                        cache.rewind(by: 1)
                        let last = prefixIds[prefixIds.count - 1]
                        let arr = MLXArray([Int32(last)], [1, 1])
                        let logits = traceSpan(tracer, "prefill_tail") {
                            model.forwardCached(arr, cache: cache)
                        }
                        return PromptPrefill(
                            cache: cache,
                            lastLogits: logits[0..., logits.shape[1] - 1, 0...],
                            tokenDType: arr.dtype
                        )
                    }
                    let arr = MLXArray(remaining.map { Int32($0) }, [1, remaining.count])
                    let logits = traceSpan(tracer, "prefill_tail") {
                        model.forwardCached(arr, cache: cache)
                    }
                    return PromptPrefill(
                        cache: cache,
                        lastLogits: logits[0..., logits.shape[1] - 1, 0...],
                        tokenDType: arr.dtype
                    )
                } catch {
                    fputs("warning: serve prompt cache load failed (\(error)); rebuilding\n", stderr)
                }
            }

            let cache = KVCache(nLayers: config.nLayers)
            let prefixArr = MLXArray(prefixIds.map { Int32($0) }, [1, prefixIds.count])
            let prefixLogits = traceSpan(tracer, "prefill_prefix") {
                model.forwardCached(prefixArr, cache: cache)
            }
            tracer?.setCache(enabled: true, hit: false, prefixTokens: prefixIds.count)
            do {
                try traceSpan(tracer, "prompt_cache_save") {
                    try cache.saveToDisk(to: paths.cache)
                }
                let (bytes, _) = cache.totalBytes { _ in 4 }
                KVCachePersist.writeMeta(key, to: paths.meta,
                                         tokens: cache.currentLength,
                                         bytes: bytes)
            } catch {
                fputs("warning: serve prompt cache save failed: \(error)\n", stderr)
            }

            let remaining = Array(kept.dropFirst(prefixIds.count))
            if remaining.isEmpty {
                return PromptPrefill(
                    cache: cache,
                    lastLogits: prefixLogits[0..., prefixLogits.shape[1] - 1, 0...],
                    tokenDType: prefixArr.dtype
                )
            }
            let arr = MLXArray(remaining.map { Int32($0) }, [1, remaining.count])
            let logits = traceSpan(tracer, "prefill_tail") {
                model.forwardCached(arr, cache: cache)
            }
            return PromptPrefill(
                cache: cache,
                lastLogits: logits[0..., logits.shape[1] - 1, 0...],
                tokenDType: arr.dtype
            )
        }

        /// Encode the prompt, run a KV-cached generation loop, decode the
        /// completion, return text + token counts. Cache is built fresh
        /// each call — we don't keep state across HTTP requests because
        /// the harness sends independent prompts. Caching here is purely
        /// the *per-request* speedup (O(T) per step instead of O(T²)).
        ///
        /// Why KV-cached and not the simple growing-concat forward:
        /// uncached decode with a LoRA-wrapped Q/K/V projection appears
        /// to leak MLX graph nodes (per-layer per-step), and at ~500
        /// generated tokens the unified-memory pressure kills the
        /// process silently (no stderr, no stack). See
        /// `docs/prds/factory-serve-long-gen-crash.md`. Bare-base serve
        /// happens to survive because the per-step graph is smaller;
        /// the LoRA path tips it over. The KV-cached path matches
        /// `hf-load --sample`, which has always worked at 500+ tokens
        /// with the same base+adapter.
        func generate(prompt: String, maxTokens: Int, temperature: Float,
                       stop: [String], grammar: ServeGrammarSpec? = nil,
                       extraStopTokenIds: Set<Int> = [],
                       cachePrefix: String? = nil,
                       tracer: InferenceTracer? = nil) throws -> (String, Int, Int)
        {
            let promptIds = traceSpan(tracer, "tokenize_prompt") {
                tokenizer.encode(prompt)
            }
            if promptIds.isEmpty {
                return ("", 0, 0)
            }

            // Bound context: drop the head of the prompt if it overflows.
            // lm-eval's MMLU-Pro prompts can hit ~3K tokens — beyond the
            // model's default contextLength. For 0-shot tasks we cope by
            // truncating from the left so the most relevant tail survives.
            // Leave room for ≥1 generated token inside the window.
            let ctxCap = maxContext
            let promptCap = max(1, ctxCap - 1)
            let kept: [Int]
            if promptIds.count > promptCap {
                kept = Array(promptIds[(promptIds.count - promptCap)..<promptIds.count])
            } else {
                kept = promptIds
            }

            // Fresh KV cache; prefill on the bounded prompt, then per-step
            // [B,1] forwards keep the graph bounded.
            let prefill = try prefillPromptCache(kept: kept, cachePrefix: cachePrefix, tracer: tracer)
            let cache = prefill.cache
            var lastLogits = prefill.lastLogits
            let tokenDType = prefill.tokenDType
            let constraint = try traceSpan(tracer, "constraint_init") {
                try makeConstraint(grammar)
            }
            let activeStopTokenIds = eosTokenIds.union(extraStopTokenIds)

            var generated: [Int] = []
            generated.reserveCapacity(maxTokens)
            for tokenIndex in 0..<maxTokens {
                if constraint?.isComplete == true { break }
                let sampled = sampleToken(from: lastLogits, temperature: temperature,
                                          constraint: constraint)
                let tokenInt = sampled.token
                if activeStopTokenIds.contains(tokenInt) { break }
                generated.append(tokenInt)
                let nextId = MLXArray([Int32(tokenInt)], [1, 1])

                var decodeMs = 0.0
                // Stop-string detection: re-decode the running tail and check
                // whether any user-supplied stop string appears. We slice the
                // *decoded* output rather than ids because BPE tokens don't
                // align with characters.
                if !stop.isEmpty {
                    let decodeStart = InferenceTracer.nowNs()
                    let renderedSoFar = tokenizer.decode(generated)
                    decodeMs += InferenceTracer.ms(from: decodeStart, to: InferenceTracer.nowNs())
                    if stop.contains(where: { !$0.isEmpty && renderedSoFar.contains($0) }) {
                        // Trim everything from (and including) the stop string.
                        if let trimmed = trimAtStop(renderedSoFar, stops: stop) {
                            return (trimmed, promptIds.count, generated.count)
                        }
                    }
                }
                if cache.currentLength >= ctxCap { break }
                let modelStart = InferenceTracer.nowNs()
                let logits = model.forwardCached(nextId.asType(tokenDType), cache: cache)
                let modelMs = InferenceTracer.ms(from: modelStart, to: InferenceTracer.nowNs())
                lastLogits = logits[0..., 0, 0...]
                tracer?.addToken(index: tokenIndex,
                                 modelMs: modelMs,
                                 constraintMs: sampled.constraintMs,
                                 decodeMs: decodeMs)
            }
            let text = traceSpan(tracer, "tokenizer_decode_final") {
                tokenizer.decode(generated)
            }
            return (text, promptIds.count, generated.count)
        }

        private func trimAtStop(_ text: String, stops: [String]) -> String? {
            // Find the earliest matching stop in the decoded text and trim there.
            var earliest: String.Index? = nil
            for s in stops where !s.isEmpty {
                if let r = text.range(of: s) {
                    if earliest == nil || r.lowerBound < earliest! {
                        earliest = r.lowerBound
                    }
                }
            }
            guard let cut = earliest else { return nil }
            return String(text[..<cut])
        }

        /// OpenAI ChatCompletion `messages` → flat prompt. We use ChatML-ish
        /// formatting because that's what tinygpt's SFT templates can match;
        /// if your model was trained on a different template, prefer the
        /// `/v1/completions` endpoint and pass an already-formatted prompt.
        private func renderChatMessages(_ messages: [[String: Any]]) -> String {
            var out = ""
            if let toolsSystemPrompt {
                out += "<|im_start|>system\n\(toolsSystemPrompt)<|im_end|>\n"
            }
            for m in messages {
                let role = (m["role"] as? String) ?? "user"
                let content = (m["content"] as? String) ?? ""
                out += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
            out += "<|im_start|>assistant\n"
            return out
        }

        private func renderChatPromptCachePrefix(_ messages: [[String: Any]]) -> String? {
            var out = ""
            if let toolsSystemPrompt {
                out += "<|im_start|>system\n\(toolsSystemPrompt)<|im_end|>\n"
            }
            for m in messages {
                let role = (m["role"] as? String) ?? "user"
                guard role == "system" || role == "developer" else { break }
                let content = (m["content"] as? String) ?? ""
                out += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
            return out.isEmpty ? nil : out
        }

        // MARK: HTTP response helpers

        private func respond(clientFd: Int32, status: Int, body: String) {
            let statusText = httpStatusText(status)
            let bytes = [UInt8](body.utf8)
            var head = "HTTP/1.1 \(status) \(statusText)\r\n"
            head += "Content-Type: text/plain; charset=utf-8\r\n"
            head += "Content-Length: \(bytes.count)\r\n"
            head += "Connection: close\r\n"
            head += "\r\n"
            writeAll(clientFd: clientFd, data: Data(head.utf8))
            writeAll(clientFd: clientFd, data: Data(bytes))
        }

        private func respondJSON(clientFd: Int32, status: Int, payload: [String: Any]) {
            let data: Data
            do {
                data = try JSONSerialization.data(withJSONObject: payload)
            } catch {
                respond(clientFd: clientFd, status: 500, body: "json encode failed: \(error)")
                return
            }
            var head = "HTTP/1.1 \(status) \(httpStatusText(status))\r\n"
            head += "Content-Type: application/json; charset=utf-8\r\n"
            head += "Content-Length: \(data.count)\r\n"
            head += "Connection: close\r\n"
            head += "\r\n"
            writeAll(clientFd: clientFd, data: Data(head.utf8))
            writeAll(clientFd: clientFd, data: data)
        }

        /// Write all bytes to the client socket. Returns true on success,
        /// false if the peer disconnected mid-write (write returned 0 or
        /// -1 with errno=EPIPE/ECONNRESET). Callers that care about
        /// peer-gone (the streaming endpoints) check this to abort the
        /// generation loop early.
        @discardableResult
        private func writeAll(clientFd: Int32, data: Data) -> Bool {
            var ok = true
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { ok = false; return }
                var sent = 0
                while sent < data.count {
                    let n = Darwin.write(clientFd, base.advanced(by: sent), data.count - sent)
                    if n <= 0 { ok = false; return }
                    sent += n
                }
            }
            return ok
        }

        private func httpStatusText(_ code: Int) -> String {
            switch code {
            case 200: return "OK"
            case 400: return "Bad Request"
            case 404: return "Not Found"
            case 500: return "Internal Server Error"
            default:  return "Status"
            }
        }
    }

    // MARK: Tokenizer wrapper

    /// `TokenizerBox` lets `Server` work with either ByteTokenizer or
    /// HFTokenizer without the rest of the file caring. Same shape as
    /// Sample.swift's logic, factored for reuse.
    public enum TokenizerBox: @unchecked Sendable {
        case byteLevel
        case hf(HFTokenizer)

        func encode(_ text: String) -> [Int] {
            switch self {
            case .byteLevel: return [UInt8](text.utf8).map { Int($0) }
            case .hf(let t):
                do { return try t.encode(text) }
                catch { return [] }
            }
        }
        func decode(_ ids: [Int]) -> String {
            switch self {
            case .byteLevel:
                let bytes = ids.compactMap { (id: Int) -> UInt8? in
                    guard id >= 0 && id < 256 else { return nil }
                    return UInt8(id)
                }
                return String(decoding: bytes, as: UTF8.self)
            case .hf(let t):
                return t.decode(ids)
            }
        }
    }
}

// MARK: - Serve-side constrained decoding

protocol ServeByteFSM: AnyObject {
    var isComplete: Bool { get }
    func cloneForServe() -> ServeByteFSM
    func acceptBytes(_ bytes: [UInt8]) -> Bool
}

extension JSONSchemaFSM: ServeByteFSM {
    func cloneForServe() -> ServeByteFSM { clone() }
}

struct ServeGrammarSpec {
    enum Kind {
        case jsonSchema(JSONSchemaNode)
        case paceToolTags
    }

    let kind: Kind

    static func parse(_ text: String) throws -> ServeGrammarSpec {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{") {
            let data = Data(trimmed.utf8)
            return ServeGrammarSpec(kind: .jsonSchema(try JSONSchemaNode.from(data: data)))
        }
        if trimmed.contains("root ::="), trimmed.contains("[POINT:") {
            return ServeGrammarSpec(kind: .paceToolTags)
        }
        throw NSError(domain: "tinygpt.serve.grammar", code: 1,
                      userInfo: [NSLocalizedDescriptionKey:
                        "unsupported grammar: pass a JSON Schema or grammars/pace-tool-tags.gbnf"])
    }
}

final class ServeConstraint {
    private let masker: ServeTokenMasker
    private let fsm: ServeByteFSM

    var isComplete: Bool { fsm.isComplete }

    init(spec: ServeGrammarSpec, tokenizer: Serve.TokenizerBox, vocabSize: Int) {
        let decodeId: (Int) -> String = { id in tokenizer.decode([id]) }
        self.masker = ServeTokenMasker(vocabSize: vocabSize, decodeId: decodeId)
        switch spec.kind {
        case .jsonSchema(let node):
            self.fsm = JSONSchemaFSM(rootSchema: node)
        case .paceToolTags:
            self.fsm = PaceToolTagFSM()
        }
    }

    func mask() -> [Float] {
        masker.mask(for: fsm)
    }

    func commit(tokenId: Int) {
        _ = masker.commit(tokenId: tokenId, into: fsm)
    }
}

final class ServeTokenMasker {
    private let tokenBytes: [[UInt8]]

    init(vocabSize: Int, decodeId: (Int) -> String) {
        var out: [[UInt8]] = []
        out.reserveCapacity(vocabSize)
        for id in 0..<vocabSize {
            out.append(Array(decodeId(id).utf8))
        }
        self.tokenBytes = out
    }

    func mask(for fsm: ServeByteFSM) -> [Float] {
        var out = [Float](repeating: -.infinity, count: tokenBytes.count)
        if fsm.isComplete { return out }
        for id in 0..<tokenBytes.count {
            let bytes = tokenBytes[id]
            if bytes.isEmpty { continue }
            let probe = fsm.cloneForServe()
            if probe.acceptBytes(bytes) {
                out[id] = 0
            }
        }
        return out
    }

    @discardableResult
    func commit(tokenId: Int, into fsm: ServeByteFSM) -> Bool {
        guard tokenId >= 0 && tokenId < tokenBytes.count else { return false }
        return fsm.acceptBytes(tokenBytes[tokenId])
    }
}

final class PaceToolTagFSM: ServeByteFSM {
    private var bytes: [UInt8] = []

    var isComplete: Bool {
        guard let text = String(bytes: bytes, encoding: .utf8),
              let bracket = text.firstIndex(of: "[")
        else { return false }
        return Self.isCompleteTag(String(text[bracket...]))
    }

    func cloneForServe() -> ServeByteFSM {
        let copy = PaceToolTagFSM()
        copy.bytes = bytes
        return copy
    }

    func acceptBytes(_ newBytes: [UInt8]) -> Bool {
        var candidate = bytes
        for byte in newBytes {
            candidate.append(byte)
            guard Self.isViable(candidate) else { return false }
        }
        bytes = candidate
        return true
    }

    private static func isViable(_ bytes: [UInt8]) -> Bool {
        guard let text = String(bytes: bytes, encoding: .utf8) else { return false }
        guard let bracket = text.firstIndex(of: "[") else {
            return text.unicodeScalars.allSatisfy(isTextScalar)
        }
        let before = text[..<bracket]
        guard before.unicodeScalars.allSatisfy(isTextScalar) else { return false }
        let tag = String(text[bracket...])
        return isTagPrefix(tag)
    }

    private static func isTextScalar(_ s: UnicodeScalar) -> Bool {
        if s.value == 0x0A || s.value == 0x0D || s.value == 0x09 { return false }
        if s == "[" || s == "]" { return false }
        return s.value >= 0x20 && s.value <= 0x7E
    }

    private static func isTagPrefix(_ tag: String) -> Bool {
        if "[POINT:none]".hasPrefix(tag) { return true }
        if tag.hasPrefix("[POINT:") {
            return coordsLabelPrefix(String(tag.dropFirst("[POINT:".count)))
        }
        if "[POINT:".hasPrefix(tag) { return true }
        if tag.hasPrefix("[CLICK:") {
            return coordsLabelPrefix(String(tag.dropFirst("[CLICK:".count)))
        }
        if "[CLICK:".hasPrefix(tag) { return true }
        if tag.hasPrefix("[TYPE:") {
            return bodyPrefix(String(tag.dropFirst("[TYPE:".count)),
                              min: 1, max: 200, first: typeScalar, rest: typeScalar)
        }
        if "[TYPE:".hasPrefix(tag) { return true }
        if tag.hasPrefix("[SCROLL:") {
            return scrollPrefix(String(tag.dropFirst("[SCROLL:".count)))
        }
        if "[SCROLL:".hasPrefix(tag) { return true }
        if tag.hasPrefix("[KEY:") {
            return bodyPrefix(String(tag.dropFirst("[KEY:".count)),
                              min: 1, max: 30, first: keyScalar, rest: keyScalar)
        }
        if "[KEY:".hasPrefix(tag) { return true }
        if tag.hasPrefix("[OPEN_APP:") {
            return bodyPrefix(String(tag.dropFirst("[OPEN_APP:".count)),
                              min: 1, max: 31, first: alphaScalar, rest: appScalar)
        }
        if "[OPEN_APP:".hasPrefix(tag) { return true }
        return false
    }

    private static func coordsLabelPrefix(_ s: String) -> Bool {
        let scalars = Array(s.unicodeScalars)
        var i = 0
        let x = consumeDigits(scalars, &i, max: 4)
        if i == scalars.count { return x <= 4 }
        guard x >= 1, scalars[i] == "," else { return false }
        i += 1
        let y = consumeDigits(scalars, &i, max: 4)
        if i == scalars.count { return y <= 4 }
        guard y >= 1, scalars[i] == ":" else { return false }
        i += 1
        return bodyPrefix(Array(scalars[i...]), min: 1, max: 41,
                          first: alphaScalar, rest: labelScalar)
    }

    private static func scrollPrefix(_ s: String) -> Bool {
        for dir in ["up", "down", "left", "right"] {
            if dir.hasPrefix(s) { return true }
            if s.hasPrefix(dir + ":") {
                let rest = String(s.dropFirst((dir + ":").count))
                return digitBodyPrefix(rest, min: 1, max: 4)
            }
        }
        return false
    }

    private static func digitBodyPrefix(_ s: String, min: Int, max: Int) -> Bool {
        return bodyPrefix(s, min: min, max: max, first: digitScalar, rest: digitScalar)
    }

    private static func bodyPrefix(_ s: String, min: Int, max: Int,
                                   first: (UnicodeScalar) -> Bool,
                                   rest: (UnicodeScalar) -> Bool) -> Bool {
        return bodyPrefix(Array(s.unicodeScalars), min: min, max: max, first: first, rest: rest)
    }

    private static func bodyPrefix(_ scalars: [UnicodeScalar], min: Int, max: Int,
                                   first: (UnicodeScalar) -> Bool,
                                   rest: (UnicodeScalar) -> Bool) -> Bool {
        if scalars.isEmpty { return true }
        var count = 0
        for (idx, scalar) in scalars.enumerated() {
            if scalar == "]" {
                return idx == scalars.count - 1 && count >= min
            }
            let ok = count == 0 ? first(scalar) : rest(scalar)
            guard ok else { return false }
            count += 1
            if count > max { return false }
        }
        return true
    }

    private static func consumeDigits(_ scalars: [UnicodeScalar], _ i: inout Int, max: Int) -> Int {
        var count = 0
        while i < scalars.count, digitScalar(scalars[i]) {
            count += 1
            if count > max { return count }
            i += 1
        }
        return count
    }

    private static func isCompleteTag(_ tag: String) -> Bool {
        return tag.range(of: #"^\[(POINT:none|POINT:[0-9]{1,4},[0-9]{1,4}:[A-Za-z][A-Za-z0-9 _-]{0,40}|CLICK:[0-9]{1,4},[0-9]{1,4}:[A-Za-z][A-Za-z0-9 _-]{0,40}|TYPE:[A-Za-z0-9 _.,!?'"-]{1,200}|SCROLL:(up|down|left|right):[0-9]{1,4}|KEY:[A-Za-z+]{1,30}|OPEN_APP:[A-Za-z][A-Za-z0-9 ]{0,30})\]$"#,
                         options: .regularExpression) != nil
    }

    private static func digitScalar(_ s: UnicodeScalar) -> Bool {
        s.value >= 48 && s.value <= 57
    }

    private static func alphaScalar(_ s: UnicodeScalar) -> Bool {
        (s.value >= 65 && s.value <= 90) || (s.value >= 97 && s.value <= 122)
    }

    private static func keyScalar(_ s: UnicodeScalar) -> Bool {
        alphaScalar(s) || s == "+"
    }

    private static func appScalar(_ s: UnicodeScalar) -> Bool {
        alphaScalar(s) || digitScalar(s) || s == " "
    }

    private static func labelScalar(_ s: UnicodeScalar) -> Bool {
        alphaScalar(s) || digitScalar(s) || s == " " || s == "_" || s == "-"
    }

    private static func typeScalar(_ s: UnicodeScalar) -> Bool {
        alphaScalar(s) || digitScalar(s) || s == " " || s == "_" || s == "-" ||
        s == "." || s == "," || s == "!" || s == "?" || s == "'" || s == "\""
    }
}

// MARK: - HTTP request parser
//
// Hand-rolled because we want zero deps. Reads the request line, header
// block (until "\r\n\r\n"), then drains Content-Length bytes of body.
// No chunked encoding (lm-eval-harness doesn't use it).

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    static func read(from fd: Int32) -> HTTPRequest? {
        var buf = Data()
        var chunk = [UInt8](repeating: 0, count: 4096)
        // Read until we see the end of the header block.
        while true {
            let n = chunk.withUnsafeMutableBufferPointer { ptr -> Int in
                Darwin.read(fd, ptr.baseAddress, ptr.count)
            }
            if n <= 0 { return nil }
            buf.append(chunk, count: n)
            if let _ = buf.range(of: Data("\r\n\r\n".utf8)) {
                break
            }
            if buf.count > 16 * 1024 * 1024 { return nil }
        }
        guard let headerEnd = buf.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let headerData = buf.prefix(upTo: headerEnd.lowerBound)
        let headerText = String(decoding: headerData, as: UTF8.self)
        let lines = headerText.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard let firstLine = lines.first else { return nil }
        let parts = firstLine.split(separator: " ", maxSplits: 2)
        guard parts.count >= 2 else { return nil }
        let method = String(parts[0])
        let path = String(parts[1])
        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            if let colon = line.firstIndex(of: ":") {
                let k = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
                let v = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                headers[k] = v
            }
        }

        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        var body = buf.suffix(from: headerEnd.upperBound)
        while body.count < contentLength {
            let n = chunk.withUnsafeMutableBufferPointer { ptr -> Int in
                Darwin.read(fd, ptr.baseAddress, ptr.count)
            }
            if n <= 0 { break }
            body.append(chunk, count: n)
        }
        // Trim if we read past Content-Length (extra pipelined bytes).
        if body.count > contentLength {
            body = body.prefix(contentLength)
        }
        return HTTPRequest(method: method, path: path, headers: headers, body: Data(body))
    }
}
