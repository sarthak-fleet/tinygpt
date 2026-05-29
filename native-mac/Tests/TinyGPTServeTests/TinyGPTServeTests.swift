import Foundation
import XCTest
import MLX
import TinyGPTIO
@testable import TinyGPTServe
@testable import TinyGPTModel

/// Verifies the OpenAI-compatible HTTP server boots, responds on the
/// well-known endpoints, and produces JSON in the expected shape.
///
/// SAME RUN CAVEAT AS TinyGPTModelTests: the MLX runtime needs the Metal
/// library compiled, which only Xcode does. Run via:
///   `xcodebuild test -scheme TinyGPT -destination "platform=macOS"`
/// `swift test` will fail at module load with "Failed to load the default
/// metallib." (the test itself is fine, MLX's init isn't).
///
/// The tests require a `.tinygpt` checkpoint on disk. We probe two paths:
///   • `TINYGPT_TEST_MODEL` env var
///   • `/tmp/flagship-huge.tinygpt` (the flagship checkpoint the user has
///     locally — same one driving the rest of the eval wiring).
///
/// If neither is present we `XCTSkip` so the suite stays green in CI
/// without bundling a multi-megabyte fixture in-repo. The HTTP-parser unit
/// test (`test_httpRequestParser_*`) doesn't need a model and always runs.
final class TinyGPTServeTests: XCTestCase {

    // MARK: HTTP parser (no model needed)

    /// Direct test of the HTTP request parser — the smallest unit of the
    /// server stack that's testable without MLX. We write a request to a
    /// pipe and call `HTTPRequest.read(from:)` on the read end.
    func test_httpRequestParser_acceptsPostWithJSONBody() throws {
        let raw = "POST /v1/chat/completions HTTP/1.1\r\n" +
                  "Content-Type: application/json\r\n" +
                  "Content-Length: 17\r\n" +
                  "\r\n" +
                  "{\"prompt\":\"hi\"}\r\n"
        let req = try parseRawHTTPRequest(raw)
        XCTAssertEqual(req.method, "POST")
        XCTAssertEqual(req.path, "/v1/chat/completions")
        XCTAssertEqual(req.headers["content-type"], "application/json")
        XCTAssertEqual(req.headers["content-length"], "17")
        XCTAssertEqual(req.body.count, 17)
    }

    func test_httpRequestParser_acceptsGet() throws {
        let raw = "GET /v1/models HTTP/1.1\r\n" +
                  "Host: 127.0.0.1\r\n" +
                  "\r\n"
        let req = try parseRawHTTPRequest(raw)
        XCTAssertEqual(req.method, "GET")
        XCTAssertEqual(req.path, "/v1/models")
        XCTAssertEqual(req.body.count, 0)
    }

    private func parseRawHTTPRequest(_ raw: String) throws -> HTTPRequest {
        // Build a pipe(2), write the request to the writer end, then ask
        // `HTTPRequest.read(from:)` to consume it from the reader end.
        var fds = [Int32](repeating: 0, count: 2)
        let p = fds.withUnsafeMutableBufferPointer { buf in
            pipe(buf.baseAddress)
        }
        XCTAssertEqual(p, 0)
        let readFd = fds[0]
        let writeFd = fds[1]
        defer { close(readFd); close(writeFd) }

        let bytes = [UInt8](raw.utf8)
        _ = bytes.withUnsafeBufferPointer { ptr -> Int in
            write(writeFd, ptr.baseAddress, ptr.count)
        }
        // Close the writer so `read` sees EOF when it's drained the body.
        close(writeFd)
        guard let req = HTTPRequest.read(from: readFd) else {
            throw NSError(domain: "test", code: 1)
        }
        return req
    }

    // MARK: Live server (requires a model checkpoint)

    /// Bring up the server against the user's real checkpoint and verify
    /// `/v1/models` returns "tinygpt" — proves the bind, accept, and
    /// HTTP-response path all work.
    func test_modelsEndpoint_returnsTinygptModel() throws {
        let modelPath = try resolveModelPathOrSkip()
        let server = try Serve.start(modelPath: modelPath, host: "127.0.0.1", port: 0)
        defer { server.stop() }

        let url = URL(string: "http://127.0.0.1:\(server.port)/v1/models")!
        let (data, status) = blockingGET(url: url)
        XCTAssertEqual(status, 200)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["object"] as? String, "list")
        let list = json?["data"] as? [[String: Any]]
        XCTAssertEqual(list?.first?["id"] as? String, "tinygpt")
    }

    /// Verifies a /v1/chat/completions request returns a well-formed
    /// ChatCompletion. We don't assert on the content (it's whatever the
    /// loaded model produces) but we do require the JSON shape to match
    /// what lm-evaluation-harness's `local-chat-completions` expects.
    func test_chatCompletions_returnsValidJSON() throws {
        let modelPath = try resolveModelPathOrSkip()
        let server = try Serve.start(modelPath: modelPath, host: "127.0.0.1", port: 0)
        defer { server.stop() }

        let url = URL(string: "http://127.0.0.1:\(server.port)/v1/chat/completions")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "model": "tinygpt",
            "messages": [["role": "user", "content": "hi"]],
            "max_tokens": 4,
            "temperature": 0.0,
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, status) = blockingRequest(req)
        XCTAssertEqual(status, 200)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["object"] as? String, "chat.completion")
        XCTAssertEqual(json?["model"] as? String, "tinygpt")
        let choices = json?["choices"] as? [[String: Any]]
        XCTAssertNotNil(choices?.first?["message"] as? [String: Any])
        XCTAssertEqual((choices?.first?["message"] as? [String: Any])?["role"] as? String, "assistant")
        let usage = json?["usage"] as? [String: Any]
        XCTAssertNotNil(usage?["prompt_tokens"])
        XCTAssertNotNil(usage?["completion_tokens"])
    }

    /// Verifies the /v1/completions endpoint. Same shape requirements as
    /// chat but with `text` instead of `message`.
    func test_completions_returnsValidJSON() throws {
        let modelPath = try resolveModelPathOrSkip()
        let server = try Serve.start(modelPath: modelPath, host: "127.0.0.1", port: 0)
        defer { server.stop() }

        let url = URL(string: "http://127.0.0.1:\(server.port)/v1/completions")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "model": "tinygpt",
            "prompt": "ROMEO",
            "max_tokens": 4,
            "temperature": 0.0,
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, status) = blockingRequest(req)
        XCTAssertEqual(status, 200)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["object"] as? String, "text_completion")
        let choices = json?["choices"] as? [[String: Any]]
        XCTAssertNotNil(choices?.first?["text"] as? String)
    }

    // MARK: - Helpers

    /// Returns a path to a usable `.tinygpt` (or HF dir) checkpoint, or
    /// `XCTSkip`s the calling test if none is reachable. We keep the
    /// test suite green for fresh clones without forcing a multi-MB
    /// fixture into the repo.
    private func resolveModelPathOrSkip() throws -> String {
        if let env = ProcessInfo.processInfo.environment["TINYGPT_TEST_MODEL"],
           FileManager.default.fileExists(atPath: env) {
            return env
        }
        let flagship = "/tmp/flagship-huge.tinygpt"
        if FileManager.default.fileExists(atPath: flagship) {
            return flagship
        }
        throw XCTSkip("No TINYGPT_TEST_MODEL set and /tmp/flagship-huge.tinygpt missing.")
    }

    /// Tiny blocking URLSession wrapper. We avoid async/await in tests to
    /// keep the failure mode obvious — if a request hangs, the test times
    /// out at the XCTest level, not in a Task continuation.
    private func blockingGET(url: URL) -> (Data, Int) {
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = 60
        return blockingRequest(req)
    }

    private func blockingRequest(_ req: URLRequest) -> (Data, Int) {
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var resultData = Data()
        nonisolated(unsafe) var resultStatus = -1
        let task = URLSession.shared.dataTask(with: req) { data, response, _ in
            if let data = data { resultData = data }
            if let http = response as? HTTPURLResponse { resultStatus = http.statusCode }
            sem.signal()
        }
        task.resume()
        _ = sem.wait(timeout: .now() + 120)
        return (resultData, resultStatus)
    }
}
