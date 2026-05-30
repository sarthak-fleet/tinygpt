import Foundation

/// HuggingFace Datasets Hub API client.
///
/// Tinygpt is becoming an on-device SLM factory; training data is the
/// bottleneck. HF Hub hosts the world's training corpora (xLAM, Hermes
/// function-calling, OpenHermes-2.5, MetaMath, UltraFeedback, etc.).
/// This client provides the read-side moat: resolve a dataset by id,
/// list its shards via the Hub API, stream them to a local cache, and
/// hand back URLs for downstream conversion to tinygpt's JSONL / plain
/// formats.
///
/// References:
///   - Datasets server API:  https://huggingface.co/docs/datasets-server
///   - LFS resolve endpoint: https://huggingface.co/<id>/resolve/<rev>/<path>
///   - Auto-converted parquet:
///     https://huggingface.co/api/datasets/<id>/parquet/<config>/<split>/<n>.parquet
///
/// Auth: if `HF_TOKEN` is set in the environment we send it as a Bearer
/// token. Gated datasets (Llama-3 tokenizer, some preference data) fail
/// with HTTP 401/403 if unauthenticated — we surface a clean message
/// rather than dump JSON to stderr.
public enum HFDatasets {

    // MARK: - Errors

    public enum HFError: Error, CustomStringConvertible {
        case http(status: Int, url: String, body: String)
        case malformedResponse(String)
        case network(String)
        case missingAuth(datasetId: String)
        case notFound(datasetId: String)
        case ioError(String)
        public var description: String {
            switch self {
            case .http(let s, let u, let b):
                let snippet = b.count > 200 ? String(b.prefix(200)) + "..." : b
                return "HTTP \(s) for \(u)\n  body: \(snippet)"
            case .malformedResponse(let s): return "malformed HF response: \(s)"
            case .network(let s): return "network error: \(s)"
            case .missingAuth(let id):
                return """
                dataset '\(id)' returned 401/403. This means one of:
                  - the dataset is gated and you need to accept its license
                  - the dataset is private (you need an HF_TOKEN with read access)
                  - the id is misspelled (HF Hub returns 401, not 404, for unknown ids
                    to avoid leaking which datasets exist)
                If the id is correct, set:
                    export HF_TOKEN=hf_xxx
                Get one at https://huggingface.co/settings/tokens.
                """
            case .notFound(let id):
                return "dataset '\(id)' not found on HF Hub (or you don't have access)"
            case .ioError(let s): return "I/O error: \(s)"
            }
        }
    }

    // MARK: - Types

    /// One file in a dataset repo, as reported by the HF API.
    public struct Sibling: Sendable {
        public let rfilename: String          // path relative to the repo root
        public let size: Int?                 // bytes, when reported
        public init(rfilename: String, size: Int? = nil) {
            self.rfilename = rfilename; self.size = size
        }
    }

    /// Metadata returned by `GET /api/datasets/<id>`.
    /// `cardData` is intentionally omitted from the typed shape — it's a
    /// free-form JSON blob and dragging it through Sendable would require
    /// wrapping `Any`. If callers need it, fetch it via `info(id:)`'s raw
    /// HTTP path (we keep the API surface minimal).
    public struct DatasetInfo: Sendable {
        public let id: String                 // canonical "owner/name"
        public let sha: String?
        public let lastModified: String?
        public let downloads: Int?
        public let likes: Int?
        public let tags: [String]
        public let siblings: [Sibling]
        public let gated: Bool                // gated=true means HF_TOKEN required even for read
        public let `private`: Bool

        /// Total size in bytes of all siblings (best-effort — `size`
        /// may be nil for older snapshots).
        public var totalBytes: Int {
            siblings.compactMap(\.size).reduce(0, +)
        }
    }

    /// One row of the auto-converted parquet endpoint. Hub auto-converts
    /// most public datasets; some (huge ones, weird schemas) don't have
    /// the conversion. We fall back to the raw siblings list in that case.
    public struct ParquetFile: Sendable {
        public let url: String
        public let filename: String
        public let config: String
        public let split: String
        public let size: Int?
    }

    // MARK: - Public API

    /// Fetch dataset info for `id` (e.g. "Salesforce/xlam-function-calling-60k").
    /// Throws `HFError.notFound` on 404, `HFError.missingAuth` on 401/403.
    public static func info(id: String) throws -> DatasetInfo {
        let url = "https://huggingface.co/api/datasets/\(id)"
        let (data, status) = try httpGet(url)
        if status == 404 { throw HFError.notFound(datasetId: id) }
        if status == 401 || status == 403 { throw HFError.missingAuth(datasetId: id) }
        guard status == 200 else {
            throw HFError.http(status: status, url: url, body: String(data: data, encoding: .utf8) ?? "")
        }
        return try decodeInfo(data: data, id: id)
    }

    /// List the auto-converted parquet shards for a dataset, if HF has
    /// converted it. Returns an empty array when the conversion endpoint
    /// 404s (i.e. dataset isn't convertible / too big / etc).
    public static func parquetFiles(id: String) throws -> [ParquetFile] {
        let url = "https://huggingface.co/api/datasets/\(id)/parquet"
        let (data, status) = try httpGet(url)
        if status == 404 { return [] }
        if status == 401 || status == 403 { throw HFError.missingAuth(datasetId: id) }
        guard status == 200 else {
            throw HFError.http(status: status, url: url, body: String(data: data, encoding: .utf8) ?? "")
        }
        return try decodeParquetIndex(data: data, id: id)
    }

    /// Resolve a sibling filename to its raw download URL (LFS-redirected
    /// for large files).
    public static func resolveFileURL(id: String, filename: String, revision: String = "main") -> String {
        // HF urlencodes the path components; in practice dataset names and
        // siblings are filesystem-safe (no spaces, no #). We percent-encode
        // each path segment defensively.
        let encodedFilename = filename
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        return "https://huggingface.co/datasets/\(id)/resolve/\(revision)/\(encodedFilename)"
    }

    /// Default cache root: `~/.cache/tinygpt/datasets/`.
    /// We use `HF_HOME` only as a hint — never write into HuggingFace's own
    /// cache layout, so users can `rm -rf` tinygpt's cache without
    /// affecting their `~/.cache/huggingface` directory.
    public static func cacheRoot() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let override = env["TINYGPT_DATASET_CACHE"], !override.isEmpty {
            return URL(fileURLWithPath: (override as NSString).expandingTildeInPath, isDirectory: true)
        }
        let home = env["HOME"].map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/tinygpt/datasets", isDirectory: true)
    }

    /// `cacheRoot / <id>` — `id` may contain '/' which we keep as a
    /// subdirectory split. The cache dir is created on demand.
    public static func cacheDir(for id: String) throws -> URL {
        let dir = cacheRoot().appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Download a single file from a dataset. Resumes by skipping if the
    /// destination already exists with the expected size (when known).
    /// Progress reporting goes to stderr — we don't have a TUI here.
    @discardableResult
    public static func downloadFile(
        id: String,
        filename: String,
        expectedSize: Int? = nil,
        progress: ((Int64, Int64) -> Void)? = nil
    ) throws -> URL {
        let dir = try cacheDir(for: id)
        let dest = dir.appendingPathComponent(filename)
        // Make sure subdirs (e.g. "data/train-00000-of-00010.parquet") exist.
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        // Resume: if size matches expected, skip. If expected is unknown,
        // we skip only if the file is non-empty (best-effort).
        if FileManager.default.fileExists(atPath: dest.path) {
            let attrs = (try? FileManager.default.attributesOfItem(atPath: dest.path)) ?? [:]
            let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
            if let expected = expectedSize, size == expected, size > 0 {
                progress?(Int64(size), Int64(size))
                return dest
            }
            if expectedSize == nil && size > 0 {
                // Best-effort: trust the cached file if we don't know its
                // expected size. Pass `expectedSize` to force a re-download.
                progress?(Int64(size), Int64(size))
                return dest
            }
        }
        let urlString = resolveFileURL(id: id, filename: filename)
        try streamDownload(urlString: urlString, to: dest, progress: progress)
        return dest
    }

    // MARK: - HTTP plumbing

    /// Synchronous GET that returns (data, status). We deliberately keep
    /// this synchronous so the CLI flow reads top-to-bottom; if/when this
    /// goes into the SwiftUI app we'll lift it to async.
    ///
    /// Swift 6 strict concurrency: result/error are exchanged through a
    /// small class instead of captured `var`s (closure capture of `var`
    /// is illegal across actor isolation).
    public static func httpGet(_ urlString: String) throws -> (Data, Int) {
        guard let url = URL(string: urlString) else {
            throw HFError.malformedResponse("bad URL: \(urlString)")
        }
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        req.setValue("tinygpt/0.1 (+https://github.com/sarthak/tinygpt)", forHTTPHeaderField: "User-Agent")

        let box = ResultBox()
        let sema = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: req) { data, response, err in
            defer { sema.signal() }
            if let err = err { box.error = err; return }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            box.data = data ?? Data()
            box.status = status
        }
        task.resume()
        sema.wait()
        if let error = box.error { throw HFError.network("\(error)") }
        guard let data = box.data else { throw HFError.network("no response") }
        return (data, box.status)
    }

    /// Streaming download. URLSession's downloadTask handles arbitrarily
    /// large files by writing to disk as bytes arrive, so we can pull
    /// multi-GB parquet shards without blowing memory.
    public static func streamDownload(
        urlString: String,
        to dest: URL,
        progress: ((Int64, Int64) -> Void)?
    ) throws {
        guard let url = URL(string: urlString) else {
            throw HFError.malformedResponse("bad URL: \(urlString)")
        }
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        req.setValue("tinygpt/0.1", forHTTPHeaderField: "User-Agent")

        let delegate = DownloadDelegate(destination: dest, progress: progress)
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 60 * 60 * 6   // 6h for multi-GB shards
        let session = URLSession(configuration: cfg, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let task = session.downloadTask(with: req)
        task.resume()
        delegate.semaphore.wait()
        if let err = delegate.error { throw HFError.network("\(err)") }
        if delegate.statusCode == 401 || delegate.statusCode == 403 {
            throw HFError.missingAuth(datasetId: url.path)
        }
        if delegate.statusCode == 404 {
            throw HFError.http(status: 404, url: urlString, body: "not found")
        }
        if delegate.statusCode != 0 && delegate.statusCode >= 400 {
            throw HFError.http(status: delegate.statusCode, url: urlString, body: "")
        }
    }

    // MARK: - JSON decode

    /// Decode the `/api/datasets/<id>` JSON into a typed DatasetInfo. We
    /// parse permissively — HF adds new fields frequently and we shouldn't
    /// fail on schema drift.
    static func decodeInfo(data: Data, id: String) throws -> DatasetInfo {
        let obj: [String: Any]
        do {
            obj = (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        } catch { throw HFError.malformedResponse("\(error)") }
        let canonicalId = (obj["id"] as? String) ?? id
        let sha = obj["sha"] as? String
        let lastModified = obj["lastModified"] as? String
        let downloads = (obj["downloads"] as? NSNumber)?.intValue
        let likes = (obj["likes"] as? NSNumber)?.intValue
        let tags = (obj["tags"] as? [String]) ?? []
        let gated: Bool = {
            // HF returns `gated`: false | "auto" | "manual" — coerce
            // anything non-falsy to true.
            if let b = obj["gated"] as? Bool { return b }
            if let s = obj["gated"] as? String { return !s.isEmpty && s != "false" }
            return false
        }()
        let `private` = (obj["private"] as? Bool) ?? false
        var siblings: [Sibling] = []
        for sib in (obj["siblings"] as? [[String: Any]] ?? []) {
            guard let name = sib["rfilename"] as? String else { continue }
            let size = (sib["size"] as? NSNumber)?.intValue
            siblings.append(Sibling(rfilename: name, size: size))
        }
        return DatasetInfo(id: canonicalId, sha: sha, lastModified: lastModified,
                           downloads: downloads, likes: likes, tags: tags,
                           siblings: siblings, gated: gated, private: `private`)
    }

    /// Decode the `/api/datasets/<id>/parquet` JSON. The shape is:
    ///   { "<config>": { "<split>": ["url1", "url2", ...] } }
    static func decodeParquetIndex(data: Data, id: String) throws -> [ParquetFile] {
        let obj: Any
        do {
            obj = try JSONSerialization.jsonObject(with: data)
        } catch { throw HFError.malformedResponse("\(error)") }
        var result: [ParquetFile] = []
        // Top level is a {config -> {split -> [url]}} dict.
        guard let configs = obj as? [String: Any] else { return [] }
        for (config, splitsAny) in configs {
            guard let splits = splitsAny as? [String: Any] else { continue }
            for (split, urlsAny) in splits {
                guard let urls = urlsAny as? [String] else { continue }
                for url in urls {
                    let name = url.split(separator: "/").last.map(String.init) ?? url
                    result.append(ParquetFile(url: url, filename: name, config: config, split: split, size: nil))
                }
            }
        }
        return result
    }
}

/// Reference cell shared with closures so we can mutate fields under
/// Swift 6 strict concurrency (closures can't capture `var`s). Used by
/// the synchronous `httpGet` wrapper.
private final class ResultBox: @unchecked Sendable {
    var data: Data?
    var status: Int = 0
    var error: Error?
}

/// URLSessionDownloadDelegate that drives our synchronous streamDownload
/// helper. Writes the temp file to its final destination on completion.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let destination: URL
    let progress: ((Int64, Int64) -> Void)?
    let semaphore = DispatchSemaphore(value: 0)
    var error: Error?
    var statusCode: Int = 0

    init(destination: URL, progress: ((Int64, Int64) -> Void)?) {
        self.destination = destination
        self.progress = progress
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // Capture HTTP status before moving the file — a 404 still produces
        // a "download" with the JSON error body.
        if let resp = downloadTask.response as? HTTPURLResponse {
            self.statusCode = resp.statusCode
        }
        if statusCode >= 400 {
            return  // don't move error bodies into the cache
        }
        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
        } catch {
            self.error = error
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        progress?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error { self.error = error }
        if let resp = task.response as? HTTPURLResponse, statusCode == 0 {
            self.statusCode = resp.statusCode
        }
        semaphore.signal()
    }
}
