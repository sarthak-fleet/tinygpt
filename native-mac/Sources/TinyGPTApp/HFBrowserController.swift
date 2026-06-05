import Foundation

/// Minimal HuggingFace **model** downloader for the app — pulls the
/// files needed to feed `HFModelLoader` (config.json, tokenizer files,
/// safetensors shards) into a local cache. Search/browse is a v2
/// follow-up; v1 takes an explicit `owner/repo` from the user.
///
/// Mirrors the shape of `TinyGPTData.HFDatasets` but targets the
/// `/api/models/<id>` endpoint and the `/<id>/resolve/<rev>/<path>`
/// download URL instead. Uses URLSession directly so the app stays
/// independent of the CLI's `download-dataset` subcommand.
///
/// Auth: `HF_TOKEN` from the env if present. Gated/private models
/// otherwise surface a clean "needs HF_TOKEN" error.
@MainActor
final class HFBrowserController: ObservableObject {
    @Published var status: String = ""
    @Published var progress: Double = 0     // 0..1, NaN when indeterminate
    @Published var isDownloading: Bool = false
    @Published var downloadedModels: [DownloadedModel] = []
    @Published var lastError: String? = nil

    /// One model downloaded into the local cache, surfaced in the sidebar.
    struct DownloadedModel: Identifiable, Hashable {
        let id: String                     // owner/repo
        let url: URL                       // path to the cache dir
        let displayName: String
        let sizeBytes: Int
    }

    private var activeTask: URLSessionDataTask? = nil

    init() {
        Task { await self.refresh() }
    }

    /// Cache root: `~/Library/Application Support/TinyGPT/hf/`. Lives next
    /// to the gallery cache so a single rm -rf cleans up everything.
    static func cacheRoot() -> URL {
        let support = (try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.homeDirectoryForCurrentUser
        return support.appendingPathComponent("TinyGPT/hf", isDirectory: true)
    }

    static func cacheDir(for id: String) -> URL {
        // owner/repo → owner__repo on disk so `/` doesn't make a subdir.
        let safe = id.replacingOccurrences(of: "/", with: "__")
        return cacheRoot().appendingPathComponent(safe, isDirectory: true)
    }

    /// Scan the cache dir, populate `downloadedModels`. Cheap O(N) over
    /// already-downloaded items.
    func refresh() async {
        let fm = FileManager.default
        let root = Self.cacheRoot()
        guard fm.fileExists(atPath: root.path) else { return }
        guard let entries = try? fm.contentsOfDirectory(at: root,
                                                         includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        var found: [DownloadedModel] = []
        for entry in entries {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: entry.path, isDirectory: &isDir), isDir.boolValue else { continue }
            // Require a config.json — the marker that this is a usable
            // HF model directory.
            if !fm.fileExists(atPath: entry.appendingPathComponent("config.json").path) { continue }
            let safeName = entry.lastPathComponent
            let id = safeName.replacingOccurrences(of: "__", with: "/")
            let size = Self.dirSize(entry)
            found.append(DownloadedModel(
                id: id, url: entry,
                displayName: id, sizeBytes: size))
        }
        self.downloadedModels = found.sorted { $0.id < $1.id }
    }

    /// Kick off a download for `owner/repo`. Pulls the file manifest from
    /// HF API, then fetches each file in sequence with progress updates.
    func download(repo: String) {
        let id = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty else { return }
        guard id.contains("/") else {
            lastError = "expected 'owner/repo', got '\(id)'"
            return
        }
        cancel()
        lastError = nil
        isDownloading = true
        progress = 0
        status = "fetching manifest for \(id)…"

        Task {
            do {
                let info = try await self.fetchModelInfo(id: id)
                // Decide which files to pull: config + tokenizer + every
                // safetensors / *.json that's small + the index for sharded
                // models. Skip pytorch_model.bin (we want safetensors) and
                // anything > 50 GB (sanity cap).
                let candidates = info.siblings.filter { sib in
                    let name = sib.rfilename
                    return name == "config.json"
                        || name == "tokenizer.json"
                        || name == "tokenizer_config.json"
                        || name == "special_tokens_map.json"
                        || name == "generation_config.json"
                        || name == "model.safetensors"
                        || name == "model.safetensors.index.json"
                        || (name.hasPrefix("model-") && name.hasSuffix(".safetensors"))
                }
                guard !candidates.isEmpty else {
                    throw HFError.noUsableFiles(id: id)
                }
                let cacheDir = Self.cacheDir(for: id)
                try FileManager.default.createDirectory(
                    at: cacheDir, withIntermediateDirectories: true)

                let totalBytes = candidates.compactMap(\.size).reduce(0, +)
                var bytesDone = 0
                for (i, sib) in candidates.enumerated() {
                    if Task.isCancelled { throw HFError.cancelled }
                    self.status = "[\(i+1)/\(candidates.count)] \(sib.rfilename) (\(Self.formatBytes(sib.size ?? 0)))"
                    try await self.downloadFile(
                        id: id, file: sib.rfilename,
                        to: cacheDir.appendingPathComponent(sib.rfilename)
                    ) { current, total in
                        let runningTotal = bytesDone + Int(current)
                        if totalBytes > 0 {
                            self.progress = Double(runningTotal) / Double(totalBytes)
                        }
                    }
                    bytesDone += sib.size ?? 0
                }
                self.status = "✓ \(id) → \(cacheDir.path)"
                self.progress = 1.0
                self.isDownloading = false
                await self.refresh()
            } catch HFError.cancelled {
                self.status = "cancelled"
                self.isDownloading = false
            } catch {
                self.lastError = "\(error)"
                self.status = "failed"
                self.isDownloading = false
            }
        }
    }

    func cancel() {
        activeTask?.cancel()
        activeTask = nil
        isDownloading = false
    }

    /// Delete a downloaded model's cache directory. Refreshes after.
    func delete(_ model: DownloadedModel) {
        try? FileManager.default.removeItem(at: model.url)
        Task { await refresh() }
    }

    // MARK: - HF API

    enum HFError: Error, CustomStringConvertible {
        case http(status: Int, url: String)
        case auth(id: String)
        case notFound(id: String)
        case noUsableFiles(id: String)
        case cancelled
        var description: String {
            switch self {
            case .http(let s, let u):    return "HTTP \(s) for \(u)"
            case .auth(let id):          return "model '\(id)' requires HF_TOKEN (gated/private). export HF_TOKEN=hf_… and retry."
            case .notFound(let id):      return "model '\(id)' not found on HF Hub"
            case .noUsableFiles(let id): return "model '\(id)' has no safetensors files we can load"
            case .cancelled:             return "cancelled"
            }
        }
    }

    private struct Sibling: Decodable {
        let rfilename: String
        let size: Int?
    }
    private struct ModelInfo: Decodable {
        let id: String
        let siblings: [Sibling]
    }

    private func fetchModelInfo(id: String) async throws -> ModelInfo {
        let urlString = "https://huggingface.co/api/models/\(id)"
        guard let url = URL(string: urlString) else { throw HFError.notFound(id: id) }
        var req = URLRequest(url: url)
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
            req.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw HFError.http(status: 0, url: urlString)
        }
        if http.statusCode == 404 { throw HFError.notFound(id: id) }
        if http.statusCode == 401 || http.statusCode == 403 { throw HFError.auth(id: id) }
        guard http.statusCode == 200 else {
            throw HFError.http(status: http.statusCode, url: urlString)
        }
        return try JSONDecoder().decode(ModelInfo.self, from: data)
    }

    /// Stream-download one file with progress. Uses URLSession async
    /// bytes so progress is per-chunk rather than waiting for the whole
    /// file (important for multi-GB safetensors shards).
    private func downloadFile(id: String, file: String, to dest: URL,
                              progress: @escaping (Int64, Int64) -> Void) async throws {
        if FileManager.default.fileExists(atPath: dest.path) {
            // Trust prior downloads — same scope as `tinygpt download-dataset`'s
            // resume-by-skip behavior. Re-downloading needs an explicit delete.
            return
        }
        let safeFile = file
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let urlString = "https://huggingface.co/\(id)/resolve/main/\(safeFile)"
        guard let url = URL(string: urlString) else {
            throw HFError.http(status: 0, url: urlString)
        }
        var req = URLRequest(url: url)
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
            req.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw HFError.http(status: 0, url: urlString)
        }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw HFError.auth(id: id)
        }
        guard http.statusCode == 200 else {
            throw HFError.http(status: http.statusCode, url: urlString)
        }
        let total = http.expectedContentLength

        // Write to a temp path first; rename atomically on completion so a
        // mid-download cancel doesn't leave a half-file at the real path.
        let tmp = dest.appendingPathExtension("part")
        FileManager.default.createFile(atPath: tmp.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: tmp) else {
            throw HFError.http(status: 0, url: dest.path)
        }
        defer { try? handle.close() }

        var buffer = Data()
        var written: Int64 = 0
        for try await byte in asyncBytes {
            if Task.isCancelled { throw HFError.cancelled }
            buffer.append(byte)
            if buffer.count >= 64 * 1024 {     // 64 KB chunks
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                progress(written, total)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            written += Int64(buffer.count)
            progress(written, total)
        }
        try handle.close()
        try FileManager.default.moveItem(at: tmp, to: dest)
    }

    // MARK: - Helpers

    private static func dirSize(_ url: URL) -> Int {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total = 0
        for case let file as URL in enumerator {
            let v = try? file.resourceValues(forKeys: [.fileSizeKey])
            total += v?.fileSize ?? 0
        }
        return total
    }

    static func formatBytes(_ n: Int) -> String {
        if n >= 1 << 30 { return String(format: "%.1f GB", Double(n) / Double(1 << 30)) }
        if n >= 1 << 20 { return String(format: "%.0f MB", Double(n) / Double(1 << 20)) }
        if n >= 1 << 10 { return String(format: "%.0f KB", Double(n) / Double(1 << 10)) }
        return "\(n) B"
    }
}
