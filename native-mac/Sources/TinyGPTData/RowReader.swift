import Foundation

/// Reader for the shard formats HF datasets ship in.
///
/// Status:
///   - JSONL / JSON / NDJSON  → fully supported (line-by-line)
///   - Parquet                → header detection only (NOT full decode)
///   - Arrow                  → header detection only (NOT full decode)
///
/// The parquet path is the elephant in the room: Apple ships no parquet
/// decoder, and the spec is non-trivial (mixed encodings, dictionary
/// page indirection, snappy/zstd compression). Rather than pull in a
/// large C++ dep right now, we:
///   1. Prefer HF's `dataset-server` JSON API for small subsets (the
///      Hub auto-converts and serves rows as JSON).
///   2. Tag the parquet/arrow files as downloaded-but-undecoded and
///      print a clear message — full parquet support is a follow-up.
///
/// JSONL covers a huge fraction of the registry (xLAM, Hermes, MetaMath,
/// OpenHermes-2.5 in their JSON-shipped variants, etc.). Parquet support
/// is the next deliverable — see docs/hf_datasets_integration.md.
public enum RowReader {

    public enum ShardFormat: Sendable, Equatable {
        case jsonl       // one JSON object per line
        case json        // single JSON document; either an array of objects or {"data": [...]}
        case parquet
        case arrow
        case unknown
    }

    /// Guess the shard format from filename + (optionally) magic bytes.
    public static func detectFormat(url: URL) -> ShardFormat {
        let name = url.lastPathComponent.lowercased()
        if name.hasSuffix(".jsonl") || name.hasSuffix(".jsonl.gz") || name.hasSuffix(".ndjson") { return .jsonl }
        if name.hasSuffix(".json")  { return .json }
        if name.hasSuffix(".parquet") { return .parquet }
        if name.hasSuffix(".arrow") || name.hasSuffix(".feather") { return .arrow }

        // Magic-byte fallback.
        if let fh = try? FileHandle(forReadingFrom: url) {
            defer { try? fh.close() }
            if let head = try? fh.read(upToCount: 8) {
                // parquet magic: "PAR1" at start, also at end.
                if head.starts(with: [0x50, 0x41, 0x52, 0x31]) { return .parquet }
                // arrow: "ARROW1\0\0" — but we only have first 8 bytes.
                if head.starts(with: [0x41, 0x52, 0x52, 0x4f, 0x57, 0x31, 0x00, 0x00]) { return .arrow }
                // JSONL/JSON: starts with '{' or '['.
                if head.first == 0x7b || head.first == 0x5b { return name.contains("jsonl") ? .jsonl : .json }
            }
        }
        return .unknown
    }

    /// Iterate rows in a shard. Calls `visit` once per row; returning
    /// `false` from `visit` aborts the iteration (used for "first row
    /// sniff" during format detection).
    ///
    /// Returns the number of rows visited. Throws on I/O / parse errors;
    /// silent skip on per-row JSON errors so one corrupt line doesn't
    /// trash a whole corpus.
    @discardableResult
    public static func readRows(url: URL, format: ShardFormat? = nil,
                                visit: (_ row: [String: Any]) -> Bool) throws -> Int {
        let fmt = format ?? detectFormat(url: url)
        switch fmt {
        case .jsonl:
            return try readJSONL(url: url, visit: visit)
        case .json:
            return try readJSON(url: url, visit: visit)
        case .parquet, .arrow:
            // We don't have a decoder. Surface the situation as an empty
            // iteration with a stderr note — the caller already cached
            // the file, so the user can decode it externally.
            fputs("note: \(url.lastPathComponent) is \(fmt) — parquet/arrow decoding is not yet implemented. " +
                  "The file is cached at \(url.path). " +
                  "Use `parquet-tools` or `python -c 'import pandas; pandas.read_parquet(...).to_json(...)'` " +
                  "to convert, then re-run with --input <jsonl>.\n", stderr)
            return 0
        case .unknown:
            throw HFDatasets.HFError.ioError("unknown shard format for \(url.lastPathComponent)")
        }
    }

    /// Streaming JSONL reader. Reads the file as UTF-8 bytes and
    /// breaks on '\n'. For shards up to a few GB this is fine; multi-GB
    /// shards we'd want a buffered LineReader, but that's premature now.
    private static func readJSONL(url: URL, visit: ([String: Any]) -> Bool) throws -> Int {
        guard let fh = try? FileHandle(forReadingFrom: url) else {
            throw HFDatasets.HFError.ioError("could not open \(url.path)")
        }
        defer { try? fh.close() }
        var count = 0
        var buffer = Data()
        let chunkSize = 1 << 20   // 1 MiB
        while true {
            let chunk = (try? fh.read(upToCount: chunkSize)) ?? Data()
            if chunk.isEmpty { break }
            buffer.append(chunk)
            // Drain complete lines from the buffer.
            while let lf = buffer.firstIndex(of: 0x0a) {
                let line = buffer.subdata(in: buffer.startIndex..<lf)
                buffer.removeSubrange(buffer.startIndex...lf)
                guard !line.isEmpty else { continue }
                if let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any] {
                    count += 1
                    if !visit(obj) { return count }
                }
            }
        }
        // Trailing line without LF.
        if !buffer.isEmpty,
           let obj = try? JSONSerialization.jsonObject(with: buffer) as? [String: Any] {
            count += 1
            _ = visit(obj)
        }
        return count
    }

    /// Single-document JSON reader. Accepts either:
    ///   - `[{...}, {...}]` — top-level array of row dicts
    ///   - `{"data": [...]}` — HF dataset-server response shape
    private static func readJSON(url: URL, visit: ([String: Any]) -> Bool) throws -> Int {
        guard let data = try? Data(contentsOf: url) else {
            throw HFDatasets.HFError.ioError("could not read \(url.path)")
        }
        let obj = try JSONSerialization.jsonObject(with: data)
        var rows: [[String: Any]] = []
        if let arr = obj as? [[String: Any]] { rows = arr }
        if let dict = obj as? [String: Any], let arr = dict["rows"] as? [[String: Any]] {
            // HF rows-API shape:  {"rows": [{"row_idx", "row": {...}}, ...]}
            rows = arr.compactMap { $0["row"] as? [String: Any] }
        }
        if let dict = obj as? [String: Any], let arr = dict["data"] as? [[String: Any]] { rows = arr }
        var count = 0
        for r in rows {
            count += 1
            if !visit(r) { return count }
        }
        return count
    }
}
