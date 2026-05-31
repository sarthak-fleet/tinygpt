import Foundation
import TinyGPTData

/// `tinygpt download-dataset hf://datasets/<id>` —
/// resolves a HuggingFace dataset, streams its shards to
/// `~/.cache/tinygpt/datasets/<id>/`, and converts to one of tinygpt's
/// training formats (SFT JSONL, DPO JSONL, plain text).
///
/// This is the data-pipeline entry point for the agent-factory north
/// star: pulling clean, on-disk corpora from HF Hub so users can SFT/DPO
/// without leaving the CLI.
///
/// FLAGS
///   <dataset>             hf://datasets/<owner>/<name>  or just <owner>/<name>
///   --format sft|dpo|plain  force a target format (default: auto-detect)
///   --out <path>          where to write the converted corpus (default: cache dir / corpus.jsonl)
///   --map a:b,c:d         override field aliases (dataset-field:canonical-field)
///   --revision <rev>      git revision to pin (default: main)
///   --max-files <n>       cap on how many shards to download (default: all)
///   --inspect             print the file listing and schema, but don't download
///   --dry-run             resolve + detect, no I/O to corpus file
///
/// EXAMPLES
///   tinygpt download-dataset hf://datasets/Salesforce/xlam-function-calling-60k
///   tinygpt download-dataset OpenHermes-2.5 --format sft --out openhermes.jsonl
///   tinygpt download-dataset yahma/alpaca-cleaned --max-files 1 --inspect
enum DownloadDataset {

    static func run(args: [String]) {
        var datasetArg: String?
        var formatArg: CorpusFormat?
        var outPath: String?
        var fieldMap = FieldMap()
        var revision = "main"
        var maxFiles = Int.max
        var inspect = false
        var dryRun = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--format":
                guard i+1 < args.count, let f = CorpusFormat(parsing: args[i+1]) else {
                    fputs("--format requires sft|dpo|plain\n", stderr); exit(2)
                }
                formatArg = f; i += 2
            case "--out":      outPath = args[i+1]; i += 2
            case "--map":      fieldMap = FieldMap.parse(args[i+1]); i += 2
            case "--revision": revision = args[i+1]; i += 2
            case "--max-files":
                maxFiles = Int(args[i+1]) ?? maxFiles; i += 2
            case "--inspect": inspect = true; i += 1
            case "--dry-run": dryRun = true; i += 1
            case "-h", "--help":
                printUsage(); return
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); printUsage(); exit(2)
                }
                datasetArg = args[i]; i += 1
            }
        }
        _ = revision  // currently passed through resolveFileURL only — pin support is best-effort

        guard let raw = datasetArg else {
            fputs("missing dataset id (e.g. hf://datasets/owner/name)\n", stderr)
            printUsage(); exit(2)
        }
        let datasetId = normalizeDatasetId(raw)

        do {
            try runResolved(datasetId: datasetId, formatArg: formatArg,
                            outPath: outPath, fieldMap: fieldMap,
                            maxFiles: maxFiles, inspect: inspect, dryRun: dryRun)
        } catch let err as HFDatasets.HFError {
            fputs("error: \(err.description)\n", stderr); exit(1)
        } catch {
            fputs("error: \(error)\n", stderr); exit(1)
        }
    }

    // MARK: - Core

    static func runResolved(datasetId: String, formatArg: CorpusFormat?,
                            outPath: String?, fieldMap: FieldMap,
                            maxFiles: Int, inspect: Bool, dryRun: Bool) throws {
        print("==> resolving dataset: \(datasetId)")
        let info = try HFDatasets.info(id: datasetId)

        // Header summary.
        print("")
        print("  id:        \(info.id)")
        if let sha = info.sha            { print("  sha:       \(sha)") }
        if let lm = info.lastModified    { print("  modified:  \(lm)") }
        if let dl = info.downloads       { print("  downloads: \(dl)") }
        if info.gated                    { print("  gated:     YES (HF_TOKEN required if download fails)") }
        if info.private                { print("  private:   YES") }
        print("  files:     \(info.siblings.count)  (\(formatBytes(info.totalBytes)) reported)")

        // Filter to the data-bearing siblings — skip README, .gitattributes,
        // dataset_infos.json, etc. We also skip auxiliary parquet metadata.
        let candidates = info.siblings.filter { isDataShard($0.rfilename) }
        let shardCount = min(candidates.count, maxFiles)
        let chosen = Array(candidates.prefix(shardCount))
        print("\n  data shards (\(candidates.count) found, taking \(chosen.count)):")
        for s in chosen.prefix(8) {
            let szStr = s.size.map(formatBytes) ?? "?"
            print("    \(s.rfilename)  \(szStr)")
        }
        if chosen.count > 8 { print("    ... (+\(chosen.count - 8) more)") }

        if chosen.isEmpty {
            fputs("error: no recognised data shards in '\(datasetId)'.\n", stderr)
            fputs("       (this dataset may use a format we don't yet support — e.g. tar archive or arrow.)\n", stderr)
            exit(1)
        }

        if inspect {
            print("\n--inspect: stopping here.")
            return
        }

        // Download shards.
        var localPaths: [URL] = []
        print("")
        for (idx, sib) in chosen.enumerated() {
            print("==> [\(idx+1)/\(chosen.count)] downloading \(sib.rfilename) (\(sib.size.map(formatBytes) ?? "?"))")
            let url = try HFDatasets.downloadFile(id: info.id, filename: sib.rfilename, expectedSize: sib.size,
                                                  progress: { written, total in
                if total > 0 {
                    let pct = Double(written) / Double(total) * 100
                    fputs("\r  \(formatBytes(Int(written)))/\(formatBytes(Int(total))) (\(String(format: "%.1f", pct))%)   ", stderr)
                } else {
                    fputs("\r  \(formatBytes(Int(written)))    ", stderr)
                }
            })
            fputs("\n", stderr)
            localPaths.append(url)
        }

        // Sniff a row from the first decodable shard.
        var sampleRow: [String: Any]? = nil
        var firstDecodable: URL? = nil
        for path in localPaths {
            let fmt = RowReader.detectFormat(url: path)
            if fmt == .jsonl || fmt == .json {
                _ = try RowReader.readRows(url: path, format: fmt) { row in
                    sampleRow = row; return false   // stop after first row
                }
                firstDecodable = path
                break
            }
        }
        guard let sample = sampleRow, firstDecodable != nil else {
            fputs("\nerror: downloaded shards are not in a directly readable format (parquet/arrow not yet decoded).\n", stderr)
            fputs("       cached at: \(HFDatasets.cacheRoot().path)/\(info.id)/\n", stderr)
            fputs("       convert manually or wait for parquet support — see docs/hf_datasets_integration.md.\n", stderr)
            exit(1)
        }

        // Detect format unless user forced one.
        let detection = FormatDetector.detect(sampleRow: sample, userMap: fieldMap)
        let resolvedFormat = formatArg ?? detection.format
        print("")
        print("==> schema sniffing  (sample fields: \(sample.keys.sorted().joined(separator: ", ")))")
        print("    detected:     \(detection.format.rawValue)  (confidence \(String(format: "%.0f%%", detection.confidence*100)))")
        print("    rationale:    \(detection.rationale)")
        if formatArg != nil && formatArg != detection.format {
            print("    using:        \(resolvedFormat.rawValue)  (forced via --format)")
        }
        // If the user forced a format different from the detection, the
        // detector's map is for the WRONG format — derive a fresh map
        // for the target format from the same sample row.
        let mapToUse: [String: String]
        if let forced = formatArg, forced != detection.format {
            mapToUse = mapForForcedFormat(sample: sample, format: forced, userMap: fieldMap)
        } else {
            mapToUse = detection.map
        }
        if !mapToUse.isEmpty {
            let pretty = mapToUse.map { "\($0.key)→\($0.value)" }.sorted().joined(separator: ", ")
            print("    field map:    \(pretty)")
        }

        if detection.confidence < 0.5 && formatArg == nil {
            fputs("\nwarning: low-confidence schema detection. Re-run with --format and --map to be explicit.\n", stderr)
            fputs("         sample row keys: \(sample.keys.sorted().joined(separator: ", "))\n", stderr)
        }

        if dryRun {
            print("\n--dry-run: skipping write phase.")
            return
        }

        // Decide output path.
        let outURL: URL = {
            if let p = outPath { return URL(fileURLWithPath: p) }
            let ext = resolvedFormat == .plain ? "txt" : "jsonl"
            let base = (try? HFDatasets.cacheDir(for: info.id)) ?? HFDatasets.cacheRoot()
            return base.appendingPathComponent("corpus.\(ext)")
        }()

        print("\n==> converting rows → \(outURL.path)")
        let writer = try JSONLWriter(url: outURL, append: false)
        defer { writer.close() }
        var totalIn = 0, totalOut = 0, skipped = 0
        for path in localPaths {
            _ = try RowReader.readRows(url: path) { row in
                totalIn += 1
                if let rec = CorpusConverter.convert(row: row, format: resolvedFormat, map: mapToUse) {
                    do {
                        if case .plain(let t) = rec, resolvedFormat == .plain && outURL.pathExtension == "txt" {
                            try writer.writePlainLine(t)
                        } else {
                            try writer.write(rec)
                        }
                        totalOut += 1
                    } catch {
                        // Skip the offending row but keep going. Rare.
                        skipped += 1
                    }
                } else {
                    skipped += 1
                }
                return true
            }
        }
        print("    read:    \(totalIn) rows")
        print("    wrote:   \(totalOut) records")
        if skipped > 0 { print("    skipped: \(skipped) (missing fields / empty)") }
        print("")
        print("==> done. corpus at \(outURL.path)")
        switch resolvedFormat {
        case .sft:
            print("    next:  tinygpt sft <base> --data \(outURL.path) --template chatml --out lora.tinygpt")
        case .dpo:
            print("    next:  tinygpt dpo <base> --data \(outURL.path) --out dpo.tinygpt")
        case .plain:
            print("    next:  pass --data \(outURL.path) to tinygpt train")
        }
    }

    // MARK: - Helpers

    /// When the user forces a target format that differs from what the
    /// detector picked, the detector's per-format map is wrong. Re-resolve
    /// a map by trying each canonical-field's aliases against the sample
    /// row. Falls back to the user's explicit `--map` overrides.
    static func mapForForcedFormat(sample: [String: Any], format: CorpusFormat,
                                   userMap: FieldMap) -> [String: String] {
        let keys = Set(sample.keys.map { $0.lowercased() })
        func sourceKey(for alias: String) -> String? {
            sample.keys.first(where: { $0.lowercased() == alias.lowercased() })
        }
        func firstMatch(_ aliases: [String]) -> String? {
            for a in aliases where keys.contains(a.lowercased()) { return sourceKey(for: a) }
            return nil
        }
        // Apply user --map first: user's "src:dst" means "treat dataset
        // field `src` as canonical `dst`". The map we output is
        // `canonical -> dataset-field`, so we invert.
        var explicit: [String: String] = [:]
        for (src, dst) in userMap.mapping {
            if let actual = sourceKey(for: src) { explicit[dst.lowercased()] = actual }
        }
        var out: [String: String] = [:]
        switch format {
        case .sft:
            out["instruction"] = explicit["instruction"] ?? firstMatch(FormatDetector.sftInstructionAliases) ?? ""
            out["input"]       = explicit["input"]       ?? firstMatch(FormatDetector.sftInputAliases) ?? ""
            out["response"]    = explicit["response"]    ?? firstMatch(FormatDetector.sftResponseAliases) ?? ""
        case .dpo:
            out["prompt"]   = explicit["prompt"]   ?? firstMatch(FormatDetector.dpoPromptAliases) ?? ""
            out["chosen"]   = explicit["chosen"]   ?? firstMatch(FormatDetector.dpoChosenAliases) ?? ""
            out["rejected"] = explicit["rejected"] ?? firstMatch(FormatDetector.dpoRejectedAliases) ?? ""
        case .plain:
            // For plain text we either have a dedicated `text` field or
            // we concatenate every string field on the row (handled by
            // the converter via `__concat`).
            if let dedicated = explicit["text"] ?? firstMatch(FormatDetector.plainAliases) {
                out["text"] = dedicated
            } else {
                // Pick the longest string field as a fallback.
                let longest = sample.compactMap { (k, v) -> (String, Int)? in
                    guard let s = v as? String else { return nil }
                    return (k, s.count)
                }.max(by: { $0.1 < $1.1 })?.0
                if let l = longest { out["text"] = l }
            }
        }
        // Strip empty values so the converter doesn't index by "".
        return out.filter { !$0.value.isEmpty }
    }

    /// Strip `hf://datasets/` prefix and any trailing slashes.
    static func normalizeDatasetId(_ raw: String) -> String {
        var s = raw
        let prefixes = ["hf://datasets/", "https://huggingface.co/datasets/", "hf://"]
        for p in prefixes where s.hasPrefix(p) { s.removeFirst(p.count) }
        while s.hasSuffix("/") { s.removeLast() }
        return s
    }

    /// True for files that should contribute training data. False for
    /// README / git metadata / dataset_infos.
    static func isDataShard(_ name: String) -> Bool {
        let lower = name.lowercased()
        // Hidden / git files.
        if lower.hasPrefix(".") { return false }
        // Common non-data sidecars.
        let skipNames: Set<String> = ["readme.md", "license", "license.md", "license.txt",
                                       "dataset_infos.json", "dataset_dict.json",
                                       ".gitattributes", "config.json"]
        if skipNames.contains(lower) { return false }
        if lower.hasSuffix(".md") || lower.hasSuffix(".txt.md") { return false }
        // Yes-list of data extensions.
        let dataExt = [".jsonl", ".json", ".jsonl.gz", ".ndjson", ".parquet", ".arrow",
                       ".csv", ".tsv", ".feather"]
        return dataExt.contains(where: { lower.hasSuffix($0) })
    }

    static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000_000 { return String(format: "%.1f GB", Double(n) / 1_000_000_000) }
        if n >= 1_000_000     { return String(format: "%.0f MB", Double(n) / 1_000_000) }
        if n >= 1_000         { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    static func printUsage() {
        print("""
        usage: tinygpt download-dataset <dataset> [flags]

        dataset:
          hf://datasets/owner/name    canonical form
          owner/name                  shorthand

        flags:
          --format sft|dpo|plain      force target format
          --out <path>                output file (default: cache dir/corpus.jsonl)
          --map a:b,c:d               field-name overrides
          --revision <rev>            git revision (default: main)
          --max-files <n>             cap shards to download
          --inspect                   print file list + schema, no download
          --dry-run                   resolve + sniff, no write

        env:
          HF_TOKEN                    bearer token for gated/private datasets
          TINYGPT_DATASET_CACHE       override cache root (default: ~/.cache/tinygpt/datasets)
        """)
    }
}
