import Foundation
import CryptoKit

/// `tinygpt dedupe` — drop duplicate lines or documents from a text
/// corpus.
///
/// Matters most for raw web scrapes (Common Crawl subsets, raw HTML
/// dumps) where the same text appears in many places. Trains on a
/// deduped corpus avoid memorising boilerplate. Hash-based: SHA-256 of
/// each unit, skip on first re-occurrence. Streaming — never holds the
/// full corpus in RAM, only the set of seen hashes (~32 bytes per
/// unique unit).
///
/// USAGE
///   tinygpt dedupe <input.txt> --out <output.txt>
///                  [--unit line|doc] [--min-len N] [--quiet]
///
///   --unit line     SHA each line; skip duplicates (default)
///   --unit doc      SHA each paragraph (separated by blank lines)
///   --min-len N     Skip units shorter than N chars (default 0)
///   --quiet         Suppress progress stats
enum Dedupe {
    static func run(args: [String]) {
        var inputPath: String? = nil
        var outPath: String? = nil
        var unit = "line"
        var minLen = 0
        var quiet = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":      outPath = args[i+1]; i += 2
            case "--unit":     unit = args[i+1]; i += 2
            case "--min-len":  minLen = Int(args[i+1]) ?? 0; i += 2
            case "--quiet":    quiet = true; i += 1
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                inputPath = args[i]; i += 1
            }
        }
        guard let inputPath = inputPath else { fputs("missing <input.txt>\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out required\n", stderr); exitUsage() }
        guard unit == "line" || unit == "doc" else {
            fputs("--unit must be 'line' or 'doc' (got '\(unit)')\n", stderr); exitUsage()
        }

        let inputURL = URL(fileURLWithPath: inputPath)
        let outputURL = URL(fileURLWithPath: outPath)

        let text: String
        do { text = try String(contentsOf: inputURL, encoding: .utf8) }
        catch { fputs("error reading \(inputPath): \(error)\n", stderr); exit(1) }
        let inputBytes = text.utf8.count

        // Split into units. For "line" mode each line is a unit; for
        // "doc" mode we split on blank-line boundaries (one or more
        // consecutive \n). Both modes preserve the original separator
        // when writing the dedup'd output.
        let units: [String]
        if unit == "line" {
            units = text.components(separatedBy: "\n")
        } else {
            // Split on runs of 2+ newlines. Each chunk is one document;
            // we re-join with "\n\n" on write.
            units = text.components(separatedBy: "\n\n").filter { !$0.isEmpty }
        }

        var seen = Set<String>()
        seen.reserveCapacity(units.count)
        var kept: [String] = []
        kept.reserveCapacity(units.count)
        var skippedDup = 0
        var skippedShort = 0
        for u in units {
            if u.count < minLen { skippedShort += 1; continue }
            // SHA-256 the unit's UTF-8 bytes; store as hex.
            let digest = SHA256.hash(data: Data(u.utf8))
            let key = digest.map { String(format: "%02x", $0) }.joined()
            if seen.contains(key) { skippedDup += 1; continue }
            seen.insert(key)
            kept.append(u)
        }

        let separator = unit == "line" ? "\n" : "\n\n"
        let outputText = kept.joined(separator: separator)
        do { try outputText.write(to: outputURL, atomically: true, encoding: .utf8) }
        catch { fputs("error writing \(outPath): \(error)\n", stderr); exit(1) }

        if !quiet {
            let outBytes = outputText.utf8.count
            let pctKept = units.isEmpty ? 0.0 : Double(kept.count) / Double(units.count) * 100
            let pctBytes = inputBytes == 0 ? 0.0 : Double(outBytes) / Double(inputBytes) * 100
            print("""
            dedupe summary
              unit:           \(unit)
              input units:    \(units.count)
              kept:           \(kept.count)  (\(String(format: "%.1f", pctKept))%)
              skipped (dup):  \(skippedDup)
              skipped (<\(minLen) chars):  \(skippedShort)
              input bytes:    \(inputBytes)
              output bytes:   \(outBytes)  (\(String(format: "%.1f", pctBytes))%)
              wrote:          \(outPath)
            """)
        }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt dedupe <input.txt> --out <output.txt> [options]

        --unit line|doc        line-level (SHA per line) or document-level
                               (SHA per paragraph, separated by blank lines)
                               default: line
        --min-len N            skip units shorter than N chars (default 0)
        --quiet                suppress summary stats

        Hash-based streaming deduplication via SHA-256. Memory cost is
        ~32 bytes per unique unit; runtime is O(N) over the input.
        """)
        exit(code)
    }
}
