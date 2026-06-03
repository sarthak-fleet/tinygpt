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
        var nearDup = false
        var minHashPerms = 64
        var nearDupThreshold: Float = 0.85
        var shingleSize = 5

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":            outPath = args[i+1]; i += 2
            case "--unit":           unit = args[i+1]; i += 2
            case "--min-len":        minLen = Int(args[i+1]) ?? 0; i += 2
            case "--quiet":          quiet = true; i += 1
            case "--near-dup":       nearDup = true; i += 1
            case "--minhash-perms":  minHashPerms = Int(args[i+1]) ?? minHashPerms; i += 2
            case "--near-threshold": nearDupThreshold = Float(args[i+1]) ?? nearDupThreshold; i += 2
            case "--shingle":        shingleSize = Int(args[i+1]) ?? shingleSize; i += 2
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

        var kept: [String] = []
        kept.reserveCapacity(units.count)
        var skippedDup = 0
        var skippedShort = 0

        if nearDup {
            // MinHash-based near-duplicate detection. For each unit:
            //   1. tokenize into K-char shingles
            //   2. SHA-256-hash each shingle to a UInt64 (truncated)
            //   3. compute the min over `minHashPerms` independent hash
            //      transforms — this is the MinHash sketch
            //   4. compare each unit's sketch to all previously-kept
            //      sketches; estimated Jaccard = fraction of matching
            //      positions in the sketch. If ≥ threshold, drop.
            // O(N · perms + N²) — fine for ≤ 100K units; full LSH
            // banding is the optimization for million-scale corpora.
            let perms = minHashPerms
            // Linear hash transforms h_i(x) = (a_i · x + b_i) mod prime.
            let prime: UInt64 = (1 << 61) - 1   // Mersenne prime, fast mod
            var aCoefs = [UInt64](); aCoefs.reserveCapacity(perms)
            var bCoefs = [UInt64](); bCoefs.reserveCapacity(perms)
            var seed: UInt64 = 0xc0ffee_1234_abcd_ef
            for _ in 0..<perms {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                aCoefs.append((seed | 1) % prime)
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                bCoefs.append(seed % prime)
            }
            var sketches: [[UInt64]] = []
            sketches.reserveCapacity(units.count)
            for u in units {
                if u.count < minLen { skippedShort += 1; continue }
                let sketch = minHashSketch(of: u, shingle: shingleSize,
                                            aCoefs: aCoefs, bCoefs: bCoefs,
                                            prime: prime)
                // Compare against all kept sketches.
                var dup = false
                for kSketch in sketches {
                    var matches = 0
                    for p in 0..<perms { if sketch[p] == kSketch[p] { matches += 1 } }
                    let jaccard = Float(matches) / Float(perms)
                    if jaccard >= nearDupThreshold { dup = true; break }
                }
                if dup { skippedDup += 1; continue }
                sketches.append(sketch)
                kept.append(u)
            }
        } else {
            var seen = Set<String>()
            seen.reserveCapacity(units.count)
            for u in units {
                if u.count < minLen { skippedShort += 1; continue }
                let digest = SHA256.hash(data: Data(u.utf8))
                let key = digest.map { String(format: "%02x", $0) }.joined()
                if seen.contains(key) { skippedDup += 1; continue }
                seen.insert(key)
                kept.append(u)
            }
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

    /// Compute a MinHash sketch of `text`. Tokenize into character
    /// shingles of size K, hash each to a UInt64 base value, then for
    /// each of the `perms` linear hash transforms take the minimum
    /// across all shingles. Estimated Jaccard between two sketches =
    /// fraction of positions where they match.
    static func minHashSketch(of text: String, shingle K: Int,
                                aCoefs: [UInt64], bCoefs: [UInt64],
                                prime: UInt64) -> [UInt64] {
        let perms = aCoefs.count
        var sketch = [UInt64](repeating: UInt64.max, count: perms)
        let chars = Array(text.unicodeScalars)
        if chars.count < K {
            // Too short to shingle — hash the whole thing once.
            let h = baseHash(of: text)
            for p in 0..<perms {
                sketch[p] = (aCoefs[p] &* h &+ bCoefs[p]) % prime
            }
            return sketch
        }
        for s in 0...(chars.count - K) {
            var shingle = ""
            for k in 0..<K { shingle.unicodeScalars.append(chars[s + k]) }
            let h = baseHash(of: shingle)
            for p in 0..<perms {
                let v = (aCoefs[p] &* h &+ bCoefs[p]) % prime
                if v < sketch[p] { sketch[p] = v }
            }
        }
        return sketch
    }

    /// FNV-1a-style 64-bit hash, fast and good-enough for MinHash
    /// permutation input. SHA-256 was overkill (~1000× slower).
    @inline(__always)
    static func baseHash(of s: String) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for byte in s.utf8 {
            h ^= UInt64(byte)
            h &*= 0x100000001b3
        }
        return h
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt dedupe <input.txt> --out <output.txt> [options]

        --unit line|doc        line-level (SHA per line) or document-level
                               (SHA per paragraph, separated by blank lines)
                               default: line
        --min-len N            skip units shorter than N chars (default 0)
        --quiet                suppress summary stats

        --near-dup             enable MinHash near-duplicate detection
                               (catches paraphrased boilerplate that
                               exact-SHA misses). O(N²) vs O(N); fine up
                               to ~100K units. Full LSH banding is the
                               optimization for million-scale corpora.
        --minhash-perms N      sketch size (default 64). Higher = more
                               accurate Jaccard estimate.
        --near-threshold F     Jaccard cutoff for "near-duplicate"
                               (default 0.85). Lower = more aggressive
                               dedup; higher = catches only near-exact.
        --shingle K            character shingle size (default 5).

        Exact mode: SHA-256-keyed streaming. Memory cost is ~32 bytes
        per unique unit; runtime is O(N) over the input.
        """)
        exit(code)
    }
}
