import Foundation
import CryptoKit

/// Persistent BPE-token cache for `tinygpt train`.
///
/// Tokenising a 500 MB+ corpus through SentencePiece / BPE costs 10-30
/// minutes on M-series silicon. Every Mega/Huge re-train (and every
/// `--resume` after a crash) paid that toll until we cached. With the
/// cache, the second invocation jumps straight to step 1.
///
/// File layout: tightly-packed Int32 little-endian token ids, written
/// once via Data.write(atomic:) so a crash mid-write never leaves a
/// truncated cache.
///
/// Cache key: SHA-256 of (corpus path, tokenizer dir, file size, mtime,
/// vocab size), truncated to 8 hex chars. Any of those changing
/// invalidates the cache silently — wrong file, edited file, swapped
/// tokenizer, vocab-size mismatch all force a re-tokenize.
enum TokenCache {

    /// Where the cache file lives for a given (corpus, tokenizer) pair.
    /// Co-located with the corpus so it's obvious what it belongs to and
    /// gets cleaned up alongside corpus moves. Returns nil if the corpus
    /// can't be `stat`'d (we can't compute the digest without size/mtime).
    static func cacheURL(corpus: URL, tokenizerDir: URL, vocabSize: Int) -> URL? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: corpus.path),
              let size = attrs[.size] as? NSNumber,
              let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970
        else { return nil }
        var hasher = SHA256()
        hasher.update(data: Data(corpus.path.utf8))
        hasher.update(data: Data(tokenizerDir.path.utf8))
        var sz = size.int64Value
        hasher.update(data: Data(bytes: &sz, count: MemoryLayout<Int64>.size))
        var mt = mtime
        hasher.update(data: Data(bytes: &mt, count: MemoryLayout<Double>.size))
        var vs = Int32(vocabSize)
        hasher.update(data: Data(bytes: &vs, count: MemoryLayout<Int32>.size))
        let digest = hasher.finalize()
        let hex = digest.prefix(4).map { String(format: "%02x", $0) }.joined()
        return corpus.appendingPathExtension("tokens-\(hex)")
    }

    /// Load a previously-written cache. Returns nil on missing or
    /// truncated files — caller re-tokenizes.
    static func read(_ url: URL) -> [Int32]? {
        guard let data = try? Data(contentsOf: url, options: .alwaysMapped) else { return nil }
        let n = data.count / MemoryLayout<Int32>.size
        guard n > 0 else { return nil }
        return data.withUnsafeBytes { ptr -> [Int32] in
            Array(UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(to: Int32.self),
                count: n))
        }
    }

    /// Persist tokens atomically. Failures are silent at the call site
    /// (next run just re-tokenizes) — we never fail training on a
    /// cache-write hiccup.
    static func write(_ tokens: [Int32], to url: URL) throws {
        let data = tokens.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url, options: .atomic)
    }
}
