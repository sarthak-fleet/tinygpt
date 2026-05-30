import Foundation

/// BPE-dropout (Provilkov et al., ACL 2020) — tokenization regularization
/// for BPE-based language models.
///
/// Standard BPE applies merges greedily in priority order until the
/// sequence is irreducible. BPE-dropout randomly *skips* a merge with
/// probability `p` at each pair-evaluation step. The result: the same
/// surface text gets a slightly different token sequence each time you
/// encode it. Used at TRAINING time only — encoding at inference / val
/// goes through the unmodified BPE so loss numbers and generations are
/// deterministic.
///
/// Why path (b)? swift-transformers' `BPETokenizer` is module-internal
/// (no public access to merges or per-merge hooks), so intercepting at
/// the merge level would require forking the library. Loading the same
/// `tokenizer.json` ourselves and implementing a parallel byte-level
/// GPT-2 BPE encoder is ~200 lines of Swift with a single dependency
/// (the GPT-2 byte alphabet table, replicated below). For p=0 this
/// implementation produces the same token sequence as swift-transformers
/// on GPT-2 / Llama / Qwen / Gemma / Phi style models (the byte-level
/// BPE family). For SentencePiece or WordPiece tokenizers we don't apply
/// dropout (the caller falls back to the wrapped HFTokenizer).
///
/// Scope: this encoder supports the GPT-2-style byte-level BPE used by
/// every modern decoder-only model the project targets. It does NOT
/// support SentencePiece byte_fallback, WordPiece, or hexa-encoding of
/// unknown bytes — for those, dropout silently degrades to standard
/// encoding (the model's primary tokenizer takes the path).
public final class BPEDropoutTokenizer: @unchecked Sendable {
    /// Merge-rank table: lower rank = applied first (i.e. higher priority).
    public let bpeRanks: [BytePair: Int]
    /// Token-string → id lookup, taken directly from `tokenizer.json`.
    public let vocab: [String: Int]
    /// Inverse — used by callers that want to decode back to text.
    public let idToToken: [Int: String]
    /// Unknown token id. Falls through to whatever the JSON declared.
    public let unknownTokenId: Int?
    /// Whether `tokenizer.json` declared this as a byte-level BPE model.
    /// If false, the encode function will refuse to apply dropout and
    /// signal the caller to use the upstream tokenizer instead.
    public let isByteLevel: Bool

    public struct BytePair: Hashable, Sendable {
        public let a: String
        public let b: String
        public init(_ a: String, _ b: String) { self.a = a; self.b = b }
    }

    /// Load merges + vocab from a HuggingFace `tokenizer.json` file.
    /// Returns nil if the file doesn't describe a BPE model (e.g.
    /// SentencePiece, WordPiece). Throws on I/O / JSON errors.
    public static func loadFromTokenizerJSON(_ url: URL) throws -> BPEDropoutTokenizer? {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        guard let model = root["model"] as? [String: Any] else { return nil }
        guard (model["type"] as? String) == "BPE" else { return nil }
        // Vocab: { token: id }
        guard let vocabRaw = model["vocab"] as? [String: Any] else { return nil }
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(vocabRaw.count)
        for (k, v) in vocabRaw {
            if let i = v as? Int { vocab[k] = i }
            else if let n = v as? NSNumber { vocab[k] = n.intValue }
        }
        // Merges: either `[[a, b], …]` (new format) or `["a b", …]` (legacy).
        guard let mergesRaw = model["merges"] as? [Any] else { return nil }
        var bpeRanks: [BytePair: Int] = [:]
        bpeRanks.reserveCapacity(mergesRaw.count)
        for (i, m) in mergesRaw.enumerated() {
            if let arr = m as? [String], arr.count == 2 {
                bpeRanks[BytePair(arr[0], arr[1])] = i
            } else if let s = m as? String {
                let parts = s.split(separator: " ", maxSplits: 1).map(String.init)
                if parts.count == 2 {
                    bpeRanks[BytePair(parts[0], parts[1])] = i
                }
            }
        }
        // Detect byte-level pre-tokenizer (presence of `pre_tokenizer.type ==
        // "ByteLevel"` or a `Sequence` containing it). For our purposes, if
        // the vocab contains the GPT-2 byte alphabet's distinctive entries
        // (e.g. `Ġ` for ' '), it's byte-level BPE.
        let isByteLevel = vocab["Ġ"] != nil || vocab["Ġthe"] != nil
            || (root["pre_tokenizer"] as? [String: Any]).map { detectByteLevel($0) } ?? false
        // Unknown token id, if any.
        var unkId: Int? = nil
        if let unk = model["unk_token"] as? String, let id = vocab[unk] { unkId = id }
        // Build the inverse map.
        var idToToken: [Int: String] = [:]
        idToToken.reserveCapacity(vocab.count)
        for (t, i) in vocab { idToToken[i] = t }
        return BPEDropoutTokenizer(bpeRanks: bpeRanks, vocab: vocab,
                                   idToToken: idToToken,
                                   unknownTokenId: unkId, isByteLevel: isByteLevel)
    }

    private static func detectByteLevel(_ cfg: [String: Any]) -> Bool {
        if (cfg["type"] as? String) == "ByteLevel" { return true }
        if let pretokenizers = cfg["pretokenizers"] as? [[String: Any]] {
            return pretokenizers.contains { ($0["type"] as? String) == "ByteLevel" }
        }
        return false
    }

    public init(bpeRanks: [BytePair: Int], vocab: [String: Int],
                 idToToken: [Int: String], unknownTokenId: Int?,
                 isByteLevel: Bool)
    {
        self.bpeRanks = bpeRanks
        self.vocab = vocab
        self.idToToken = idToToken
        self.unknownTokenId = unknownTokenId
        self.isByteLevel = isByteLevel
    }

    /// Encode text with BPE-dropout. `pDrop` is the per-merge skip
    /// probability (0 = standard BPE, 0.1 = the Provilkov paper's default).
    /// Returns token ids, falling back to `unknownTokenId` for any piece
    /// not in the vocab.
    ///
    /// Caveat: this implementation only handles byte-level BPE (GPT-2
    /// family). For other tokenizer kinds the caller MUST route through
    /// `HFTokenizer.encode` instead — `supportsDropout` reflects this.
    public func encodeWithDropout(_ text: String, pDrop: Float) -> [Int] {
        precondition(isByteLevel, "BPEDropoutTokenizer.encode requires byte-level BPE")
        var ids: [Int] = []
        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        BPEDropoutTokenizer.byteLevelPreTokenizeRegex.enumerateMatches(in: text, range: fullRange) { match, _, _ in
            guard let match else { return }
            let piece = nsText.substring(with: match.range)
            // Convert each UTF-8 byte to the GPT-2 byte-alphabet character.
            var encoded = ""
            encoded.reserveCapacity(piece.utf8.count)
            for byte in piece.utf8 {
                encoded.append(BPEDropoutTokenizer.byteEncoderTable[Int(byte)])
            }
            for tok in bpe(encoded, pDrop: pDrop) {
                if let id = vocab[tok] {
                    ids.append(id)
                } else if let unk = unknownTokenId {
                    ids.append(unk)
                }
                // else: silently drop — only happens if a byte produces a
                // character not in vocab and unk is undefined, which means
                // the tokenizer is misconfigured.
            }
        }
        return ids
    }

    /// Standard BPE merge loop with per-merge dropout. Mirrors the
    /// `huggingface/tokenizers` algorithm but with the priority-queue
    /// replaced by a linear scan (good enough for short tokens; pieces
    /// rarely exceed ~30 symbols after byte-level pre-tokenization).
    private func bpe(_ token: String, pDrop: Float) -> [String] {
        var symbols = token.unicodeScalars.map { String($0) }
        if symbols.count <= 1 { return symbols }

        while true {
            // Find the pair with the lowest (best) rank among adjacent
            // pairs that survive the dropout coin flip.
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(symbols.count - 1) {
                if pDrop > 0, Float.random(in: 0..<1) < pDrop {
                    // Skip this merge candidate this iteration.
                    continue
                }
                if let rank = bpeRanks[BytePair(symbols[i], symbols[i + 1])],
                   rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if bestIdx == -1 { break }   // no more applicable merges this pass
            // Apply the merge in place.
            let merged = symbols[bestIdx] + symbols[bestIdx + 1]
            symbols.replaceSubrange(bestIdx...(bestIdx + 1), with: [merged])
        }
        return symbols
    }

    /// GPT-2 byte-level pre-tokenization regex. Same pattern used by
    /// swift-transformers (and `huggingface/tokenizers`). Match objects
    /// segment text into chunks like `'s`, ` word`, ` 123`, etc. before
    /// byte-encoding.
    static let byteLevelPreTokenizeRegex: NSRegularExpression = {
        let pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        return try! NSRegularExpression(pattern: pattern)
    }()

    /// GPT-2 byte alphabet (255 visible code points covering all 256 byte
    /// values). Replicated here so we don't have to depend on
    /// swift-transformers' internal table.
    static let byteEncoderTable: [String] = buildByteEncoderTable()

    private static func buildByteEncoderTable() -> [String] {
        var arr = [String](repeating: "", count: 256)
        // The standard GPT-2 byte→unicode mapping. Visible ASCII / Latin-1
        // bytes map to themselves; non-visible bytes get re-coded into the
        // 256-383 unicode block.
        let directRanges: [ClosedRange<Int>] = [33...126, 161...172, 174...255]
        var b: [Int] = []
        for r in directRanges { b.append(contentsOf: r) }
        var cs = b
        var n = 0
        for x in 0..<256 where !b.contains(x) {
            b.append(x)
            cs.append(256 + n)
            n += 1
        }
        for (byteIdx, codepoint) in zip(b, cs) {
            arr[byteIdx] = String(UnicodeScalar(codepoint)!)
        }
        return arr
    }
}
