import Foundation
@preconcurrency import Tokenizers
@preconcurrency import Hub

/// Abstract tokenizer interface. Two implementations:
///   - `ByteTokenizer`: our default — 1 byte = 1 token, vocab = 256
///   - `HFTokenizer`: wraps `huggingface/swift-transformers`, supports
///     BPE / SentencePiece / tiktoken from a HF model directory
///
/// Both expose the same encode/decode pair, so the rest of the model
/// code doesn't care which is in use.
public protocol TGTokenizer: Sendable {
    /// Number of distinct token ids the tokenizer can produce.
    /// Must match the model's `vocabSize` for the embedding lookup to
    /// be in range.
    var vocabSize: Int { get }

    /// Encode a UTF-8 string into a list of integer token ids.
    func encode(_ text: String) throws -> [Int]

    /// Decode token ids back to a UTF-8 string.
    func decode(_ ids: [Int]) -> String
}

/// Byte-level tokenizer — what our from-scratch models use. Every UTF-8
/// byte is its own token id 0..255. Round-trip is exact for any input.
public struct ByteTokenizer: TGTokenizer {
    public var vocabSize: Int { 256 }
    public init() {}

    public func encode(_ text: String) throws -> [Int] {
        return [UInt8](text.utf8).map { Int($0) }
    }

    public func decode(_ ids: [Int]) -> String {
        let bytes = ids.compactMap { (id: Int) -> UInt8? in
            guard id >= 0 && id < 256 else { return nil }
            return UInt8(id)
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}

/// Wraps HuggingFace's `swift-transformers` tokenizer. Loads from a
/// local HF model directory that contains `tokenizer.json` (and
/// optionally `tokenizer_config.json`).
///
/// Supports the full HF spectrum: BPE (Llama, Mistral, Phi, Qwen, LFM,
/// Gemma), SentencePiece (T5 family), and WordPiece (BERT family).
/// The implementation auto-detects the right kind from `tokenizer.json`.
public final class HFTokenizer: TGTokenizer {
    private let tokenizer: Tokenizer
    public let vocabSize: Int

    /// Load a tokenizer from a HF model directory.
    /// Expects at minimum `tokenizer.json` to be present.
    public static func load(from url: URL) async throws -> HFTokenizer {
        // `AutoTokenizer.from(modelFolder:)` reads tokenizer.json +
        // tokenizer_config.json from a local directory and constructs
        // the right tokenizer kind automatically.
        let tokenizer = try await AutoTokenizer.from(modelFolder: url)
        return HFTokenizer(tokenizer: tokenizer)
    }

    private init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
        // The HF library exposes vocab via the underlying tokenizer's
        // `tokenizer.vocabSize` if available; fall back to a max-id scan.
        // For now, use the constants in tokenizer_config — we'll wire a
        // precise value later if needed.
        self.vocabSize = 0  // filled in by caller from HF config.json's vocab_size
    }

    public func encode(_ text: String) throws -> [Int] {
        return tokenizer.encode(text: text)
    }

    public func decode(_ ids: [Int]) -> String {
        return tokenizer.decode(tokens: ids)
    }
}
