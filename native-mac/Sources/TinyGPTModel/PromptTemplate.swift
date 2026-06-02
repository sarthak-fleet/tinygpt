import Foundation

/// Prompt templates for supervised fine-tuning (SFT). Each template knows
/// how to format an (instruction, optional input, response) triple into
/// a single chat-style training string AND to mark which substring is the
/// "response" portion — that's the part the model should be scored
/// against, with the instruction tokens masked out of the loss.
///
/// Templates included:
///   - `.chatml`   — Used by SmolLM2, Qwen, and many newer chat models.
///                    `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…<|im_end|>`
///   - `.alpaca`   — The Stanford Alpaca format, ubiquitous for SFT datasets.
///                    `### Instruction:\n…\n\n### Response:\n…`
///   - `.llama`    — Meta's Llama 2/3 chat format, simpler `[INST] … [/INST] …`
///   - `.plain`    — No markers, just prompt + " " + response. For ablation.
public enum PromptTemplate: String, Sendable {
    case chatml, alpaca, llama, plain

    public init?(name: String) {
        guard let v = PromptTemplate(rawValue: name.lowercased()) else { return nil }
        self = v
    }

    /// Render an (instruction, input, response) triple as a single string
    /// plus the byte offset at which the response begins (for masking).
    ///
    /// `input` is the optional "context for the instruction" field that
    /// some datasets carry (e.g., Alpaca's text-to-be-summarized). Empty
    /// strings are dropped — the templates fold the instruction + input
    /// into one logical user turn.
    public func render(instruction: String, input: String, response: String)
        -> (fullText: String, responseStart: Int)
    {
        let userTurn: String
        if input.isEmpty {
            userTurn = instruction
        } else {
            userTurn = "\(instruction)\n\n\(input)"
        }
        switch self {
        case .chatml:
            // Detect inline `system: ...` prefix and split it into a
            // proper <|im_start|>system block. Datasets that convert
            // OpenAI-style `[{role, content}, ...]` arrays to a flat
            // `instruction` string (notably hermes-function-calling-v1)
            // bury the system role as a "system: ..." prefix; without
            // this split the system content gets wrapped in the user turn
            // and inference-time prompts have to mimic the buried shape,
            // which is a real footgun.
            let (systemBlock, userContent) = Self.splitChatmlSystem(userTurn)
            let preface = "\(systemBlock)<|im_start|>user\n\(userContent)<|im_end|>\n<|im_start|>assistant\n"
            return (preface + response + "<|im_end|>", preface.utf8.count)
        case .alpaca:
            let preface = "### Instruction:\n\(userTurn)\n\n### Response:\n"
            return (preface + response, preface.utf8.count)
        case .llama:
            let preface = "[INST] \(userTurn) [/INST] "
            return (preface + response, preface.utf8.count)
        case .plain:
            let preface = "\(userTurn) "
            return (preface + response, preface.utf8.count)
        }
    }

    /// Look for a `system: ...` prefix at the start of the text and
    /// peel it off into a `<|im_start|>system ... <|im_end|>` block.
    /// Returns `(systemBlock, userContent)`. If no system prefix is
    /// present, returns `("", text)`. Matches case-insensitively on the
    /// `system:` / `user:` role markers since dataset converters aren't
    /// consistent. Handles the common patterns:
    ///
    ///     "system: You are X.\nuser: hello"
    ///     "System: You are X.\n\nUser: hello"
    ///     "system: only a system msg"   (no user boundary; everything
    ///                                    becomes system, user empty)
    static func splitChatmlSystem(_ text: String) -> (system: String, user: String) {
        let trimmed = String(text.drop { $0 == " " || $0 == "\t" || $0 == "\n" })
        let head = String(trimmed.prefix(7)).lowercased()
        guard head == "system:" else { return ("", text) }
        let afterSystem = String(String(trimmed.dropFirst(7))
            .drop { $0 == " " || $0 == "\t" })
        // Find the first role boundary (newline followed by "user:" /
        // "assistant:" / "system:"). Search via the lowercased copy and
        // translate the offset back as an Int — String.Index from one
        // String can't be used on another even when they're char-equal
        // after lowercasing.
        let lowered = afterSystem.lowercased()
        let boundaries = ["\nuser:", "\nassistant:", "\nsystem:"]
        var bestOffset: Int? = nil
        var bestLen: Int = 0
        for marker in boundaries {
            if let r = lowered.range(of: marker) {
                let off = lowered.distance(from: lowered.startIndex, to: r.lowerBound)
                if bestOffset == nil || off < bestOffset! {
                    bestOffset = off
                    bestLen = marker.count
                }
            }
        }
        let systemContent: String
        let userContent: String
        if let off = bestOffset {
            let split = afterSystem.index(afterSystem.startIndex, offsetBy: off)
            let userStart = afterSystem.index(split, offsetBy: bestLen)
            systemContent = String(afterSystem[..<split])
            userContent = String(afterSystem[userStart...])
        } else {
            systemContent = afterSystem
            userContent = ""
        }
        let sysTrim = systemContent.trimmingCharacters(in: .whitespacesAndNewlines)
        let userTrim = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        let block = "<|im_start|>system\n\(sysTrim)<|im_end|>\n"
        return (block, userTrim)
    }
}

/// One SFT training example: the full tokenized sequence + a boolean
/// mask marking which positions belong to the response (loss is
/// computed only on `mask == true` positions).
///
/// Sequence shape: `[T]` int32 tokens, `[T]` Bool mask. T is variable
/// per example; the SFTCorpus pads/truncates batches to a common length.
public struct SFTExample: Sendable {
    public let tokens: [Int32]
    public let responseMask: [Bool]
    public init(tokens: [Int32], responseMask: [Bool]) {
        precondition(tokens.count == responseMask.count,
                     "tokens and mask must match length")
        self.tokens = tokens
        self.responseMask = responseMask
    }
}
