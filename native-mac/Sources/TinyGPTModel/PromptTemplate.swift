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
            let preface = "<|im_start|>user\n\(userTurn)<|im_end|>\n<|im_start|>assistant\n"
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
