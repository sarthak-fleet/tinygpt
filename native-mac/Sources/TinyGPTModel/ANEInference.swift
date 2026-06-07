import Foundation
#if canImport(CoreML)
@preconcurrency import CoreML

/// Inference paths that route through Core ML so eligible ops dispatch to
/// the Apple Neural Engine (16-core, ~38 TOPS on M3+). Decode throughput is
/// expected to be 3-10× the MLX-Swift Metal path for the same model once
/// the stateful KV-cache variant (M3) lands.
///
/// Two model families live here:
///
///   1. `TinyGPTANE` — the original from-scratch byte-level path. Input
///      is `tokens: [1, T] Int32` (byte IDs 0-255), output is logits of
///      shape `[1, T, 256]`. Kept for the gallery models.
///
///   2. `Qwen3ANE` — the HF-arch path used by the Pace specialist
///      (Qwen3-0.6B + baked LoRA). Input is `input_ids: [1, T] Int32`
///      (BPE token IDs), output is logits of shape `[1, T, vocab]`.
///      Used by `tinygpt ane-validate` and `tinygpt serve --coreml`.
///
/// Both classes are stateless (full-prompt-at-once). The M3 stateful
/// variant adds KV-cache state via CoreML 8+ `MLState` — gated behind
/// macOS 15+ at runtime; we'll add it as a new `Qwen3ANEStateful` class
/// when that lands so the two execution shapes stay clearly separated.

@available(macOS 14.0, *)
public final class TinyGPTANE {
    private let model: MLModel
    public let contextLength: Int

    public init(model: MLModel, contextLength: Int) {
        self.model = model
        self.contextLength = contextLength
    }

    /// Load a compiled .mlpackage / .mlmodelc and configure it for ANE.
    public static func load(url: URL) async throws -> TinyGPTANE {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all  // .all = CPU + GPU + ANE; runtime picks per op
        let compiled: URL
        if url.pathExtension == "mlmodelc" {
            compiled = url
        } else {
            compiled = try await MLModel.compileModel(at: url)
        }
        let model = try await MLModel.load(contentsOf: compiled, configuration: cfg)
        // Pull context length from the input description ("tokens" shape).
        let inputDesc = model.modelDescription.inputDescriptionsByName["tokens"]
        let ctx = inputDesc?.multiArrayConstraint?.shape.last?.intValue ?? 256
        return TinyGPTANE(model: model, contextLength: ctx)
    }

    /// Run one forward pass on a token sequence (left-padded with 0s to
    /// `contextLength`). Returns logits over the last position — `[256]`
    /// next-byte distribution. Caller does sampling.
    public func predict(tokens: [UInt8]) throws -> [Float] {
        // Build a fixed-length input. Truncate from the left if longer.
        let T = contextLength
        var ids = [Int32](repeating: 0, count: T)
        let src = Array(tokens.suffix(T))
        for (i, b) in src.enumerated() {
            ids[T - src.count + i] = Int32(b)
        }
        let arr = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        for i in 0..<T {
            arr[i] = NSNumber(value: ids[i])
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["tokens": arr])
        let out = try model.prediction(from: provider)
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "TinyGPTANE", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "no logits output"])
        }
        // logits shape: [1, T, 256]. We want logits[0, T-1, :].
        let V = 256
        let base = (T - 1) * V
        var result = [Float](repeating: 0, count: V)
        for i in 0..<V {
            result[i] = Float(truncating: logits[base + i])
        }
        return result
    }
}

/// CoreML inference for HF-arch models (Qwen3, Llama-class). Built around
/// the .mlpackage emitted by `scripts/ane/qwen3_to_coreml.py`:
///
///   input  → input_ids: [1, T] Int32, T = traced max_prompt_length
///   output → logits:    [1, T, vocab_size] Float16/Float32
///
/// Prompt-prefill semantics: caller pads to T with a chosen padding token
/// (Qwen3 uses ID 0 for `<|endoftext|>` which is fine padding; the model
/// only reads positions ≤ kept_length under causal masking). We extract
/// the logit slice at the LAST KEPT position (right before padding kicks in)
/// for next-token sampling.
@available(macOS 14.0, *)
public final class Qwen3ANE {
    public let model: MLModel
    /// Traced max prompt length (the fixed T the .mlpackage was converted with).
    public let maxPromptLength: Int
    /// Output vocab size, pulled from the logits shape at load time.
    public let vocabSize: Int
    /// Which compute path the runtime is told to favour. Kept for
    /// observability — Xcode Instruments → Core ML still shows the
    /// per-op dispatch.
    public let computeUnits: MLComputeUnits

    public init(model: MLModel, maxPromptLength: Int, vocabSize: Int,
                computeUnits: MLComputeUnits) {
        self.model = model
        self.maxPromptLength = maxPromptLength
        self.vocabSize = vocabSize
        self.computeUnits = computeUnits
    }

    public enum ANELoadError: Error, CustomStringConvertible {
        case missingInput(String)
        case missingOutput(String)
        case unexpectedShape(String, [Int])
        public var description: String {
            switch self {
            case .missingInput(let n): return "model has no input named '\(n)'"
            case .missingOutput(let n): return "model has no output named '\(n)'"
            case .unexpectedShape(let n, let s): return "tensor '\(n)' has unexpected shape \(s)"
            }
        }
    }

    /// Load + compile a .mlpackage with the requested compute_units. Returns
    /// the wrapped model + the introspected shapes.
    public static func load(url: URL,
                             computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> Qwen3ANE {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        // Performance hints. The defaults are usually right for ANE
        // (CoreML compiles a per-device plan at load), but we make our
        // intent explicit.
        cfg.allowLowPrecisionAccumulationOnGPU = true

        let compiled: URL
        if url.pathExtension == "mlmodelc" {
            compiled = url
        } else {
            compiled = try await MLModel.compileModel(at: url)
        }
        let model = try await MLModel.load(contentsOf: compiled, configuration: cfg)

        // Introspect input shape: [1, T].
        guard let inputDesc = model.modelDescription.inputDescriptionsByName["input_ids"] else {
            throw ANELoadError.missingInput("input_ids")
        }
        guard let inputConstraint = inputDesc.multiArrayConstraint else {
            throw ANELoadError.missingInput("input_ids (no multiArrayConstraint)")
        }
        let inShape = inputConstraint.shape.map { $0.intValue }
        guard inShape.count == 2, inShape[0] == 1 else {
            throw ANELoadError.unexpectedShape("input_ids", inShape)
        }
        let T = inShape[1]

        // Introspect output shape: [1, T, vocab].
        guard let outputDesc = model.modelDescription.outputDescriptionsByName["logits"] else {
            throw ANELoadError.missingOutput("logits")
        }
        guard let outputConstraint = outputDesc.multiArrayConstraint else {
            throw ANELoadError.missingOutput("logits (no multiArrayConstraint)")
        }
        let outShape = outputConstraint.shape.map { $0.intValue }
        guard outShape.count == 3, outShape[0] == 1, outShape[1] == T else {
            throw ANELoadError.unexpectedShape("logits", outShape)
        }
        let vocab = outShape[2]

        return Qwen3ANE(model: model, maxPromptLength: T, vocabSize: vocab,
                         computeUnits: computeUnits)
    }

    /// Run one forward pass on a token sequence (truncated from the LEFT
    /// to fit `maxPromptLength`, then padded on the RIGHT with `padTokenId`).
    /// Returns the logits at the last KEPT position (the next-token
    /// distribution at the end of the prompt).
    ///
    /// Why right-pad and slice from the right edge of the kept prefix:
    /// the model is causal, so positions beyond the prompt see only
    /// padding tokens — which the model has never been trained to attend
    /// past — and their logits aren't meaningful. The position we want is
    /// `kept_length - 1`, where the prompt actually ends.
    public func predictNextLogits(tokens: [Int32],
                                    padTokenId: Int32 = 0) throws -> [Float] {
        let T = maxPromptLength
        let src = Array(tokens.suffix(T))
        guard !src.isEmpty else {
            throw NSError(domain: "Qwen3ANE", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "empty token list"])
        }
        let kept = src.count

        // Build the padded input as an MLMultiArray of Int32 length T.
        // We construct via `dataPointer` for speed — at T=2048 the
        // per-element NSNumber path would be ~2× slower than the memcpy.
        let arr = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let bufPtr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<kept { bufPtr[i] = src[i] }
        for i in kept..<T { bufPtr[i] = padTokenId }

        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_ids": arr])
        let out = try model.prediction(from: provider)
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "Qwen3ANE", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "no logits output"])
        }
        // logits is [1, T, vocab]. Extract row at position (kept - 1).
        // CoreML returns either Float32 or Float16 — handle both via
        // NSNumber's float bridge (which converts internally).
        let V = vocabSize
        let base = (kept - 1) * V
        var result = [Float](repeating: 0, count: V)
        // Fast-path: dataType is .float32 → memcpy of the row.
        if logits.dataType == .float32 {
            let lp = logits.dataPointer.assumingMemoryBound(to: Float.self)
            result.withUnsafeMutableBufferPointer { rp in
                _ = memcpy(rp.baseAddress, lp + base, V * MemoryLayout<Float>.size)
            }
        } else if logits.dataType == .float16 {
            // Float16 → Float32 conversion via the bit-pattern recipe.
            let lp = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<V {
                result[i] = Float(Float16(bitPattern: lp[base + i]))
            }
        } else {
            // Fallback for unexpected dtypes (Double / Int32) — slow path.
            for i in 0..<V {
                result[i] = Float(truncating: logits[base + i])
            }
        }
        return result
    }

    /// Argmax — top-1 next token ID. Returns the index over the vocab.
    public func predictNextToken(tokens: [Int32], padTokenId: Int32 = 0) throws -> Int {
        let logits = try predictNextLogits(tokens: tokens, padTokenId: padTokenId)
        var best = 0
        var bestV: Float = -Float.greatestFiniteMagnitude
        for i in 0..<logits.count {
            if logits[i] > bestV { bestV = logits[i]; best = i }
        }
        return best
    }
}

/// Stateful CoreML inference for the M3 Qwen3 .mlpackage.
///
/// Surface differs from `Qwen3ANE`:
///   - Inputs: input_ids `[1, T_new]` int32, causal_mask `[1, 1, T_new, end_step]`
///     fp16, position_offset `[1]` int32
///   - State: opaque MLState (consolidated k_cache + v_cache slots)
///   - Outputs: logits `[1, T_new, vocab]` fp16
///
/// Usage pattern (decode loop):
///   1. `state = try ane.makeState()`  — fresh per request
///   2. Prefill the prompt at once:    `_ = try ane.forward(state, ids: promptIds, positionOffset: 0)`
///   3. Repeat:                        `nextLogits = try ane.forward(state, ids: [lastToken], positionOffset: ctxLen)`
///
/// The mlpackage was traced with RangeDim on both query length and
/// end_step, so the same model handles prefill (T_new > 1) and decode
/// (T_new = 1).
@available(macOS 15.0, *)
public final class Qwen3ANEStateful: @unchecked Sendable {
    public let model: MLModel
    public let maxSeqLen: Int
    public let vocabSize: Int
    public let computeUnits: MLComputeUnits

    public init(model: MLModel, maxSeqLen: Int, vocabSize: Int,
                computeUnits: MLComputeUnits) {
        self.model = model; self.maxSeqLen = maxSeqLen
        self.vocabSize = vocabSize; self.computeUnits = computeUnits
    }

    public enum LoadError: Error, CustomStringConvertible {
        case noInput(String)
        case noState(String)
        case noOutput(String)
        public var description: String {
            switch self {
            case .noInput(let n): return "no input '\(n)'"
            case .noState(let n): return "no state '\(n)'"
            case .noOutput(let n): return "no output '\(n)'"
            }
        }
    }

    public static func load(url: URL,
                             computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                             defaultMaxSeq: Int = 256) async throws -> Qwen3ANEStateful {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        cfg.allowLowPrecisionAccumulationOnGPU = true
        let compiled: URL
        if url.pathExtension == "mlmodelc" {
            compiled = url
        } else {
            compiled = try await MLModel.compileModel(at: url)
        }
        let model = try await MLModel.load(contentsOf: compiled, configuration: cfg)
        // CoreML's high-level Swift API doesn't expose state shapes
        // (`stateDescriptionsByName` returns descriptions whose constraint
        // is nil for state slots — the shape lives in the underlying
        // protobuf spec). Rather than parse that, we accept the
        // `defaultMaxSeq` from the caller (the value passed to
        // qwen3_to_coreml.py at convert time, typically 256) and probe
        // the vocab via a 1-token forward.
        let maxSeq = defaultMaxSeq
        // Probe vocab with a 1-token forward.
        let probe = try MLMultiArray(shape: [1, 1], dataType: .int32)
        probe.dataPointer.assumingMemoryBound(to: Int32.self)[0] = 0
        let mask = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
        mask.dataPointer.assumingMemoryBound(to: UInt16.self)[0] = Float16(0).bitPattern
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos.dataPointer.assumingMemoryBound(to: Int32.self)[0] = 0
        let state = model.makeState()
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": probe, "causal_mask": mask, "position_offset": pos
        ])
        let out = try await model.prediction(from: provider, using: state)
        var vocab = 0
        if let logits = out.featureValue(for: "logits")?.multiArrayValue,
           logits.shape.count >= 3 {
            vocab = logits.shape.last?.intValue ?? 0
        }
        guard vocab > 0 else {
            throw LoadError.noOutput("logits (vocab probe returned 0)")
        }
        return Qwen3ANEStateful(model: model, maxSeqLen: maxSeq,
                                 vocabSize: vocab, computeUnits: computeUnits)
    }

    /// Create a fresh KV cache state. Caller owns the returned MLState
    /// and must reuse it across forwards within a single decode session.
    public func makeState() -> MLState {
        return model.makeState()
    }

    /// Run one forward — prefill OR decode — and return the logits at
    /// the LAST position of the new token chunk.
    ///   ids: the new token chunk to process (length T_new ≥ 1)
    ///   positionOffset: absolute position of `ids[0]` in the full sequence
    ///                   (0 for prefill, increments per decode step)
    ///   state: MLState from `makeState()`, mutated in place
    public func forward(ids: [Int32], positionOffset: Int, state: MLState) async throws -> [Float] {
        let T_new = ids.count
        precondition(T_new >= 1, "ids must be non-empty")
        let endStep = positionOffset + T_new
        precondition(endStep <= maxSeqLen, "endStep \(endStep) exceeds maxSeqLen \(maxSeqLen)")
        // input_ids: [1, T_new] int32
        let idsArr = try MLMultiArray(shape: [1, NSNumber(value: T_new)], dataType: .int32)
        let idsPtr = idsArr.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<T_new { idsPtr[i] = ids[i] }
        // causal_mask: [1, 1, T_new, end_step] fp16 additive (-1e4 / 0).
        // For each new-token position j and key position k:
        //   mask[j, k] = 0 if k <= positionOffset + j else -1e4
        // Caching means past positions are always visible; we only need
        // to mask future positions within the new chunk.
        let maskArr = try MLMultiArray(shape: [1, 1,
                                                NSNumber(value: T_new),
                                                NSNumber(value: endStep)],
                                         dataType: .float16)
        let maskPtr = maskArr.dataPointer.assumingMemoryBound(to: UInt16.self)
        let neg = Float16(-1.0e4).bitPattern
        let zero = Float16(0).bitPattern
        for j in 0..<T_new {
            let absRow = positionOffset + j  // absolute position of this query
            let rowBase = j * endStep
            for k in 0..<endStep {
                maskPtr[rowBase + k] = (k <= absRow) ? zero : neg
            }
        }
        // position_offset: [1] int32
        let posArr = try MLMultiArray(shape: [1], dataType: .int32)
        posArr.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(positionOffset)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": idsArr,
            "causal_mask": maskArr,
            "position_offset": posArr,
        ])
        let out = try await model.prediction(from: provider, using: state)
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "Qwen3ANEStateful", code: 1)
        }
        // Slice the last position. shape: [1, T_new, vocab]
        let V = vocabSize
        let base = (T_new - 1) * V
        var result = [Float](repeating: 0, count: V)
        if logits.dataType == .float32 {
            let lp = logits.dataPointer.assumingMemoryBound(to: Float.self)
            result.withUnsafeMutableBufferPointer { rp in
                _ = memcpy(rp.baseAddress, lp + base, V * MemoryLayout<Float>.size)
            }
        } else if logits.dataType == .float16 {
            let lp = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<V { result[i] = Float(Float16(bitPattern: lp[base + i])) }
        } else {
            for i in 0..<V { result[i] = Float(truncating: logits[base + i]) }
        }
        return result
    }
}
#endif
