import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// Greedy speculative decoding (Leviathan et al., 2023, simplified).
///
/// Two models, same tokenizer:
///   - DRAFT: small, fast — generates a `K`-long candidate continuation
///     autoregressively in `K` forward passes (each small).
///   - TARGET: the model we actually want — verifies all K positions in
///     ONE parallel forward pass, returning argmax at each position.
///
/// Accept rule (greedy): the longest prefix where target's argmax at
/// position i matches the draft's token at that position. On the first
/// mismatch we substitute target's token and discard the rest. Lossless
/// vs target's greedy output — every accepted prefix is what target
/// would have produced anyway.
///
/// Speedup is K-ish on benign branches (the draft model usually agrees),
/// and falls back to ~1× target-step latency in the worst case (every
/// position mismatches). Typical 2-4× wall-clock improvement on small
/// draft + medium target pairs.
///
/// Why GREEDY first (vs full rejection-sampling spec decoding): the
/// greedy variant matches T=0 exactly without per-token Uniform draws
/// and probability ratios, so we get the simplest correct path shipped.
/// Temperature > 0 spec decoding is a follow-up (needs full p_draft +
/// p_target softmaxes + rejection sampling).
enum SpeculativeDecode {

    /// Run a single speculative step. Returns the list of newly-accepted
    /// token ids (length ≥ 1, ≤ k+1). Mutates `ids` by appending.
    ///
    /// Cost: 1 target forward over K positions + K draft forwards over 1
    /// position each (the draft's autoregressive loop). Net: ~K × draft
    /// cost + 1 × target cost — wins when target ≫ draft and the
    /// draft's K-token prediction is mostly correct.
    static func step(target: AnyModel, draft: AnyModel,
                      ids: inout [Int], k: Int, ctxCap: Int) -> [Int]
    {
        // 1. Draft proposes k tokens autoregressively (greedy). Each call
        //    starts from `ids` plus what's been proposed so far.
        var proposals: [Int] = []
        for _ in 0..<k {
            let tail = (ids + proposals).suffix(ctxCap)
            let arr = MLXArray(tail.map { Int32($0) }, [1, tail.count])
            let logits = draft(arr)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let nextId = argMax(last, axis: -1).reshaped([1])
            eval(nextId)
            proposals.append(Int(nextId.item(Int32.self)))
        }

        // 2. Target verifies in ONE parallel forward over the full prompt
        //    + k proposed tokens. At position i (0-based from the original
        //    end-of-prompt), the target's argmax predicts what token comes
        //    next. We accept proposal i iff target's argmax at that
        //    position matches.
        let withProposals = (ids + proposals).suffix(ctxCap)
        let inputArr = MLXArray(withProposals.map { Int32($0) }, [1, withProposals.count])
        let tLogits = target(inputArr)
        // Compare positions [len(prompt)-1 .. len(prompt)-1+k] of tLogits's
        // 2nd axis — argmax at that position predicts the NEXT token.
        let promptLen = withProposals.count - proposals.count
        var accepted: [Int] = []
        var lastTargetTok = -1
        for i in 0..<proposals.count {
            // Position in tLogits where the (i+1)th token would be predicted:
            //   the model's output at index promptLen - 1 + i predicts token at promptLen + i.
            let pos = promptLen - 1 + i
            let row = tLogits[0..., pos, 0...]
            let argT = argMax(row, axis: -1).reshaped([1])
            eval(argT)
            let tTok = Int(argT.item(Int32.self))
            if tTok == proposals[i] {
                accepted.append(tTok)
            } else {
                // First mismatch: take target's argmax, stop here.
                accepted.append(tTok)
                ids.append(contentsOf: accepted)
                return accepted
            }
            lastTargetTok = tTok
        }
        // All k accepted: free bonus — sample target's argmax at the
        // last verified position too (predicts the (k+1)th token).
        _ = lastTargetTok
        let bonusPos = promptLen - 1 + proposals.count
        let bonusRow = tLogits[0..., bonusPos, 0...]
        let bonusArg = argMax(bonusRow, axis: -1).reshaped([1])
        eval(bonusArg)
        accepted.append(Int(bonusArg.item(Int32.self)))
        ids.append(contentsOf: accepted)
        return accepted
    }

    /// Stochastic speculative decoding (Leviathan et al., 2023 — full
    /// recipe with rejection sampling). Lossless wrt sampling from
    /// `softmax(target_logits / T)` at the same temperature — every
    /// accepted continuation is distributed identically to plain
    /// target sampling.
    ///
    /// For each draft proposal at position i:
    ///   - p_target(x) = softmax(target_logits[i] / T)
    ///   - p_draft(x) = softmax(draft_logits[i]  / T)
    ///   - accept with probability min(1, p_target(t) / p_draft(t))
    ///   - on rejection, resample from the residual
    ///     max(0, p_target - p_draft), normalized
    ///
    /// All-accepted case: bonus token sampled from p_target at the
    /// next position — that's the "free token" speculative decoding
    /// pays out when the draft agrees with target.
    ///
    /// Why this is the right call for T > 0: greedy spec dec is only
    /// equivalent to T=0 sampling; for T > 0 you need this rejection
    /// loop to keep the output distribution lossless. Same correctness
    /// guarantee as Leviathan's Theorem 3.5.
    static func stepStochastic(target: AnyModel, draft: AnyModel,
                                ids: inout [Int], k: Int, ctxCap: Int,
                                temperature: Float) -> [Int] {
        let T = max(temperature, 1e-5)

        // 1. Draft proposes k tokens by SAMPLING (not argmax). Retain
        //    the full softmax distribution at each step so we can
        //    compute p_draft(proposed) and the rejection residual.
        var proposals: [Int] = []
        var draftProbs: [MLXArray] = []  // each shape [vocab]
        for _ in 0..<k {
            let tail = (ids + proposals).suffix(ctxCap)
            let arr = MLXArray(tail.map { Int32($0) }, [1, tail.count])
            let logits = draft(arr)
            let lastLogits = logits[0..., logits.shape[1] - 1, 0...]
            let scaled = lastLogits / MLXArray(T)
            let probs = softmax(scaled, axis: -1).reshaped([scaled.shape[scaled.shape.count - 1]])
            eval(probs)
            draftProbs.append(probs)
            // MLXRandom.categorical takes logits, so pass the unnormalized scaled logits.
            let nextId = MLXRandom.categorical(scaled).reshaped([1])
            eval(nextId)
            proposals.append(Int(nextId.item(Int32.self)))
        }

        // 2. Target verifies in ONE parallel forward over prompt + k proposals.
        let withProposals = (ids + proposals).suffix(ctxCap)
        let inputArr = MLXArray(withProposals.map { Int32($0) }, [1, withProposals.count])
        let tLogits = target(inputArr)
        let promptLen = withProposals.count - proposals.count

        // 3. Accept/reject per position.
        var accepted: [Int] = []
        for i in 0..<proposals.count {
            let pos = promptLen - 1 + i
            let row = tLogits[0..., pos, 0...]
            let scaled = row / MLXArray(T)
            let pTarget = softmax(scaled, axis: -1)
                .reshaped([scaled.shape[scaled.shape.count - 1]])
            eval(pTarget)

            let tok = proposals[i]
            let pT_tok = pTarget[tok].item(Float.self)
            let pD_tok = draftProbs[i][tok].item(Float.self)
            let ratio: Float = pT_tok / max(pD_tok, 1e-10)
            let u: Float = Float.random(in: 0..<1)

            if u <= min(1.0, ratio) {
                accepted.append(tok)
            } else {
                // Resample from the residual: max(0, p_target - p_draft)
                // normalized. Theorem 3.5 of Leviathan et al. — the
                // resulting distribution exactly matches sampling from
                // p_target alone.
                let residRaw = MLX.maximum(pTarget - draftProbs[i], MLXArray(Float(0)))
                let residSum = MLX.sum(residRaw).item(Float.self)
                let newTok: Int
                if residSum > 1e-10 {
                    let resid = residRaw / MLXArray(residSum)
                    // Sample from a categorical distribution given probs:
                    // log(p) + Gumbel noise, then argmax. The Gumbel trick.
                    let logResid = MLX.log(resid + MLXArray(Float(1e-20)))
                    let sampled = MLXRandom.categorical(logResid).reshaped([1])
                    eval(sampled)
                    newTok = Int(sampled.item(Int32.self))
                } else {
                    // Pathological: p_target ≤ p_draft pointwise. Fall back
                    // to a fresh sample from p_target directly.
                    let sampled = MLXRandom.categorical(scaled).reshaped([1])
                    eval(sampled)
                    newTok = Int(sampled.item(Int32.self))
                }
                accepted.append(newTok)
                ids.append(contentsOf: accepted)
                return accepted
            }
        }

        // 4. All k accepted: bonus token sampled from p_target at the
        //    "next" position. The free token speculative decoding
        //    earns when the draft and target agree on the full burst.
        let bonusPos = promptLen - 1 + proposals.count
        let bonusRow = tLogits[0..., bonusPos, 0...]
        let bonusScaled = bonusRow / MLXArray(T)
        let bonusId = MLXRandom.categorical(bonusScaled).reshaped([1])
        eval(bonusId)
        accepted.append(Int(bonusId.item(Int32.self)))
        ids.append(contentsOf: accepted)
        return accepted
    }
}
