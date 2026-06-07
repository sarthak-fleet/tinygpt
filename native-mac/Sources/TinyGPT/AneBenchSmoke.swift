import Foundation
import TinyGPTIO
import TinyGPTModel
#if canImport(CoreML)
import CoreML
@preconcurrency import Tokenizers
#endif

/// `tinygpt ane-bench-smoke` — quick decode-rate measurement on a
/// stateless Qwen3 .mlpackage. NOT a full benchmark — just enough to
/// tell us whether ANE dispatch is happening at all.
///
/// What "decode" means here: loop the stateless model, appending the
/// argmax token to the prompt each step. This is O(T²) — every step
/// recomputes the full prefix — so it's pessimistic vs the eventual
/// stateful path. The numbers tell us:
///   - tok/s on the stateless path (lower bound for the stateful path)
///   - whether ANE is engaging (compare ane / gpu / cpu units)
///
/// Approval: this is a single short loop (default 16 tokens × ~T=256),
/// not a multi-minute benchmark. Falls within the "OK to run" envelope
/// per the brief's resource-discipline rule.
///
/// USAGE
///   tinygpt ane-bench-smoke \
///       --coreml /path/to/pace.mlpackage \
///       --hf-dir /path/to/hf-dir \
///       --prompt "..." --tokens 16 --compute-units ane
enum AneBenchSmoke {
    static func run(args: [String]) {
        var coremlPath: String? = nil
        var hfDirPath: String? = nil
        var prompt: String = "The capital of France is"
        var nTokens: Int = 16
        var computeUnits: String = "ane"
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--coreml": coremlPath = args[i+1]; i += 2
            case "--hf-dir": hfDirPath = args[i+1]; i += 2
            case "--prompt": prompt = args[i+1]; i += 2
            case "--tokens": nTokens = Int(args[i+1]) ?? nTokens; i += 2
            case "--compute-units": computeUnits = args[i+1]; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        guard let coremlPath = coremlPath, let hfDirPath = hfDirPath else { exitUsage() }
#if canImport(CoreML)
        runImpl(coremlPath: coremlPath, hfDirPath: hfDirPath,
                prompt: prompt, nTokens: nTokens, computeUnits: computeUnits)
#else
        fputs("CoreML not available\n", stderr); exit(1)
#endif
    }

#if canImport(CoreML)
    private static func runImpl(coremlPath: String, hfDirPath: String,
                                  prompt: String, nTokens: Int, computeUnits: String) {
        let coremlURL = URL(fileURLWithPath: coremlPath)
        let hfDir = URL(fileURLWithPath: hfDirPath)

        let tok: Tokenizer
        do { tok = try blockingLoadTokenizer(from: hfDir) }
        catch { fputs("tokenizer load: \(error)\n", stderr); exit(1) }
        var ids = tok.encode(text: prompt).map { Int32($0) }
        print("""

        tinygpt ane-bench-smoke
        ----------------------------------------------------
        mlpackage:     \(coremlURL.path)
        prompt tokens: \(ids.count)
        decode tokens: \(nTokens)
        compute units: \(computeUnits)
        """)
        let cu = mlComputeUnits(from: computeUnits)
        let ane: Qwen3ANE
        do {
            ane = try blockingLoadQwen3ANE(url: coremlURL, computeUnits: cu)
        } catch {
            fputs("ANE load failed: \(error)\n", stderr); exit(1)
        }
        print("loaded — maxPromptLength=\(ane.maxPromptLength), vocab=\(ane.vocabSize)\n")

        // Warm up — first call includes the compile / dispatch plan cost.
        // We discard its timing.
        do {
            _ = try ane.predictNextToken(tokens: ids)
        } catch {
            fputs("warmup failed: \(error)\n", stderr); exit(1)
        }

        var stepTimes: [Double] = []
        let loopStart = Date()
        for step in 0..<nTokens {
            let t0 = Date()
            let next: Int
            do { next = try ane.predictNextToken(tokens: ids) }
            catch { fputs("decode step \(step) failed: \(error)\n", stderr); exit(1) }
            let dt = -t0.timeIntervalSinceNow
            stepTimes.append(dt)
            ids.append(Int32(next))
            // Print incremental token. The first token may include a small
            // tokenizer-cache warmup but it's negligible vs the model fwd.
            let piece = tok.decode(tokens: [next])
            print(String(format: "  step %2d: %.3fs  %.0f tok/s  %@",
                          step, dt, 1.0 / max(dt, 1e-6), piece))
        }
        let totalT = -loopStart.timeIntervalSinceNow

        // Summary stats. The first step is roughly the same shape as
        // subsequent steps in the stateless setup (we always do a full
        // T=maxPromptLength forward), so we report mean / median.
        let sorted = stepTimes.sorted()
        let median = sorted[sorted.count / 2]
        let mean = stepTimes.reduce(0, +) / Double(stepTimes.count)
        let fastest = sorted.first ?? 0
        print("""

        decode summary (stateless, full-prefix-each-step — pessimistic):
          total:    \(String(format: "%.2f", totalT))s for \(nTokens) tokens
          mean:     \(String(format: "%.3f", mean))s/tok   (\(String(format: "%.1f", 1/mean)) tok/s)
          median:   \(String(format: "%.3f", median))s/tok   (\(String(format: "%.1f", 1/median)) tok/s)
          fastest:  \(String(format: "%.3f", fastest))s/tok   (\(String(format: "%.1f", 1/fastest)) tok/s)

        notes:
          - This is the stateless path. Every step re-encodes the full
            T=\(ane.maxPromptLength) prefix → O(T) per step inside the model.
          - The M3 stateful path will use coremltools.StateType for a
            persistent KV cache, dropping per-step compute to O(1) and
            should be 5-10× faster than these numbers.
          - If these numbers are < 10 tok/s, ANE dispatch may not be
            engaging. Profile via Xcode Instruments → Core ML to confirm.
          - For comparison, MLX-Swift on M-series is currently ~30-50 tok/s
            on this model (KV-cached).
        """)
    }

    private static func blockingLoadTokenizer(from dir: URL) throws -> Tokenizer {
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var boxed: Tokenizer? = nil
        nonisolated(unsafe) var error: Error? = nil
        Task.detached {
            do { boxed = try await AutoTokenizer.from(modelFolder: dir) }
            catch let e { error = e }
            sem.signal()
        }
        sem.wait()
        if let e = error { throw e }
        guard let t = boxed else {
            throw NSError(domain: "AneBenchSmoke", code: 99,
                          userInfo: [NSLocalizedDescriptionKey: "tokenizer load nil"])
        }
        return t
    }

    private static func blockingLoadQwen3ANE(url: URL,
                                              computeUnits: MLComputeUnits) throws -> Qwen3ANE {
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var boxed: Qwen3ANE? = nil
        nonisolated(unsafe) var error: Error? = nil
        Task.detached {
            do { boxed = try await Qwen3ANE.load(url: url, computeUnits: computeUnits) }
            catch let e { error = e }
            sem.signal()
        }
        sem.wait()
        if let e = error { throw e }
        guard let m = boxed else {
            throw NSError(domain: "AneBenchSmoke", code: 99,
                          userInfo: [NSLocalizedDescriptionKey: "ANE load nil"])
        }
        return m
    }

    private static func mlComputeUnits(from s: String) -> MLComputeUnits {
        switch s.lowercased() {
        case "ane":  return .cpuAndNeuralEngine
        case "gpu":  return .cpuAndGPU
        case "all":  return .all
        case "cpu":  return .cpuOnly
        default:    return .cpuAndNeuralEngine
        }
    }
#endif

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt ane-bench-smoke --coreml <pkg> --hf-dir <dir> [options]

          --coreml PATH         the .mlpackage to bench
          --hf-dir PATH         HF dir for the matching tokenizer
          --prompt STR          starting prompt (default: "The capital of France is")
          --tokens N            number of decode steps (default: 16)
          --compute-units OPT   ane (default) | gpu | all | cpu

        Loops the stateless .mlpackage `N` times to measure per-token decode
        cost on the M2 path. PESSIMISTIC vs the eventual stateful (M3) path —
        every step re-encodes the full T-token prefix.
        """)
        exit(code)
    }
}
