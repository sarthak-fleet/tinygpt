import Foundation
import TinyGPTIO
import TinyGPTModel
#if canImport(CoreML)
import CoreML
@preconcurrency import Tokenizers
#endif

/// `tinygpt ane-validate` — sanity check a converted Pace .mlpackage
/// against the MLX-Swift reference.
///
/// Runs the prompt through both paths at temperature 0:
///   - MLX-Swift via TinyGPTModelHF (CoreGPU, fp32, the production
///     `tinygpt hf-load --sample` codepath)
///   - CoreML via `Qwen3ANE` (CPU+ANE, fp16 inside the .mlpackage)
///
/// Reports:
///   - top-1 token agreement (the strict gate)
///   - top-5 overlap (a softer "the model didn't go off the rails" check)
///   - logit cosine similarity at the last position (numerical health)
///
/// USAGE
///   tinygpt ane-validate \\
///       --coreml /path/to/pace.mlpackage \\
///       --hf-dir /path/to/baked-hf-dir \\
///       --prompt "The capital of France is" \\
///       --compute-units ane
///
/// EXPECTED OUTPUT (M2 acceptance)
///   ✓ top-1 agreement (MLX=12095 ' Paris'  ·  ANE=12095 ' Paris')
///     top-5 overlap: 5/5
///     logit cos-sim: 0.9994
enum AneValidate {
    static func run(args: [String]) {
        var coremlPath: String? = nil
        var hfDirPath: String? = nil
        var prompt: String = "The capital of France is"
        var computeUnits: String = "ane"
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--coreml": coremlPath = args[i+1]; i += 2
            case "--hf-dir": hfDirPath = args[i+1]; i += 2
            case "--prompt": prompt = args[i+1]; i += 2
            case "--compute-units": computeUnits = args[i+1]; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        guard let coremlPath = coremlPath, let hfDirPath = hfDirPath else {
            exitUsage()
        }
#if canImport(CoreML)
        run(coremlPath: coremlPath, hfDirPath: hfDirPath, prompt: prompt, computeUnits: computeUnits)
#else
        fputs("CoreML not available on this platform\n", stderr); exit(1)
#endif
    }

#if canImport(CoreML)
    private static func run(coremlPath: String, hfDirPath: String,
                             prompt: String, computeUnits: String) {
        let coremlURL = URL(fileURLWithPath: coremlPath)
        let hfDir = URL(fileURLWithPath: hfDirPath)

        // 1. Load the HF tokenizer + encode the prompt.
        let tok: Tokenizer
        do {
            tok = try blockingLoadTokenizer(from: hfDir)
        } catch {
            fputs("tokenizer load failed: \(error)\n", stderr); exit(1)
        }
        let ids = tok.encode(text: prompt).map { Int32($0) }
        print("""

        tinygpt ane-validate
        --------------------------------------------------------------
        prompt:        \(prompt.debugDescription)
        encoded ids:   \(ids)   (\(ids.count) tokens)
        mlpackage:     \(coremlURL.path)
        hf-dir:        \(hfDir.path)
        compute units: \(computeUnits)
        """)

        // 2. MLX path — load the HF model and run a single forward.
        print("\n[1/3] MLX path — loading HF model + tokenizer …")
        let mlxResult = mlxLastLogits(hfDir: hfDir, ids: ids)
        let mlxTop1 = mlxResult.top1Id
        print("      ✓ MLX top-1 = \(mlxTop1)  ·  ' \(tok.decode(tokens: [mlxTop1]))'")

        // 3. ANE path — load .mlpackage + run a single forward.
        print("\n[2/3] ANE path — loading .mlpackage with computeUnits=\(computeUnits) …")
        let cu = mlComputeUnits(from: computeUnits)
        let aneLogits: [Float]
        let aneTop1: Int
        do {
            let ane = try blockingLoadQwen3ANE(url: coremlURL, computeUnits: cu)
            print("      ✓ loaded — maxPromptLength=\(ane.maxPromptLength), vocab=\(ane.vocabSize)")
            if ids.count > ane.maxPromptLength {
                fputs("WARN: prompt (\(ids.count) tokens) longer than traced max (\(ane.maxPromptLength)); will truncate from left\n", stderr)
            }
            let logits = try ane.predictNextLogits(tokens: ids)
            aneLogits = logits
            aneTop1 = argmax(logits)
        } catch {
            fputs("ANE inference failed: \(error)\n", stderr); exit(1)
        }
        print("      ✓ ANE top-1 = \(aneTop1)  ·  ' \(tok.decode(tokens: [aneTop1]))'")

        // 4. Comparison report.
        print("\n[3/3] comparison")
        let mlxTop5 = topK(mlxResult.lastLogits, k: 5)
        let aneTop5 = topK(aneLogits, k: 5)
        let top5Overlap = Set(mlxTop5).intersection(Set(aneTop5)).count
        let cos = cosineSimilarity(mlxResult.lastLogits, aneLogits)

        print("      top-1 MLX  : \(mlxTop1)  ' \(tok.decode(tokens: [mlxTop1]))'")
        print("      top-1 ANE  : \(aneTop1)  ' \(tok.decode(tokens: [aneTop1]))'")
        print("      top-5 MLX  : \(mlxTop5)")
        print("      top-5 ANE  : \(aneTop5)")
        print("      top-5 overlap: \(top5Overlap)/5")
        print(String(format: "      logit cos-sim: %.6f", cos))

        if mlxTop1 == aneTop1 {
            print("\n✓ top-1 AGREEMENT — ANE path matches MLX reference on this prompt.")
        } else {
            print("\n✗ top-1 MISMATCH — MLX (\(mlxTop1)) vs ANE (\(aneTop1)).")
            print("  Investigate: did weights load correctly? Is the arch correct?")
            print("  See logit cos-sim above — values close to 1.0 mean small fp16 noise;")
            print("  values < 0.99 typically mean an arch / weight bug.")
            exit(1)
        }
    }

    // MARK: - MLX forward (single shot, last-position logits)

    private struct MLXResult {
        let lastLogits: [Float]
        let top1Id: Int
    }

    private static func mlxLastLogits(hfDir: URL, ids: [Int32]) -> MLXResult {
        // We deliberately don't reuse Sample.swift's full pipeline here —
        // we want a clean, allocation-light path that returns the raw
        // last-position logits as a `[Float]`. Use the HF loader + a
        // single `callAsFunction` invocation.
        let result: HFModelLoader.LoadResult
        do {
            result = try HFModelLoader.load(from: hfDir)
        } catch {
            fputs("HFModelLoader.load failed: \(error)\n", stderr); exit(1)
        }
        let model = result.model
        // Build the input MLXArray and forward.
        // We import MLX inside this function to avoid leaking it as an
        // unconditional dep — at the package level Sample.swift already
        // pulls MLX so this is free.
        let logitsArray: [Float] = mlxForwardLast(model: model, ids: ids)
        let argmaxId = argmax(logitsArray)
        return MLXResult(lastLogits: logitsArray, top1Id: argmaxId)
    }

    // MARK: - Tokenizer loading (sync bridge)

    private static func blockingLoadTokenizer(from dir: URL) throws -> Tokenizer {
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var boxed: Tokenizer? = nil
        nonisolated(unsafe) var error: Error? = nil
        Task.detached {
            do {
                boxed = try await AutoTokenizer.from(modelFolder: dir)
            } catch let e { error = e }
            sem.signal()
        }
        sem.wait()
        if let e = error { throw e }
        guard let t = boxed else {
            throw NSError(domain: "AneValidate", code: 99,
                          userInfo: [NSLocalizedDescriptionKey: "tokenizer load returned nil"])
        }
        return t
    }

    // MARK: - ANE loading (sync bridge)

    private static func blockingLoadQwen3ANE(url: URL,
                                              computeUnits: MLComputeUnits) throws -> Qwen3ANE {
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var boxed: Qwen3ANE? = nil
        nonisolated(unsafe) var error: Error? = nil
        Task.detached {
            do {
                boxed = try await Qwen3ANE.load(url: url, computeUnits: computeUnits)
            } catch let e { error = e }
            sem.signal()
        }
        sem.wait()
        if let e = error { throw e }
        guard let m = boxed else {
            throw NSError(domain: "AneValidate", code: 99,
                          userInfo: [NSLocalizedDescriptionKey: "ANE load returned nil"])
        }
        return m
    }

    // MARK: - helpers

    private static func argmax(_ logits: [Float]) -> Int {
        var best = 0
        var bestV: Float = -Float.greatestFiniteMagnitude
        for i in 0..<logits.count {
            if logits[i] > bestV { bestV = logits[i]; best = i }
        }
        return best
    }

    private static func topK(_ logits: [Float], k: Int) -> [Int] {
        let n = logits.count
        var indices = Array(0..<n)
        indices.sort { logits[$0] > logits[$1] }
        return Array(indices.prefix(k))
    }

    private static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<n {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let d = (na.squareRoot() * nb.squareRoot())
        return d > 0 ? dot / d : 0
    }

    private static func mlComputeUnits(from s: String) -> MLComputeUnits {
        switch s.lowercased() {
        case "ane":  return .cpuAndNeuralEngine
        case "gpu":  return .cpuAndGPU
        case "all":  return .all
        case "cpu":  return .cpuOnly
        default:
            fputs("unknown --compute-units \(s); using ane\n", stderr)
            return .cpuAndNeuralEngine
        }
    }
#endif

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt ane-validate --coreml <pkg.mlpackage> --hf-dir <dir> [options]

          --coreml PATH         the .mlpackage produced by qwen3_to_coreml.py
          --hf-dir PATH         HF directory whose weights match the .mlpackage
                                (baked dir from `tinygpt bake-lora`, OR the
                                base if the .mlpackage was converted from base)
          --prompt STR          parity prompt (default: "The capital of France is")
          --compute-units OPT   ane (default) | gpu | all | cpu

        Compares top-1 next-token between MLX-Swift (reference, fp32) and
        CoreML (.mlpackage, fp16). Exits 0 on top-1 agreement, 1 on mismatch.
        Also reports top-5 overlap + last-position logit cosine similarity
        for numerical health.
        """)
        exit(code)
    }
}
