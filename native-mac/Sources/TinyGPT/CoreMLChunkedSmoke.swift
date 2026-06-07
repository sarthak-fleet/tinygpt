import Foundation
#if canImport(CoreML)
@preconcurrency import CoreML
import TinyGPTModel
import Tokenizers

/// `tinygpt coreml-chunked-smoke` — drive a chunked-block Qwen3 ANE bundle
/// end-to-end from Swift. Measures prefill + decode tok/s and verifies
/// the model emits a coherent continuation.
///
/// This is the Swift parity of `scripts/ane/m8_chained_decode.py`. The
/// Python driver hit 17-18 tok/s; Swift removes the per-call ml.predict
/// overhead (~1ms × 28 = 28ms/token of waste) and should land at 30+ tok/s.
@available(macOS 15.0, *)
enum CoreMLChunkedSmoke {
    /// Sync entry: bridges the async runner via DispatchSemaphore, matching
    /// the pattern other CoreML-touching subcommands use.
    static func run(args: [String]) {
        let sem = DispatchSemaphore(value: 0)
        Task.detached {
            await runAsync(args: args)
            sem.signal()
        }
        sem.wait()
    }

    static func runAsync(args: [String]) async {
        var chunkedDir: String? = nil
        var hfDir: String? = nil
        var prompt = "The capital of France is"
        var maxSeq = 128
        var steps = 8
        var cu = MLComputeUnits.cpuAndNeuralEngine

        var i = 0
        while i < args.count {
            let a = args[i]
            func next() -> String? { i += 1; return i < args.count ? args[i] : nil }
            switch a {
            case "--chunked-dir": chunkedDir = next()
            case "--hf-dir":      hfDir = next()
            case "--prompt":      if let v = next() { prompt = v }
            case "--max-seq":     if let v = next(), let n = Int(v) { maxSeq = n }
            case "--steps":       if let v = next(), let n = Int(v) { steps = n }
            case "--compute-units":
                if let v = next() {
                    switch v.lowercased() {
                    case "ane":     cu = .cpuAndNeuralEngine
                    case "gpu":     cu = .cpuAndGPU
                    case "all":     cu = .all
                    case "cpu":     cu = .cpuOnly
                    default: break
                    }
                }
            case "-h", "--help":
                printHelp(); return
            default:
                fputs("unknown flag: \(a)\n", stderr); printHelp(); exit(2)
            }
            i += 1
        }
        guard let chunkedDir, let hfDir else {
            printHelp(); exit(2)
        }

        let cdURL = URL(fileURLWithPath: (chunkedDir as NSString).expandingTildeInPath)
        let hfURL = URL(fileURLWithPath: (hfDir as NSString).expandingTildeInPath)

        do {
            print("[setup] loading chunked model from \(cdURL.lastPathComponent)...")
            let t0 = Date()
            let model = try await Qwen3ANEChunked.load(
                chunkedDir: cdURL, hfDir: hfURL,
                computeUnits: cu, defaultMaxSeq: maxSeq)
            print("        loaded \(model.nLayers) blocks + embed/norm in \(String(format: "%.2f", -t0.timeIntervalSinceNow))s")
            print("        n_layers=\(model.nLayers)  hidden=\(model.hiddenSize)  vocab=\(model.vocabSize)")

            // Tokenize using HF tokenizer.json in the HF dir.
            print("[setup] loading tokenizer...")
            let tok = try await AutoTokenizer.from(modelFolder: hfURL)
            let ids = tok.encode(text: prompt).map { Int32($0) }
            print("        tokens: \(ids)")

            let states = model.makeStates()

            // Prefill: feed one token at a time (simpler than batched prefill).
            print("[run] prefill \(ids.count) tokens through \(model.nLayers) ANE blocks...")
            let prefStart = Date()
            var lastLogits: [Float] = []
            for (pos, tid) in ids.enumerated() {
                lastLogits = try await model.forward(
                    ids: [tid], positionOffset: pos, states: states)
            }
            let prefSec = -prefStart.timeIntervalSinceNow
            let prefRate = Double(ids.count) / prefSec
            print(String(format: "      prefill: %.0fms total, %.1f tok/s", prefSec * 1000, prefRate))

            // First decode = argmax of logits after prefill.
            var nextID = argmax(lastLogits)
            print("[run] first token: id=\(nextID)  '\(tok.decode(tokens: [nextID]))'")
            var generated: [Int] = ids.map(Int.init) + [nextID]

            // Steady-state decode.
            print("[run] decode \(steps - 1) more tokens...")
            let decStart = Date()
            for _ in 0..<(steps - 1) {
                let pos = generated.count - 1
                let logits = try await model.forward(
                    ids: [Int32(nextID)], positionOffset: pos, states: states)
                nextID = argmax(logits)
                generated.append(nextID)
            }
            let decSec = -decStart.timeIntervalSinceNow
            let decRate = Double(steps - 1) / decSec

            print("")
            print("=== generated ===")
            print("  prompt + completion: '\(tok.decode(tokens: generated))'")
            print(String(format: "  prefill tok/s: %.1f", prefRate))
            print(String(format: "  decode tok/s:  %.1f", decRate))
        } catch {
            fputs("error: \(error)\n", stderr); exit(1)
        }
    }

    static func argmax(_ x: [Float]) -> Int {
        var best = 0; var bestV: Float = -Float.infinity
        for i in 0..<x.count where x[i] > bestV { best = i; bestV = x[i] }
        return best
    }

    static func printHelp() {
        let s = """
        usage: tinygpt coreml-chunked-smoke --chunked-dir <dir> --hf-dir <dir> [opts]

        --chunked-dir DIR     dir containing m8-block-{0..N-1}.mlpackage files
                              (output of scripts/ane/m8_block_export.py)
        --hf-dir DIR          HF directory with config.json + safetensors
                              (the BAKED dir if running a LoRA-merged specialist)
        --prompt TEXT         prompt to decode (default: "The capital of France is")
        --max-seq N           must match the max_seq used at export time (default 128)
        --steps N             how many tokens to decode after prefill (default 8)
        --compute-units OPT   ane | gpu | all | cpu (default: ane)
        """
        print(s)
    }
}
#endif
