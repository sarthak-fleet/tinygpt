import Foundation
import TinyGPTIO
import TinyGPTBench
import TinyGPTServe

/// CLI entry point. Mirrors `python_ref/load_tinygpt.py --inspect`.
/// Subcommands:
///   tinygpt inspect <path>     — print the file's manifest + metadata
///   tinygpt validate <path>    — read and re-encode, exit 0 iff round-trips
///                                bit-identically (sanity check for writers)
///
/// Once the model + training milestones land, this entry point grows
/// `train` and `sample` subcommands. For M1 it's read-only.
@main
struct TinyGPT {
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let cmd = args.first else {
            printUsage()
            exit(2)
        }
        switch cmd {
        case "inspect":
            guard let path = args.dropFirst().first else {
                fputs("inspect: missing <path>\n", stderr); exit(2)
            }
            if path == "-h" || path == "--help" {
                print("usage: tinygpt inspect <path.tinygpt>\n  Prints the file's manifest + metadata.")
                exit(0)
            }
            run { try inspect(path: path) }
        case "validate":
            guard let path = args.dropFirst().first else {
                fputs("validate: missing <path>\n", stderr); exit(2)
            }
            if path == "-h" || path == "--help" {
                print("usage: tinygpt validate <path.tinygpt>\n  Round-trip check: read → encode → byte-compare.\n  Exit 0 iff the file re-encodes bit-identically.")
                exit(0)
            }
            run { try validate(path: path) }
        case "bench":
            // Inference-side LLM benchmark harness (Bench360-modelled).
            // See docs/benchmark_harness_design.md.
            Benchmark.run(args: Array(args.dropFirst()))
        case "infer-heatmap":
            InferHeatmap.run(args: Array(args.dropFirst()))
        case "bench-train":
            // Legacy training-throughput benchmark vs the WebGPU browser
            // baseline. Used to be `tinygpt bench` before the inference
            // harness shipped — preserved under a new name.
            Bench.run(args: Array(args.dropFirst()))
        case "train":
            Train.run(args: Array(args.dropFirst()))
        case "eval":
            Eval.run(args: Array(args.dropFirst()))
        case "eval-indic":
            // Indic-language eval harness: MILU (multi-choice across 11
            // Indic langs) + IndicGenBench (29-lang generative tasks).
            // Required by Wave 4 before claiming Hindi/multilingual
            // support. See docs/research/indic_evals.md.
            EvalIndic.run(args: Array(args.dropFirst()))
        case "finetune":
            Finetune.run(args: Array(args.dropFirst()))
        case "sft":
            SFT.run(args: Array(args.dropFirst()))
        case "dpo":
            DPO.run(args: Array(args.dropFirst()))
        case "distill":
            Distill.run(args: Array(args.dropFirst()))
        case "synthesize":
            Synthesize.run(args: Array(args.dropFirst()))
        case "tokenize-train":
            TokenizeTrain.run(args: Array(args.dropFirst()))
        case "rerank-train":
            RerankTrain.run(args: Array(args.dropFirst()))
        case "rerank-eval":
            RerankEval.run(args: Array(args.dropFirst()))
        case "es":
            ES.run(args: Array(args.dropFirst()))
        case "laser":
            LASER.run(args: Array(args.dropFirst()))
        case "hqq":
            HQQ.run(args: Array(args.dropFirst()))
        case "gptq":
            GPTQWorker.run(args: Array(args.dropFirst()))
        case "prune-unstructured":
            PruneUnstructured.run(args: Array(args.dropFirst()))
        case "prune-structured":
            PruneStructured.run(args: Array(args.dropFirst()))
        case "score-bench":
            fputs("[deprecated] score-bench is now run-bench; please migrate\n", stderr)
            RunBench.run(args: Array(args.dropFirst()))
        case "run-bench":
            RunBench.run(args: Array(args.dropFirst()))
        case "magpie":
            Magpie.run(args: Array(args.dropFirst()))
        case "tuned-lens":
            TunedLens.run(args: Array(args.dropFirst()))
        case "linear-probe":
            LinearProbe.run(args: Array(args.dropFirst()))
        case "dedupe":
            Dedupe.run(args: Array(args.dropFirst()))
        case "rome":
            ROME.run(args: Array(args.dropFirst()))
        case "memit":
            MEMIT.run(args: Array(args.dropFirst()))
        case "bon":
            BestOfN.run(args: Array(args.dropFirst()))
        case "sae":
            SAE.run(args: Array(args.dropFirst()))
        case "sae-explore":
            SaeExplore.run(args: Array(args.dropFirst()))
        case "sae-to-saelens":
            SaeToSaelens.run(args: Array(args.dropFirst()))
        case "train-quality-classifier":
            QualityClassifier.runTrain(args: Array(args.dropFirst()))
        case "quality-filter":
            QualityClassifier.runFilter(args: Array(args.dropFirst()))
        case "filter":
            Filter.run(args: Array(args.dropFirst()))
        case "eval-compare":
            EvalCompare.run(args: Array(args.dropFirst()))
        case "eval-gate":
            EvalGateCommand.run(args: Array(args.dropFirst()))
        case "run-lm-eval":
            RunLmEval.run(args: Array(args.dropFirst()))
        case "eval-mteb":
            EvalMTEB.run(args: Array(args.dropFirst()))
        case "merge":
            Merge.run(args: Array(args.dropFirst()))
        case "bake-lora":
            // Fold a .lora adapter into the base safetensors so the merged
            // dir is drop-in for HF tools / CoreML conversion. See
            // BakeLora.swift for math + dtype handling.
            BakeLora.run(args: Array(args.dropFirst()))
        case "ane-validate":
            // Compare MLX (reference, fp32) vs CoreML (.mlpackage, fp16/ANE)
            // top-1 token at the last position. Used as the M2 acceptance
            // gate for the ANE conversion pipeline.
            AneValidate.run(args: Array(args.dropFirst()))
        case "ane-bench-smoke":
            // Quick decode-rate measurement on the stateless .mlpackage.
            // Pessimistic vs M3 stateful path but tells us whether ANE
            // dispatch is engaging at all.
            AneBenchSmoke.run(args: Array(args.dropFirst()))
        case "coreml-serve":
            // Minimal OpenAI-compatible HTTP server backed by a Qwen3
            // stateful CoreML .mlpackage. See CoreMLServe.swift for the
            // PRD-flag-vs-sibling-command deviation note (the canonical
            // PRD asked for `serve --coreml`; we ship it as a sibling to
            // honor the "don't touch existing serve" rule).
            CoreMLServe.run(args: Array(args.dropFirst()))
        case "coreml-chunked-smoke":
            // Drive a chunked-block ANE bundle end-to-end from Swift.
            // M8 path: 28 single-block .mlpackages sidestep the ANE
            // multi-layer crash. See Qwen3ANEChunked.swift.
            if #available(macOS 15.0, *) {
                CoreMLChunkedSmoke.run(args: Array(args.dropFirst()))
            } else {
                fputs("coreml-chunked-smoke requires macOS 15+\n", stderr); exit(1)
            }
        case "eval-bfcl":
            EvalBFCL.run(args: Array(args.dropFirst()))
        case "eval-tau-bench":
            EvalTauBench.run(args: Array(args.dropFirst()))
        case "eval-humaneval":
            EvalHumanEval.run(args: Array(args.dropFirst()))
        case "judge":
            JudgeShim.run(args: Array(args.dropFirst()))
        case "patch":
            Patch.run(args: Array(args.dropFirst()))
        case "causal-trace":
            CausalTrace.run(args: Array(args.dropFirst()))
        case "gguf-inspect":
            GGUFInspect.run(args: Array(args.dropFirst()))
        case "gguf-load":
            GGUFLoad.run(args: Array(args.dropFirst()))
        case "gguf-extract":
            GGUFExtract.run(args: Array(args.dropFirst()))
        case "to-coreml":
            ToCoreML.run(args: Array(args.dropFirst()))
        case "to-safetensors":
            ToSafetensors.run(args: Array(args.dropFirst()))
        case "train-heads":
            TrainHeads.run(args: Array(args.dropFirst()))
        case "compare":
            Compare.run(args: Array(args.dropFirst()))
        case "hf-inspect":
            HFInspect.run(args: Array(args.dropFirst()))
        case "hf-load":
            HFLoad.run(args: Array(args.dropFirst()))
        case "download-dataset":
            DownloadDataset.run(args: Array(args.dropFirst()))
        case "list-datasets":
            ListDatasets.run(args: Array(args.dropFirst()))
        case "fetch-github":
            FetchGitHub.run(args: Array(args.dropFirst()))
        case "prep-data":
            PrepData.run(args: Array(args.dropFirst()))
        case "extractor-data":
            ExtractorData.run(args: Array(args.dropFirst()))
        case "train-extractor":
            TrainExtractor.run(args: Array(args.dropFirst()))
        case "extract":
            Extract.run(args: Array(args.dropFirst()))
        case "push":
            CloudPush.run(args: Array(args.dropFirst()))
        case "pull":
            CloudPull.run(args: Array(args.dropFirst()))
        case "cloud":
            CloudList.run(args: Array(args.dropFirst()))
        case "escalate":
            Escalate.run(args: Array(args.dropFirst()))
        case "sample":
            Sample.run(args: Array(args.dropFirst()))
        case "agent":
            Agent.run(args: Array(args.dropFirst()))
        case "screen":
            Screen.run(args: Array(args.dropFirst()))
        case "ax-capture":
            AXCapture.run(args: Array(args.dropFirst()))
        case "serve":
            Serve.run(args: Array(args.dropFirst()))
        case "debug-names":
            DebugNames.run(args: Array(args.dropFirst()))
        case "debug-load":
            DebugNames.compareLoaded(args: Array(args.dropFirst()))
        case "debug-logits":
            DebugNames.logits(args: Array(args.dropFirst()))
        case "debug-dtypes":
            DebugNames.dtypes(args: Array(args.dropFirst()))
        case "debug-loss":
            DebugNames.sanityLoss(args: Array(args.dropFirst()))
        case "vlm-smoke":
            // VLM specialist — Milestone 1 smoke (vision encoder load +
            // forward). See docs/prds/factory-vision-specialist.md.
            VLMSmoke.run(args: Array(args.dropFirst()))
        case "-h", "--help":
            printUsage()
        default:
            fputs("unknown subcommand: \(cmd)\n\n", stderr)
            printUsage()
            exit(2)
        }
    }

    private static func run(_ body: () throws -> Void) {
        do {
            try body()
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func printUsage() {
        print("""
        tinygpt — native-side CLI for the .tinygpt file format and training

        usage:
          tinygpt inspect <path>     print manifest + metadata for a .tinygpt file
          tinygpt validate <path>    round-trip check: read → encode → byte-compare
          tinygpt bench [flags]      inference-side LLM benchmark harness (Bench360-modelled)
          tinygpt infer-heatmap <trace.json> render an inference latency heatmap
          tinygpt bench-train [flags] training-throughput benchmark vs. WebGPU baseline
          tinygpt synthesize [flags] label prompt JSONL via an OpenAI-compatible teacher
          tinygpt tokenize-train [flags] train a domain BPE tokenizer.json
          tinygpt rerank-train [flags] train a lightweight reranker artifact
          tinygpt rerank-eval [flags]  evaluate a reranker and emit E0 rows
          tinygpt screen <sub> ...   Mac screen-reading scaffold (Wave 2.6)
                                     subs: capture | tree | both
                                     see `tinygpt screen --help` for flags
          tinygpt ax-capture [flags] capture PNG + AX JSON pairs for VLM data

        file format documented in Sources/TinyGPTIO/TinyGPTFile.swift.
        bench flags documented in `tinygpt bench --help`.
        bench harness design documented in docs/benchmark_harness_design.md.
        """)
    }

    static func inspect(path: String) throws {
        let url = URL(fileURLWithPath: path)
        let file = try TinyGPTFileReader.read(url)
        let header = file.header

        print("\nFile: \(url.path)")
        print(String(repeating: "-", count: 64))
        print("Version: \(file.version)")
        print("Step:    \(file.step)")

        print("\nConfig:")
        if let v = header.config.layers   { print("  layers     \(v)") }
        if let v = header.config.dModel   { print("  dModel     \(v)") }
        if let v = header.config.ctx      { print("  ctx        \(v)") }
        if let v = header.config.heads    { print("  heads      \(v)") }
        if let v = header.config.dMlp     { print("  dMlp       \(v)") }
        if let v = header.config.batchSize { print("  batchSize  \(v)") }
        if let v = header.config.backend  { print("  backend    \(v)") }

        if let fl = header.finalLoss {
            var line = "  final loss"
            if let t = fl.train { line += "  train \(format(t))" }
            if let v = fl.val   { line += ", val \(format(v))" }
            if let s = fl.step  { line += " @ step \(s)" }
            print(line)
        }

        if let sample = header.sample, !sample.isEmpty {
            let snippet = sample.prefix(80)
            print("\n  sample: \(snippet.debugDescription)")
        }

        print("\nTensors (\(file.tensors.count)):")
        var total = 0
        for tensor in file.tensors {
            let n = tensor.entry.elementCount
            total += n
            let shape = "\(tensor.entry.shape)"
            print("  \(pad(tensor.entry.name, 40))  \(pad(shape, 20))  \(formatCount(n))")
        }
        print(String(repeating: "-", count: 64))
        print("  total parameters: \(formatCount(total))")
    }

    static func validate(path: String) throws {
        let url = URL(fileURLWithPath: path)
        let original = try Data(contentsOf: url)
        let file = try TinyGPTFileReader.decode(original, source: url)
        let reencoded = try TinyGPTFileWriter.encode(file)

        if original == reencoded {
            print("OK: \(url.lastPathComponent) round-trips bit-identically (\(original.count) bytes)")
            return
        }

        // If the bytes differ, the most common causes are JSON key ordering
        // (the browser doesn't sort keys) and re-encoded loss-history (passed
        // through as raw JSON, whose nested objects also re-order). Fall back
        // to a semantic comparison of the modeled fields.
        let reDecoded = try TinyGPTFileReader.decode(reencoded, source: url)
        let headersMatch =
            file.header.config == reDecoded.header.config
            && file.header.manifest == reDecoded.header.manifest
            && file.header.savedAt == reDecoded.header.savedAt
            && file.header.finalLoss == reDecoded.header.finalLoss
            && file.header.sample == reDecoded.header.sample
            && file.header.weightDtype == reDecoded.header.weightDtype
            && file.header.includesOptimizerState == reDecoded.header.includesOptimizerState
            && file.header.stateByteLength == reDecoded.header.stateByteLength
        let tensorsMatch =
            file.step == reDecoded.step
            && file.tensors.count == reDecoded.tensors.count
            && zip(file.tensors, reDecoded.tensors).allSatisfy { a, b in
                a.weight == b.weight && a.adamM == b.adamM && a.adamV == b.adamV
            }
        if headersMatch && tensorsMatch {
            print("OK (semantic): \(url.lastPathComponent) round-trips with reordered JSON keys")
            print("    original \(original.count) bytes, re-encoded \(reencoded.count) bytes")
            return
        }
        fputs("FAIL: \(url.lastPathComponent) does not round-trip\n", stderr)
        if !headersMatch { fputs("  reason: header field mismatch\n", stderr) }
        if !tensorsMatch { fputs("  reason: tensor body mismatch\n", stderr) }
        exit(1)
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private static func formatCount(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
}
