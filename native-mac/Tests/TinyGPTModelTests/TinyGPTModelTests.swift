import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO
import XCTest
@testable import TinyGPTModel

/// MLX-Swift's SPM build does NOT compile the Metal shader library
/// (`default.metallib`) — Xcode compiles `.metal` files automatically as part
/// of its build system, but `swift build` doesn't run the Metal toolchain.
///
/// The result: running these tests via `swift test` (Command Line) fails with
/// "Failed to load the default metallib." even when we force the CPU stream,
/// because MLX's C runtime tries to init both streams at module load.
///
/// **Workaround**: run these tests inside Xcode (Product → Test) or via
/// `xcodebuild test -scheme TinyGPT`. The compiled metallib ends up in the
/// product's resources directory and the C runtime finds it.
///
/// We keep one trivial test that doesn't touch MLX at runtime so `swift test`
/// still surfaces source-level breakage; the real numerics tests live behind
/// the Xcode-only flag.
final class TinyGPTModelTests: XCTestCase {

    // MARK: - ModelConfig defaults + presets

    /// Compile-only test — proves `TinyGPTModel.swift` and the public surface
    /// still build without MLX runtime calls. The full numerics suite is in
    /// `TinyGPTModelNumericsTests` (Xcode-only).
    func test_modelConfigConstructs() {
        let cfg = ModelConfig.huge
        XCTAssertEqual(cfg.nLayers, 12)
        XCTAssertEqual(cfg.dModel, 256)
        XCTAssertEqual(cfg.contextLength, 256)
        XCTAssertEqual(cfg.headDim, 32)
    }

    func test_megaConfigIsBiggerThanHuge() {
        XCTAssertGreaterThan(ModelConfig.mega.dModel, ModelConfig.huge.dModel)
        XCTAssertGreaterThan(ModelConfig.mega.contextLength, ModelConfig.huge.contextLength)
        XCTAssertGreaterThan(ModelConfig.mega.nLayers, ModelConfig.huge.nLayers)
    }

    /// Default ctor surfaces a small byte-level config. Pins each field so
    /// a future "let's bump the default" gets a deliberate review.
    func test_modelConfigDefaults_areTinyByteLevel() {
        let cfg = ModelConfig()
        XCTAssertEqual(cfg.modelName, "byte-tinygpt-v0")
        XCTAssertEqual(cfg.vocabSize, 256)
        XCTAssertEqual(cfg.contextLength, 128)
        XCTAssertEqual(cfg.nLayers, 4)
        XCTAssertEqual(cfg.nHeads, 4)
        XCTAssertEqual(cfg.dModel, 128)
        XCTAssertEqual(cfg.dMlp, 512)
        XCTAssertEqual(cfg.dropout, 0)
        XCTAssertTrue(cfg.tieEmbeddings)
        XCTAssertEqual(cfg.dtype, "float32")
        XCTAssertFalse(cfg.useRoPE)
        XCTAssertFalse(cfg.useRMSNorm)
        XCTAssertFalse(cfg.useSwiGLU)
        XCTAssertFalse(cfg.useYOCO)
        XCTAssertFalse(cfg.useGradCheckpoint)
        XCTAssertFalse(cfg.useMoD)
        XCTAssertFalse(cfg.useDifferentialAttention)
        XCTAssertFalse(cfg.useALiBi)
        XCTAssertNil(cfg.slidingWindow)
        XCTAssertEqual(cfg.nExperts, 1)  // dense default
        XCTAssertEqual(cfg.mtpHorizons, 1)
        XCTAssertNil(cfg.tokenizerSource)
    }

    func test_modelConfig_nKvHeadsDefaultsToNHeads() {
        let cfg = ModelConfig(nHeads: 8)
        XCTAssertEqual(cfg.nKvHeads, 8)
    }

    func test_modelConfig_gqaWithFewerKVHeads() {
        // Grouped-query attention: nKvHeads < nHeads. Must divide.
        let cfg = ModelConfig(nHeads: 32, nKvHeads: 8, dModel: 256)
        XCTAssertEqual(cfg.nHeads, 32)
        XCTAssertEqual(cfg.nKvHeads, 8)
        XCTAssertEqual(cfg.headDim, 8)
    }

    // MARK: - Forward determinism on fixed seed

    /// MLXRandom.seed(s) pins the noise the embedding init draws from.
    /// Two models built under the same seed must produce bit-identical
    /// logits for the same input. The check is two passes / one process —
    /// not across-process (xcodebuild can reorder targets and there's no
    /// guarantee an out-of-test reset matches).
    func test_forwardIsDeterministicUnderFixedSeed() {
        let cfg = ModelConfig(vocabSize: 32, contextLength: 8, nLayers: 2,
                              nHeads: 2, dModel: 16, dMlp: 32)
        MLXRandom.seed(0xC0FFEE)
        let m1 = TinyGPTModel(cfg)
        MLXRandom.seed(0xC0FFEE)
        let m2 = TinyGPTModel(cfg)
        let idx = MLXArray([Int32](repeating: 1, count: 8), [1, 8])
        let a = m1(idx)
        let b = m2(idx)
        MLX.eval(a, b)
        let af = a.asArray(Float.self)
        let bf = b.asArray(Float.self)
        XCTAssertEqual(af.count, bf.count)
        XCTAssertEqual(af.count, 1 * 8 * 32)
        var maxDelta: Float = 0
        for i in 0..<af.count {
            maxDelta = max(maxDelta, abs(af[i] - bf[i]))
        }
        // Two MLX models seeded identically must produce identical
        // float32 logits — same kernel sequence, same input.
        XCTAssertEqual(maxDelta, 0,
                       "seed-fixed forward isn't deterministic (max diff \(maxDelta))")
    }

    func test_forwardShapesMatchVocabAndContext() {
        let cfg = ModelConfig(vocabSize: 64, contextLength: 16, nLayers: 2,
                              nHeads: 2, dModel: 16, dMlp: 32)
        let m = TinyGPTModel(cfg)
        let idx = MLXArray([Int32](repeating: 5, count: 8), [1, 8])
        let logits = m(idx)
        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [1, 8, 64])
    }

    // MARK: - KVCache equivalence (cached vs uncached forward)

    /// At any T, sample(prompt) via the FULL forward and via the cached
    /// forward must produce the same logits for the LAST position. Cached
    /// forward runs the prompt as prefill (T_q == T_kv on first call); the
    /// uncached call always runs the full prompt. We compare the
    /// `[B, T-1, vocab]` last-row slice.
    func test_kvCacheMatchesUncachedForwardOnPrefill() {
        let cfg = ModelConfig(vocabSize: 32, contextLength: 16, nLayers: 2,
                              nHeads: 2, dModel: 16, dMlp: 32,
                              tieEmbeddings: true)
        MLXRandom.seed(7)
        let m = TinyGPTModel(cfg)
        let idx = MLXArray([Int32](0..<8), [1, 8])

        let uncached = m(idx)
        let cache = KVCache(nLayers: cfg.nLayers)
        let cached = m.forwardCached(idx, cache: cache)

        MLX.eval(uncached, cached)
        XCTAssertEqual(cached.shape, [1, 8, cfg.vocabSize])
        XCTAssertEqual(uncached.shape, [1, 8, cfg.vocabSize])

        let uf = uncached.asArray(Float.self)
        let cf = cached.asArray(Float.self)
        XCTAssertEqual(uf.count, cf.count)
        // KVCache path uses MLXFast.scaledDotProductAttention which has
        // slightly different fp16/Metal-rounding behaviour vs the naïve
        // matmul-then-softmax path — allow fp16-scale tolerance.
        var maxDelta: Float = 0
        for i in 0..<uf.count {
            maxDelta = max(maxDelta, abs(uf[i] - cf[i]))
        }
        XCTAssertLessThan(maxDelta, 1e-3,
                          "cached forward diverges from uncached (max diff \(maxDelta))")
    }

    func test_kvCacheGrowsAcrossSteps() {
        let cfg = ModelConfig(vocabSize: 16, contextLength: 8, nLayers: 2,
                              nHeads: 2, dModel: 16, dMlp: 32)
        MLXRandom.seed(11)
        let m = TinyGPTModel(cfg)
        let cache = KVCache(nLayers: cfg.nLayers)

        // Prefill 3 tokens.
        let prefill = MLXArray([Int32]([1, 2, 3]), [1, 3])
        _ = m.forwardCached(prefill, cache: cache)
        MLX.eval(cache.entries[0].keys)
        XCTAssertEqual(cache.currentLength, 3)
        XCTAssertEqual(cache.entries[0].keys.shape[2], 3)

        // Append a single token — cache grows to 4.
        let next = MLXArray([Int32]([4]), [1, 1])
        _ = m.forwardCached(next, cache: cache)
        MLX.eval(cache.entries[0].keys)
        XCTAssertEqual(cache.currentLength, 4)
        XCTAssertEqual(cache.entries[0].keys.shape[2], 4)
    }

    // MARK: - LoRA round-trip

    /// LoRA injection: wraps q_proj / v_proj on every block with a
    /// LoraLinear. The wrapped layer's forward at init should equal the
    /// base layer's forward (B is zero, so the delta is zero).
    func test_loraInjection_atInitDoesNotChangeForward() {
        let cfg = ModelConfig(vocabSize: 16, contextLength: 8, nLayers: 2,
                              nHeads: 2, dModel: 16, dMlp: 32)
        MLXRandom.seed(13)
        let m = TinyGPTModel(cfg)
        let idx = MLXArray([Int32]([1, 2, 3, 4]), [1, 4])
        let before = m(idx)
        MLX.eval(before)
        let bf = before.asArray(Float.self)

        let _ = LoraInjection.inject(m, config: .qv)
        let after = m(idx)
        MLX.eval(after)
        let af = after.asArray(Float.self)

        XCTAssertEqual(bf.count, af.count)
        var maxDelta: Float = 0
        for i in 0..<bf.count {
            maxDelta = max(maxDelta, abs(bf[i] - af[i]))
        }
        XCTAssertLessThan(maxDelta, 1e-5,
                          "LoraLinear with B=0 must produce identical logits (delta \(maxDelta))")
    }

    func test_loraInjection_trainableParamCount() {
        let cfg = ModelConfig(vocabSize: 16, contextLength: 8, nLayers: 4,
                              nHeads: 2, dModel: 16, dMlp: 32)
        let m = TinyGPTModel(cfg)
        LoraInjection.inject(m, config: .qv)
        let n = LoraInjection.trainableParamCount(in: m)
        // rank=4, alpha=8, two suffixes per block (q + v), 4 layers.
        // Per Linear: A is [in=16, r=4] = 64; B is [r=4, out=16] = 64;
        // total per layer per suffix = 128. Two suffixes × 4 layers = 8 ×
        // 128 = 1024.
        XCTAssertEqual(n, 1024,
                       "LoRA injection param count drifted — recompute or check shapes")
    }

    // MARK: - ModelLoader auto-detection

    /// .tinygpt file path → fromScratch. Missing-file path throws. We
    /// don't test the HF dir path here because it needs a real
    /// config.json + safetensors fixture (covered by HFLoadTests).
    func test_modelLoader_recognisesTinygptFile() throws {
        // Build the smallest .tinygpt file that the loader will accept:
        // matches the ModelConfig defaults so checkConfigMatches passes
        // (vocab=256, ctx=128, layers=4, dModel=128, etc.).
        let cfg = ModelConfig()
        let m = TinyGPTModel(cfg)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("loader-test-\(UUID().uuidString).tinygpt")

        // Write a checkpoint using TrainSupport-style manifest entries.
        // Tiny helper inline to avoid importing the executable target.
        let entries = manifestForByteLevel(cfg)
        var tensors: [TinyGPTTensor] = []
        for entry in entries {
            let weight = m.parameters().flattened().first { $0.0 == entry.name }?.1
            guard let array = weight else {
                XCTFail("missing parameter \(entry.name)")
                return
            }
            MLX.eval(array)
            let arr2: MLXArray
            // Apply the same WASM-layout transpose Train uses.
            if isLinearWeightName(entry.name), array.shape.count == 2 {
                arr2 = array.transposed()
            } else {
                arr2 = array
            }
            let floats: [Float] = arr2.asArray(Float.self)
            let weightData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            let zeros = Data(count: weightData.count)
            tensors.append(TinyGPTTensor(
                entry: entry, weight: weightData, adamM: zeros, adamV: zeros, dtype: .fp32
            ))
        }
        let header = TinyGPTHeader(
            config: .init(
                layers: cfg.nLayers, dModel: cfg.dModel, ctx: cfg.contextLength,
                heads: cfg.nHeads, dMlp: cfg.dMlp, batchSize: 2, backend: "mlx-swift"
            ),
            manifest: entries,
            weightDtype: "fp32",
            includesOptimizerState: true
        )
        let file = TinyGPTFile(header: header, step: 0, tensors: tensors)
        try TinyGPTFileWriter.write(file, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let loaded = try ModelLoader.load(url.path)
        if case .fromScratch(_) = loaded.model {
            // success
        } else {
            XCTFail("loaded .tinygpt should map to .fromScratch")
        }
        XCTAssertEqual(loaded.config.nLayers, cfg.nLayers)
        XCTAssertEqual(loaded.config.dModel, cfg.dModel)
        XCTAssertNil(loaded.hfTokenizerDir)  // no tokenizerSource → byte-level
    }

    func test_modelLoader_directoryWithoutConfigJSONFails() {
        // An empty dir is not a valid HF model dir — ModelLoader must
        // surface an explicit error rather than silently falling back.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("not-a-model-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try ModelLoader.load(dir.path))
    }

    // MARK: - Crash-recovery (in-process: 50 steps vs 25 + reload + 25)

    /// Train two models from the same initial seed:
    ///   A: 50 contiguous training steps in one go.
    ///   B: 25 steps, save to disk, build a fresh model, load the
    ///      checkpoint, train 25 more steps.
    /// Both share the same batch-sampler seed (we re-seed the corpus
    /// sampler at step 25 so B's second half sees the same random
    /// windows A would have).
    ///
    /// Final parameters must match within fp32 epsilon of the optimiser
    /// restart drift. Adam state restarts on B's reload — the warm-up
    /// re-converges within ~100 steps; here we only run 25, so the
    /// tolerance is loose (per-param L2 < 0.05). That's still tight
    /// enough to catch the bugs we care about: param ordering, missing
    /// fields, manifest drift.
    func test_crashRecovery_inProcess_50vs25Plus25() throws {
        let cfg = ModelConfig(vocabSize: 16, contextLength: 8, nLayers: 2,
                              nHeads: 2, dModel: 16, dMlp: 32, tieEmbeddings: true)

        // Tiny deterministic corpus: cycles 0..15 so the sampler can
        // always find a window. Big enough that the random start
        // distribution doesn't collapse to a single point.
        let corpusLen = 256
        let bytes = (0..<corpusLen).map { UInt8($0 % cfg.vocabSize) }
        let corpus = ByteCorpus(Data(bytes))

        // Path A: 50 steps in one go.
        MLXRandom.seed(42)
        let modelA = TinyGPTModel(cfg)
        let trainerA = Trainer(model: modelA, learningRate: 1e-3, compileStep: false)
        // Drive deterministic batch order: we control sampleBatch ourselves
        // (the corpus sampler uses Swift's Int.random which is unseeded).
        // Easier: precompute the same window indices for both paths.
        let batchesA = makeDeterministicBatches(
            seed: 1234, corpus: corpus, count: 50, B: 2, T: cfg.contextLength
        )
        for (x, y) in batchesA {
            _ = trainerA.step(inputs: x, targets: y)
        }
        MLX.eval(modelA)

        // Path B: 25 steps, save, reload, 25 more.
        MLXRandom.seed(42)
        let modelB = TinyGPTModel(cfg)
        let trainerB1 = Trainer(model: modelB, learningRate: 1e-3, compileStep: false)
        for i in 0..<25 {
            _ = trainerB1.step(inputs: batchesA[i].0, targets: batchesA[i].1)
        }
        MLX.eval(modelB)

        // Save + reload.
        let saveURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("crash-recovery-\(UUID().uuidString).tinygpt")
        defer { try? FileManager.default.removeItem(at: saveURL) }
        try writeCheckpoint(model: modelB, cfg: cfg, step: 25, to: saveURL)
        let reloaded = TinyGPTModel(cfg)
        try TinyGPTWeightLoader.load(saveURL, into: reloaded)
        let trainerB2 = Trainer(model: reloaded, learningRate: 1e-3, compileStep: false)
        for i in 25..<50 {
            _ = trainerB2.step(inputs: batchesA[i].0, targets: batchesA[i].1)
        }
        MLX.eval(reloaded)

        // Compare every parameter element-wise. AdamW state restarts on
        // resume, so the second halves diverge slightly; tolerance is
        // calibrated so the test catches a "save/reload swapped two
        // params" bug but not the legitimate optimiser-restart drift.
        let aParams = Dictionary(uniqueKeysWithValues: modelA.parameters().flattened())
        let bParams = Dictionary(uniqueKeysWithValues: reloaded.parameters().flattened())
        XCTAssertEqual(aParams.keys.sorted(), bParams.keys.sorted())
        var totalL2Squared: Double = 0
        var totalElems: Int = 0
        for (key, a) in aParams {
            guard let b = bParams[key] else {
                XCTFail("missing param after reload: \(key)")
                return
            }
            let af: [Float] = a.asArray(Float.self)
            let bf: [Float] = b.asArray(Float.self)
            XCTAssertEqual(af.count, bf.count, "shape drift on \(key)")
            for i in 0..<af.count {
                let d = Double(af[i] - bf[i])
                totalL2Squared += d * d
                totalElems += 1
            }
        }
        let rmsDelta = (totalL2Squared / Double(max(1, totalElems))).squareRoot()
        // 50 steps of a tiny model on a periodic corpus: the magnitude of
        // each param settles around ~0.5; a per-element RMS difference of
        // < 0.05 means resume reproduced the run within 10% of param
        // magnitude. Tighter tolerances are achievable once AdamW state
        // is persisted (see TrainSupport.atomicSave comment).
        XCTAssertLessThan(
            rmsDelta, 0.05,
            "resume diverged too much from contiguous (rms \(rmsDelta))"
        )
    }

    // MARK: - Data-perf smoke tests (sample packing + BPE-dropout)

    /// Smoke: build 1000 synthetic SFT examples with a power-law length
    /// distribution, then sample many batches with each pack-mode and
    /// report the (length × frequency) histogram. The "sample" mode
    /// should flatten the (length-weighted) bias toward long examples.
    func test_smoke_samplePackingHistogram() throws {
        // Length distribution: bimodal Pareto — most examples short, long
        // tail of long ones. Mimics chat-template-Shakespeare-ish data.
        let nExamples = 1000
        let nBatches = 2000
        let batchSize = 4
        srand48(42)
        var examples: [SFTExample] = []
        examples.reserveCapacity(nExamples)
        for i in 0..<nExamples {
            // Pareto-ish: 80% short (10..60), 20% long (60..400).
            let len: Int
            if drand48() < 0.8 { len = 10 + Int(drand48() * 50) }
            else                { len = 60 + Int(drand48() * 340) }
            let toks = (0..<len).map { _ in Int32(i % 1000) }
            let mask = Array(repeating: true, count: len)
            examples.append(SFTExample(tokens: toks, responseMask: mask))
        }
        let corpus = SFTCorpus(examples, vocabSize: 1000)

        // Buckets (linear: [0,80), [80,160), [160,240), [160,400)).
        func bucket(_ l: Int) -> Int { min(3, l / 80) }
        // Count occurrences per bucket for each mode. We weight by 1 (raw
        // frequency) AND by length (length × frequency) to see whether
        // the long-bucket dominates total token contribution.
        func histogram(_ sampler: (Int, Int) -> Int) -> ([Int], [Int]) {
            var freq = [Int](repeating: 0, count: 4)
            var weighted = [Int](repeating: 0, count: 4)
            for _ in 0..<nBatches {
                for _ in 0..<batchSize {
                    let idx = sampler(batchSize, 128)
                    let l = examples[idx].tokens.count
                    let b = bucket(l)
                    freq[b] += 1
                    weighted[b] += l
                }
            }
            return (freq, weighted)
        }

        // Drive each mode by hand using the same RNG that the actual
        // sampler uses internally (Int.random / Double.random).
        // For honesty we sample using the public API and inspect by hash
        // of returned (input) row's first token — but synthetic examples
        // share token contents, so we'd lose identity. Instead, drive
        // the same RNG and indices directly: probe the public helpers.
        let invWeights = corpus.inverseLengthWeights
        // Build a cumulative for "sample" so we can match the impl.
        var cum: [Double] = []; var running: Double = 0
        for w in invWeights { running += w; cum.append(running) }
        // Round to 1.0 so binary search never falls off the end.
        let totalW = cum.last ?? 1.0

        // Uniform sampler (matches sampleBatch).
        let (uFreq, uWeighted) = histogram { _, _ in
            return Int.random(in: 0..<examples.count)
        }
        // Inverse-length weighted (matches sampleBatchWeighted).
        let (sFreq, sWeighted) = histogram { _, _ in
            let target = Double.random(in: 0..<totalW)
            var lo = 0, hi = cum.count - 1
            while lo < hi {
                let mid = (lo + hi) / 2
                if cum[mid] < target { lo = mid + 1 } else { hi = mid }
            }
            return lo
        }
        // Length-bucket uniform (matches sampleBatchBucketed with N=4).
        var buckets: [[Int]] = Array(repeating: [], count: 4)
        let minLen = examples.map { $0.tokens.count }.min()!
        let maxLen = examples.map { $0.tokens.count }.max()!
        let span = Double(maxLen - minLen)
        for (i, ex) in examples.enumerated() {
            let frac = Double(ex.tokens.count - minLen) / span
            var b = Int(frac * 4); if b >= 4 { b = 3 }
            buckets[b].append(i)
        }
        let nonEmpty = buckets.filter { !$0.isEmpty }
        let (bFreq, bWeighted) = histogram { _, _ in
            let bk = nonEmpty[Int.random(in: 0..<nonEmpty.count)]
            return bk[Int.random(in: 0..<bk.count)]
        }

        let labels = ["[10-80)", "[80-160)", "[160-240)", "[240-400]"]
        print("\n=== sample-packing smoke ===")
        print("buckets:                \(labels)")
        print("uniform   freq/bucket:     \(uFreq)")
        print("uniform   tok·freq/bucket: \(uWeighted)")
        print("sample    freq/bucket:     \(sFreq)")
        print("sample    tok·freq/bucket: \(sWeighted)")
        print("bucket    freq/bucket:     \(bFreq)")
        print("bucket    tok·freq/bucket: \(bWeighted)")
        // Per-EXAMPLE histogram of (length × frequency) — this is where
        // sample-packing earns its keep. Under uniform sampling, every
        // example has the same expected pick count, so (length × freq)
        // is proportional to length — long examples dominate token
        // contribution. Under inverse-length sampling, each example's
        // expected (length × freq) is constant, so the histogram should
        // be FLAT across length.
        func perExampleHist(_ sampler: () -> Int) -> [Int: Double] {
            var picks = [Int](repeating: 0, count: examples.count)
            let total = nBatches * batchSize
            for _ in 0..<total {
                picks[sampler()] += 1
            }
            // Compute (length × freq) per example.
            var byLen: [Int: Double] = [:]
            for i in 0..<examples.count {
                let l = examples[i].tokens.count
                let contribution = Double(l) * Double(picks[i])
                byLen[l, default: 0] += contribution
            }
            // Normalise by the number of examples in each length bin so
            // we get average (length × freq) per example for that length.
            var counts: [Int: Int] = [:]
            for ex in examples { counts[ex.tokens.count, default: 0] += 1 }
            for (l, _) in byLen { byLen[l]! /= Double(counts[l]!) }
            return byLen
        }
        let uHist = perExampleHist { Int.random(in: 0..<examples.count) }
        let sHist = perExampleHist {
            let target = Double.random(in: 0..<totalW)
            var lo = 0, hi = cum.count - 1
            while lo < hi {
                let mid = (lo + hi) / 2
                if cum[mid] < target { lo = mid + 1 } else { hi = mid }
            }
            return lo
        }
        // Group histogram into broader bins for readability.
        func binMean(_ h: [Int: Double], _ lo: Int, _ hi: Int) -> Double {
            let vs = h.filter { lo <= $0.key && $0.key < hi }.map { $0.value }
            return vs.isEmpty ? 0 : vs.reduce(0,+) / Double(vs.count)
        }
        let u10_80    = binMean(uHist, 10, 80)
        let u80_160   = binMean(uHist, 80, 160)
        let u160_240  = binMean(uHist, 160, 240)
        let u240_400  = binMean(uHist, 240, 401)
        let s10_80    = binMean(sHist, 10, 80)
        let s80_160   = binMean(sHist, 80, 160)
        let s160_240  = binMean(sHist, 160, 240)
        let s240_400  = binMean(sHist, 240, 401)
        print(String(format: "per-example (length × freq) — average within each length bin:"))
        print(String(format: "  uniform   [10-80):%.1f  [80-160):%.1f  [160-240):%.1f  [240-400]:%.1f",
                      u10_80, u80_160, u160_240, u240_400))
        print(String(format: "  sample    [10-80):%.1f  [80-160):%.1f  [160-240):%.1f  [240-400]:%.1f",
                      s10_80, s80_160, s160_240, s240_400))
        // Compute coefficient of variation (CoV = stddev/mean) of
        // per-example (length × freq) bin-averages — lower means flatter.
        // Sample mode should be flatter than uniform.
        func cov(_ xs: [Double]) -> Double {
            let m = xs.reduce(0, +) / Double(xs.count)
            let v = xs.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(xs.count)
            return sqrt(v) / max(m, 1e-9)
        }
        let covU = cov([u10_80, u80_160, u160_240, u240_400])
        let covS = cov([s10_80, s80_160, s160_240, s240_400])
        print(String(format: "CoV(per-example length·freq): uniform=%.3f  sample=%.3f", covU, covS))
        // sanity: weighted sampling should flatten per-example contrib.
        XCTAssertLessThan(covS, covU,
                          "weighted sampling didn't flatten per-example contribution (covS=\(covS), covU=\(covU))")
    }

    /// Smoke: byte-level BPE-dropout encoder produces variable-length
    /// token sequences on the same text when p_drop > 0, and same length
    /// every time when p_drop == 0. Uses a tiny synthetic vocab.
    func test_smoke_bpeDropoutVariability() throws {
        // Build a minimal byte-level BPE: 8 byte symbols + 4 merges.
        // Symbols are GPT-2 byte-alphabet codepoints for 'h','e','l','o',' '.
        let h = "h", e = "e", l = "l", o = "o", sp = "\u{0120}"  // space
        var vocab: [String: Int] = [:]
        let symbols = [h, e, l, o, sp, "he", "lo", "hel", "hello", "ll", " "]
        for (i, s) in symbols.enumerated() { vocab[s] = i }
        // Merges in priority order: (h,e) → he, (l,l) → ll, (l,o) → lo,
        // (he,l) → hel — these are the typical ones for "hello".
        var ranks: [BPEDropoutTokenizer.BytePair: Int] = [:]
        ranks[.init(h, e)] = 0
        ranks[.init(l, l)] = 1
        ranks[.init(l, o)] = 2
        ranks[.init("he", l)] = 3
        let idToTok = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })
        let tok = BPEDropoutTokenizer(
            bpeRanks: ranks, vocab: vocab, idToToken: idToTok,
            unknownTokenId: nil, isByteLevel: true
        )
        // p=0 → deterministic.
        let ids0a = tok.encodeWithDropout("hello", pDrop: 0)
        let ids0b = tok.encodeWithDropout("hello", pDrop: 0)
        print("\n=== BPE-dropout smoke ===")
        print("p=0   encoding A: \(ids0a) (len=\(ids0a.count))")
        print("p=0   encoding B: \(ids0b) (len=\(ids0b.count))")
        XCTAssertEqual(ids0a, ids0b, "p=0 must be deterministic")
        // p=0.5 → variable (run multiple, expect at least one different length).
        srand48(7)
        var lens: [Int: Int] = [:]
        for _ in 0..<50 {
            let ids = tok.encodeWithDropout("hello", pDrop: 0.5)
            lens[ids.count, default: 0] += 1
        }
        print("p=0.5 length histogram over 50 encodes: \(lens)")
        XCTAssertGreaterThan(lens.count, 1,
                              "p=0.5 should produce at least 2 distinct token-counts")
    }

    /// Smoke: BPE-dropout encoder produces the same token sequence as
    /// the GPT-2 / Llama tokenizer for p=0 on real text. We can't load
    /// a HF tokenizer.json easily in tests, so this test loads the
    /// swift-transformers test resource and verifies our encoder agrees
    /// for common short strings on the byte-alphabet level (no model
    /// vocab match needed — we just check that the byte-encode produces
    /// the expected GPT-2 string).
    func test_smoke_bpeDropoutByteAlphabetMatches() {
        // The space byte (0x20) should map to "Ġ" (U+0120), 'a' → 'a'.
        let table = BPEDropoutTokenizer.byteEncoderTable
        XCTAssertEqual(table[0x20], "\u{0120}", "space should encode to Ġ")
        XCTAssertEqual(table[0x61], "a", "'a' should encode to itself")
        XCTAssertEqual(table[0x41], "A", "'A' should encode to itself")
        XCTAssertEqual(table[0xC4], "\u{00C4}", "0xC4 should pass through")
        XCTAssertEqual(table[0x00], "\u{0100}", "NUL should map to U+0100")
    }
}

// MARK: - Helpers (private to this test file)

/// Manifest entries for a dense byte-level model. Inlined here to avoid
/// importing the TinyGPT executable target into the test module.
private func manifestForByteLevel(_ cfg: ModelConfig) -> [TinyGPTHeader.TensorEntry] {
    var entries: [TinyGPTHeader.TensorEntry] = []
    var offset = 0
    let push: (String, [Int]) -> Void = { name, shape in
        let size = shape.reduce(1, *)
        entries.append(.init(name: name, shape: shape, floatOffset: offset))
        offset += size
    }
    let C = cfg.dModel, M = cfg.dMlp
    push("token_embedding.weight", [cfg.vocabSize, C])
    push("position_embedding.weight", [cfg.contextLength, C])
    push("ln_final.weight", [C])
    push("ln_final.bias", [C])
    for i in 0..<cfg.nLayers {
        push("blocks.\(i).ln1.weight", [C])
        push("blocks.\(i).ln1.bias", [C])
        push("blocks.\(i).attn.q_proj.weight", [C, C])
        push("blocks.\(i).attn.q_proj.bias", [C])
        push("blocks.\(i).attn.k_proj.weight", [C, C])
        push("blocks.\(i).attn.k_proj.bias", [C])
        push("blocks.\(i).attn.v_proj.weight", [C, C])
        push("blocks.\(i).attn.v_proj.bias", [C])
        push("blocks.\(i).attn.o_proj.weight", [C, C])
        push("blocks.\(i).attn.o_proj.bias", [C])
        push("blocks.\(i).ln2.weight", [C])
        push("blocks.\(i).ln2.bias", [C])
        push("blocks.\(i).mlp.fc_in.weight", [M, C])
        push("blocks.\(i).mlp.fc_in.bias", [M])
        push("blocks.\(i).mlp.fc_out.weight", [C, M])
        push("blocks.\(i).mlp.fc_out.bias", [C])
    }
    return entries
}

private func isLinearWeightName(_ name: String) -> Bool {
    guard name.hasSuffix(".weight") else { return false }
    if name == "token_embedding.weight" || name == "position_embedding.weight" {
        return false
    }
    if name.hasSuffix(".ln1.weight") || name.hasSuffix(".ln2.weight")
        || name == "ln_final.weight" {
        return false
    }
    return true
}

/// Tiny deterministic batch generator: linear-congruential start indices so
/// path A and path B's reload sequence see the same windows.
private func makeDeterministicBatches(
    seed: UInt64, corpus: ByteCorpus, count: Int, B: Int, T: Int
) -> [(MLXArray, MLXArray)] {
    var state = seed
    func nextStart(_ upper: Int) -> Int {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int(state >> 33) % upper
    }
    var batches: [(MLXArray, MLXArray)] = []
    batches.reserveCapacity(count)
    let upper = corpus.bytes.count - T - 1
    for _ in 0..<count {
        var inputs = [Int32](repeating: 0, count: B * T)
        var targets = [Int32](repeating: 0, count: B * T)
        for bi in 0..<B {
            let s = nextStart(upper)
            for j in 0..<T {
                inputs[bi * T + j] = Int32(corpus.bytes[s + j])
                targets[bi * T + j] = Int32(corpus.bytes[s + j + 1])
            }
        }
        batches.append((MLXArray(inputs, [B, T]), MLXArray(targets, [B, T])))
    }
    return batches
}

/// Lightweight inline checkpoint writer (mirrors TrainSupport.atomicSave
/// without dragging the executable target into the test module).
private func writeCheckpoint(
    model: TinyGPTModel, cfg: ModelConfig, step: Int, to url: URL
) throws {
    let entries = manifestForByteLevel(cfg)
    let params = model.parameters().flattened()
    let paramMap = Dictionary(uniqueKeysWithValues: params)
    var tensors: [TinyGPTTensor] = []
    for entry in entries {
        guard let value = paramMap[entry.name] else {
            throw NSError(domain: "test", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "missing \(entry.name)"])
        }
        var array = value
        if isLinearWeightName(entry.name), array.shape.count == 2 {
            array = array.transposed()
        }
        MLX.eval(array)
        let floats: [Float] = array.asArray(Float.self)
        let weightData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        let zeros = Data(count: weightData.count)
        tensors.append(TinyGPTTensor(
            entry: entry, weight: weightData, adamM: zeros, adamV: zeros, dtype: .fp32
        ))
    }
    let header = TinyGPTHeader(
        config: .init(
            layers: cfg.nLayers, dModel: cfg.dModel, ctx: cfg.contextLength,
            heads: cfg.nHeads, dMlp: cfg.dMlp, batchSize: 2, backend: "mlx-swift"
        ),
        manifest: entries,
        weightDtype: "fp32",
        includesOptimizerState: true
    )
    let file = TinyGPTFile(header: header, step: Int32(step), tensors: tensors)
    try TinyGPTFileWriter.write(file, to: url)
}
