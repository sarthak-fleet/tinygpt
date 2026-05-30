import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Byte-level corpus loader. Materialises the whole corpus into memory as
/// a single `[UInt8]` and serves random `(B, T+1)` windows so each step
/// trains on a different chunk. Matches the browser's CPU-side sampler.
public final class ByteCorpus: Sendable {
    public let bytes: [UInt8]

    public init(_ data: Data) {
        self.bytes = Array(data)
    }

    public convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        self.init(data)
    }

    /// Sample a batch: `(input [B, T] int32, target [B, T] int32)`.
    /// `target = input shifted by 1` so the model predicts next-byte.
    public func sampleBatch(batchSize B: Int, contextLength T: Int) -> (MLXArray, MLXArray) {
        let (inputs, targets) = sampleBatchRaw(batchSize: B, contextLength: T)
        return (MLXArray(inputs, [B, T]), MLXArray(targets, [B, T]))
    }

    /// Generate the raw Int32 windows without materialising MLXArrays. Used
    /// by the prefetcher so the CPU-side sampling runs concurrently with
    /// the GPU's previous-step compute.
    public func sampleBatchRaw(batchSize B: Int, contextLength T: Int) -> ([Int32], [Int32]) {
        precondition(bytes.count > T + 1, "corpus too small for context \(T)")
        var inputs = [Int32](repeating: 0, count: B * T)
        var targets = [Int32](repeating: 0, count: B * T)
        for i in 0..<B {
            let start = Int.random(in: 0..<(bytes.count - T - 1))
            for j in 0..<T {
                inputs[i * T + j] = Int32(bytes[start + j])
                targets[i * T + j] = Int32(bytes[start + j + 1])
            }
        }
        return (inputs, targets)
    }
}

/// Token-id corpus loader. Same `sampleBatch` interface as `ByteCorpus`,
/// but the underlying buffer is already-tokenised `Int32` ids — used when
/// fine-tuning an HF model whose embedding table is BPE-indexed (vocab in
/// the tens of thousands), so feeding raw bytes would index into a tiny
/// slice of the vocab and train the LoRA against a wrong distribution.
///
/// Callers build it from any tokenizer (typically `HFTokenizer.encode`)
/// — the corpus doesn't care which scheme produced the ids.
public final class TokenizedCorpus: Sendable {
    public let tokens: [Int32]
    public let vocabSize: Int

    public init(tokens: [Int32], vocabSize: Int) {
        self.tokens = tokens
        self.vocabSize = vocabSize
    }

    /// Sample a batch: `(input [B, T] int32, target [B, T] int32)`.
    public func sampleBatch(batchSize B: Int, contextLength T: Int) -> (MLXArray, MLXArray) {
        let (inputs, targets) = sampleBatchRaw(batchSize: B, contextLength: T)
        return (MLXArray(inputs, [B, T]), MLXArray(targets, [B, T]))
    }

    public func sampleBatchRaw(batchSize B: Int, contextLength T: Int) -> ([Int32], [Int32]) {
        precondition(tokens.count > T + 1, "tokenized corpus too small for context \(T) (got \(tokens.count) tokens)")
        var inputs = [Int32](repeating: 0, count: B * T)
        var targets = [Int32](repeating: 0, count: B * T)
        for i in 0..<B {
            let start = Int.random(in: 0..<(tokens.count - T - 1))
            for j in 0..<T {
                inputs[i * T + j] = tokens[start + j]
                targets[i * T + j] = tokens[start + j + 1]
            }
        }
        return (inputs, targets)
    }

    /// Hold out the last `valSplit` fraction as a validation set. Same
    /// semantics as `TrainSupport.splitCorpus` for byte corpora.
    public func split(valSplit: Double) -> (train: TokenizedCorpus, val: TokenizedCorpus?) {
        guard valSplit > 0, valSplit < 0.5 else { return (self, nil) }
        let total = tokens.count
        let valCount = max(1, Int(Double(total) * valSplit))
        let trainEnd = total - valCount
        let train = TokenizedCorpus(tokens: Array(tokens[0..<trainEnd]), vocabSize: vocabSize)
        let val = TokenizedCorpus(tokens: Array(tokens[trainEnd..<total]), vocabSize: vocabSize)
        return (train, val)
    }
}

/// Streaming-tokenization corpus for BPE-dropout training. Stores the
/// source text plus a `BPEDropoutTokenizer`, and re-tokenises a random
/// text window for each batch. Output shape matches `TokenizedCorpus.
/// sampleBatch`, so callers swap in transparently.
///
/// Why "streaming"? With BPE-dropout enabled (`pDrop > 0`), the same
/// character window yields a slightly different token sequence on each
/// re-encode. Caching tokens up-front and slicing — the path the static
/// `TokenizedCorpus` takes — would defeat the regulariser. The cost is
/// ~5-15× slower batch construction; for from-scratch transformer
/// training the GPU step still dominates, but on huge batches you'll
/// notice it.
///
/// When pDrop == 0 this still re-tokenises each batch and is therefore
/// strictly slower than the cached path — use it only when dropout is on.
public final class StreamingTokenizedCorpus: @unchecked Sendable {
    public let text: String
    public let pDrop: Float
    public let vocabSize: Int
    public let tokenizer: BPEDropoutTokenizer
    /// Cumulative-byte index over the text, scanned via UTF-8 to a String
    /// for safe substring indexing. Stored as UTF-8 view so random window
    /// starts don't crash on multi-byte boundaries.
    private let utf8Bytes: [UInt8]

    public init(text: String, tokenizer: BPEDropoutTokenizer,
                 vocabSize: Int, pDrop: Float)
    {
        self.text = text
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.pDrop = pDrop
        self.utf8Bytes = [UInt8](text.utf8)
    }

    /// Pull a random text window, tokenise it with dropout, take the
    /// first T+1 tokens. If a window happens to produce fewer than T+1
    /// tokens (rare; pathological strings of long-merge runs), re-sample.
    /// Retries are capped to avoid pathological loops on tiny corpora.
    public func sampleBatch(batchSize B: Int, contextLength T: Int) -> (MLXArray, MLXArray) {
        let (inputs, targets) = sampleBatchRaw(batchSize: B, contextLength: T)
        return (MLXArray(inputs, [B, T]), MLXArray(targets, [B, T]))
    }

    public func sampleBatchRaw(batchSize B: Int, contextLength T: Int) -> ([Int32], [Int32]) {
        var inputs = [Int32](repeating: 0, count: B * T)
        var targets = [Int32](repeating: 0, count: B * T)
        // Heuristic: assume ~3 bytes per token on average for English BPE
        // — over-shoot by 2× so retries are rare. T+1 tokens need ~3(T+1)
        // bytes; we'll grab 8(T+1) to be safe.
        let windowBytes = max(64, 8 * (T + 1))
        let totalBytes = utf8Bytes.count
        precondition(totalBytes > windowBytes, "corpus too small for streaming dropout (need \(windowBytes) bytes, have \(totalBytes))")
        for i in 0..<B {
            var attempt = 0
            while attempt < 4 {
                let start = Int.random(in: 0..<(totalBytes - windowBytes))
                // Snap to a valid UTF-8 boundary by stepping forward until
                // we find a non-continuation byte (top two bits != 0b10).
                var s = start
                while s < totalBytes && (utf8Bytes[s] & 0xC0) == 0x80 { s += 1 }
                let e = min(s + windowBytes, totalBytes)
                let slice = utf8Bytes[s..<e]
                guard let chunk = String(bytes: slice, encoding: .utf8) else {
                    attempt += 1; continue
                }
                let ids = tokenizer.encodeWithDropout(chunk, pDrop: pDrop)
                if ids.count < T + 1 { attempt += 1; continue }
                for j in 0..<T {
                    inputs[i * T + j] = Int32(ids[j])
                    targets[i * T + j] = Int32(ids[j + 1])
                }
                break
            }
            // If we never got a full window, leave the row zeroed — loss
            // contribution is unavoidable but cheap and rare.
        }
        return (inputs, targets)
    }

    /// Hold out the last `valSplit` fraction as a validation set —
    /// matches `TokenizedCorpus.split`. The validation corpus is wrapped
    /// in a `TokenizedCorpus` (no dropout) for deterministic val loss.
    public func split(valSplit: Double) -> (train: StreamingTokenizedCorpus, val: TokenizedCorpus?) {
        guard valSplit > 0, valSplit < 0.5 else { return (self, nil) }
        let total = utf8Bytes.count
        let valCount = max(1, Int(Double(total) * valSplit))
        let trainEnd = total - valCount
        // Snap split point to UTF-8 boundary.
        var split = trainEnd
        while split < total && (utf8Bytes[split] & 0xC0) == 0x80 { split += 1 }
        let trainText = String(bytes: utf8Bytes[0..<split], encoding: .utf8) ?? text
        let valText = String(bytes: utf8Bytes[split..<total], encoding: .utf8) ?? ""
        let train = StreamingTokenizedCorpus(text: trainText, tokenizer: tokenizer,
                                              vocabSize: vocabSize, pDrop: pDrop)
        // Val tokens encoded once, no dropout, then frozen.
        let valIds = tokenizer.encodeWithDropout(valText, pDrop: 0).map { Int32($0) }
        let val = valIds.count > 2
            ? TokenizedCorpus(tokens: valIds, vocabSize: vocabSize)
            : nil
        return (train, val)
    }
}

/// Async batch prefetcher — pipelines CPU-side batch construction with the
/// previous step's GPU compute. Maintains one pre-built batch ahead.
public actor BatchPrefetcher {
    private let corpus: ByteCorpus
    private let batchSize: Int
    private let contextLength: Int

    public init(corpus: ByteCorpus, batchSize: Int, contextLength: Int) {
        self.corpus = corpus
        self.batchSize = batchSize
        self.contextLength = contextLength
    }

    public func next() -> ([Int32], [Int32]) {
        corpus.sampleBatchRaw(batchSize: batchSize, contextLength: contextLength)
    }
}

/// Bounded-queue batch pipeline. Spins one producer thread that calls the
/// supplied sampler closure to build the next batch (MLXArray construction
/// included) while the main training thread is running forward/backward
/// on the previous batch. Up to `capacity` batches sit in flight.
///
/// This is the live re-wiring of the prefetch idea documented around
/// `Train.swift` — see the comment block above the `sampleTrainBatch`
/// call site. The previous implementation was dropped on the assumption
/// that MLXArray construction blocks the same thread the GPU dispatch
/// runs on. Empirically that's true for *eager* eval, but the train
/// step lazily builds a graph and only calls `eval` at the end — so the
/// pipeline DOES overlap useful CPU work (random sampling + Int32 fill
/// + MLXArray buffer copy) with the kernel-launch tail of the previous
/// step.
///
/// Thread safety: a single producer thread fills the queue under
/// `lock`; the consumer (training loop) drains under the same lock.
/// Two semaphores form a bounded-buffer producer/consumer pair so the
/// producer blocks when the queue is full and the consumer blocks when
/// it's empty.
public final class BatchPipeline: @unchecked Sendable {
    public typealias Sampler = (Int, Int) -> (MLXArray, MLXArray)

    private let sampler: Sampler
    private let batchSize: Int
    private let contextLength: Int

    private let queueLock = NSLock()
    private var queue: [(MLXArray, MLXArray)] = []
    private let slotsFree: DispatchSemaphore
    private let slotsFilled: DispatchSemaphore
    private var stopped: Bool = false
    private let producer: DispatchQueue

    public init(sampler: @escaping Sampler, batchSize: Int, contextLength: Int,
                capacity: Int = 2) {
        precondition(capacity >= 1, "capacity must be ≥ 1")
        self.sampler = sampler
        self.batchSize = batchSize
        self.contextLength = contextLength
        self.slotsFree = DispatchSemaphore(value: capacity)
        self.slotsFilled = DispatchSemaphore(value: 0)
        // Producer runs at `.utility` so it doesn't fight the training
        // thread for the P-core. Background CPU work; the consumer
        // dictates the cadence anyway.
        self.producer = DispatchQueue(
            label: "tinygpt.trainer.batch-pipeline",
            qos: .utility
        )
        producer.async { [weak self] in self?.produceLoop() }
    }

    private func produceLoop() {
        while true {
            slotsFree.wait()
            // Check stop AFTER acquiring a slot — covers the `stop()`
            // call racing with us between iterations.
            queueLock.lock()
            let stop = stopped
            queueLock.unlock()
            if stop { return }
            let (x, y) = sampler(batchSize, contextLength)
            queueLock.lock()
            queue.append((x, y))
            queueLock.unlock()
            slotsFilled.signal()
        }
    }

    /// Pop the next batch. Blocks if none is ready.
    public func next() -> (MLXArray, MLXArray) {
        slotsFilled.wait()
        queueLock.lock()
        let batch = queue.removeFirst()
        queueLock.unlock()
        slotsFree.signal()
        return batch
    }

    /// Stop the producer thread synchronously. Drains any in-flight
    /// queued batches BEFORE returning so the consumer is the last
    /// thread that touches the MLXArrays — important because MLXArrays
    /// constructed on a background thread will be released on whichever
    /// thread drops the last reference. Mixing thread origins can
    /// confuse MLX's evaluator at shutdown (without this drain we saw
    /// SIGTRAP on macOS 26 / mlx-swift 0.25 between the last training
    /// step and the post-run banner).
    public func stop() {
        queueLock.lock()
        if stopped { queueLock.unlock(); return }
        stopped = true
        queueLock.unlock()
        // Wake the producer if it's blocked on a full queue.
        slotsFree.signal()
        // Drain any queued batches on the consumer thread. Each pop
        // signals `slotsFree` which lets the producer make forward
        // progress (it'll observe `stopped` and exit).
        while true {
            if slotsFilled.wait(timeout: .now() + .milliseconds(50)) == .timedOut {
                break
            }
            queueLock.lock()
            if !queue.isEmpty { _ = queue.removeFirst() }
            queueLock.unlock()
            slotsFree.signal()
        }
    }
}

/// Global L2-norm gradient clipping. Computes `‖g‖₂` across every
/// parameter, then uniformly scales each leaf by `min(1, maxNorm / ‖g‖₂)`.
/// Standard transformer-LM training stability lever — without it, the
/// occasional spike (early steps, rare token in a long sequence) can
/// blow up bf16 weights past the point the optimiser recovers from.
///
/// All ops are MLX ops, so this composes cleanly inside `compile`.
public func clipGradNorm(_ grads: ModuleParameters, maxNorm: Float) -> ModuleParameters {
    var sumSq = MLXArray(Float(0))
    for (_, g) in grads.flattened() {
        sumSq = sumSq + (g * g).sum()
    }
    let norm = MLX.sqrt(sumSq)
    // scale ≤ 1 ALWAYS (we never amplify) — `minimum(1, ratio)`.
    let scale = MLX.minimum(MLXArray(Float(1)),
                            MLXArray(maxNorm) / (norm + MLXArray(Float(1e-6))))
    return grads.mapValues { g in g * scale }
}

/// AdamW + value-and-grad train loop. One `step()` call does a full
/// forward + backward + optimiser update and returns the scalar loss.
///
/// Supports gradient accumulation via `accumulatedStep(microBatches:)` —
/// runs N micro-batches, sums gradients element-wise, divides by N, and
/// applies one optimizer update. Useful when the effective batch you
/// want exceeds memory: ctx=1024 × B=8 might OOM, but ctx=1024 × B=2
/// repeated 4× gives the same effective batch with ¼ the memory cost.
public final class Trainer {
    public let model: TinyGPTModel
    /// Generic optimiser handle — any of AdamW, Lion, Sophia, Muon, or
    /// Adafactor (the latter wrapped in `AdafactorAdapter`). Exposed
    /// through the `Optimizer & LearningRateMutable` composition so
    /// the schedule code can both step it and adjust its learning rate.
    public let optimizer: any Optimizer & LearningRateMutable
    /// Which optimiser kind backs `optimizer`, for diagnostics + the
    /// per-step LR-scheduler bookkeeping. Defaults to .adamw when the
    /// older AdamW-only initializer path is used.
    public let optimizerKind: OptimizerKind
    public private(set) var stepCount: Int = 0
    /// L2 norm cap for gradient clipping. `nil` = off; `1.0` is the
    /// transformer-LM default.
    public let gradClipNorm: Float?

    /// GaLore manager — `nil` when GaLore is disabled (the common case
    /// today). When non-nil, every step projects 2-D weight gradients
    /// through a rank-R basis before the optimiser sees them.
    /// See `GaLore.swift` for the details.
    public let galore: GaLoreManager?

    /// Layer-wise LR decay factor (default 1.0 = no decay). When < 1,
    /// each block's gradient is multiplied by `factor^(L - 1 - i)` so
    /// deeper layers get the full LR. Cheap — one MLX scalar multiply
    /// per leaf.
    public let lrLayerDecay: Float

    /// Compiled (graph-traced) train step. MLX-Swift's `compile` traces the
    /// step the first time it's called and reuses the kernel-launch sequence
    /// thereafter — the single biggest win over an interpreted train loop.
    private let trainStepFn: (MLXArray, MLXArray) -> MLXArray
    private let gradFn: (TinyGPTModel, MLXArray, MLXArray) -> (MLXArray, ModuleParameters)
    private let useCompile: Bool

    /// Compiled accumulation step — set when `init(..., accumMicroBatches:)` is
    /// non-nil. When present, it traces a single graph that loops N times
    /// inside the trace; the Swift caller batches the micro-batches into a
    /// flat array and dispatches them as one MLX kernel sequence. See
    /// ``accumulatedStepCompiled``.
    ///
    /// `nil` on the legacy (uncompiled) accumulation path. The shape of the
    /// compiled trace depends on N, so the trainer must be re-built if the
    /// caller wants a different N — but in practice `--accum N` is set
    /// once per run.
    private let accumStepFn: (([MLXArray]) -> [MLXArray])?
    /// N for `accumStepFn`. Fixed at trainer-init time so the trace closes
    /// cleanly (`compile` doesn't trace a Swift `for` loop with a runtime
    /// upper bound).
    public let compiledAccumN: Int?

    public init(
        model: TinyGPTModel,
        learningRate: Float = 3e-4,
        weightDecay: Float = 0.1,
        betas: (Float, Float) = (0.9, 0.95),
        eps: Float = 1e-8,
        compileStep: Bool = true,
        gradClipNorm: Float? = nil,
        optimizer optimizerKind: OptimizerKind = .adamw,
        galore: GaLoreManager? = nil,
        lrLayerDecay: Float = 1.0,
        /// When non-nil, build a `CompiledAdamW`-backed step where LR can
        /// change between calls without re-tracing. Today only the AdamW
        /// optimiser kind goes down this path; the call site is responsible
        /// for falling back to the standard `Trainer` when the user picks a
        /// non-AdamW optimiser. Passing `true` with a non-AdamW kind is a
        /// programmer error and traps at init.
        useCompiledLR: Bool = false,
        /// When non-nil, fold gradient accumulation into a single compiled
        /// trace of `accumMicroBatches` micro-batches. N is fixed-at-compile
        /// so the graph shape is stable. The caller's `accumulatedStep`
        /// must pass exactly N micro-batches — guarded by precondition.
        accumMicroBatches: Int? = nil
    ) {
        self.model = model
        self.useCompile = compileStep
        self.gradClipNorm = gradClipNorm
        self.optimizerKind = optimizerKind
        self.galore = galore
        self.lrLayerDecay = lrLayerDecay
        // Compile-with-mutable-LR path: build CompiledAdamW directly so
        // the LR lives as an MLXArray in the optimiser's `innerState()`.
        // Once captured by `compile(inputs: [opt], ...)`, mutating the LR
        // is just an `_updateInternal` away — no re-trace.
        //
        // We only support AdamW down this path; falling back to the
        // standard factory for everything else preserves Lion/Sophia/Muon/
        // Adafactor exactly as before (those flow through the standard
        // uncompiled scheduler path in Train.swift).
        if useCompiledLR {
            precondition(optimizerKind == .adamw,
                         "useCompiledLR currently supports only --optimizer adamw (got \(optimizerKind.rawValue))")
            self.optimizer = CompiledAdamW(
                learningRate: learningRate,
                betas: betas,
                eps: eps,
                weightDecay: weightDecay
            )
        } else {
            self.optimizer = makeOptimizer(
                kind: optimizerKind,
                learningRate: learningRate,
                weightDecay: weightDecay,
                betas: betas,
                eps: eps
            )
        }
        // value_and_grad of the loss function, captured to apply via optimizer.
        // The closure captures `model` by reference — MLX's autograd
        // discovers parameters through `Module.trainableParameters()`.
        let lossFn = { (m: TinyGPTModel, x: MLXArray, y: MLXArray) -> MLXArray in
            m.loss(x, y)
        }
        let gradFn = valueAndGrad(model: model, lossFn)
        self.gradFn = gradFn
        let optimizer = self.optimizer
        let m = model
        let clip = gradClipNorm
        let layerDecay = lrLayerDecay
        let nLayers = model.config.nLayers
        let galoreMgr = galore

        // GaLore mutates projector state out-of-graph, so it MUST live on
        // the uncompiled path. Layer-wise LR decay is graph-pure (just a
        // scalar multiply per leaf) and stays compile-safe.
        //
        // The single-step compile lives here (whether or not the LR is
        // mutable). For `useCompiledLR == true`, the optimiser's LR
        // MLXArray is part of `inputs:` state, so the trace stays valid
        // when the scheduler mutates it via `optimizer.learningRate = …`.
        // For `useCompiledLR == false`, this preserves the original
        // constant-LR fast path bit-for-bit.
        let canCompile = compileStep && galoreMgr == nil

        if canCompile {
            // Compile the full train step so MLX traces it once and reuses
            // the kernel-launch sequence thereafter. `inputs:` and `outputs:`
            // are model and optimizer so the compile knows to handle their
            // updated state across re-invocations. Clip + layer-LR scaling
            // happen INSIDE the traced graph, so they cost ~nothing per
            // step after the first.
            let compiled = compile(
                inputs: [m, optimizer],
                outputs: [m, optimizer]
            ) { (x: MLXArray, y: MLXArray) -> MLXArray in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                optimizer.update(model: m, gradients: processed)
                return loss
            }
            self.trainStepFn = compiled
        } else {
            self.trainStepFn = { (x: MLXArray, y: MLXArray) -> MLXArray in
                let (loss, grads) = gradFn(m, x, y)
                var processed = grads
                processed = clip.map { clipGradNorm(processed, maxNorm: $0) } ?? processed
                // GaLore projection happens AFTER clipping so the norm cap
                // sees the raw gradient (the rank-R version is by
                // definition a contraction — clipping it twice is fine
                // but unnecessary).
                if let g = galoreMgr {
                    processed = g.processGradients(processed)
                }
                if layerDecay < 0.9999 {
                    processed = scaleLayerwiseLR(processed, decay: layerDecay, nLayers: nLayers)
                }
                optimizer.update(model: m, gradients: processed)
                return loss
            }
        }

        // -------------------------------------------------------------
        // Compiled gradient-accumulation step. Built only if the caller
        // passed `accumMicroBatches: N` and the rest of the compile gate
        // is on. The compiled trace takes a flat array of 2N MLXArrays
        // — (x0,y0,x1,y1,…,x_{N-1},y_{N-1}) — and folds the entire
        // accumulation loop INSIDE the trace. The Swift caller then
        // dispatches one kernel-launch sequence per optimiser step
        // instead of N separate `gradFn` calls.
        //
        // GaLore is incompatible with this path (out-of-graph state).
        // -------------------------------------------------------------
        if let N = accumMicroBatches, N > 1, canCompile {
            self.compiledAccumN = N
            let nF = Float(N)
            self.accumStepFn = compile(
                inputs: [m, optimizer],
                outputs: [m, optimizer]
            ) { (xs: [MLXArray]) -> [MLXArray] in
                // Pair up the inputs back into (x_i, y_i) tuples.
                // `xs` shape: 2N flat — even indices x, odd indices y.
                precondition(xs.count == 2 * N,
                             "compiled accum step expects exactly 2N arrays (got \(xs.count) for N=\(N))")
                var accumGrads: ModuleParameters? = nil
                var lossSum = MLXArray(Float(0))
                for i in 0..<N {
                    let x = xs[2 * i]
                    let y = xs[2 * i + 1]
                    let (loss, grads) = gradFn(m, x, y)
                    lossSum = lossSum + loss
                    if let acc = accumGrads {
                        // Element-wise sum across leaves. Two-dict
                        // mapValues visits the matching leaf in `grads`
                        // and folds it in.
                        accumGrads = acc.mapValues(grads) { a, b in a + (b ?? a) }
                    } else {
                        accumGrads = grads
                    }
                }
                // Mean → clip → layer-LR decay → optimiser update.
                let scale = MLXArray(1.0 / nF)
                var avg = accumGrads!.mapValues { (g: MLXArray) -> MLXArray in g * scale }
                if let cn = clip { avg = clipGradNorm(avg, maxNorm: cn) }
                if layerDecay < 0.9999 {
                    avg = scaleLayerwiseLR(avg, decay: layerDecay, nLayers: nLayers)
                }
                optimizer.update(model: m, gradients: avg)
                // Return the mean loss as a one-element array.
                return [lossSum / MLXArray(nF)]
            }
        } else {
            self.accumStepFn = nil
            self.compiledAccumN = nil
        }
    }

    /// One training step. Returns the scalar batch loss.
    public func step(inputs: MLXArray, targets: MLXArray) -> Float {
        let loss = trainStepFn(inputs, targets)
        // Force eager evaluation so the lazy graph doesn't grow across steps.
        // `model` and `optimizer` both conform to `Updatable`; eval walks
        // their parameters / state.
        eval(loss, model, optimizer)
        stepCount += 1
        return loss.item(Float.self)
    }

    /// Gradient-accumulated step. Runs every micro-batch through the loss
    /// + gradient function, sums the gradients element-wise across
    /// micro-batches, then divides by N and applies a single optimizer
    /// update. The returned scalar is the mean loss across micro-batches.
    ///
    /// When the trainer was built with `accumMicroBatches: N`, the entire
    /// accumulation loop runs inside ONE compiled trace — N micro-batches
    /// in, mean loss out. That's the fast path; it's ~30-50% faster on
    /// accum>1 than the legacy host-loop fallback below because the trace
    /// dispatches a single kernel sequence to the GPU instead of crossing
    /// the host/MLX boundary N times.
    ///
    /// When `accumMicroBatches` is nil, this falls back to the original
    /// Swift `for`-loop path: each micro-batch gets a separate `gradFn`
    /// call, gradients are summed element-wise in host code, then a
    /// single optimiser update closes the step. That path is required
    /// for GaLore (out-of-graph projector state) and for any non-AdamW
    /// optimiser, which today doesn't go through `useCompiledLR`.
    public func accumulatedStep(microBatches: [(MLXArray, MLXArray)]) -> Float {
        precondition(!microBatches.isEmpty, "accumulatedStep needs ≥1 micro-batch")
        // Fast path: compile-folded accumulation. Builds a flat (x_i, y_i)
        // sequence and dispatches the compiled trace once.
        if let fn = accumStepFn, let N = compiledAccumN, microBatches.count == N {
            var flat: [MLXArray] = []
            flat.reserveCapacity(2 * N)
            for (x, y) in microBatches { flat.append(x); flat.append(y) }
            let outs = fn(flat)
            let loss = outs[0]
            eval(loss, model, optimizer)
            stepCount += 1
            return loss.item(Float.self)
        }
        var accumGrads: ModuleParameters? = nil
        var lossSum: Float = 0
        let n = microBatches.count
        for (x, y) in microBatches {
            let (loss, grads) = gradFn(model, x, y)
            eval(loss)
            lossSum += loss.item(Float.self)
            if let accum = accumGrads {
                // Sum element-wise. mapValues with two dicts visits matching
                // leaves; we add the corresponding MLXArrays. Each call
                // returns a NEW ModuleParameters; the old one becomes garbage.
                accumGrads = accum.mapValues(grads) { a, b in a + (b ?? a) }
            } else {
                accumGrads = grads
            }
        }
        // Mean: divide accumulated sum by micro-batch count, then update.
        let scale = MLXArray(1.0 / Float(n))
        var avg = accumGrads!.mapValues { (g: MLXArray) -> MLXArray in g * scale }
        if let cn = gradClipNorm {
            avg = clipGradNorm(avg, maxNorm: cn)
        }
        // GaLore projection — runs once per *optimiser update*, not once
        // per micro-batch (the projection is linear, so projecting the
        // mean is exactly the mean of the projections — same answer,
        // cheaper).
        if let g = galore {
            avg = g.processGradients(avg)
        }
        if lrLayerDecay < 0.9999 {
            avg = scaleLayerwiseLR(avg, decay: lrLayerDecay, nLayers: model.config.nLayers)
        }
        optimizer.update(model: model, gradients: avg)
        eval(model, optimizer)
        stepCount += 1
        return lossSum / Float(n)
    }
}
