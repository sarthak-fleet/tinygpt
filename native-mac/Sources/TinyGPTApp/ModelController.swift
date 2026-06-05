import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// Owns the loaded model + the in-flight generation. Views observe
/// `@Published` properties; the controller serialises model operations onto
/// its own task queue so the UI never blocks.
@MainActor
final class ModelController: ObservableObject {
    @Published var loadedItem: GalleryItem? = nil
    @Published var status: String = "no model loaded"
    @Published var paramCount: Int = 0
    @Published var deviceName: String = ""
    @Published var generated: String = ""
    @Published var isGenerating: Bool = false
    @Published var tokensPerSec: Double = 0
    @Published var evalResult: String? = nil
    @Published var isEvaluating: Bool = false
    /// Completed completions in this session. New runs append a fresh
    /// HistoryItem; the live in-flight buffer is `generated` above.
    /// Cleared via `clearHistory()` or naturally on app restart.
    @Published var history: [HistoryItem] = []

    /// One past prompt + completion + the sampler settings that produced
    /// it. Reproducible at a click.
    struct HistoryItem: Identifiable, Hashable {
        let id = UUID()
        let timestamp: Date
        let modelId: String
        let modelName: String
        let prompt: String
        let output: String
        let temperature: Float
        let topK: Int
        let repetitionPenalty: Float
        let tokensGenerated: Int
        let tokensPerSec: Double
    }

    private var model: TinyGPTModel? = nil
    private var modelConfig: ModelConfig? = nil
    private var generationTask: Task<Void, Never>? = nil

    init() {
        deviceName = "\(Device.defaultDevice())"
    }

    func load(_ item: GalleryItem) async {
        cancelGeneration()
        status = "loading \(item.displayName)…"
        loadedItem = nil
        model = nil
        modelConfig = nil

        // Stay on the main actor — the model isn't Sendable, and the load
        // work is small (read ~20MB file, decode fp16 to fp32, build the
        // module). The actual heavy MLX evaluation happens on the GPU
        // stream independently. SwiftUI yields to the run loop between
        // statements, so the UI stays responsive enough.
        let url = item.url
        do {
            let file = try TinyGPTFileReader.read(url)
            let h = file.header.config
            let cfg = ModelConfig(
                vocabSize: 256,
                contextLength: h.ctx ?? 256,
                nLayers: h.layers ?? 12,
                nHeads: h.heads ?? 8,
                dModel: h.dModel ?? 256,
                dMlp: h.dMlp ?? 1024
            )
            let m = TinyGPTModel(cfg)
            try TinyGPTWeightLoader.load(file, into: m)
            self.model = m
            self.modelConfig = cfg
            self.loadedItem = item
            self.paramCount = m.numParameters()
            self.status = "ready — \(formatParams(self.paramCount)) parameters on \(self.deviceName)"
        } catch {
            self.status = "failed to load \(item.displayName): \(error)"
        }
    }

    /// Stream-generate tokens. Cancel any in-flight generation first.
    /// `topK` 0 disables top-k filtering; `repetitionPenalty` 1.0 disables
    /// the repetition-penalty pass. The completed run is archived to
    /// `history` once finished (or on cancel after at least one token).
    func generate(prompt: String, maxTokens: Int, temperature: Float,
                  topK: Int = 0, repetitionPenalty: Float = 1.0) {
        cancelGeneration()
        guard let model, let cfg = modelConfig else { return }
        generated = prompt
        isGenerating = true
        tokensPerSec = 0
        status = "generating…"

        let item = loadedItem
        generationTask = Task {
            await runGenerate(model: model, cfg: cfg,
                              prompt: prompt, maxTokens: maxTokens,
                              temperature: temperature,
                              topK: topK,
                              repetitionPenalty: repetitionPenalty,
                              archiveTo: item)
        }
    }

    func clearHistory() { history.removeAll() }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if isGenerating {
            isGenerating = false
            status = "ready"
        }
    }

    /// Score the loaded model on the given UTF-8 corpus. Sets `evalResult`
    /// when finished. Mirrors `tinygpt eval` semantics: cross-entropy loss
    /// + bits-per-byte + perplexity over N random windows.
    func evaluate(corpus: Data, batches: Int = 20) {
        guard let model, let cfg = modelConfig else {
            evalResult = "no model loaded"; return
        }
        isEvaluating = true
        evalResult = nil
        Task { @MainActor in
            let byteCorpus = ByteCorpus(corpus)
            var lossSum: Float = 0
            var count = 0
            for _ in 0..<batches {
                let (x, y) = byteCorpus.sampleBatch(batchSize: 8, contextLength: cfg.contextLength)
                let l = model.loss(x, y)
                eval(l)
                lossSum += l.item(Float.self)
                count += 1
            }
            let avg = lossSum / Float(count)
            let bpb = avg / log(Float(2))
            let ppl = exp(avg)
            self.evalResult = String(
                format: "loss %.3f · BPB %.3f · perplexity %.1f (over %d batches × 8 × %d tokens)",
                avg, bpb, ppl, count, cfg.contextLength
            )
            self.isEvaluating = false
        }
    }

    private func runGenerate(model: TinyGPTModel, cfg: ModelConfig,
                             prompt: String, maxTokens: Int, temperature: Float,
                             topK: Int, repetitionPenalty: Float,
                             archiveTo item: GalleryItem?) async {
        let promptBytes = [UInt8](prompt.utf8)
        var idx = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])
        // Track recent tokens for the repetition-penalty pass. Bound the
        // window so memory doesn't grow with the run; the last 256 tokens
        // are what matter for byte-level repetition.
        var recent: [Int32] = promptBytes.map { Int32($0) }
        let recentWindow = 256

        let t0 = Date()
        var streamed = 0
        for _ in 0..<maxTokens {
            if Task.isCancelled { break }
            let T = idx.shape.last!
            let lo = max(0, T - cfg.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            var last = logits[0..., logits.shape[1] - 1, 0...].reshaped([logits.shape[2]])

            // Repetition penalty (Keskar et al. 2019 — CTRL): divide logits
            // for tokens that appeared in the recent window by `penalty`.
            // For negative logits the standard CTRL trick is to MULTIPLY
            // instead; we keep it symmetric by branching per-sign so
            // |logit| always shrinks for repeated tokens.
            if repetitionPenalty > 1.0 && !recent.isEmpty {
                let tail = recent.suffix(recentWindow)
                let uniq = Set(tail)
                // Pull the logits row to host, modify, push back. The vocab
                // is small (256 for byte-level gallery models) so this is
                // cheap; the alternative — gather + scatter in MLX — adds
                // graph nodes for no real win.
                eval(last)
                var floats = last.asArray(Float.self)
                for tok in uniq {
                    let i = Int(tok)
                    if i >= 0 && i < floats.count {
                        if floats[i] > 0 { floats[i] /= repetitionPenalty }
                        else            { floats[i] *= repetitionPenalty }
                    }
                }
                last = MLXArray(floats, [floats.count])
            }

            let nextId: MLXArray
            if temperature <= 0 {
                nextId = argMax(last, axis: -1).reshaped([1, 1])
            } else {
                var scaled = last / MLXArray(temperature)
                // Top-K filter: keep only the K largest logits, set the
                // rest to -inf. K==0 or K>=vocab disables.
                let vocab = scaled.shape.last!
                if topK > 0 && topK < vocab {
                    let asc = sorted(scaled, axis: -1)       // ascending
                    let thr = asc[vocab - topK]              // k-th largest
                    eval(thr)
                    let thrF = thr.item(Float.self)
                    scaled = MLX.where(
                        scaled .< MLXArray(thrF),
                        MLXArray(-Float.infinity),
                        scaled)
                }
                nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
            }
            eval(nextId)
            let id = Int(nextId.item(Int32.self))
            recent.append(Int32(id))
            idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
            streamed += 1

            // Stream one token at a time to the UI; coalesce updates via
            // an actor-isolated assign to avoid SwiftUI churn (re-render
            // per token is fine at 100 tok/s).
            let glyph: String
            if let scalar = UnicodeScalar(id), id >= 9 {
                glyph = String(scalar)
            } else {
                glyph = ""
            }
            await MainActor.run { [glyph] in
                self.generated.append(glyph)
                let elapsed = -t0.timeIntervalSinceNow
                if elapsed > 0 {
                    self.tokensPerSec = Double(streamed) / elapsed
                }
            }
        }
        await MainActor.run {
            self.isGenerating = false
            self.status = "done — \(streamed) tokens at \(String(format: "%.0f", self.tokensPerSec)) tok/s"
            // Archive the run — except for trivial zero-token cancels, which
            // would clutter the history list.
            if streamed > 0, let item {
                let outputOnly = String(self.generated.dropFirst(prompt.count))
                self.history.append(HistoryItem(
                    timestamp: Date(),
                    modelId: item.id,
                    modelName: item.displayName,
                    prompt: prompt,
                    output: outputOnly,
                    temperature: temperature,
                    topK: topK,
                    repetitionPenalty: repetitionPenalty,
                    tokensGenerated: streamed,
                    tokensPerSec: self.tokensPerSec
                ))
            }
        }
    }

    private func formatParams(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}
