import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// Owns a training run and streams progress to the UI. Lives on the main
/// actor so views can subscribe directly; training-step work happens
/// inline (MLX is GPU-async, so the per-step cost on the main thread
/// is mostly orchestration, not compute).
@MainActor
final class TrainController: ObservableObject {
    @Published var lossHistory: [LossPoint] = []
    @Published var status: String = "configure a run and press Start"
    @Published var stepCount: Int = 0
    @Published var targetSteps: Int = 1000
    @Published var stepsPerSec: Double = 0
    @Published var isTraining: Bool = false
    @Published var currentLoss: Float = 0
    @Published var sampleText: String = ""
    @Published var presetIdx: Int = 0  // index into Self.presets

    // Today's CLI flags surfaced in the UI:
    enum LRSchedule: String, CaseIterable, Identifiable {
        case cosine, wsd, constant
        var id: Self { self }
    }
    @Published var lrSchedule: LRSchedule = .cosine
    @Published var maxLR: Float = 3e-4
    @Published var minLR: Float = 3e-5
    @Published var warmupSteps: Int = 100
    @Published var seedText: String = ""              // empty = random
    @Published var spikeDetectEnabled: Bool = true
    @Published var spikeAlert: String? = nil          // most recent spike message

    /// Picks the user can choose between. Each is a (name, config) pair.
    /// Browser-reachable sizes at the top; Mac-only behemoth sizes below.
    static let presets: [(name: String, cfg: ModelConfig)] = [
        ("Tiny ·   842K",  ModelConfig(vocabSize: 256, contextLength: 128, nLayers: 4,
                                       nHeads: 4, dModel: 128, dMlp: 512)),
        ("Small · 2.4M",   ModelConfig(vocabSize: 256, contextLength: 256, nLayers: 6,
                                       nHeads: 6, dModel: 192, dMlp: 768)),
        ("Huge ·  9.6M",   ModelConfig.huge),
        ("Mega ·   76M",   ModelConfig.mega),
        ("Behemoth · 404M (Mac only)",  ModelConfig.behemoth),
        ("Titan · 1.3B (Mac only)",     ModelConfig.titan),
    ]

    private var trainTask: Task<Void, Never>? = nil

    func start(corpus: Data) {
        cancel()
        let cfg = Self.presets[presetIdx].cfg
        let presetName = Self.presets[presetIdx].name
        lossHistory.removeAll()
        stepCount = 0
        currentLoss = 0
        sampleText = ""
        spikeAlert = nil
        isTraining = true

        // Seed RNG before any model construction so init is reproducible.
        // Treat blank or non-numeric seedText as "no seed" (random init).
        let parsedSeed: UInt64? = UInt64(seedText.trimmingCharacters(in: .whitespaces))
        if let s = parsedSeed { MLXRandom.seed(s) }

        let seedTag = parsedSeed.map { "seed \($0)" } ?? "random init"
        status = "building \(presetName) (\(formatParams(estimateParams(cfg))) params) · \(seedTag)…"

        trainTask = Task {
            await runTraining(corpus: corpus, cfg: cfg, presetName: presetName)
        }
    }

    func cancel() {
        trainTask?.cancel()
        trainTask = nil
        if isTraining {
            isTraining = false
            status = "stopped at step \(stepCount)"
        }
    }

    private func runTraining(corpus: Data, cfg: ModelConfig, presetName: String) async {
        let model = TinyGPTModel(cfg)
        // Snapshot the LR config — `@Published` vars can mutate from the
        // UI mid-run but the trainer needs a stable view.
        let schedule = self.lrSchedule
        let maxLRSnap = self.maxLR
        let minLRSnap = self.minLR
        let warmupSnap = self.warmupSteps
        let total = self.targetSteps
        // WSD decay window — 10% of total steps by default, matches the
        // CLI's --decay-steps auto. The stable middle phase makes WSD
        // friendly to mid-run resume; the decay is fast at the end.
        let decaySteps = max(1, total / 10)

        // Start the optimizer at the LR the schedule says step 0 should
        // be at (typically the first warmup tick, not maxLR).
        let initialLR = lrAt(step: 0, schedule: schedule, maxLR: maxLRSnap,
                              minLR: minLRSnap, warmup: warmupSnap,
                              decaySteps: decaySteps, total: total)
        let trainer = Trainer(model: model, learningRate: initialLR, compileStep: true)
        let byteCorpus = ByteCorpus(corpus)
        let batchSize = batchSizeFor(cfg)

        let scheduleTag = schedule == .constant ? "constant" :
                          schedule == .wsd ? "wsd(warmup \(warmupSnap), decay \(decaySteps))" :
                          "cosine(warmup \(warmupSnap))"
        status = "training \(presetName) · batch \(batchSize) · ctx \(cfg.contextLength) · \(scheduleTag)"
        let t0 = Date()

        // Spike detector reuses TinyGPTModel.LossSpikeDetector — same
        // primitive the CLI uses, so the UI and CLI agree on what
        // counts as a spike.
        var spikeDetector = LossSpikeDetector(window: 50, factor: 3.0)

        for step in 0..<total {
            if Task.isCancelled { break }

            // Schedule the LR for this step (skip when constant — the
            // initial value already matches).
            if schedule != .constant {
                trainer.optimizer.learningRate = lrAt(
                    step: step, schedule: schedule,
                    maxLR: maxLRSnap, minLR: minLRSnap,
                    warmup: warmupSnap, decaySteps: decaySteps, total: total)
            }

            let (x, y) = byteCorpus.sampleBatch(batchSize: batchSize, contextLength: cfg.contextLength)
            let loss = trainer.step(inputs: x, targets: y)

            self.stepCount = step + 1
            self.currentLoss = loss
            if step % 5 == 0 || step == total - 1 {
                self.lossHistory.append(LossPoint(step: step + 1, loss: loss))
            }
            let elapsed = -t0.timeIntervalSinceNow
            if elapsed > 0 { self.stepsPerSec = Double(step + 1) / elapsed }

            // Loss-spike check. The detector silently warms up for its
            // window (50 steps); a triggered spike publishes a banner
            // the UI can render. v1 is observe-only — the operator
            // decides whether to Stop and resume with a lower LR.
            if spikeDetectEnabled {
                let (spike, ma) = spikeDetector.observe(loss: loss, step: step)
                if spike {
                    self.spikeAlert = String(
                        format: "spike at step %d: loss %.3f vs moving-avg %.3f (>%.1f×)",
                        step + 1, loss, ma, 3.0)
                }
            }

            await Task.yield()
        }
        self.isTraining = false
        self.status = "done — \(stepCount) steps in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s, final loss \(String(format: "%.3f", currentLoss))"
    }

    /// Dispatch the LR for a step across the three schedule modes. WSD
    /// reuses the same `lrAtWSD` helper the CLI uses so behavior matches
    /// `tinygpt train --lr-schedule wsd`. Cosine is inlined here (small
    /// + the equivalent CLI helper lives in `TrainSupport` which is in
    /// the CLI executable target, not reachable from the app).
    private func lrAt(step: Int, schedule: LRSchedule, maxLR: Float, minLR: Float,
                       warmup: Int, decaySteps: Int, total: Int) -> Float {
        switch schedule {
        case .constant:
            return maxLR
        case .cosine:
            if step < warmup {
                return maxLR * Float(step + 1) / Float(max(1, warmup))
            }
            if step >= total { return minLR }
            let progress = Double(step - warmup) / Double(max(1, total - warmup))
            let cos = 0.5 * (1.0 + Foundation.cos(.pi * progress))
            return minLR + (maxLR - minLR) * Float(cos)
        case .wsd:
            return lrAtWSD(step: step, total: total, warmup: warmup,
                            decaySteps: decaySteps, maxLR: maxLR, minLR: minLR)
        }
    }

    /// Conservative batch sizes — Mega at B=8 is tight on a 16 GB box; smaller
    /// presets can go larger but the perf difference is small at this scale.
    private func batchSizeFor(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 512 { return 4 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }

    private func estimateParams(_ cfg: ModelConfig) -> Int {
        let v = cfg.vocabSize, c = cfg.dModel, ctx = cfg.contextLength
        let m = cfg.dMlp, l = cfg.nLayers
        return v*c + ctx*c + 2*c + l*(4*(c*c + c) + 4*c + 2*(c*m + m + c))
    }

    private func formatParams(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n)/1_000_000) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n)/1_000) }
        return "\(n)"
    }
}
