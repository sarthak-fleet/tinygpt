import Foundation

// MARK: - Warmup-stable-decay (WSD) schedule

/// Warmup-stable-decay (WSD) learning rate, MiniCPM / SmolLM-style.
///
/// - `0 ≤ step < warmup`: linear ramp from 0 → maxLR
/// - `warmup ≤ step < total − decaySteps`: stable at maxLR
/// - `total − decaySteps ≤ step < total`: 1−√(t) decay from maxLR → minLR
/// - `step ≥ total`: minLR
///
/// The 1−√(t) decay shape (Hu et al., 2024, MiniCPM §4.3) decays faster
/// than half-cosine in the early decay window and is empirically the
/// better choice for the final-anneal phase on small models.
///
/// The stable middle phase makes WSD friendly to mid-run resume and to
/// extending pretraining without re-tuning a cosine envelope. Use
/// `decaySteps` as your annealing window — switch corpus to a curated
/// high-quality subset when step crosses `total − decaySteps`.
public func lrAtWSD(step: Int, total: Int, warmup: Int, decaySteps: Int,
                    maxLR: Float, minLR: Float) -> Float {
    if step < warmup {
        return maxLR * Float(step + 1) / Float(max(1, warmup))
    }
    if step >= total { return minLR }
    let decayStart = total - max(0, decaySteps)
    if step < decayStart { return maxLR }
    let progress = Float(step - decayStart) / Float(max(1, decaySteps))
    let shape = Float(Foundation.sqrt(Double(progress)))
    return maxLR - (maxLR - minLR) * shape
}

// MARK: - Loss spike detector

/// Sliding-window loss spike detector. v1 is **observe-only**: each
/// `observe(loss:step:)` call returns whether the latest loss exceeds
/// `factor × moving-average` over the last `window` steps. The caller
/// chooses the response (log, save an emergency checkpoint, pause).
///
/// Auto-rollback to a prior checkpoint is a v2 follow-up — the current
/// Adam-state-doesn't-persist limitation means a rollback already implies
/// a partial restart pain (see `--resume` docs on `tinygpt train`). v1
/// gives the operator an early warning so they can investigate or
/// `--resume` with a lower LR.
public struct LossSpikeDetector {
    public let window: Int
    public let factor: Float
    private var buf: [Float] = []
    private var lastSpikeStep: Int = -1

    public init(window: Int = 50, factor: Float = 3.0) {
        self.window = max(2, window)
        self.factor = max(1.01, factor)
        self.buf.reserveCapacity(self.window)
    }

    /// Observe one step's loss. Returns `(isSpike, movingAverage)`. The
    /// detector silently warms up for the first `window` observations.
    /// Sustained spikes are debounced: the next signal can fire at earliest
    /// `window/2` steps after the last one.
    public mutating func observe(loss: Float, step: Int) -> (spike: Bool, ma: Float) {
        guard buf.count >= window else {
            buf.append(loss)
            return (false, 0)
        }
        let sum = buf.reduce(0, +)
        let ma = sum / Float(buf.count)
        buf.removeFirst()
        buf.append(loss)
        let cooled = (step - lastSpikeStep) > window / 2
        let isSpike = loss.isFinite && ma.isFinite && loss > factor * ma && cooled
        if isSpike { lastSpikeStep = step }
        return (isSpike, ma)
    }
}
