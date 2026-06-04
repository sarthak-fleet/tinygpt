import Foundation

/// Append-only JSONL emitter for a training run. One JSON object per line.
///
/// Schema (any field may be absent on a given line — readers must be tolerant):
/// ```
/// {"t": <unix-s>, "type": "step", "step": Int, "loss": Float,
///  "lr": Float, "step_per_s": Float?, "ma": Float?, "spike": Bool?,
///  "peak_rss_mb": Float?}
/// {"t": <unix-s>, "type": "val",  "step": Int, "val_loss": Float}
/// {"t": <unix-s>, "type": "meta", "preset": String, "depth": Int?,
///  "lr_schedule": String, "warmup": Int, "max_lr": Float, "min_lr": Float,
///  "decay_steps": Int?, "params": Int, "batch": Int, "ctx": Int}
/// ```
///
/// Robustness:
/// - One JSON object per line; readers can safely skip a malformed line.
/// - The file is opened append-only; concurrent runs append to disjoint
///   paths by convention. A crash mid-write leaves earlier lines intact.
/// - Flushed every `flushEvery` writes (default 50) to bound IO churn but
///   still survive a SIGKILL with most of the data.
/// - All Float values that aren't finite are written as `null` so the
///   viewer doesn't choke on a NaN/Inf payload mid-stream.
final class TrainLog {
    private let url: URL
    private let fh: FileHandle
    private let flushEvery: Int
    private var sinceFlush: Int = 0
    private let startedAt: Date

    init?(path: String, flushEvery: Int = 50) {
        let url = URL(fileURLWithPath: path)
        let fm = FileManager.default
        // Ensure parent exists.
        let dir = url.deletingLastPathComponent()
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        // Create-or-truncate at the start of a run; the previous log (if
        // any) is rotated to <path>.prev so it's not lost.
        if fm.fileExists(atPath: url.path) {
            let prev = URL(fileURLWithPath: url.path + ".prev")
            _ = try? fm.removeItem(at: prev)
            _ = try? fm.moveItem(at: url, to: prev)
        }
        if !fm.createFile(atPath: url.path, contents: nil) { return nil }
        guard let fh = try? FileHandle(forWritingTo: url) else { return nil }
        self.url = url
        self.fh = fh
        self.flushEvery = max(1, flushEvery)
        self.startedAt = Date()
    }

    deinit { close() }

    /// Best-effort flush + close. Idempotent — safe to call from the
    /// run completion path AND from a SIGINT handler shim.
    func close() {
        try? fh.synchronize()
        try? fh.close()
    }

    /// Write a single line. Always ends with `\n`. Non-finite floats
    /// become `null`. Strings are UTF-8-escaped via JSONSerialization.
    private func writeLine(_ dict: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(dict),
              let data = try? JSONSerialization.data(withJSONObject: dict,
                                                     options: [.sortedKeys])
        else { return }
        var payload = data
        payload.append(0x0A)  // '\n'
        try? fh.write(contentsOf: payload)
        sinceFlush += 1
        if sinceFlush >= flushEvery {
            try? fh.synchronize()
            sinceFlush = 0
        }
    }

    /// Sanitize a Float — non-finite → NSNull so the JSON serializer
    /// emits `null` instead of choking. (NaN/Inf aren't valid JSON.)
    private static func safe(_ f: Float?) -> Any {
        guard let f, f.isFinite else { return NSNull() }
        return f
    }

    private static func safe(_ d: Double?) -> Any {
        guard let d, d.isFinite else { return NSNull() }
        return d
    }

    private var elapsed: Double { -startedAt.timeIntervalSinceNow }

    // MARK: - Public emit methods

    func meta(preset: String, depth: Int?, lrSchedule: String, warmup: Int,
              maxLR: Float, minLR: Float, decaySteps: Int?,
              totalSteps: Int, params: Int, batch: Int, ctx: Int,
              seed: UInt64? = nil) {
        var d: [String: Any] = [
            "t": Date().timeIntervalSince1970,
            "type": "meta",
            "preset": preset,
            "lr_schedule": lrSchedule,
            "warmup": warmup,
            "max_lr": Self.safe(maxLR),
            "min_lr": Self.safe(minLR),
            "total_steps": totalSteps,
            "params": params,
            "batch": batch,
            "ctx": ctx,
        ]
        if let depth { d["depth"] = depth }
        if let decaySteps { d["decay_steps"] = decaySteps }
        if let seed { d["seed"] = "\(seed)" }  // string-encode UInt64 (JSON max safe int is 2^53-1)
        writeLine(d)
    }

    func step(step: Int, loss: Float, lr: Float,
              stepPerSec: Double? = nil, peakRssMB: Double? = nil,
              ma: Float? = nil, spike: Bool? = nil, valLoss: Float? = nil) {
        var d: [String: Any] = [
            "t": Date().timeIntervalSince1970,
            "elapsed_s": Self.safe(elapsed),
            "type": "step",
            "step": step,
            "loss": Self.safe(loss),
            "lr": Self.safe(lr),
        ]
        if let stepPerSec { d["step_per_s"] = Self.safe(stepPerSec) }
        if let peakRssMB  { d["peak_rss_mb"] = Self.safe(peakRssMB) }
        if let ma         { d["ma"] = Self.safe(ma) }
        if let spike      { d["spike"] = spike }
        if let valLoss    { d["val_loss"] = Self.safe(valLoss) }
        writeLine(d)
    }

    func val(step: Int, valLoss: Float) {
        writeLine([
            "t": Date().timeIntervalSince1970,
            "elapsed_s": Self.safe(elapsed),
            "type": "val",
            "step": step,
            "val_loss": Self.safe(valLoss),
        ])
    }

    func done(finalStep: Int, finalLoss: Float, totalSeconds: Double) {
        writeLine([
            "t": Date().timeIntervalSince1970,
            "elapsed_s": Self.safe(elapsed),
            "type": "done",
            "step": finalStep,
            "loss": Self.safe(finalLoss),
            "total_s": Self.safe(totalSeconds),
        ])
        close()
    }
}
