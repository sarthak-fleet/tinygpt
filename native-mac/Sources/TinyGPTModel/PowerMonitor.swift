import Foundation
#if canImport(IOKit)
import IOKit.ps
#endif

/// Lightweight power + thermal + battery monitor for pausable training.
///
/// Why this exists: training on a Mac can draw more power than a USB-C
/// monitor's PD output can supply, causing battery discharge while
/// "charging." Thermal pressure can also force the SoC to throttle,
/// slowing everything down. This module exposes enough state for the
/// training loop to detect these conditions and cooperatively pause
/// (via the existing SIGINT atomic checkpoint path).
///
/// Each query is cheap (~milliseconds). The training loop polls between
/// macro-batches, not per-step.
public enum PowerMonitor {

    /// Snapshot of current power state.
    public struct Snapshot {
        public let batteryPercent: Int?       // 0-100, nil if no battery
        public let batteryCharging: Bool?     // true = charging, false = discharging, nil if no battery
        public let onACPower: Bool            // true if wall power connected (regardless of charging direction)
        public let thermalPressure: ThermalPressure
        public let timestamp: Date

        public var summary: String {
            var s = "thermal=\(thermalPressure.label)"
            if let bp = batteryPercent { s += " battery=\(bp)%" }
            if let c = batteryCharging { s += " " + (c ? "charging" : "discharging") }
            if onACPower { s += " ac=on" }
            return s
        }
    }

    public enum ThermalPressure: Int, CaseIterable {
        case nominal = 0
        case fair = 1
        case serious = 2
        case critical = 3

        public var label: String {
            switch self {
            case .nominal:  return "nominal"
            case .fair:     return "fair"
            case .serious:  return "serious"
            case .critical: return "critical"
            }
        }

        /// Returns true if we should consider pausing.
        public var isPaused: Bool { rawValue >= ThermalPressure.serious.rawValue }
    }

    /// Read the current snapshot.
    public static func sample() -> Snapshot {
        let thermal = readThermalPressure()
        let (pct, charging, ac) = readBatteryState()
        return Snapshot(
            batteryPercent: pct,
            batteryCharging: charging,
            onACPower: ac,
            thermalPressure: thermal,
            timestamp: Date()
        )
    }

    // MARK: - Thermal

    private static func readThermalPressure() -> ThermalPressure {
        let state = ProcessInfo.processInfo.thermalState
        switch state {
        case .nominal:  return .nominal
        case .fair:     return .fair
        case .serious:  return .serious
        case .critical: return .critical
        @unknown default: return .nominal
        }
    }

    // MARK: - Battery / AC

    /// Returns (batteryPercent, charging, onACPower).
    /// On a desktop / Mac mini with no battery, returns (nil, nil, true).
    private static func readBatteryState() -> (Int?, Bool?, Bool) {
        #if canImport(IOKit)
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else {
            return (nil, nil, true) // no power info available — assume desktop
        }
        guard let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [CFTypeRef] else {
            return (nil, nil, true)
        }

        var pct: Int? = nil
        var charging: Bool? = nil
        var ac: Bool = false

        for src in sources {
            guard let info = IOPSGetPowerSourceDescription(snapshot, src)?.takeUnretainedValue() as? [String: AnyObject]
            else { continue }
            if let pState = info[kIOPSPowerSourceStateKey] as? String {
                ac = (pState == kIOPSACPowerValue)
            }
            if let isCharging = info[kIOPSIsChargingKey] as? Bool {
                charging = isCharging
            }
            if let cur = info[kIOPSCurrentCapacityKey] as? Int,
               let max = info[kIOPSMaxCapacityKey] as? Int, max > 0 {
                pct = (cur * 100) / max
            }
        }
        return (pct, charging, ac)
        #else
        return (nil, nil, true)
        #endif
    }

    /// Should we pause training right now?
    ///
    /// Returns a tuple (shouldPause, reason). The reason is `nil` when
    /// the answer is no.
    public static func shouldPause(snapshot s: Snapshot,
                                    cfg: PauseConfig) -> (Bool, String?) {
        // Thermal pressure: serious or worse → pause.
        if cfg.thermalPause && s.thermalPressure.isPaused {
            return (true, "thermal pressure \(s.thermalPressure.label)")
        }
        // Battery discharge despite AC power → pause. Means the adapter
        // can't keep up; continuing burns battery + likely throttles.
        if cfg.batteryDischargePause, s.onACPower, s.batteryCharging == false,
           let pct = s.batteryPercent, pct < 95 {
            return (true, "AC connected but discharging (battery \(pct)%, charging=false)")
        }
        // Standalone battery threshold: low battery without AC → pause.
        if cfg.batteryLowPause, !s.onACPower, let pct = s.batteryPercent,
           pct < cfg.batteryLowThreshold {
            return (true, "battery \(pct)% (below \(cfg.batteryLowThreshold)% threshold) and not on AC")
        }
        return (false, nil)
    }

    public struct PauseConfig {
        public var thermalPause: Bool = true
        public var batteryDischargePause: Bool = true
        public var batteryLowPause: Bool = true
        public var batteryLowThreshold: Int = 20

        public init() {}
    }
}
