import XCTest
@testable import TinyGPTModel

/// Pure-logic tests for the WSD scheduler and loss-spike detector.
/// No MLX runtime calls — compiles and runs under `swift test` if the
/// rest of the test target does. These are the kind of tests that catch
/// the off-by-one in a decay-phase boundary or a stale buffer in the
/// spike detector — both bugs that would silently survive a release build
/// and only surface during a training run.
final class TrainSchedHelpersTests: XCTestCase {

    // MARK: - WSD

    /// At step 0, the WSD scheduler should return a small positive LR
    /// (the first warmup tick), NOT maxLR or 0.
    func test_wsd_step0_isFirstWarmupTick() {
        let lr = lrAtWSD(step: 0, total: 1000, warmup: 100, decaySteps: 100,
                          maxLR: 1e-3, minLR: 1e-5)
        XCTAssertEqual(lr, 1e-3 / 100, accuracy: 1e-9,
                       "step 0 should be the first warmup tick: maxLR/warmup")
    }

    /// At the last warmup step, LR should equal maxLR.
    func test_wsd_endOfWarmup_isMaxLR() {
        // step + 1 == warmup at step = warmup - 1 (last warmup step).
        let lr = lrAtWSD(step: 99, total: 1000, warmup: 100, decaySteps: 100,
                          maxLR: 1e-3, minLR: 1e-5)
        XCTAssertEqual(lr, 1e-3, accuracy: 1e-9,
                       "end-of-warmup should equal maxLR")
    }

    /// Anywhere in the stable phase, LR should be exactly maxLR.
    func test_wsd_stablePhase_holdsAtMaxLR() {
        for step in [100, 250, 500, 800, 899] {
            let lr = lrAtWSD(step: step, total: 1000, warmup: 100, decaySteps: 100,
                              maxLR: 1e-3, minLR: 1e-5)
            XCTAssertEqual(lr, 1e-3, accuracy: 1e-9,
                           "stable phase (step \(step)) should hold at maxLR")
        }
    }

    /// At the start of the decay window, LR is still maxLR; at the end,
    /// it's minLR. Monotone non-increasing in between (1-sqrt is monotone).
    func test_wsd_decayPhase_endpointsAndMonotone() {
        let total = 1000, warmup = 100, decaySteps = 100
        let maxLR: Float = 1e-3, minLR: Float = 1e-5

        // Start of decay: progress=0, shape=0, LR=maxLR.
        let atStart = lrAtWSD(step: total - decaySteps,
                               total: total, warmup: warmup, decaySteps: decaySteps,
                               maxLR: maxLR, minLR: minLR)
        XCTAssertEqual(atStart, maxLR, accuracy: 1e-9,
                       "decay-window start should equal maxLR")

        // Penultimate step: very close to minLR.
        let nearEnd = lrAtWSD(step: total - 1,
                               total: total, warmup: warmup, decaySteps: decaySteps,
                               maxLR: maxLR, minLR: minLR)
        XCTAssertGreaterThan(nearEnd, minLR,
                             "near-end should still be > minLR (haven't crossed total yet)")
        XCTAssertLessThan(nearEnd - minLR, 1e-4,
                          "near-end should be within shouting distance of minLR")

        // Monotone non-increasing across the decay window.
        var prev: Float = .greatestFiniteMagnitude
        for step in (total - decaySteps)..<total {
            let lr = lrAtWSD(step: step, total: total, warmup: warmup,
                              decaySteps: decaySteps,
                              maxLR: maxLR, minLR: minLR)
            XCTAssertLessThanOrEqual(lr, prev + 1e-9,
                                     "WSD decay should be monotone non-increasing at step \(step)")
            prev = lr
        }
    }

    /// Past the final step, the scheduler clamps to minLR.
    func test_wsd_pastTotal_clampsToMinLR() {
        let lr = lrAtWSD(step: 5000, total: 1000, warmup: 100, decaySteps: 100,
                          maxLR: 1e-3, minLR: 1e-5)
        XCTAssertEqual(lr, 1e-5, accuracy: 1e-9,
                       "past total should clamp to minLR")
    }

    /// 1-sqrt decay is faster than cosine in the early decay window.
    /// At progress=0.25, 1-sqrt gives a 50% reduction; cosine gives ~15%.
    /// This test pins the WSD-specific shape.
    func test_wsd_decayShape_isOneMinusSqrt() {
        // Decay window: steps 900..1000 (100 steps).
        // At step 925: progress = 25/100 = 0.25, sqrt = 0.5, LR = max - 0.5*(max-min).
        let lr = lrAtWSD(step: 925, total: 1000, warmup: 100, decaySteps: 100,
                          maxLR: 1.0, minLR: 0.0)
        XCTAssertEqual(lr, 0.5, accuracy: 1e-6,
                       "at progress=0.25, 1-sqrt shape should yield 0.5 of the range")
    }

    /// decaySteps == 0 should never trigger the decay branch — degenerate
    /// case stays at maxLR until clamp.
    func test_wsd_zeroDecaySteps_isAllStableThenClamp() {
        let lr = lrAtWSD(step: 500, total: 1000, warmup: 100, decaySteps: 0,
                          maxLR: 1e-3, minLR: 1e-5)
        XCTAssertEqual(lr, 1e-3, accuracy: 1e-9,
                       "decaySteps=0 should hold at maxLR through total")
    }

    // MARK: - LossSpikeDetector

    /// During warmup (first `window` observations), the detector returns
    /// `(false, 0)` and never triggers.
    func test_spike_warmup_neverTriggers() {
        var d = LossSpikeDetector(window: 10, factor: 3.0)
        for step in 0..<10 {
            let (spike, ma) = d.observe(loss: 1.0, step: step)
            XCTAssertFalse(spike, "no spike during warmup (step \(step))")
            XCTAssertEqual(ma, 0, "moving-avg=0 during warmup")
        }
    }

    /// A constant-loss stream past warmup should never trigger a spike.
    func test_spike_constantLoss_neverTriggers() {
        var d = LossSpikeDetector(window: 5, factor: 3.0)
        for step in 0..<5 { _ = d.observe(loss: 2.0, step: step) }
        for step in 5..<50 {
            let (spike, _) = d.observe(loss: 2.0, step: step)
            XCTAssertFalse(spike, "constant loss should not trigger spike (step \(step))")
        }
    }

    /// A sudden 5x jump after warmup should trigger.
    func test_spike_suddenJump_triggers() {
        var d = LossSpikeDetector(window: 5, factor: 3.0)
        for step in 0..<5 { _ = d.observe(loss: 1.0, step: step) }
        let (spike, ma) = d.observe(loss: 5.0, step: 5)
        XCTAssertTrue(spike, "5x jump over flat baseline should trigger")
        XCTAssertEqual(ma, 1.0, accuracy: 1e-6, "moving-avg of flat 1.0 = 1.0")
    }

    /// A 2x jump under factor=3.0 should NOT trigger (calibration check).
    func test_spike_subThreshold_doesNotTrigger() {
        var d = LossSpikeDetector(window: 5, factor: 3.0)
        for step in 0..<5 { _ = d.observe(loss: 1.0, step: step) }
        let (spike, _) = d.observe(loss: 2.0, step: 5)
        XCTAssertFalse(spike, "2x rise under factor=3.0 should not trigger")
    }

    /// After a spike fires, the detector debounces for `window/2` steps —
    /// a sustained spike doesn't fire repeatedly.
    func test_spike_debounces() {
        var d = LossSpikeDetector(window: 10, factor: 3.0)
        for step in 0..<10 { _ = d.observe(loss: 1.0, step: step) }

        let (firstSpike, _) = d.observe(loss: 5.0, step: 10)
        XCTAssertTrue(firstSpike, "first 5x jump should fire")

        // Steps 11..14 are inside the debounce window (window/2 = 5).
        // The MA is climbing as 5.0s push out 1.0s, so spike comparisons
        // weaken too — but even if they didn't, debounce should mask.
        var firedDuringDebounce = false
        for step in 11..<15 {
            let (s, _) = d.observe(loss: 5.0, step: step)
            if s { firedDuringDebounce = true }
        }
        XCTAssertFalse(firedDuringDebounce,
                       "sustained spike should not re-fire inside debounce window")
    }

    /// NaN and infinity in the input never trigger (defensive — finite
    /// check exists in observe).
    func test_spike_naNAndInfinity_doNotTrigger() {
        var d = LossSpikeDetector(window: 5, factor: 3.0)
        for step in 0..<5 { _ = d.observe(loss: 1.0, step: step) }
        let (sNaN, _) = d.observe(loss: .nan, step: 5)
        XCTAssertFalse(sNaN, "NaN should not trigger spike")
        let (sInf, _) = d.observe(loss: .infinity, step: 6)
        XCTAssertFalse(sInf, "+Inf should not trigger spike")
    }

    /// Constructor clamps degenerate args: window < 2 → 2, factor ≤ 1 → 1.01.
    func test_spike_constructorClamps() {
        let d1 = LossSpikeDetector(window: 0, factor: 0.5)
        XCTAssertGreaterThanOrEqual(d1.window, 2)
        XCTAssertGreaterThan(d1.factor, 1.0)

        let d2 = LossSpikeDetector(window: 1, factor: 1.0)
        XCTAssertGreaterThanOrEqual(d2.window, 2)
        XCTAssertGreaterThan(d2.factor, 1.0)
    }
}
