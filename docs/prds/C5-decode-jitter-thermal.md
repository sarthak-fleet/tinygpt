---
name: C5 decode jitter under thermal load
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier C (C5)
related_prds: app-train-controls-thermal.md (training-side thermal companion),
              docs/research/mac_decode_baseline_m5pro.md (steady-state baseline this PRD stress-tests)
---

# PRD — Measure decode tok/s degradation under sustained thermal load

## Goal

Add `scripts/bench_decode_thermal.py` that runs the existing decode
bench in a sustained loop for N minutes (default 30), captures
per-window p99 tok/s + die temperature + power draw via
`powermetrics`, and reports the steady-state vs cold-start delta.
Answers "does the M5 Pro throttle hard enough to break the realtime
floor under sustained agent load?" — a question the current 25-run
baseline can't answer because it warms up too briefly.

## Why now

- `mac_decode_baseline_m5pro.md` reports cold/warm metrics from 20-25
  run windows (~30s total). Real-world agent sessions run minutes,
  not seconds. We don't actually know whether decode degrades
  meaningfully under sustained load.
- The shipped train-side thermal controls
  (`app-train-controls-thermal.md`) handle pretrain throttling.
  Inference-side is the missing companion.
- Cheap to write (~1 day per the PLAN estimate). The bench harness
  already exists; this PRD wraps it in a loop + adds the thermal
  sidecar.

## Scope — in

- `scripts/bench_decode_thermal.py` — wraps `bench_decode.py` in a
  per-minute loop:
  - Run a 60-second window of decode (5–10 requests).
  - Read `powermetrics --samplers thermal --interval 1000` for the
    matching window (requires sudo — script prompts on first run).
  - Emit one row per window: `{minute, p99_tok_s, p99_ttft_ms,
    die_temp_c, package_power_w, throttle_pct}`.
- Default duration: 30 minutes. Override via `--minutes N`.
- Output: a per-minute JSONL + a single summary JSON suitable for
  pasting into the decode-baseline doc.
- New decode-baseline doc section: "Run 6 — sustained 30 min thermal"
  with the per-minute curve.
- One sentence in `docs/agent_runtime.md` flagging the curve for
  long-session Agent users.

## Scope — out

- **Automatic thermal-policy adjustment** in serve. The bench answers
  "do we have a problem"; the policy work is a follow-up if the data
  says yes.
- **Cross-machine baseline.** M5 Pro only; the doc already notes the
  hardware envelope.
- **Power-budget enforcement** (a la macOS Low Power Mode). Manual
  user toggle for V1.

## Files to touch

| File | Change |
|---|---|
| `scripts/bench_decode_thermal.py` | new — the wrapper |
| `docs/research/mac_decode_baseline_m5pro.md` | "Run 6 — sustained 30 min thermal" section + curve |
| `docs/agent_runtime.md` | one-paragraph thermal note pointing at the curve |
| `docs/research/data/decode-thermal-<model>.jsonl` | per-run artifact |

## Don't touch

- `scripts/bench_decode.py` — the new script consumes it as a child
  process, no API changes.

## Acceptance criteria

- [ ] `scripts/bench_decode_thermal.py --model google/gemma-3-12b
  --minutes 30 --rss-pid <pid>` runs 30 windows and produces a
  per-minute JSONL.
- [ ] The summary JSON contains:
  - cold p99 tok/s (first minute)
  - steady p99 tok/s (minutes 5–30 mean)
  - delta_pct (positive = degradation)
  - peak die temperature
  - throttle_pct over the run (fraction of windows where power
    < nominal; sourced from `powermetrics` flags)
- [ ] If `powermetrics` isn't available (no sudo), the script
  degrades to tok/s-only — no crash; thermal columns NULL'd.
- [ ] One row appended to `mac_decode_baseline_m5pro.md` Run 6 with
  the curve.

## Reference patterns

- `scripts/bench_decode.py` — the inner harness (just shipped). The
  wrapper invokes it via `subprocess.run`.
- `scripts/duty-cycle-throttle.sh` — existing throttling logic for
  training; the powermetrics invocation pattern is here.
- `app-train-controls-thermal.md` — companion PRD on the training
  side.

## Open questions

- Whether to use `powermetrics` or `sysctl machdep.xcpm.cpu_thermal_level`
  for thermal data. **Recommendation:** powermetrics — richer data
  (per-CPU + GPU + ANE thermals), already covered by an existing
  permission prompt in our app.
- Whether to gate on a specific delta threshold (e.g. "fail if steady
  tok/s drops > 30% from cold"). **Recommendation:** no gate in V1 —
  the bench is diagnostic, not pass/fail. Add a threshold once we've
  seen the curve and have a calibrated expectation.
