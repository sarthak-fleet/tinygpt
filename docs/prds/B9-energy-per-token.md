---
name: B9 energy J/token measurement
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B9)
related_prds: C5-decode-jitter-thermal.md (powermetrics infra; energy sits next to thermals)
---

# PRD — J/token instrumentation via powermetrics

## Goal

Measure energy-per-generated-token across the model zoo and across
training runs. Surface a `j_per_token` column on the SLM leaderboard
and a per-epoch number on training runs. Answers "what does it cost
in laptop battery to use / train this specialist?" — a question
real users actually ask.

## Why now

- The hardware bench framework is the right home for this — it's
  one more column next to TTFT / ITL / tok/s / RSS. The bench harness
  doesn't measure it.
- Long-running training on battery is increasingly a thing; even
  a rough J/step number lets users budget overnight training honestly.
- powermetrics needs sudo, so this PRD documents the auth shim
  alongside the measurement.

## Scope — in

- `scripts/bench_energy.py` — wraps either `bench_decode.py` (decode)
  or a `tinygpt train --steps N` (training) in a powermetrics sidecar
  that samples package power at 1 Hz, integrates over the window,
  and divides by tokens generated (or steps run).
- Output JSON gains `joules_total`, `joules_per_token`, `joules_per_step`.
- Optional integration with the leaderboard aggregator — new
  optional `energy_json:` field on the manifest row.
- One-time sudo helper: `scripts/setup_powermetrics_sudoers.sh`
  walks the user through adding a passwordless sudo rule for
  `/usr/bin/powermetrics` only.

## Scope — out

- **Cross-device energy accounting** (CPU vs GPU vs ANE breakdown).
  Package power is the V1 number; per-engine is V2.
- **Training-loss-per-joule** optimization passes — V1 measures,
  doesn't optimize.

## Files to touch

| File | Change |
|---|---|
| `scripts/bench_energy.py` | new — sidecar wrapper |
| `scripts/setup_powermetrics_sudoers.sh` | new — one-time auth helper |
| `scripts/build_slm_leaderboard.py` | optional `energy_json` field in manifest reader |
| `docs/research/mac_decode_baseline_m5pro.md` | add energy column to Run 5/6 |
| `docs/PLAN.md` | B9 ⬜ → ✅ on ship |

## Acceptance criteria

- [ ] `scripts/bench_energy.py --decode --model google/gemma-3-12b
  --n 20` produces a `joules_per_token` number ± 10% on rerun.
- [ ] `scripts/bench_energy.py --train --steps 100
  --base shakespeare.tinygpt` produces a `joules_per_step` number.
- [ ] Without sudo set up, the script prints a clear "run
  `setup_powermetrics_sudoers.sh` once" error.
- [ ] Energy row appears on the leaderboard when present.

## Reference patterns

- `scripts/duty-cycle-throttle.sh` — existing powermetrics
  invocation; same auth pattern.
- C5 decode-jitter-thermal — companion (thermal + energy are
  often co-measured).

## Open questions

- Whether to integrate into `tinygpt bench` itself (Swift-side)
  vs keep it a script wrapper. **Recommendation:** Python wrapper —
  powermetrics has too much shell glue to comfortably go in Swift.
