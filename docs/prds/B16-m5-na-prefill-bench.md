---
name: B16 M5 Neural Accelerator prefill benchmark
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B16)
related_prds: docs/research/mac_decode_baseline_m5pro.md (decode baseline; this PRD is the prefill counterpart)
---

# PRD — Verify Apple's claimed 3.5–4× M5-vs-M4 prefill speedup

## Goal

Verify, on TinyGPT's actual MLX path, the [Apple ML Research,
2026](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
claim that the M5 Neural Accelerator delivers 3.5–4× faster prefill
vs M4 on the same workload. Bump `mlx-swift` to the latest (0.31.4+)
and benchmark; report a confirmed-or-refuted number with a bench
row in the decode-baseline doc.

Half-day. Free win if confirmed; the bump is reversible.

## Why now

- Current pin: `mlx-swift 0.31.3` on macOS 26.5 / M5 Pro (well past
  the 26.2 floor Apple cites). Whether the M5 NA paths are already
  hot or still gated by a version bump is the open question.
- If Apple's claim holds, TinyGPT inherits a 3.5–4× prefill speedup
  with no code changes — material for the bigger specialist models
  in the model zoo.
- If the claim doesn't hold on our path, the negative finding is
  itself informative (and surfaces a question for Apple's MLX team).

## Scope — in

- Run `scripts/bench_decode.py` against a Mega-class model
  (~960M) with `--prompt-tokens 2048` to make prefill dominate
  the wall-clock. Capture baseline numbers with current pin.
- Bump `Package.swift` to the latest `mlx-swift` release. Rebuild;
  rerun bench. Capture the after numbers.
- Report: TTFT delta, prefill tok/s delta. Pass = ≥ 2× improvement
  in prefill tok/s. Fail = no significant change OR regression.
- New row in `mac_decode_baseline_m5pro.md` — "Run 7 — M5 NA
  prefill" — with before/after.

## Scope — out

- **Decode-side measurement** — prefill is the variable here;
  decode stays where it is.
- **M4 cross-machine comparison.** We're on M5 Pro; the M5 vs M4
  claim is Apple's; we measure the M5 NA's contribution within
  M5 Pro by gating the right MLX features.
- **MLX-fast custom ops.** V1 = stock MLX.

## Files to touch

| File | Change |
|---|---|
| `native-mac/Package.swift` | mlx-swift version bump |
| `docs/research/mac_decode_baseline_m5pro.md` | Run 7 (before/after) |
| `docs/decision_log.md` | one-line entry on the bump |
| `docs/PLAN.md` | B16 ⬜ → ✅ + delta |

## Acceptance criteria

- [ ] Before/after prefill tok/s captured on M5 Pro, same prompt,
  same model.
- [ ] Documented as ≥ 2× win → adopt, < 2× → roll back the bump
  with a decision-log entry.

## Reference patterns

- `scripts/bench_decode.py` — already supports per-run timing.
- [Apple ML Research M5 LLMs post](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
  — the source claim.

## Open questions

- Whether the M5 NA path requires explicit MLX feature flags or
  is auto-detected. **Recommendation:** read the mlx-swift release
  notes around 0.31.4 first; default-on if available.
