---
name: B14 speculative decoding (Mini-Llama draft for Mega target)
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B14)
related_prds: factory-serve-mlx-compile-specdec.md (existing serve-side spec-dec wiring),
              factory-ane-inference-pace.md (ANE routing context for the Mini-Llama draft)
---

# PRD — Vanilla speculative decoding with a Mini-Llama draft

## Goal

Wire `--draft-model <path>` into `tinygpt serve` and `tinygpt sample`
so a small "draft" model proposes K tokens per step and the target
model verifies them in one parallel forward. Expect ~1.6–2× decode
tok/s on Mega (~960M target) using a Tiny-class draft (~10M params),
matching the [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
paper.

This is distinct from the *Medusa* / *EAGLE-2* spec-dec already shipped
(those train extra heads on the target itself). Vanilla spec-dec uses
two *independent* models — the cheaper option when you already have
a small model trained on the same domain.

## Why now

- The model zoo already includes Tiny + Small + Medium + Huge + Mega +
  flagship. Each smaller-larger pair is a natural draft-target
  candidate; we've never tried it.
- The MLX-Swift forward path already supports batched token
  verification — the kv-cache + scaledDotProductAttention path
  generates `next_n_logits` for a given `prompt + draft` in one shot.
  Spec-dec is "compute the draft, batch-verify on target, accept
  prefix, fall back on first reject" — all primitives shipped.
- The numerics-gate framework (`docs/precision.md`) is the no-quality-
  regression discipline this needs from day one. Spec-dec must yield
  byte-identical output to greedy target on T=0, within KL ε on T>0.

## Scope — in

- `Sources/TinyGPTServe/SpecDec.swift` — new file with the
  accept/reject loop. Takes target + draft + K (draft length).
- `tinygpt serve --draft-model <path> --draft-k 4` flag.
- `tinygpt sample --draft-model <path>` for offline CLI use.
- Numerics gate: `evals/specdec-numerics.swift` asserts
  `serve(target, draft) == serve(target)` byte-for-byte at T=0 on a
  fixed 100-prompt set. Gate-fail → spec-dec disabled, fall back to
  plain decode (silent), match the f16-compute / coop-matrix pattern.
- Per-request `usage` block gains `draft_proposed`, `draft_accepted`,
  `accept_rate` — visible in the response and traceable.

## Scope — out

- **Auto-pick draft model** based on target's tokenizer + size. V1
  takes a path; the matchmaker is V2 (cheap heuristic: smallest
  model with same tokenizer + ≥80% target ppl on a calibration set).
- **Tree decoding** (multiple draft branches). Leviathan-style linear
  spec-dec covers ~80% of the lift; tree adds another 1.3× at 2×
  the code complexity. Defer.
- **Draft model on ANE while target on GPU** — heterogeneous routing
  is filed at `factory-ane-inference-pace.md` and waits on the
  rumored Stateful Models API.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTServe/SpecDec.swift` | new — accept/reject loop |
| `Sources/TinyGPTServe/Serve.swift` | `--draft-model` arg; plumb optional draft into the generate path |
| `Sources/TinyGPT/Sample.swift` | mirror flag for offline use |
| `evals/specdec-numerics.swift` | new — T=0 byte-equality + T>0 KL ε gate |
| `evals/specdec-throughput.sh` | new — measure tok/s ratio (spec / plain) on Tiny-draft + Mega-target |
| `docs/precision.md` | append a "vanilla spec-dec" gate row |
| `docs/PLAN.md` | flip B14 ⬜ → ✅ + measured tok/s lift |

## Don't touch

- `Sources/TinyGPTServe/Serve.swift` line ranges 1059–1135 and
  1596–1675 — the existing Medusa/EAGLE-2 paths. Vanilla spec-dec is
  a sibling; don't refactor them unless extending without breaking is
  impossible.
- The shipped numerics-gate harness — extend, don't fork.

## Acceptance criteria

- [ ] T=0 byte-equality gate passes on 100 prompts × Mega target with
  Tiny draft. Failure → spec-dec is silently disabled at boot, serve
  prints a one-line warning.
- [ ] Decode tok/s on Mega improves by ≥ 1.4× under spec-dec vs plain
  on the standard 64-prompt × 128-gen workload from
  `docs/research/mac_decode_baseline_m5pro.md`. Documented as a row.
- [ ] Per-request `usage` reports `accept_rate ≥ 0.6` on the same
  workload (lower acceptance = the draft isn't well-matched to the
  target; user can re-train or pick a different draft).
- [ ] `evals/specdec-throughput.sh` passes in CI.

## Reference patterns

- `Sources/TinyGPTServe/Medusa.swift` — the existing spec-dec path
  carries the accept-prefix-reject-on-first-mismatch invariant. Copy
  the loop shape; replace the "Medusa heads predict the next K
  tokens" step with "draft model predicts the next K tokens via its
  own forward".
- [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192), §3 —
  the rejection-sampling math. T=0 reduces to the simple
  "accept iff exact match"; T>0 needs the importance-weighted accept.
  Both paths covered in the paper.

## Open questions

- Initial draft-target pairing. **Recommendation:** Tiny (8M) as draft
  for Mega (960M) — 100:1 ratio is what the Leviathan paper used.
  Smaller drafts ship sooner; can re-tune K once the loop works.
- Whether to make `--draft-k` adaptive (high accept-rate → bump K,
  low → drop K). **Recommendation:** ship K=4 fixed; adaptive is
  100 LOC for a ~5% gain, defer.
