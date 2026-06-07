---
name: serve — wire MLX compile + speculative decoding (queued after #260)
status: superseded-by-corrected-plan-2026-06-07
owner: unassigned (parallel-agent task — Swift)
created: 2026-06-08
priority: P1 — 4-8× decode speedup (compounds with prompt cache + Q4)
queued-after: factory-serve-prompt-cache.md (same Serve.swift)
---

# PRD — MLX compile + speculative decoding in serve

## Corrected status — 2026-06-07

This PRD should no longer be treated as an executable implementation ticket.
It is superseded by a corrected three-part plan:

1. **Measure MLX compile first**: add a tiny benchmark around
   `AnyModel.forwardCached` and only wire `compile(...)` into serve if
   shape-specialized decode actually improves latency without increasing
   graph retention. The original claim that `Sample.swift` already had a
   serving-ready compile pattern was false.
2. **Move draft-model speculative decode into `TinyGPTModel`**: the current
   `SpeculativeDecode.swift` lives in the executable target, so
   `TinyGPTServe` cannot reuse it. Library extraction should preserve the
   existing two-model greedy/stochastic algorithm before any serve work.
3. **Gate grammar + speculation correctness**: serve may only accept a
   speculated token if it advances the same byte-FSM state as ordinary
   token-by-token decoding. Until the grammar interaction has a test, do not
   claim spec-dec compatibility with JSON Schema or Pace tool tags.

Recommended replacement PRDs:

- `serve-forwardcached-compile-measurement.md`
- `model-speculative-decode-library-extract.md`
- `serve-draft-specdecode-no-grammar.md`
- `serve-specdecode-grammar-correctness.md`

The current serve code remains on the KV-cached `forwardCached` path. No
speculative serve path was shipped from this stale PRD.

## 2026-06-07 blocker note

Do not assign this as written.

Findings from repo inspection:
- `Sample.swift` does **not** currently use `MLX.compile`, so there is no
  sample-side compile pattern to port.
- `SpeculativeDecode.swift` lives in the executable target, not
  `TinyGPTModel` / `TinyGPTServe`, so serve cannot reuse it without first
  moving the primitive into a library target.
- Self-speculation with the same model is not an obvious speed win and must
  be designed against the KV-cache path.
- Grammar-constrained speculation needs a correctness design: proposed tokens
  must be accepted only if they advance the FSM exactly like plain decode.

Next honest PRD: split this into (1) measure/validate MLX inference compile
API for `AnyModel.forwardCached`, (2) move reusable speculative decode into
`TinyGPTModel`, and (3) add serve speculative decode only after the grammar
interaction is specified.

## Goal

Two related serve-side perf wins that should land together since they
touch the same decode loop:

1. **MLX compile** — Sample.swift uses `MLX.compile` for ~2× faster
   decode. Port to serve. Module-level cache the compiled function so
   the cost is paid once per process.

2. **Speculative decoding** — Spec dec primitive ships in TinyGPT
   (B14, Sources/TinyGPTModel/MedusaHeads.swift,
   EagleDraft.swift, KVCache.swift). Wire it into serve so requests
   automatically use it. Default draft: theme-completer (if shared
   tokenizer) OR self-speculation. 2-3× decode speedup.

Combined with prompt cache + Q4 base, target full-response latency
drops from ~1200ms → ~175ms.

## Scope — in

### 1. MLX compile

Sample.swift line ~280-310 (search for `MLX.compile`). The pattern:
- Build a compile-shape signature (model + input shapes)
- Cache the compiled forward function in a static Module
- Use it in the decode loop instead of raw `model.forwardCached(...)`

Mirror in `Serve.swift`'s `generate` and `generateStreaming`.

### 2. Speculative decoding

Two modes:

- **Self-speculation** (no draft model): use the same model to draft
  N tokens, verify them in batched parallel. Simplest; minimal infra.
- **Mini-draft** (separate draft): a tiny model emits draft tokens,
  primary model verifies. Faster but requires draft model with same
  tokenizer.

For v1 of this PRD: ship self-speculation only. Mini-draft as v2.

CLI flag: `--speculative-k <N>` (default 4, 0 = disabled).

### 3. Cache `--compile` opt-out

Add `--no-compile` to disable MLX compile when needed for debugging.
Default: ON.

## Scope — out

- Mini-draft model loading (v2)
- Tree-based speculation (Medusa heads; v3)
- Speculative + grammar interaction (correctness gates; investigate, may need extra checks)

## Acceptance

1. Smoke: serve loads with compile + speculative-k=4
2. Decode speed: ≥2× current rate measured on the standard latency test
3. Same output bytes as non-spec-dec (greedy temperature=0 should be
   identical bit-for-bit)
4. Grammar enforcement still works (the FSM mask must apply to
   speculated tokens before they're accepted)
5. Build clean; existing serve tests pass

## Files involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTServe/Serve.swift` | Compile cache + speculative loop integration |
| `native-mac/Sources/TinyGPTModel/MedusaHeads.swift` etc | Reference only; don't edit |

## Estimated effort

**~3-4 hours.** Compile alone is ~1 hr. Speculative loop with grammar
correctness is the trickier piece.

## Why queued after #260

Both #260 (prompt cache) and this PRD touch the same decode functions
in Serve.swift. To avoid merge conflicts: ship #260 first, then this.

If you assign this elf BEFORE #260 finishes, instruct them to wait /
rebase after #260 lands.
