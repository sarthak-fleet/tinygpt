---
name: Cleanup — consolidate Score.swift into E0-conformant run-bench
status: shipped-2026-06-06
owner: unassigned (parallel-agent task — focused refactor)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (reinvention cleanup audit)
---

# PRD — `tinygpt score-bench` → `tinygpt run-bench` (E0-conformant)

## Goal

Migrate the useful work in `Score.swift` (782 lines) — perplexity scoring +
2 synthetic exact-match tasks — into a new `RunBench.swift` that emits
E0-conformant JSONL rows consumable by `tinygpt eval-compare`. Decommission
the legacy manifest-patching code path.

## Why now

The 2026-06-05 eval pipeline rebuild shipped E0 (`EvalCompare.Row` schema)
and E3 (`run-lm-eval` lm-evaluation-harness wrapper). `Score.swift` predates
this and writes to a different schema — the browser-leaderboard manifest at
`browser/public/gallery/manifest.json` via surgical JSON patching. Result:
two parallel eval-output formats, no shared comparison view, fragile JSON
surgery, no batch-append safety.

Consolidation gives us one canonical eval-row shape, eliminates ~400 lines
of manifest-patching code, and makes the synthetic-task benchmarks
visible in the same `eval-compare` views as the lm-eval-harness numbers.

## Scope — in

1. **New file**: `native-mac/Sources/TinyGPT/RunBench.swift` with subcommand
   `tinygpt run-bench`.
2. **Two evaluation modes**, both emitting `EvalCompare.Row` JSONL:
   - **Perplexity** — corpus-based loss + perplexity for a model. Inputs:
     `--model <ckpt>`, `--corpus <txt-or-jsonl>`, `--ctx N`, `--batch N`.
     Output rows: one per (model, corpus) with metric `perplexity` and
     `loss`.
   - **Task exact-match** — runs the `sort-6` and `reverse-16` synthetic
     tasks (port the deterministic Mulberry32 RNG from Score.swift to
     keep byte-for-byte parity with the browser scorer). Output rows: one
     per (model, task) with metric `acc` and a `n_examples` count.
3. **Subcommand wiring** in `TinyGPT.swift`: route `run-bench` → `RunBench.run`.
4. **Extract `greedyGenerate`** into a public utility (`Sources/TinyGPTModel/
   GenerationUtils.swift` or similar) — currently duplicated in `Score.swift`
   and `EvalIndic.swift`. Both should call the shared one.
5. **Deprecate `score-bench`**: keep the subcommand for one release as an
   alias that prints a warning and routes to `run-bench` with default args.
6. **Delete**: `Score.swift`'s manifest-patching code (`patchManifest()`
   and surrounding JSON-surgery helpers). The manifest update step, if
   still needed, becomes a post-eval shell or Python script that reads
   the E0 JSONL and rewrites `manifest.json` from it.

## Scope — out (v2)

- New synthetic tasks beyond sort-6 / reverse-16 — port only what exists.
- Browser leaderboard UI changes — the existing `/eval-leaderboard.astro`
  (shipped today) consumes E0 JSONL, so the new output should drop in.
- Migration of historical `manifest.json` benchmark scores — leave the
  manifest as-is for now; new runs go to JSONL.

## Acceptance criteria

1. `tinygpt run-bench --help` lists both modes with sensible defaults.
2. End-to-end smoke:
   ```
   tinygpt run-bench --model /tmp/huge-base-v1.tinygpt \
       --perplexity /tmp/fineweb-edu.txt \
       --tasks sort-6,reverse-16 \
       --out /tmp/run-bench-smoke.jsonl
   ```
   - Produces ≥3 JSONL rows (1 perplexity + 1 per task)
   - Each row passes `EvalCompare.Row` decoding
   - `tinygpt eval-compare /tmp/run-bench-smoke.jsonl --by task` renders
3. `tinygpt score-bench …` still works (legacy alias) but prints
   `[deprecated] score-bench is now run-bench; please migrate`
4. `GenerationUtils.greedyGenerate` is called from both `RunBench.swift`
   and `EvalIndic.swift`; the private copies in those files are removed
5. Build passes; existing tests pass; no regression in `tinygpt eval-compare`

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/RunBench.swift` |
| **create** | `native-mac/Sources/TinyGPTModel/GenerationUtils.swift` (extract `greedyGenerate`) |
| **modify** | `native-mac/Sources/TinyGPT/TinyGPT.swift` — add `run-bench` dispatch + deprecate `score-bench` |
| **modify** | `native-mac/Sources/TinyGPT/EvalIndic.swift` — call shared `greedyGenerate` |
| **delete (gradual)** | `native-mac/Sources/TinyGPT/Score.swift` — remove after deprecation period |
| **don't touch** | `EvalCompare.swift`, `RunLmEval.swift`, `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Estimated effort

**~2-3 days focused work** with OSS-pattern adoption (E0 row shape already
exists; no schema work).

## Coordination

PR description must include:
1. Smoke command output (≥3 rows in JSONL + `eval-compare` table)
2. Confirmation that `score-bench` legacy alias still works
3. Build + existing-test passing
4. Line count comparison (Score.swift before vs RunBench.swift after)

Maintainer marks Score.swift as deprecated in `docs/PLAN.md`, plans final
removal one release later.

## Known risks

- **Manifest update flow may have downstream consumers.** Check
  `browser/` and `scripts/` for anything that reads
  `manifest.json` benchmark fields. Document any breakage and provide
  a shell shim if needed.
- **Mulberry32 RNG parity** with browser scorer is critical — synthetic
  tasks must produce byte-for-byte same prompts as the JS implementation.
  Port the RNG exactly.

## Source links

- Sub-agent audit: see `docs/sessions/2026-06-06-mac-specialist-platform.md`
  reinvention-cleanup section
- E0 schema: `Sources/TinyGPT/EvalCompare.swift`
- E3 wrapper pattern: `Sources/TinyGPT/RunLmEval.swift`
