---
name: Cleanup — partial migration of data-pipeline Swift code to datatrove
status: shipped-2026-06-06
owner: unassigned (parallel-agent task — Python wrapper + Swift trim)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (reinvention cleanup audit)
---

# PRD — partial migration: ExtractorData + GitHubCorpus → datatrove

## Goal

Migrate the **generic** parts of `ExtractorData.swift` (525 lines) and
`GitHubCorpus.swift` (525 lines) to a Python shim built on HuggingFace's
`datatrove` library. Keep the **TinyGPT-specific** parts in Swift
(synthesis via cloud LLM, MLX-Swift training integration).

Net result: ~600-800 lines of Swift code can be retired; gain parallel
execution + integrated dedup/quality-filtering via datatrove.

## Why now

The 2026-06-06 reinvention audit found these two files re-implement
patterns that datatrove already solves at scale:

| Capability | Our Swift code | datatrove equivalent |
|---|---|---|
| Multi-format ingestion (BFCL, τ-bench, GitHub) | manual field-alias resolution | `HFReader` + schema detection |
| Truncation / metadata capture | manual | `JsonlWriter` + post-processors |
| Dedup (MinHash) | separate `tinygpt dedupe` command | built-in `MinhashDedup` filter |
| Quality filtering | separate `tinygpt quality-filter` command | built-in `FunctionFilter` |
| Parallel execution | none (single-threaded) | Ray / Slurm / local pool |

Current code maxes out at ~10K examples/hour because it's single-threaded.
With datatrove's parallel executors, throughput grows ~10-100×, which
matters when generating distillation datasets at 100K+ rows.

## Scope — in

1. **New Python module**: `scripts/data-prep/` with:
   - `Cargo.toml`-equivalent: `pyproject.toml` declaring `datatrove`,
     `requests` (for GitHub API), `ratelimit`
   - `prep_data.py` CLI: `python prep_data.py --bfcl <path> --tau-bench
     <path> --github <owner/repo> --kinds issues-prs,reviews,commits
     --dedup --quality-filter --out <jsonl>`
   - Internal modules: `bfcl_reader.py`, `tau_bench_reader.py`,
     `github_reader.py` (each implements a datatrove `Reader`)
2. **Datatrove pipeline composition** in `prep_data.py`:
   ```
   pipeline = [
       BFCLReader(bfcl_path) if --bfcl
     | TauBenchReader(tau_path) if --tau-bench
     | GitHubReader(repo, kinds) if --github
     | MinhashDedupFilter() if --dedup
     | QualityFilter(...) if --quality-filter
     | JsonlWriter(out)
   ]
   ```
3. **Trim Swift code**:
   - `ExtractorData.swift`: keep `synthesise()` (cloud LLM query) +
     wiring to `train-extractor`. Remove BFCL/τ-bench parsing
     (delegated to Python shim).
   - `GitHubCorpus.swift`: delete entirely (Python shim replaces it).
   - `tinygpt extractor-data` subcommand: keep flag-compatible; route
     non-synthesis paths to the Python shim via subprocess.
   - `tinygpt fetch-github`: deprecate; print a one-liner pointing at
     the Python script.
4. **Migration helper**: a one-line wrapper `tinygpt prep-data ...` that
   shells out to `python scripts/data-prep/prep_data.py`.

## Scope — out (v2)

- Replacing `tinygpt dedupe` and `tinygpt quality-filter` — those Swift
  commands stay for standalone use; they just become optional in
  pipeline-mode where datatrove handles them.
- Distributed execution (Ray cluster, Slurm) — start with local
  multiprocess executor only.
- Migration of historical JSONL files — only new runs use the new path.

## Acceptance criteria

1. `python scripts/data-prep/prep_data.py --help` shows the documented flags.
2. End-to-end smoke (BFCL only — smallest):
   ```
   python scripts/data-prep/prep_data.py \
       --bfcl ~/.cache/tinygpt/datasets/_external/bfcl/data/ \
       --out /tmp/bfcl-prep.jsonl
   ```
   - JSONL has ≥100 rows with `{query, tool}` fields
   - Each row passes `json.loads`
3. Parity test: side-by-side run against `tinygpt extractor-data --bfcl …`
   produces same row count ±5% (small diffs OK — datatrove dedup may
   remove near-duplicates the Swift version kept).
4. Parallel speedup: with `--num-workers 4`, processes a 50K-row
   benchmark at ≥3× the single-threaded speed.
5. `tinygpt extractor-data --bfcl ...` (legacy invocation) still works
   but prints `[note] consider scripts/data-prep/prep_data.py for
   parallel + dedup`.
6. `tinygpt fetch-github ...` prints the deprecation pointer.
7. Build + existing tests pass.

## File paths

| Action | Path |
|---|---|
| **create** | `scripts/data-prep/pyproject.toml` |
| **create** | `scripts/data-prep/prep_data.py` |
| **create** | `scripts/data-prep/{bfcl,tau_bench,github}_reader.py` |
| **modify** | `native-mac/Sources/TinyGPT/ExtractorData.swift` — keep synthesis, remove BFCL/τ-bench parsing |
| **modify** | `native-mac/Sources/TinyGPT/TinyGPT.swift` — deprecate `fetch-github`, add `prep-data` shim |
| **delete (after parity)** | `native-mac/Sources/TinyGPTData/GitHubCorpus.swift` |
| **don't touch** | `EvalCompare.swift`, `RunLmEval.swift`, `Train.swift`, the Rust crates, `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Estimated effort

**~5 days focused work.**
- 1 day: project setup + datatrove BFCL Reader
- 1 day: τ-bench + GitHub Readers (latter is the harder one — issue-PR
  linkage logic needs porting from Swift)
- 1 day: dedup + quality-filter integration + parallel executor wiring
- 1 day: Swift trim + subprocess shim + deprecation messages
- 1 day: parity testing + smoke + PR

## Coordination

PR description must include:
1. Parity-test output: same input → similar row counts in old vs new
2. Speedup measurement: serial vs `--num-workers 4`
3. Line count: Swift code removed (should be ~600+ lines)
4. Build + existing-test passing

Maintainer merges Swift edits + records the migration in `docs/PLAN.md`.

## Known risks

- **Python dependency added.** TinyGPT has been mostly Swift+Rust; adding
  a Python pipeline means users need `pip install datatrove`. Mitigation:
  document clearly; ship a `requirements.txt`; consider `uv` for fast install.
- **Dedup behavior may differ.** Datatrove's MinHash params may not match
  our existing `tinygpt dedupe`. Document the parameters; compare outputs.
- **GitHub API rate limiting.** datatrove doesn't handle GitHub-specific
  rate limits out of the box; the GitHub Reader needs custom logic.
- **MLX training pipeline downstream.** `tinygpt train-extractor` reads
  the JSONL output. Schema must remain compatible (`{query, tool}` for
  router; `{instruction, response, metadata}` for SFT). Verify before
  removing Swift code.

## Source links

- Sub-agent audit: see `docs/sessions/2026-06-06-mac-specialist-platform.md`
  reinvention-cleanup section
- datatrove: https://github.com/huggingface/datatrove
- Our existing dedup primitive: `tinygpt dedupe` (#202)
- Our existing quality filter: `tinygpt quality-filter` (B10)
