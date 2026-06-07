---
name: D-tier dataset decode + verify
status: shipped-2026-06-05
owner: unassigned (parallel-agent task — low-skill, data plumbing)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier D (D2/D3/D4 follow-up)
---

# PRD — decode + verify the in-flight dataset pulls

## Goal

Decode all the parquet shards under `~/.cache/tinygpt/datasets/` that
were pulled during the 2026-06-05 session, emit clean `.jsonl` files
at predictable paths, and write a per-dataset row-count + schema summary
that the next session can grep before training a specialist.

## Why now

- Today's pulls landed source parquet for: SWE-bench_Verified,
  the-stack-smol, python_code_instructions_18k_alpaca, plus the partial
  MS-MARCO / Natural Questions (D3) that may have been mid-flight.
  None of them have flat-form JSONLs at the canonical
  `~/.cache/tinygpt/datasets/<name>.jsonl` paths yet.
- A1 specialist + B25 ScaleDown both need decoded data. Decoding now,
  during the user's 2-day training window, unblocks both.
- This is pure plumbing — no Swift, no model code. Ideal for a fresh
  agent with no codebase context.

## Scope — in

For each dataset listed under "Targets" below:

1. **Decode** the cached parquet shards to JSONL via
   `python3 scripts/parquet_to_txt.py <input-dir> <output-path> --jsonl`
   (or the eventual Rust decoder, when it lands).
2. **Verify row count** matches the published HuggingFace dataset card
   (or note the discrepancy explicitly — sometimes shards differ).
3. **Sample 3 rows** with `head -3` and confirm the columns make sense.
4. **Append a row** to `docs/dataset-inventory.md` (create if absent)
   with: dataset id, local path, row count, schema columns, intended
   use (A1 / B1 / B25 / etc.), and notes.

## Targets

| Dataset | Local cache | Output JSONL path | Intended use |
|---|---|---|---|
| `princeton-nlp/SWE-bench_Verified` | `~/.cache/tinygpt/datasets/princeton-nlp/SWE-bench_Verified/` | `~/.cache/tinygpt/datasets/swe-bench-verified.jsonl` | Code specialist eval / training |
| `bigcode/the-stack-smol` | `~/.cache/tinygpt/datasets/bigcode/the-stack-smol/` | `~/.cache/tinygpt/datasets/the-stack-smol.jsonl` | Code specialist pretrain |
| `iamtarun/python_code_instructions_18k_alpaca` | `~/.cache/tinygpt/datasets/iamtarun/python_code_instructions_18k_alpaca/` | `~/.cache/tinygpt/datasets/python-code-instr.jsonl` | Code specialist SFT |
| `NousResearch/function-calling-chatml` (if pulled) | `~/.cache/tinygpt/datasets/NousResearch/function-calling-chatml/` | `~/.cache/tinygpt/datasets/function-calling-chatml.jsonl` | A1 tool-caller SFT (alt to xlam) |
| `microsoft/ms-marco` (if pulled per D3) | `~/.cache/tinygpt/datasets/microsoft/ms-marco/` | `~/.cache/tinygpt/datasets/ms-marco.jsonl` | B25 ScaleDown pretrain |
| `google-research-datasets/natural_questions` (D3) | `~/.cache/tinygpt/datasets/google-research-datasets/natural_questions/` | `~/.cache/tinygpt/datasets/natural-questions.jsonl` | B25 ScaleDown pretrain |

If a target dir doesn't exist or is empty: note it in the inventory as
"not pulled" and move on. No need to re-run the download.

## Scope — out

- Re-pulling datasets that didn't land cleanly (out of scope — the
  user is the only one with gated-dataset HF_TOKEN; if a target needs
  it, note "gated, needs HF_TOKEN" in the inventory)
- Tokenizing the data (separate task — depends on which model gets
  trained against it)
- MinHash dedup or quality filtering (separate tasks, already shipped
  as #202 + B10)
- Any Swift / Rust code changes

## Acceptance criteria

1. Each target either has its JSONL at the listed path **OR** an
   inventory row explaining why it doesn't (e.g., "not pulled — gated,
   needs HF_TOKEN")
2. `docs/dataset-inventory.md` exists with one row per dataset and a
   markdown table format
3. For each landed JSONL, the first 3 rows are valid JSON (verifiable
   with `head -3 <path> | python3 -c "import sys,json; [json.loads(l) for l in sys.stdin]"` exits 0)
4. PR description includes the row-count summary table

## File paths

| Action | Path |
|---|---|
| **create** | `docs/dataset-inventory.md` (or append if a prior agent created it) |
| **create** | The listed `.jsonl` output files under `~/.cache/tinygpt/datasets/` |
| **don't touch** | Any Swift, Rust, browser, or planning sources |

## Estimated effort

**~1 hour** if the parquet decoder runs cleanly on the first try. The
hard part is just running the script N times with the right args.

## Inventory format

`docs/dataset-inventory.md` — markdown table, append-only:

```markdown
# Dataset inventory

Updated by agents running `docs/prds/dataset-decode-verify.md`.
The decoded JSONLs live under `~/.cache/tinygpt/datasets/` (not in
the repo — too large; this doc is the index).

| Dataset | Local JSONL | Rows | Columns | Intended use | Notes |
|---|---|---|---|---|---|
| `princeton-nlp/SWE-bench_Verified` | `swe-bench-verified.jsonl` | 500 | instance_id, problem_statement, patch, … | Code specialist eval | … |
| … |
```

## Coordination

PR description must include:
- The row-count table (paste of the inventory)
- For each dataset: the actual command run + `wc -l` output
- Confirmation that `head -3 | python3 -c 'json.loads...'` works for
  each output

## Known risks

- Some datasets may have multiple configurations (e.g., MS-MARCO has
  `passage` and `document` configs). Pick the one most relevant to
  the intended use and document the choice.
- `the-stack-smol` is sharded by language. `the-stack-smol.jsonl` may
  end up being one language only OR concatenated all — document.
- If a parquet shard fails to decode, log it but continue with others;
  partial coverage is better than nothing.
