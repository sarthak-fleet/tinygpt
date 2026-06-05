---
name: Rust parquet decoder (replaces pyarrow)
status: not-started
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 (Rust opportunities — drops pyarrow dep)
---

# PRD — `scripts/parquet-decoder/` Rust crate

## Goal

Replace `scripts/parquet_to_txt.py` (50 MB pyarrow dep) with a single
static Rust binary that decodes HuggingFace parquet shards to plain
text or JSONL — faster, no Python dependency, distributable as part
of the TinyGPT release.

## Why now

- Today's pipeline shipped a Python `parquet_to_txt.py` that needs
  `pip install pyarrow` (~50 MB on disk + a Python runtime). Every
  new dataset pull goes through it: FineWeb-Edu, UltraFeedback, MS-MARCO,
  Natural Questions, all the eval splits.
- Apache Arrow's Rust crate (`arrow` + `parquet`) is mature, statically
  linkable, ~2 MB binary. **25× smaller, 5-10× faster decode**.
- A single Rust binary at `scripts/parquet-decoder` ships in the repo,
  works on any Mac without a Python install. Future TinyGPT users can
  ingest HF datasets without bootstrapping a Python env.

## Scope — in

- New Cargo crate at `scripts/parquet-decoder/`:
  - `Cargo.toml` with `parquet` + `arrow` + `clap` + `serde_json` deps
  - `src/main.rs` with a CLI matching `parquet_to_txt.py`'s interface
- CLI surface (drop-in for the existing Python script):
  ```
  parquet-decoder <input> <output> [--field FIELD] [--jsonl] [--max-rows N]
  ```
  - `<input>`: a single `.parquet` file OR a directory (recurse, sorted)
  - `<output>`: target path
  - `--jsonl`: emit JSON-per-row (preserves all columns); default = plain
    text (one row's `text` field per record, blank-line separated)
  - `--field FIELD`: text column override; default `"text"`, fallback
    chain `text → content → instruction`
  - `--max-rows N`: cap total output (smoke runs / sampling)
- Streaming output: don't buffer the whole table in RAM — read row
  groups, write progressively. FineWeb-Edu's 2.4 GB parquet should
  decode without blowing past ~100 MB peak.

## Scope — out (v2)

- Multi-threaded shard decoding. v1 = single thread, sequential.
- Compression options (gzip output etc.). v1 = plain text / NDJSON.
- Schema inference helpers (`--inspect` to dump column types).
- Writing arrow IPC / feather output.
- Anything that requires a Python venv at runtime — entire point is
  to drop that dependency.

## Acceptance criteria

1. `cd scripts/parquet-decoder && cargo build --release` produces
   `target/release/parquet-decoder` (~2-5 MB stripped).
2. Behavior parity with the existing Python script on the smoke target:
   ```
   ./parquet-decoder \
     ~/.cache/tinygpt/datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2013-20/ \
     /tmp/fineweb-edu-rust.txt \
     --max-rows 50000
   ```
   - Output byte count within 1% of `python scripts/parquet_to_txt.py …`
     (newline conventions may vary by a few bytes)
   - `head -1 /tmp/fineweb-edu-rust.txt` matches the head of the Python
     output character-for-character
3. JSONL mode preserves every column (verify by parsing first row in
   Python and checking key set matches the parquet schema):
   ```
   ./parquet-decoder \
     ~/.cache/tinygpt/datasets/HuggingFaceH4/ultrafeedback_binarized/data/ \
     /tmp/uf-rust.jsonl \
     --jsonl
   ```
4. Speed: at least **2× faster** than the Python script on the FineWeb-Edu
   smoke (time both, compare).
5. Binary is **self-contained** — `otool -L` shows only system libs (no
   .dylib from a homebrew/conda install).

## File paths

| Action | Path |
|---|---|
| **create** | `scripts/parquet-decoder/Cargo.toml` |
| **create** | `scripts/parquet-decoder/src/main.rs` |
| **create** | `scripts/parquet-decoder/.gitignore` (just `target/`) |
| **don't touch** | `scripts/parquet_to_txt.py` (leave as fallback for one release cycle), the Swift sources, `docs/PLAN.md`, `HANDOFF.md` |
| **don't touch** | The top-level Cargo workspace — this is its own crate; no workspace.toml manipulation |

## Reference: existing Python implementation

`scripts/parquet_to_txt.py` is the spec. Key logic:

```python
table = pq.read_table(shard)
cols = table.column_names
field = (args.field if args.field in cols else
         next((c for c in ("text", "content", "instruction") if c in cols), None))

if args.jsonl:
    for batch in table.to_batches():
        for rec in batch.to_pylist():
            if args.max_rows is not None and total >= args.max_rows: break
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
else:
    col = table.column(field).to_pylist()
    for v in col:
        if args.max_rows is not None and total >= args.max_rows: break
        if not v: skipped += 1; continue
        out.write(str(v)); out.write("\n\n"); total += 1
```

Match this output verbatim.

## Dependencies (Rust)

```toml
[package]
name = "parquet-decoder"
version = "0.1.0"
edition = "2024"

[dependencies]
parquet = { version = "55", default-features = false, features = ["arrow", "snap", "zstd"] }
arrow = "55"
clap = { version = "4", features = ["derive"] }
serde_json = "1"
anyhow = "1"

[profile.release]
opt-level = "z"     # binary size — we're CPU-bound on parquet decode, not allocations
lto = "thin"
strip = true
```

Verify versions current at task start time; pin to the latest stable.

## Estimated effort

**~half-day focused.** Breakdown:

- 30 min: scaffold the crate, get `cargo build` working
- 1-2 hrs: parquet → text path (the `--text` mode is just column-select +
  iterate + write)
- 1-2 hrs: JSONL mode (arrow's `RecordBatch::iter` → serde_json::Value)
- 30 min: directory-of-parquets walker (mirror Python's `find_parquets`)
- 1 hr: smoke + diff against Python output

## Coordination

PR description must include:
1. `cargo build --release` output showing `Compiling parquet-decoder v0.1.0`
2. `ls -la target/release/parquet-decoder` showing the binary size
3. The diff command + result for the FineWeb-Edu smoke (Python vs Rust
   output should be near-byte-identical for the `text` mode)
4. A timing comparison (Python vs Rust) on the same input

Maintainer will:
- Add a `scripts/parquet_to_txt` symlink to the Rust binary (or update
  callers — there are two: `scripts/nightly/N02-huge-base-v1.sh` and
  inline in some tests)
- Mark this task done in PLAN.md / HANDOFF.md
- Decide v2 timing for retiring `parquet_to_txt.py`

## Known risks

- **Apache Arrow Rust API moves**. The 55.x crates are stable but
  major-version churn happens; pin exact versions.
- **String column edge cases**: parquet `BYTE_ARRAY` columns can be
  binary, not UTF-8. The Python script treats `.to_pylist()` strings
  as native; in Rust we use `as_string::<i32>()` and may need to
  handle invalid UTF-8 with `from_utf8_lossy`. Document the choice.
- **Snappy / Zstd codecs**: FineWeb-Edu uses snappy; UltraFeedback's
  parquet may use zstd. Both are pulled in by the feature flags above.
  Don't forget Snappy.
- **Output line endings**: Python writes `\n\n` between records in text
  mode. Rust must match — easy to drop the second `\n` and produce
  different byte counts.

## Source links

- Crate docs: https://docs.rs/parquet/55/parquet/
- Arrow Rust: https://docs.rs/arrow/55/arrow/
- Reference Python: `scripts/parquet_to_txt.py` in this repo
- HuggingFace dataset cache layout (input shape):
  `~/.cache/tinygpt/datasets/<owner>/<repo>/data/.../*.parquet`
