---
name: Rust HF downloader (parallel shards + retry)
status: not-started
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 (Rust opportunities)
---

# PRD — `scripts/hf-downloader/` Rust crate

## Goal

Replace the HTTP-download portion of `tinygpt download-dataset` (and
the App's `HFBrowserController`) with a Rust binary that does parallel
shard fetches with proper progress + retry + resume. Drops the ~270-MB
SmolLM2 weights download from "2 min serial via `hf download`" to
"30s parallel via this".

## Why now

- Multi-GB models (a Qwen3 0.6B BF16 is ~1.2 GB, sharded into 2 files;
  the larger ones are 10+ GB across many shards) take minutes serially.
  Parallel + resume is the standard.
- Today's HF downloader in Swift (App's `HFBrowserController`) is single-
  threaded URLSession with a 64 KB chunk loop. Decent but not great.
- Rust's `reqwest` + `tokio` + `indicatif` are batteries-included for
  exactly this. Single static binary, no Python.
- Pairs with the parquet-decoder Rust crate — together they form a
  "Rust data toolkit" that drops all Python deps from the data path.

## Scope — in

- New Cargo crate at `scripts/hf-downloader/`:
  - `Cargo.toml` with `reqwest` (rustls TLS), `tokio`, `serde_json`,
    `clap`, `indicatif` (progress bars), `anyhow`
  - `src/main.rs` with a CLI:
    ```
    hf-downloader <owner/repo> <out-dir> \
        [--files config.json model.safetensors tokenizer.json ...] \
        [--include 'model*.safetensors,config.json'] \
        [--token $HF_TOKEN] \
        [--concurrency 4] \
        [--retries 3]
    ```
- Hit `https://huggingface.co/api/models/<owner/repo>` for the file
  manifest (same as the App's `HFBrowserController` does today)
- Filter by `--files` (explicit list) or `--include` (glob); default
  set: `config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `*.safetensors`, `*.safetensors.index.json`
- Parallel-fetch with `--concurrency` (default 4 — enough to saturate
  HF's CDN without rate-limiting)
- Per-file progress bars via `indicatif::MultiProgress`
- Atomic writes — `<file>.part` until complete, then rename
- Resume: skip files where `<out-dir>/<file>` exists and size matches
  the manifest's reported size (when manifest reports it)
- Retry: exponential backoff up to `--retries` per file (HF's CDN
  occasionally 503s under load)

## Scope — out (v2)

- Authenticated dataset download (datasets need `/api/datasets/...`,
  models need `/api/models/...` — different endpoints). v1 = models only.
- Symlink-based caching mirroring `~/.cache/huggingface/hub/` layout.
  v1 = flat output dir.
- BFCL / τ-bench source pulls — those are repo clones, not file fetches.
- Streaming decode — leave that to the parquet-decoder crate.

## Inputs the agent has

| Resource | Location |
|---|---|
| Reference Swift impl | `native-mac/Sources/TinyGPTApp/HFBrowserController.swift` (~280 lines: manifest fetch, file URL build, chunked write, atomic rename) |
| HF API spec | https://huggingface.co/docs/hub/api |
| Auth header | `Authorization: Bearer <HF_TOKEN>` for gated/private; pass via `--token` or `HF_TOKEN` env |
| Test model (small) | `HuggingFaceTB/SmolLM2-135M` — ~270 MB total; safetensors + 4 json files |
| Test model (medium) | `Qwen/Qwen3-0.6B` — ~1.2 GB sharded across 2 safetensors |

## Acceptance criteria

1. `cd scripts/hf-downloader && cargo build --release` produces
   `target/release/hf-downloader` (~5 MB stripped)
2. Behavior parity with `hf download` on the smoke target:
   ```
   ./hf-downloader HuggingFaceTB/SmolLM2-135M /tmp/smollm2-rust/ \
       --include 'config.json,tokenizer.json,model.safetensors'
   ```
   - All 3 files present in `/tmp/smollm2-rust/`
   - sha256 matches the canonical `hf download` output
3. **Resume**: re-run the same command → all 3 files report "already
   cached" and the binary exits in <1 sec
4. **Parallel speed**: download Qwen3-0.6B (2 safetensors shards) with
   `--concurrency 2` → at least 1.5× faster than serial download
5. **Retry**: simulate a 503 by pointing at a bad path, verify the
   retry loop kicks in and exits with a clean error after 3 tries
6. Binary is self-contained: `otool -L` shows only system libs

## File paths

| Action | Path |
|---|---|
| **create** | `scripts/hf-downloader/Cargo.toml` |
| **create** | `scripts/hf-downloader/src/main.rs` |
| **create** | `scripts/hf-downloader/.gitignore` (just `target/`) |
| **don't touch** | The Swift HF downloader in the App (yet — wait for v1 parity, then maintainer decides whether to swap) |
| **don't touch** | `Sources/`, `PLAN.md`, `HANDOFF.md` |

## Dependencies

```toml
[dependencies]
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "stream", "json"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros", "fs", "io-util"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive", "env"] }
indicatif = "0.17"
anyhow = "1"
futures = "0.3"
glob = "0.3"  # for --include pattern matching

[profile.release]
opt-level = 3
lto = "thin"
strip = true
```

## Estimated effort

**~1 day focused.**

- 1 hr: scaffold + manifest fetch
- 2-3 hrs: parallel download with progress bars
- 1-2 hrs: resume / atomic write / retry
- 1 hr: smoke + sha256 verification against `hf download`

## Coordination

PR description must include:
1. `cargo build --release` output
2. Sha256 parity verification against `hf download` (paste both
   sha256sum outputs)
3. Timing comparison: serial Python vs parallel Rust (use `Qwen3-0.6B`
   as the test case — 2 shards make parallel actually matter)
4. Resume test output

Maintainer:
- Decides v2 timing for migrating App's HFBrowserController to call
  the Rust binary (or keep them as parallel impls until Swift is
  retired)
- Adds to `scripts/nightly/N01-pull-datasets.sh` if useful

## Known risks

- **HF CDN throttles aggressive parallelism**. Limit default
  `--concurrency` to 4. The HF docs recommend ≤8 for unauthenticated
  fetches.
- **Token in shell history**: prefer `HF_TOKEN` env var over `--token`.
  Document this.
- **Large file LFS redirects**: HF serves big files via S3 redirects.
  `reqwest` follows redirects by default; verify the Content-Length
  matches across the redirect.

## Source links

- HF Hub API: https://huggingface.co/docs/hub/api
- Reference Swift impl: `native-mac/Sources/TinyGPTApp/HFBrowserController.swift`
- Pattern reference (Rust async download): https://github.com/seanmonstar/reqwest/blob/master/examples/file_download.rs
