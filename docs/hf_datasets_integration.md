# HuggingFace Datasets Hub integration

`tinygpt download-dataset` and `tinygpt list-datasets` give the on-device
agent-factory direct access to HuggingFace's 100k+ dataset hub. This
document covers how to use them, how the format adapter works, the
curated registry of "good" datasets by specialist type, and the
caveats.

> **Why this matters.** Training data is the bottleneck for any
> finetuning effort. The HF Hub is where everyone publishes — xLAM and
> Hermes for function-calling, OpenHermes-2.5 / UltraChat for
> instruction-following, UltraFeedback for DPO, OpenMathReasoning for
> math, OpenThoughts for chain-of-thought. Without a clean local
> pipeline to pull and convert these, every user reinvents one. This
> module is that pipeline.

---

## Quick start

```bash
# Browse the curated registry (no network).
tinygpt list-datasets
tinygpt list-datasets --specialist tool-calling
tinygpt list-datasets --info Salesforce/xlam-function-calling-60k

# Resolve + download. Auto-detects schema and converts to JSONL.
tinygpt download-dataset hf://datasets/yahma/alpaca-cleaned

# Force a target format if auto-detect picks the wrong one.
tinygpt download-dataset hf://datasets/OpenHermes-2.5 --format sft

# Cap shards for a quick sanity test.
tinygpt download-dataset Salesforce/xlam-function-calling-60k --max-files 1

# Inspect (no download) — print file list and predicted schema.
tinygpt download-dataset hf://datasets/argilla/ultrafeedback-binarized-preferences-cleaned --inspect

# Field aliasing for unusual schemas.
tinygpt download-dataset some-owner/some-dataset --map question:instruction,solution:response
```

Output lands at `~/.cache/tinygpt/datasets/<owner>/<name>/corpus.jsonl`
(or `corpus.txt` for plain format), unless `--out` is given.

---

## What the CLI does

1. Hits `GET https://huggingface.co/api/datasets/<id>` to resolve the
   dataset and list its siblings (files in the repo).
2. Filters siblings to data shards (JSONL/JSON/parquet/arrow/csv);
   skips READMEs, dataset_infos, licenses, .gitattributes.
3. Streams each shard via `https://huggingface.co/datasets/<id>/resolve/main/<filename>`,
   writing to the local cache. Existing files with matching size are
   skipped (resume).
4. Sniffs the first row of the first decodable shard and picks a
   target format via `FormatDetector`. The user can override with
   `--format` and `--map`.
5. Iterates all shards, converts each row to a `CorpusRecord`, and
   writes the result as newline-delimited JSONL (or plain text for
   `.plain` if `--out` ends in `.txt`).

---

## Format adapter logic

The adapter looks at one row and ranks three target formats:

| Format | Output schema | Trigger |
|--------|---------------|---------|
| `sft`   | `{instruction, input?, response}` | row has `instruction`/`prompt`/`question` + `response`/`completion`/`answer`/`output` |
| `dpo`   | `{prompt, chosen, rejected}` | row has `chosen` + `rejected` (string OR chat-array) |
| `plain` | one example per line | row has `text`/`content`/`document` |

### Detection order

1. **DPO** (highest specificity) — if `chosen` and `rejected` are
   both present, it's a preference dataset. If `prompt` is missing
   but `chosen` is a chat array, we extract the prompt from the
   ShareGPT-style `[{role, content}]` structure.
2. **SFT** — if an instruction-like field and a response-like field
   are both present, it's an SFT corpus.
3. **ShareGPT-style SFT** — if `conversations` or `messages` is a
   `[{role, content}]` (or ShareGPT `[{from, value}]`) array, we
   collapse it to (joinedPriorTurns, lastAssistant).
4. **Plain** — single `text`-ish field falls back to pretraining text.
5. **Last-ditch** — if exactly one string-valued field exists, treat
   it as plain text with low confidence (`< 0.6`); the user is
   prompted to re-run with explicit flags.

### Field aliases (case-insensitive)

| Canonical | Aliases tried |
|-----------|---------------|
| `instruction` | `instruction`, `prompt`, `question`, `query`, `input_text` |
| `input` | `input`, `context`, `background` |
| `response` | `response`, `completion`, `answer`, `output`, `target`, `label`, `responses` |
| `prompt` (dpo) | `prompt`, `question`, `instruction`, `query` |
| `chosen` | `chosen`, `chosen_response`, `preferred`, `win`, `winner` |
| `rejected` | `rejected`, `rejected_response`, `dispreferred`, `loss`, `loser` |
| `text` | `text`, `content`, `document`, `raw_text`, `body` |

When the detector picks the wrong column (rare, but happens with
unusual schemas), pass `--map src_field:canonical_field,...`. For
example:

```bash
tinygpt download-dataset some-owner/some-dataset \
  --format sft \
  --map problem:instruction,solution:response
```

### Non-string values

If a row field is a JSON object or array rather than a string (e.g.
function-calling datasets store tool definitions and tool calls as
JSON), the adapter serialises it to canonical JSON with sorted keys.
Downstream SFT trainers see deterministic strings.

---

## Curated registry (`tinygpt list-datasets`)

The registry lives in
`native-mac/Sources/TinyGPTData/DatasetRegistry.swift`. Today it has
22 entries across 8 specialist categories.

| Specialist | Datasets |
|-----------|----------|
| `tool-calling` | Salesforce/xlam-function-calling-60k, NousResearch/hermes-function-calling-v1, Locutusque/function-calling-chatml |
| `debugger` | princeton-nlp/SWE-bench (+ Verified), bigcode/commitpack |
| `code` | bigcode/the-stack-smol, open-r1/codeforces-cots, iamtarun/python_code_instructions_18k_alpaca |
| `math` | nvidia/OpenMathReasoning, AI-MO/NuminaMath-CoT, meta-math/MetaMathQA |
| `reasoning` | open-thoughts/OpenThoughts-114k, open-thoughts/OpenThoughts2-1M |
| `instruct` | teknium/OpenHermes-2.5, HuggingFaceH4/ultrachat_200k, yahma/alpaca-cleaned |
| `preference` | argilla/ultrafeedback-binarized-preferences-cleaned, HuggingFaceH4/ultrafeedback_binarized, Intel/orca_dpo_pairs |
| `general` | roneneldan/TinyStories, HuggingFaceFW/fineweb-edu |

Adding a new entry is a one-line addition to
`DatasetRegistry.all` — fill in `id`, `specialists`, `format`,
`approxSize`, `license`, `notes`.

---

## Caching

- Root: `~/.cache/tinygpt/datasets/<owner>/<name>/`
- Override: set `TINYGPT_DATASET_CACHE=/path/to/cache`
- Tinygpt's cache is **separate** from HuggingFace's
  `~/.cache/huggingface/` so users can blow it away independently.
- Resume: existing files whose byte size matches the HF-reported
  `size` are skipped. If the HF API doesn't report a size (some
  datasets don't), any non-empty cached file is trusted — pass
  `--max-files N` and clear the cache to force re-download.

---

## Authentication (gated / private datasets)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

The token is sent as `Authorization: Bearer $HF_TOKEN` on every
request. Get one at <https://huggingface.co/settings/tokens>.

Without `HF_TOKEN`:
- **Public datasets** work normally.
- **Gated datasets** (e.g. some Meta models' tokenisers, some preference data)
  return HTTP 401/403. The CLI prints a clear "gated or private" error.
- **Truly non-existent IDs** also return 401 (HF's anti-enumeration
  policy). The error message calls this out.

---

## Caveats

### Parquet decoding is not yet implemented

Apple ships no parquet decoder and the spec is non-trivial (mixed
encodings, dictionary pages, snappy/zstd compression). For now:

- JSONL / JSON datasets work end-to-end.
- Parquet / Arrow shards are downloaded and cached, but
  `RowReader.readRows` prints a note to stderr and returns 0 rows.
- A user can convert a cached parquet file with:
  ```bash
  python -c "import pandas; pandas.read_parquet('shard.parquet').to_json('shard.jsonl', orient='records', lines=True)"
  ```
  and then point the next pipeline step at the JSONL.
- The HF dataset-server's auto-converted parquet endpoint is wired
  but unused for now (`HFDatasets.parquetFiles(id:)`).

**Follow-up:** wrap Apple's compression APIs + a minimal parquet
column-reader to land native parquet. Tracked separately.

### Streaming for huge datasets

For datasets >100 MB the URLSession streaming download writes
straight to disk as bytes arrive — memory stays bounded. The
conversion phase reads the file streaming line-by-line (1 MiB
chunks). However, for parquet, full streaming requires a column
reader — not yet supported.

### Schema drift

HF dataset cards drift more than model cards. The adapter:

- Auto-detects sft/dpo/plain from the **first** row of the **first**
  decodable shard. Subsequent shards are assumed to have the same
  schema (HF datasets are conventionally homogeneous).
- Surfaces low-confidence detections as a warning. If the detector
  reports < 50% confidence, the user is told to re-run with
  `--format` and `--map`.

### File-size estimates

`/api/datasets/<id>` doesn't always include sibling sizes for older
snapshots — the inspect view shows `?` in that case. The download
still works (URLSession streams without knowing total size first);
we just can't show a percentage.

### Curated-registry license check

Every registry entry has a `license` field. Some are permissive
(MIT / Apache-2.0); some are NC-only (`CC BY-NC 4.0` for xLAM and
yahma/alpaca-cleaned) — be aware before training a commercial
model. `tinygpt list-datasets --info <id>` shows the license.

---

## Architecture

```
native-mac/Sources/TinyGPTData/
  HFDatasets.swift          - HF Hub API client (URLSession + JSON)
  CorpusFormat.swift        - format types + detector + JSONL writer
  RowReader.swift           - shard readers (JSONL/JSON now; parquet TODO)
  DatasetRegistry.swift     - curated catalog by specialist

native-mac/Sources/TinyGPT/
  DownloadDataset.swift     - `tinygpt download-dataset` CLI
  ListDatasets.swift        - `tinygpt list-datasets` CLI
```

`TinyGPTData` is intentionally a **pure-Foundation library**. It
depends on nothing else in the project, so it can be unit-tested
without MLX and so the `download-dataset` / `list-datasets` CLI
subcommands launch fast (no MLX initialisation).

The CLI subcommands live under pre-switch shims in
`Sources/TinyGPT/TinyGPT.swift` (matching the existing convention
used by `score-bench`, `train-heads`, `prune-*`, `gptq`). The shim
gets removed when `TODO(hf-datasets-merge)` is resolved.

---

## Future work

- **Parquet decoder** (column reader + snappy/zstd) so the registry's
  large-shard datasets (OpenHermes-2.5, OpenThoughts2-1M,
  UltraFeedback) work end-to-end.
- **Dataset-server rows API** (`/api/datasets/<id>/rows?...`) for
  small slices of huge datasets when the user just wants a sample.
- **Schema diff CLI**: `tinygpt diff-dataset <a> <b>` to compare
  schemas between two HF datasets.
- **Config selection**: many HF datasets have multiple configs
  (subsets). Today we ignore configs — a follow-up adds
  `--config <name>`.
- **Dedup / quality filter** as a `--postprocess` flag (length
  filter, minhash dedup, language detect).
