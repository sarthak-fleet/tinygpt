# BPE-aware leaderboard scoring

How a Mac-trained BPE checkpoint gets a row on the browser leaderboard
without the leaderboard ever needing to know what "BPE" means.

## TL;DR

Run

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-smoke-bpe-score -configuration Release build

/tmp/tinygpt-smoke-bpe-score/Build/Products/Release/tinygpt \
  score-bench /tmp/flagship-huge.tinygpt \
  --benchmarks bench/benchmarks.json

cd browser && npm run build
```

That writes a `benchmarks: { … }` block into the matching entry of
`browser/public/gallery/manifest.json`, and the next `npm run build`
copies the updated manifest into `dist/gallery/manifest.json` so the
leaderboard page renders it.

## Why Mac-side (Option A)

The original `browser/score_gallery.ts` ran every benchmark through
the byte-level WASM module. It hardcoded `vocab=256` everywhere — fine
for the five browser-trained gallery cards, silently broken for any
BPE checkpoint:

- `tg_model_create(256, …)` allocated a tiny embedding table.
- `tg_import_state` tried to import a 49152×256 BPE embedding into
  that 256×256 hole → either a crash or wrong-vocab garbage logits.
- The eval loop ran anyway, computing perplexity against random bytes
  out of a tokenized text → numbers were meaningless.

Two ways out:

1. **Bring a tokenizer to the browser.** Embed HuggingFace
   tokenizers in the Node script (huggingface/tokenizers-wasm or
   @huggingface/jinja-style runtimes). Costs: ~6 MB asset, separate
   wasm boot, a third tokenizer implementation to keep in sync with
   the Mac binary and the wasm browser worker.

2. **Score on the Mac, export the JSON.** Native binary already has
   the BPE-aware `HFTokenizer` and the `ModelLoader` that
   auto-detects byte-vs-BPE from the .tinygpt header. Output is the
   exact same manifest shape the leaderboard already reads.

We picked (2). The leaderboard's contract was always "render the
numbers in `manifest.json`"; the browser never needed to *run* the
benchmark, just *display* the result.

## Components

```
bench/benchmarks.json
  source-of-truth benchmark descriptors

native-mac/Sources/TinyGPT/Score.swift
  `tinygpt score-bench` subcommand — loads model, runs benches,
  surgically patches manifest.json in place

browser/score_gallery.ts
  thin legacy shim — refreshes byte-only entries for backwards compat
  and skips BPE files with a clear "run score-bench" message

browser/public/gallery/manifest.json
  leaderboard reads this; both scorers write to it

browser/src/pages/leaderboard.astro
  fetches /gallery/manifest.json, groups by benchmark id
```

## `bench/benchmarks.json` schema

```jsonc
{
  "version": 1,
  "benchmarks": [
    {
      "id": "tinystories-ppl",          // manifest key (benchmarks.<id>)
      "name": "TinyStories PPL",        // display name
      "kind": "perplexity",             // "perplexity" | "task-exact-match"
      "lowerIsBetter": true,
      "vocabType": "any",               // "byte-only" | "bpe-only" | "any"
      "holdoutCorpus": "browser/public/benchmarks/tinystories-eval.json",
      "holdoutFormat": "stories-json",  // "stories-json" | "raw-text"
      "batches": 32,
      "batchSize": 8,
      "description": "..."
    }
    // task-exact-match shape:
    // {
    //   "id": "sort-6", "kind": "task-exact-match",
    //   "vocabType": "byte-only",
    //   "task": "sort-6" | "reverse-16",
    //   "trials": 200,
    //   "seed": 1303
    // }
  ]
}
```

Vocab compat: `byte-only` benchmarks skip BPE models (and vice versa)
by writing `null` to the manifest. The browser's existing convention
("`null` = ran but incompatible, absent = never tried") lights up the
leaderboard UI correctly — incompatible rows don't show on that tab.

## What the Swift binary does

1. **Parse `--benchmarks`**: load descriptors from JSON.
2. **Load model**: `ModelLoader.load(<path>)` → detects byte-vs-BPE
   from header.tokenizerSource and header.vocabSize. Same loader the
   `eval` subcommand uses, so any model that `tinygpt eval` can score
   `tinygpt score-bench` can also score.
3. **For each descriptor:**
   - Compatibility check via `vocabType` ⇒ write `null` if mismatched.
   - **`kind: "perplexity"`** — load corpus (BPE: encode once via
     `HFTokenizer`; byte: raw `[UInt8]`), build a `TokenizedCorpus`
     or `ByteCorpus`, draw N random windows, mean cross-entropy →
     `exp(loss)` = perplexity.
   - **`kind: "task-exact-match"`** — generate trial set
     deterministically with Mulberry32 (byte-identical to the
     `browser/score_gallery_tasks.ts` PRNG), greedy-decode each
     prompt, count exact matches.
4. **Patch the manifest**: surgical text edit (NOT
   `JSONSerialization.data(prettyPrinted: true)`, which would
   alphabetize keys and replace `: ` with ` : ` across every entry —
   massive non-reviewable diff). Implementation:
   - Locate the model entry by `"id": "<gallery-id>"` via brace-match.
   - If an existing entry has a `"benchmarks": { … }` block, replace
     it inline preserving the surrounding key order.
   - Otherwise append a fresh model entry at the end of `models[]`.

## Workflow: get a Mac-trained checkpoint onto the leaderboard

You have `~/checkpoints/my-experiment.tinygpt` (BPE or byte-level —
doesn't matter):

```bash
# 1. Build the binary
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-smoke-bpe-score -configuration Release build

# 2. Run scoring (writes to manifest.json by default)
./tinygpt score-bench ~/checkpoints/my-experiment.tinygpt \
  --benchmarks bench/benchmarks.json \
  --id my-experiment      # optional — defaults to filename stem

# 3. (optional) Inspect the diff before publishing
git diff browser/public/gallery/manifest.json

# 4. Build the site so dist/gallery/manifest.json picks it up
cd browser && npm run build

# 5. Commit + push the manifest change + (if you want) the .tinygpt file
```

`--dry-run` prints what would be written without touching the file —
useful for sanity-checking a new descriptor.

## Adding a new benchmark

1. Add a row to `bench/benchmarks.json`. Pick `vocabType` carefully
   (`any` is the right default for perplexity; byte-only for tasks
   whose prompt format depends on character-level alignment).
2. Add an entry to `browser/src/benchmarks/registry.ts` so the
   leaderboard UI knows the name and direction. The browser-side
   implementation can be a stub that throws `BenchmarkError("incompatible", …)`
   if you don't want users running it live in their tab.
3. For a perplexity benchmark, drop the holdout text under
   `browser/public/benchmarks/<name>.txt` (or `.json` with a
   `stories-json` schema) and point `holdoutCorpus` at it. BPE-only
   benches can point outside the public dir (e.g.,
   `/tmp/eval-holdout-tail.txt`) — they never get fetched from the
   browser.
4. Re-run `tinygpt score-bench` over the existing gallery to backfill
   scores.

## Caveats

- **Estimated params**: the Mac scorer's "new entry" stub estimates
  parameters from the model config rather than counting the .tinygpt
  tensors. The number ends up within ~5% of the real count for
  standard dense models; if exactness matters, copy the real number
  in by hand or run `tinygpt inspect` and update the
  `paramCount` field.
- **Single source of variance**: random window sampling for the
  perplexity benches re-seeds the global MLX RNG per call. Two
  consecutive runs of `score-bench` will produce scores within ~1%
  of each other. For reproducible regression testing use a larger
  `batches` value in the descriptor.
- **Task-exact-match with BPE**: the sort-6 / reverse-16 prompts are
  marked `byte-only` because the BPE tokenizer (SmolLM2) merges
  digit-space sequences in surprising ways. Adding a `bpe-aware`
  task variant is a future project.
- **HF model directories**: `ModelLoader` also handles
  HuggingFace-format model directories (Llama-family). They go
  through the same code path and will score fine if the descriptor's
  `vocabType` accepts BPE. The `--id` flag is useful here since
  there's no filename stem.

## Why we kept `browser/score_gallery.ts`

Three reasons:

1. CI / existing scripts call it — silently breaking them by deletion
   would be hostile.
2. The byte-only path it implements is still the path of least
   friction for the original five browser-trained models. They're
   fp32, vocab=256, fit in the WASM module's address space, and run
   in seconds.
3. Now it explicitly skips BPE models with a one-liner pointing at
   `tinygpt score-bench`. Anyone running the legacy script on a BPE
   checkpoint gets the actionable message instead of a corrupted
   number.
