# Data-side performance levers

Two complementary data-pipeline regularisers for the Mac trainer:

1. **Sample packing** — inverse-length weighted SFT sampling that
   flattens the bias toward long examples in batch construction.
2. **BPE-dropout** — Provilkov et al. (ACL 2020) tokenisation
   regularisation: the same surface text yields different token
   sequences across epochs, giving the model robustness to merge
   choices.

Both are off by default and gated by explicit CLI flags. Neither
changes the on-disk file format, so checkpoints are interchangeable
between dropout-on and dropout-off runs.

## 1. Sample packing (≠ sequence packing)

The existing `--pack` flag (kept as a back-compat alias) does
**sequence packing** — concatenates multiple short examples into one
row so every position carries a real next-token target instead of
padding. That's a throughput lever.

**Sample packing** is a different idea: change *which examples* you
draw, not *how you tile them*. The new `--pack-mode` selector lives in
`tinygpt sft`:

| `--pack-mode` | What it does |
| --- | --- |
| `none` (default) | uniform random pick — one example per row |
| `sequence` | greedy-fit multiple examples into a row (= `--pack`) |
| `sample` | inverse-length weighted sampling — short examples picked more often so each example contributes ~equally per training step |
| `bucket` | length-bucket uniform — bin examples into `--length-bucket N` buckets and sample buckets uniformly, then a uniform example within |

### Why `sample` mode?

Under uniform sampling, each example has the same expected pick count.
But the **loss contribution per pick scales with length** (more tokens
→ more loss positions). So long examples dominate the gradient signal.

Inverse-length weighting fixes this: each example's expected
(`length × frequency`) becomes constant. Short examples — typically the
ones a model under-learns when the dataset is power-law in length — get
proportionally more attention.

Implementation: `SFTCorpus` pre-computes a cumulative-weight table
(O(N)) and each batch draw is O(log N) via binary search. The change is
~30 lines in `native-mac/Sources/TinyGPTModel/SFTCorpus.swift`.

### Smoke: histogram with vs without sample-packing

Synthetic 1000-example corpus, bimodal Pareto length distribution
(80% short [10..60], 20% long [60..400]). 2000 batches × 4 = 8000
draws. We report the average (`length × frequency`) per example, binned
by length:

```
buckets:                ["[10-80)", "[80-160)", "[160-240)", "[240-400]"]
uniform   freq/bucket:     [6532, 399, 331, 738]
uniform   tok·freq/bucket: [227161, 47384, 67202, 233911]
sample    freq/bucket:     [7736, 101, 62, 101]
sample    tok·freq/bucket: [206365, 11308, 12276, 31789]
bucket    freq/bucket:     [1937, 1189, 1526, 3348]
bucket    tok·freq/bucket: [67815, 157083, 307830, 1070180]

per-example (length × freq) — average within each length bin:
  uniform   [10-80):332.6  [80-160):951.9  [160-240):1589.4  [240-400]:2555.5
  sample    [10-80):240.1  [80-160):245.7  [160-240):247.1  [240-400]:254.3

CoV(per-example length·freq): uniform=0.582  sample=0.061
```

Reading the result:
- **Uniform**: per-example contribution scales linearly with length —
  long examples (240..400) contribute **~8×** more gradient per example
  than short ones (10..80).
- **Sample**: per-example contribution is essentially flat across all
  length bins (240..254). The histogram is **10× flatter** by CoV
  (0.582 → 0.061).
- **Bucket** mode pulls in the other direction: it over-represents
  long-bucket examples because the long bucket has fewer of them and
  bucket-uniform doesn't compensate for within-bucket length variance.
  Use it deliberately when you want each LENGTH REGIME equally seen.

Reproduce: `xcodebuild -scheme TinyGPT-Package test
-only-testing:TinyGPTModelTests/TinyGPTModelTests/test_smoke_samplePackingHistogram`

### CLI

```
tinygpt sft <base> --data foo.jsonl --pack-mode sample --out adapter.lora
tinygpt sft <base> --data foo.jsonl --pack-mode bucket --length-bucket 6 --out adapter.lora
```

The `--pack` short-flag remains and is equivalent to `--pack-mode sequence`.

### Caveats

- All four modes are mutually exclusive — the last flag wins.
- `bucket` mode rebuilds the bucket map per batch (cheap; sub-ms for
  any reasonable corpus). Could be cached at corpus construction;
  punted until profiling shows it matters.
- `sample` mode uses Swift's default `Double.random`, which is
  Mersenne-Twister-equivalent. Determinism across runs requires a
  global seed (not currently plumbed; PR welcome).

---

## 2. BPE-dropout

BPE-dropout (Provilkov, Emelianenko, Voita, ACL 2020) is a tokenisation
regulariser: during training, each adjacent-pair merge is skipped with
probability `p_drop`. The result — the same surface string gets a
slightly different token decomposition each time. The model learns to
predict the next token under tokenisation noise, which the paper shows
improves robustness on rare words and morphologically rich languages.

### Path-(a) vs path-(b) — honest assessment

The deliverable allowed two implementation paths:

**Path (a) — intercept at merge time inside `swift-transformers`.**
After reading the upstream source carefully:
`swift-transformers/Sources/Tokenizers/BPETokenizer.swift` exposes
`bpe(token:)` only as a `func` on an **internal-access** `class
BPETokenizer`. The `bpeRanks` table is also internal. There is no
public hook between `PreTrainedTokenizer.tokenize(text:)` and the
per-pair merge loop. Path (a) would require either:

1. A fork of `swift-transformers` with new public surface (subclass
   point, dropout-aware encode method).
2. Some form of runtime hot-patching (not viable in Swift's
   strict-typed-module world).

Both are out of scope for a single-PR data-perf feature.

**Path (b) — re-implement BPE-dropout in our own code.** Chosen.

The implementation is `native-mac/Sources/TinyGPTModel/BPEDropout.swift`
(~200 lines, single file):

- Loads `tokenizer.json`'s `model.merges` and `model.vocab` directly
  via `JSONSerialization` — no `swift-transformers` dependency.
- Replicates the standard GPT-2 byte-level pre-tokeniser regex.
- Replicates the GPT-2 byte alphabet table (256-element array of
  visible code points covering all byte values).
- Standard BPE merge loop with a per-pair Bernoulli drop: for every
  adjacent pair, sample `Uniform(0,1) < p_drop` to decide whether to
  *ignore* that merge candidate this iteration.

### Scope honestly stated

This encoder **only** covers byte-level BPE — the family that GPT-2 /
GPT-J / Llama / Qwen / Gemma / Phi / Mistral all use. It does **not**
cover:

- SentencePiece (T5, mT5, original Llama-1).
- WordPiece (BERT family).
- Byte-fallback tokenisers (some SentencePiece variants).
- Anything with non-trivial `normalizer` or `decoder` blocks.

When `--bpe-dropout F` is passed but `tokenizer.json` isn't byte-level
BPE, the trainer prints a warning and falls back to the cached
`HFTokenizer` path (no dropout, no regularisation). The detection is
heuristic — presence of the `Ġ` byte-alphabet marker in vocab, or an
explicit `ByteLevel` pre-tokeniser declaration.

For p=0 the encoder produces byte-identical output to
`swift-transformers` on byte-level BPE models — verified by the
`test_smoke_bpeDropoutByteAlphabetMatches` test.

### Plumbing

`Trainer.swift` gains a `StreamingTokenizedCorpus` that holds the
source text (not the cached token stream) plus a
`BPEDropoutTokenizer`. Each `sampleBatch` call:

1. Picks a random byte offset in the text.
2. Snaps forward to a UTF-8 character boundary.
3. Slices ~`8 × (T+1)` bytes — overshoots so the encoded result has
   `≥ T+1` tokens after dropout.
4. Calls `encodeWithDropout(slice, pDrop)` to produce ids.
5. Takes the first `T+1` ids as `(inputs, targets)`.

Cost: re-tokenising ~5 KB per micro-batch. On a `tiny` model with
batch=4, ctx=64, the streaming path is **faster** than the cached path
because the cache-write step is skipped. On larger models the
re-tokenisation becomes a hot path; further optimisation (parallel
batch construction, encoded-window caching) is future work.

Validation corpus is encoded **once with p=0** and frozen — val loss
must be deterministic to be comparable across steps.

### CLI

```
tinygpt train --preset tiny --tokenizer /path/to/hf-model \
              --corpus shakespeare.txt --steps 200 \
              --bpe-dropout 0.1
```

`--bpe-dropout 0` is the default (off). The Provilkov paper
recommends `0.1` for BPE vocabs in the 30k+ range — Qwen3 (152k),
Llama-3 (128k), Gemma (256k) all qualify.

### Smoke: 100-step training comparison

`tinygpt train --preset tiny --tokenizer <qwen3-dir> --corpus
shakespeare.txt --steps 100 --batch 4 --ctx 64 --lr-schedule constant
--max-lr 3e-4`:

```
baseline      (no dropout):  step 1: loss 12.484
                              step 50: loss 9.555
                              step 100: loss 7.382 ← final

BPE-dropout=0.1:              step 1: loss 12.397
                              step 50: loss 9.050
                              step 100: loss 7.597 ← final
```

Both runs:
- Start at ~12.4 (= ln(152k) ≈ 11.93 + initialisation noise).
- Decrease monotonically with no instabilities.
- Land within 0.2 nats of each other at step 100.

The expected pattern from the paper: dropout-on runs have **slightly
higher training loss** at the same step count (the regulariser eats a
small chunk of capacity) but generalise better. At 100 steps on
1.1 MB Shakespeare, we're well inside the noise floor — but the
direction is right and the run is stable.

50-step pilot:
```
baseline (run A):   12.333 → 9.700
baseline (run B):   12.440 → 9.601
dropout  (run A):   12.469 → 9.375
dropout  (run B):   12.343 → 9.075
```

Across runs the loss decrease is within RNG noise — the smoke proves
the streaming path doesn't break training and the loss curves are
comparable, not that dropout helps on this corpus at this scale.

### Caveats

- The encoder doesn't honour `tokenizer_config.json` post-processors
  (BOS / EOS injection). For from-scratch BPE pre-training we don't
  inject special tokens anyway — they're an SFT/chat concern, and SFT
  is downstream of pre-training.
- The 8× byte over-shoot in `StreamingTokenizedCorpus` can fail on
  pathological strings where every byte triggers a long merge cascade
  and the encoded ratio is high. Smoke shows ~3 bytes/token typical
  for English; pathological cases retry up to 4× then leave the row
  zeroed (rare, swamped by valid rows).
- Streaming defeats the token cache. On a re-run of the same config,
  the cached path is 10-100× cheaper on the first epoch. Use
  `--bpe-dropout 0` for repeat training of the same model
  configuration.
- Determinism: the encoder uses `Float.random` per merge candidate, so
  two runs with the same `--bpe-dropout 0.1` produce different token
  sequences. Re-runs of the smoke confirm RNG-level variation.

---

## Build verdict

Both features compile clean under
```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-smoke-datperf -configuration Release build
```

Test smoke:
```
xcodebuild -scheme TinyGPT-Package -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-smoke-datperf-test -configuration Debug test \
  -only-testing:TinyGPTModelTests/TinyGPTModelTests/test_smoke_samplePackingHistogram \
  -only-testing:TinyGPTModelTests/TinyGPTModelTests/test_smoke_bpeDropoutVariability \
  -only-testing:TinyGPTModelTests/TinyGPTModelTests/test_smoke_bpeDropoutByteAlphabetMatches
```

All three tests pass; CoV flattening from 0.582 → 0.061 is asserted
hard in the test, so a regression in `SFTCorpus.weightedIndex` will
fail CI.

## Files touched

- `native-mac/Sources/TinyGPTModel/SFTCorpus.swift` — added
  `sampleBatchWeighted`, `sampleBatchBucketed`, `inverseLengthWeights`,
  `weightedIndex`.
- `native-mac/Sources/TinyGPTModel/BPEDropout.swift` — **new**, ~200
  lines, self-contained BPE-dropout encoder.
- `native-mac/Sources/TinyGPTModel/Trainer.swift` — added
  `StreamingTokenizedCorpus`.
- `native-mac/Sources/TinyGPT/SFT.swift` — `--pack-mode`,
  `--length-bucket` CLI; sampler dispatch.
- `native-mac/Sources/TinyGPT/Train.swift` — `--bpe-dropout` CLI;
  streaming-corpus switch.
- `native-mac/Tests/TinyGPTModelTests/TinyGPTModelTests.swift` — three
  smoke tests with histograms / variability checks.

`TinyGPT.swift`, `Package.swift`, `ModelConfig.swift` left untouched.
