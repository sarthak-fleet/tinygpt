---
name: B10 quality classifier on pretrain data
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B10)
related_prds: B11-wsd-schedule.md (paired training-quality win)
---

# PRD — FineWeb-Edu-style quality classifier + corpus filter

## Goal

Ship `tinygpt quality-classify <corpus.txt>` that scores every document on
an educational-quality axis, writes the per-doc scores to a sidecar, and
filters to a top-X% subset for downstream pretraining. Mirrors the
FineWeb-Edu recipe ([Penedo et al. 2024](https://arxiv.org/abs/2406.17557))
scaled down for the corpora TinyGPT trains on.

The lift, per the FineWeb-Edu paper: top-quality filtering at fixed token
budget improved downstream eval scores 2–4× more than scaling tokens alone
on small models — by far the highest "quality per dev-day" knob remaining
in our pretrain pipeline.

## Why now

- Every other corpus-quality lever has shipped (dedupe exact + MinHash,
  PPL filter via the corpus-anchored BoN scorer). The remaining quality
  axis is *intrinsic per-document quality*, which only a classifier
  surfaces.
- A1 specialist training is gated on having a quality-filtered corpus —
  pretrain runs on raw web text waste compute on low-signal docs.
- The classifier itself is trivial — a fastText-class softmax on bag-of-
  ngram embeddings, trained on a few thousand labeled examples. Most of
  the design work is already done in the FineWeb-Edu paper.

## Scope — in

- `Sources/TinyGPT/QualityClassify.swift` — new subcommand. Trains a
  small bag-of-ngram + softmax classifier on a labeled JSONL
  (`{text, label}` where `label ∈ {0, 1, 2, 3, 4, 5}` for FineWeb-Edu
  buckets) and scores a target corpus.
- Three modes: `--train <labeled.jsonl> --out <model.qcls>`, `--score
  <corpus.txt> --model <model.qcls> --out-scores <scores.jsonl>`, and
  `--filter <corpus.txt> --scores <scores.jsonl> --threshold N
  --out <filtered.txt>`.
- Sidecar format: line-aligned with the input corpus, one JSON per line
  with `{doc_id, score: float}`. Cheap to grep + cheap to re-threshold
  without re-scoring.
- Use the existing tokenizer for ngram extraction (don't add a
  vocabulary). 3-5 char ngrams (FineWeb-Edu default).
- Wire into `tinygpt dedupe`'s output: typical pipeline becomes
  `dedupe → quality-classify --score → quality-classify --filter`.

## Scope — out (explicit)

- **Active learning loop** to bootstrap labels with a teacher LLM —
  ship V1 with manually labeled data + the published FineWeb-Edu labels
  (they release the classifier weights too; we could just port those
  if the bag-of-ngram features line up).
- **Multi-axis quality** — toxicity, code-fraction, language are
  separate filters with separate PRDs (existing `pii-toxicity-filter`
  PRD covers one).
- **Continuous (regression) scoring** — V1 uses the 6-bucket softmax.
  Continuous comes later if it's needed.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/QualityClassify.swift` | new — subcommand entry point |
| `Sources/TinyGPTModel/QualityClassifier.swift` | new — classifier struct + train/score methods |
| `Sources/TinyGPT/TinyGPT.swift` | add `case "quality-classify"` (maintainer merge) |
| `evals/quality-filter-smoke.sh` | new — 100-doc smoke (train on labeled, score on tinystories shard, top-20% should overlap with manual selection ≥80%) |
| `docs/PLAN.md` | flip B10 ⬜ → ✅ on ship; note delta on a re-pretrain at fixed token budget |

## Don't touch

- `Sources/TinyGPT/TinyGPT.swift` (one switch case — maintainer)
- `docs/PLAN.md` (one status flip — maintainer)
- `Sources/TinyGPTModel/QuantizedLinear.swift` and anything in
  `TinyGPTServe/` — orthogonal subsystem.

## Acceptance criteria

- [ ] `tinygpt quality-classify --train labeled.jsonl --out model.qcls`
  fits a classifier on a held-out validation set with macro-F1 ≥ 0.55
  on the 6-bucket task (FineWeb-Edu reports ~0.6; we accept slightly
  lower since we're not finetuning the embedding).
- [ ] `tinygpt quality-classify --score corpus.txt --model model.qcls
  --out-scores scores.jsonl` runs at ≥ 5 MB/s on M5 Pro CPU.
- [ ] `tinygpt quality-classify --filter --threshold 3 ...` produces a
  filtered corpus whose downstream PPL on a held-out eval set is ≥ 2pp
  better than random sampling at the same token count. Reference
  comparison: pretrain a 22M model 5K steps on filtered vs random;
  measure PPL on shakespeare.eval.
- [ ] Smoke script `evals/quality-filter-smoke.sh` passes in CI.

## Reference patterns to copy

- `Sources/TinyGPT/Dedupe.swift` — the doc-level pipeline structure
  (read JSONL/txt → per-doc compute → emit sidecar) is the closest
  template. Same I/O shape, different scoring fn.
- `Sources/TinyGPTModel/LinearProbe.swift` — small classifier head
  with sklearn-style fit/predict. The bag-of-ngram embedding goes in
  front of an existing Linear softmax.
- FineWeb-Edu released weights at
  [HuggingFaceTB/fineweb-edu-classifier](https://huggingface.co/HuggingFaceTB/fineweb-edu-classifier).
  Their tokenizer + classifier head are small (~150 MB); a
  port-and-finetune is a cheaper first cut than train-from-scratch.

## Open questions

- Source of training labels for V1: port the fineweb-edu-classifier
  weights vs. relabel a TinyStories-sized subset with a local teacher
  (Qwen3-9B-as-judge through the shipped `tinygpt judge` shim, scored
  on the same 6-bucket rubric)? **Recommendation:** port the weights
  first, validate the macro-F1 on a small TinyGPT-corpus eval,
  re-finetune only if needed.
- Whether to add `--quality-floor N` directly to `tinygpt train` so the
  filter is applied lazily at batch time. **Recommendation:** ship the
  offline pipeline first; in-loop is V2 once we have the filter
  on disk.
