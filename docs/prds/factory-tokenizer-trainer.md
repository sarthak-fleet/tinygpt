---
name: tinygpt tokenize-train — domain-specific BPE tokenizer trainer
status: shipped-2026-06-07
owner: unassigned
created: 2026-06-07
priority: P2
---

# PRD — `tinygpt tokenize-train`

## 2026-06-07 ship note

`tinygpt tokenize-train` is implemented as a Swift wrapper over a repo-local
Rust helper in `scripts/tokenizer-trainer/` using HuggingFace `tokenizers`.

Shipped:
- `scripts/tokenizer-trainer/` Rust crate
- `tinygpt tokenize-train --corpus ... --vocab-size ... --out tokenizer.json`
- BPE tokenizer JSON output compatible with HF `tokenizer.json`

`cargo check`, `cargo build --release`, and a tiny end-to-end smoke passed.
`unigram` remains out of v1; use `bpe`.

## Goal

Add a `tokenize-train` subcommand that trains a custom BPE / SentencePiece
tokenizer on a user's domain corpus. Wraps the HuggingFace `tokenizers`
Rust crate.

Use case: when SmolLM2 BPE fragments domain vocabulary (medical jargon,
financial tickers like `$AAPL`, language not in the original corpus),
training a domain tokenizer can give 20-40% sequence-length reduction
+ sharper inductive bias.

## Why ship

Tokenizer choice is one of the few hardcoded preprocessing steps left.
Domain-specific tokenizers are real wins for niche corpora (covered
in the 2026-06-06 strategy doc tokenization section).

## Scope — in

### CLI

```
tinygpt tokenize-train \
    --corpus medical.txt \                    # one doc per line or large blob
    --vocab-size 32000 \
    --model-type bpe \                        # bpe | unigram | char
    --special-tokens "<bos>,<eos>,<pad>" \
    --out my-tokenizer.json                   # HF tokenizers format
```

### Implementation

Wrap the `tokenizers` Rust crate. Build a small Rust binary at
`scripts/tokenizer-trainer/` (matches the existing pattern of
`hf-downloader/`, `parquet-decoder/`).

CLI is a thin shell wrapping the Rust binary; emits an HF-compatible
`tokenizer.json` that loads cleanly into Swift via swift-transformers.

## Scope — out

- WordPiece (BERT-style)
- Multilingual / multimodal tokenizers
- Tokenizer alignment between two models (different problem)

## Acceptance

1. Train on FineWeb-Edu sample → 32K-vocab tokenizer
2. Verify: tokenizer.json loads via `tinygpt sample --tokenizer
   my-tokenizer.json model.tinygpt`
3. Sequence length on domain text ≤ SmolLM2 baseline (better)

## File paths

| Action | Path |
|---|---|
| **create** | `scripts/tokenizer-trainer/` Rust crate |
| **create** | `native-mac/Sources/TinyGPT/TokenizeTrain.swift` — thin wrapper |

## Estimated effort

**~2-3 days.**

## Source

- HF tokenizers: https://github.com/huggingface/tokenizers
- SentencePiece: https://github.com/google/sentencepiece
