"""
dataset_builder.py — turn raw text into training-ready data (Phase 1 / Phase 6).

STATUS: documented stub. No implementation yet.

Two outputs:

1. From-scratch training (Phase 1):
   raw text -> UTF-8 bytes -> token array (+ dataset manifest with sha256 hash).

2. LoRA fine-tuning (Phase 6):
   raw author text -> structured task examples -> JSONL, e.g.
     {"task":"continuation","title":...,"prefix":...,"completion":...}
     {"task":"rewrite","draft":...,"completion":...}
     {"task":"title","excerpt":...,"completion":...}
     {"task":"qa","question":...,"completion":...}

Cleaning steps for LoRA data: text cleaning, deduplication, train/val split,
dataset hashing, consistent prompt formatting.

Guide: docs/model_guide.md ("Dataset pipeline"), docs/lora_guide.md
       ("Fine-tuning data format", "Dataset sizes for LoRA")

TODO(phase-1): build_token_array(path) -> tokens + manifest.
TODO(phase-6): build_jsonl(examples) with dedup + train/val split + hashing.
"""
