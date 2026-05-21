"""
dataset.py — byte-level dataset pipeline (Phase 1).

STATUS: documented stub. No implementation yet.

Pipeline:
    raw text
      -> UTF-8 bytes
      -> integer token array (1 byte = 1 token, vocab 0..255)
      -> train/val split (90/10)
      -> random batch sampler
      -> (x, y) pairs

Batch construction (context_length C):
    x = tokens[i : i + C]
    y = tokens[i + 1 : i + 1 + C]

Dataset manifest (write alongside the token array — the hash makes checkpoint
resume reproducible):
    {
      "dataset_id":  "sha256 of raw bytes",
      "name":        "my_blog_posts.txt",
      "raw_bytes":   1249301,
      "token_count": 1249301,
      "tokenizer":   "byte-v1",
      "train_split": 0.9,
      "val_split":   0.1,
      "seed":        42
    }

Guide: docs/model_guide.md  ("Data requirements", "Dataset pipeline")

TODO(phase-1): load text -> bytes -> token array; compute sha256.
TODO(phase-1): get_batch(split) -> (x, y) tensors of shape [batch_size, C].
TODO(phase-1): write/read the dataset manifest JSON.
"""
