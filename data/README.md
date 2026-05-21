# data/ — datasets

## Use plain text only

Good first sources: your own notes/blog posts, public-domain text, a small
technical blog, a small codebase, short stories, README files.

Avoid initially: PDFs, HTML scraping, large websites, mixed-author corpora,
private messages, social scraping, large multilingual datasets.

## Dataset sizes (from-scratch training)

| Stage        | Size        | Purpose                     |
| ------------ | ----------- | --------------------------- |
| Smoke test   | 1–10 KB     | Check loss decreases        |
| Overfit test | 10–100 KB   | Prove gradients are correct |
| Demo dataset | 500 KB–5 MB | Realistic browser demo      |
| Stress test  | 10–100 MB   | Later only                  |

Byte-level: 1 byte ≈ 1 token, so a 1 MB text file ≈ 1 million tokens.

## LoRA datasets (structured examples)

For LoRA, do not dump raw text — build task-style examples and write JSONL:

```json
{"task":"continuation","title":"...","prefix":"...","completion":"..."}
{"task":"rewrite","draft":"...","completion":"..."}
{"task":"title","excerpt":"...","completion":"..."}
```

Aim for 300–1,000 clean examples from one consistent author. Quality beats
quantity. See `../docs/lora_guide.md` and `../docs/evaluation.md`.

## Files

- `dataset_builder.py` — turns raw text into token arrays / JSONL + manifests (stub)
- `examples/` — small sample corpora live here

`.gitignore` keeps bulk corpora out of git; `examples/` is kept.
