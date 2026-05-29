# Pretraining

The first of three training phases. See [`docs/training/index.md`](index.md)
for the overview, [`docs/training/sft.md`](sft.md) and
[`docs/training/dpo.md`](dpo.md) for what comes next.

## What it does

Given an enormous stream of raw text, predict the next token everywhere.
Loss is averaged over every position; gradients flow through every
token. The model learns grammar, vocabulary, world facts, and a
distribution over what humans tend to write.

## Math

```
L_pretrain = - (1 / N) * Σ_t  log P(x_{t+1} | x_1 … x_t)
```

where `x_1 … x_N` are the corpus tokens. Averaging over a single
contiguous corpus makes loss directly comparable across runs.

## What it needs

| Thing | Why | Where it lives |
|---|---|---|
| **Large text corpus** | ~5-20× more tokens than the model has parameters (Hoffmann/Chinchilla) | Streamed from HuggingFace via `python_ref/fetch_hf_corpus.py` |
| **BPE tokenizer** | Byte-level wastes ~4× the compute at the same coverage | `--tokenizer <hf-dir>` pointing at any HF model directory |
| **Long-run infrastructure** | A crash at hour 22 of 26 shouldn't lose 22 hours | Tier 0 safety nets in `tinygpt train`: resume, atomic save-every, SIGINT-flushes-final |
| **bf16 training** | 2× memory savings → 2× larger effective batch. See [`docs/memory_tradeoffs.md`](../memory_tradeoffs.md). | `--dtype bfloat16` |
| **Gradient accumulation** | Effective batch larger than memory budget. See [`docs/memory_tradeoffs.md`](../memory_tradeoffs.md). | `--accum N` |

## Reproduce

```bash
# 1. Stream ~500M tokens of high-quality educational web text.
source python_ref/.venv/bin/activate
python python_ref/fetch_hf_corpus.py \
    --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
    --split train --target-tokens 500M \
    --out /tmp/fineweb-edu-500M.txt

# 2. Pretrain Mega-bf16 (76M body + 25M token embedding = ~100M total).
#    B=4 × accum=4 × ctx=1024 = effective batch 16 at ~2 GB GPU memory.
#    ~23 hours on M5 Pro / 48 GB.
cd native-mac
caffeinate -di .xcode-build/Build/Products/Debug/tinygpt train \
    --preset mega \
    --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt \
    --out /tmp/mega-fineweb.tinygpt \
    --dtype bfloat16 \
    --batch 4 --accum 4 --ctx 1024 \
    --steps 30500 \
    --lr-schedule cosine --warmup 1000 \
    --max-lr 6e-4 --min-lr 6e-5 \
    --val-split 0.005 --val-every 500 --save-every 1000
```

## Expected outcome at our scale

| Tokens trained on | Predicted val loss | What it looks like |
|---|---:|---|
| 5 M (Tiny demo) | 4.9 | gibberish, fragments |
| 50 M | 4.0 | real words, broken grammar |
| **500 M (this run)** | **3.0-3.5** | coherent fragments, GPT-2-124M-class |
| 1.5 B (Chinchilla floor) | 2.5 | useful base, post-trainable |
| 5 B | 2.0 | Pythia-1.4B-class base |

A "good pretrain" is anywhere from loss ~2.5 to ~3.5. Below that, the
base is becoming useful on its own; above it, post-training is doing
nearly all the work.

## Background reading

- Hoffmann et al., 2022 (Chinchilla scaling laws).
- Kaplan et al., 2020 (Kaplan scaling laws).
