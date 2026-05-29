# The three phases of training — pretrain, SFT, DPO

A modern useful language model is the product of three distinct training
phases, each with its own dataset shape, loss function, and goal. This
section walks the whole pipeline as it exists in TinyGPT today, with the
exact commands to reproduce each step.

Three phases, in order:

| Phase | Goal | Dataset shape | Loss | Compute share at labs |
|---|---|---|---|---:|
| **[Pretrain](pretrain.md)** | Learn the structural prior of language | Continuous text | Causal next-token cross-entropy | ~50-70% |
| **[SFT](sft.md)** (supervised fine-tune) | Follow instructions in a chat format | `{prompt, response}` pairs | Same CE, but masked to response tokens only | ~5-15% |
| **[DPO](dpo.md)** (direct preference optimization) | Prefer better responses over worse ones | `{prompt, chosen, rejected}` triplets | Log-sigmoid of policy/reference log-ratio difference | ~10-30% |

The first phase produces a base model that *can complete text*; the
second teaches it to *respond to instructions*; the third teaches it to
*prefer good responses over bad*. Lab-scale models spend ~70% of total
compute on pretraining and ~30% on SFT+DPO combined; at our scale, the
ratio inverts (pretrain is cheap-but-data-limited, post-training is the
multiplier).

## End-to-end pipeline

The three phases compose into one workflow:

```bash
# Phase 1 — pretrain on FineWeb-edu 500M (~23 hr).
caffeinate -di tinygpt train --preset mega --tokenizer /tmp/smollm2 \
    --corpus /tmp/fineweb-edu-500M.txt --out /tmp/mega.tinygpt \
    --dtype bfloat16 --batch 4 --accum 4 --ctx 1024 \
    --steps 30500 --lr-schedule cosine --warmup 1000 \
    --save-every 1000 --val-split 0.005 --val-every 500

# Phase 2 — SFT on Dolly (~30 min).
tinygpt sft /tmp/mega.tinygpt --data /tmp/dolly.jsonl \
    --template chatml --steps 500 --out /tmp/mega-sft.lora

# Phase 3 — DPO on UltraFeedback (~30 min).
tinygpt dpo /tmp/mega.tinygpt --data /tmp/ultrafeedback.jsonl \
    --template chatml --beta 0.1 --steps 500 \
    --out /tmp/mega-dpo.lora

# Sample with the full stack — base + SFT + DPO adapters.
tinygpt sample /tmp/mega.tinygpt \
    --lora /tmp/mega-sft.lora --lora-weight 1.0 \
    --lora /tmp/mega-dpo.lora --lora-weight 1.0 \
    --prompt "<|im_start|>user\nExplain DPO simply.<|im_end|>\n<|im_start|>assistant\n"
```

A weekend's worth of compute on one M5 Pro produces a 100M-param
instruction-following model that scores ~2.5-3.5 on TinyStories PPL
and follows simple conversational prompts in the ChatML format. Not
GPT-quality — but a working artifact end-to-end.

## Background reading

- **Pretraining scaling laws**: Hoffmann et al., 2022 (Chinchilla);
  Kaplan et al., 2020 (Kaplan scaling laws).
- **SFT response-only loss**: standard practice since GPT-3
  fine-tuning. The mechanic of masking to the response is described
  cleanly in the [Alpaca paper](https://arxiv.org/abs/2303.18223)
  appendix.
- **DPO**: Rafailov et al., 2023 ("Direct Preference Optimization:
  Your Language Model is Secretly a Reward Model"), NeurIPS 2023.
  The original paper; the closed-form derivation in §4 is the math
  we implement.
- **Why this order**: the [LIMA paper](https://arxiv.org/abs/2305.11206)
  argues most "alignment" is shallow — pretraining does the heavy
  lifting, SFT teaches format, DPO polishes. Our pipeline structure
  matches that thesis.
