# Supervised fine-tuning (SFT)

The second of three training phases. See [`pretrain.md`](pretrain.md)
for what comes before and [`dpo.md`](dpo.md) for what comes after.

## What it does

Given a base that can complete text, teach it to follow instructions.
Same model, same forward pass, same cross-entropy loss — but the data
is `{instruction, response}` pairs templated through a chat format
(`<|im_start|>user … <|im_start|>assistant …`), and the loss is
masked to score **only the response tokens**.

## Why masking matters

Without the response-only mask, the loss includes the instruction
tokens. The gradient signal pushes the model toward predicting the
instruction back to itself — useless. With the mask, only the response
positions contribute, and the model learns "given THIS prompt, produce
THAT response."

```
L_SFT = - (1 / |R|) * Σ_{t in R}  log P(x_{t+1} | x_1 … x_t)
```

where `R` is the set of response positions. Identical math to pretrain
except for the index set.

## Templates

Three are supported by `tinygpt sft --template`:

```
chatml  (default, matches SmolLM2 / Qwen tokenizers)
  <|im_start|>user
  Capital of France?<|im_end|>
  <|im_start|>assistant
  Paris.<|im_end|>

alpaca
  ### Instruction:
  Capital of France?

  ### Response:
  Paris.

llama
  [INST] Capital of France? [/INST] Paris.
```

Use whatever template matches the tokenizer the base was trained
against. SmolLM2's tokenizer treats ChatML markers as single tokens; the
others would tokenize them as raw text.

## What datasets to use

| Dataset | Size | Style | Notes |
|---|---:|---|---|
| `databricks/databricks-dolly-15k` | 15K | hand-written instructions | High quality, small. Good first run. |
| `HuggingFaceH4/no_robots` | 10K | hand-written, diverse | Pairs well with Dolly |
| `tatsu-lab/alpaca` | 52K | GPT-generated | Broader, lower per-pair quality |
| `OpenAssistant/oasst1` | ~10K conversations | multi-turn human | Use for chat-shape SFT |

For first runs, Dolly is the canonical pick. Full catalog with URLs
and licenses in [`docs/roadmap/datasets.md`](../roadmap/datasets.md).

## Reproduce

```bash
# Tokenize Dolly into JSONL (one record per line).
python python_ref/fetch_hf_corpus.py \
    --dataset databricks/databricks-dolly-15k \
    --target-tokens 50M \
    --out /tmp/dolly.jsonl
# Hand-massage into {instruction, response} JSONL (the fetcher writes
# raw text; for SFT we want the structured form).

# SFT on top of the pretrained base. Adapter is rank-4 LoRA — adapter
# file is ~MB, base stays frozen.
.xcode-build/Build/Products/Debug/tinygpt sft \
    /tmp/mega-fineweb.tinygpt \
    --data /tmp/dolly.jsonl \
    --template chatml \
    --rank 4 --alpha 8 \
    --steps 500 \
    --out /tmp/mega-sft.lora
```

## How to know it worked

Sample with and without the adapter and compare:

```
# Base only — completes text but doesn't follow instructions
tinygpt sample /tmp/mega-fineweb.tinygpt --prompt "User: What is 2+2?" --tokens 50

# With SFT adapter — responds in the expected format
tinygpt sample /tmp/mega-fineweb.tinygpt --lora /tmp/mega-sft.lora \
    --prompt "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n" \
    --tokens 50
```

The masked-tokens count printed by `tinygpt sft` tells you how much
signal you actually trained on — for Dolly that's ~1.5 M response
tokens, vs ~3 M total prompt+response tokens. Half the data is
"context for the loss, not scored."

## Background reading

SFT response-only loss is standard practice since GPT-3 fine-tuning. The
mechanic of masking to the response is described cleanly in the
[Alpaca paper](https://arxiv.org/abs/2303.18223) appendix.
