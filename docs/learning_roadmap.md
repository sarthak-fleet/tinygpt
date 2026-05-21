# Learning roadmap

The path to learning the whole stack without drowning in complexity. This is the
curriculum; `model_guide.md`, `lora_guide.md`, and `browser_notes.md` are the
implementation detail.

This roadmap is also mirrored into the `swe-interview-prep` fleet project as 19
FSRS-tracked `ml-*` concepts (`swe-interview-prep/docs/TINYGPT_LEARNING_PATH.md`).

---

## The 9 phases

### Phase 1 — Core ML foundations
Goal: understand what the model is optimizing.
Learn: vectors, matrices, dot products, matmul, softmax, cross entropy, gradient
descent, backpropagation, Adam/AdamW, train/val split, overfitting.
Do: implement softmax and cross entropy from scratch; train logistic regression
or a tiny MLP.
Ready when: you can explain forward pass, loss, backward pass, and optimizer step
without handwaving.
Concepts: `ml-math`, `ml-gradient-descent`, `ml-backprop`, `ml-softmax-xent`, `ml-adamw`

### Phase 2 — Language modeling basics
Goal: understand next-token prediction.
Learn: tokenization, byte/char-level tokens, context windows, causal masking,
sampling, temperature, top-k, perplexity.
Do: build a bigram model and a char-level MLP; generate text; measure train/val loss.
Key idea: a language model is not "answering" — it is predicting the next token.
Concepts: `ml-tokenization`, `ml-language-modeling`, `ml-sampling`

### Phase 3 — Transformer / GPT internals
Goal: understand the architecture.
Learn: embeddings, position embeddings, self-attention, causal attention,
multi-head attention, layernorm, residuals, MLP blocks, GELU, logits.
Do: build one transformer block, then a full TinyGPT; train 0.8M on a small
corpus; overfit 10 KB; generate samples.
Concepts: `ml-embeddings`, `ml-self-attention`, `ml-multi-head`, `ml-transformer-block`

### Phase 4 — Training & debugging
Goal: know why training fails.
Learn: initialization, learning rate, gradient clipping, NaNs, exploding
gradients, under/overfitting, batch size effects, validation loss,
checkpointing, seed reproducibility.
Do: build the test suite — loss near `ln(vocab)`, tiny overfit, gradient check,
checkpoint reload, fixed-seed generation.
This is where most beginners lie to themselves. If loss is not falling on tiny
data, the model is broken — do not scale.
Concepts: `ml-training`, `ml-checkpointing`

### Phase 5 — LoRA & PEFT
Goal: understand parameter-efficient adaptation.
Learn: full fine-tuning, frozen base models, adapters, LoRA, rank, alpha, target
modules, adapter checkpoints, memorization, style-transfer limits.
Do: build a `LoRALinear`; inject into `q_proj`/`v_proj`; freeze base; train
adapter only; save/load adapter; compare base vs LoRA outputs.
Concepts: `ml-lora`

### Phase 6 — Data engineering for style adaptation
Goal: prepare clean text that teaches the model something specific.
Learn: text cleaning, deduplication, train/val split, dataset hashing, prompt
formatting, continuation/rewrite/title/Q&A examples, memorization tests.
Do: produce a clean JSONL dataset (300–1,000 examples) of structured tasks.
Concepts: `ml-data-engineering`

### Phase 7 — Browser systems
Goal: move training/inference into the browser safely.
Learn: TypedArray, ArrayBuffer, Web Workers, transferable objects, WASM,
Emscripten, WASM SIMD, OPFS, IndexedDB, storage quota, feature detection.
Do: a browser app that trains a tiny model in a Worker; UI stays responsive;
checkpoint survives reload.
Concepts: `ml-browser-runtime`

### Phase 8 — WebGPU
Goal: accelerate tensor-heavy operations.
Learn: WGSL, GPU buffers, bind groups, compute pipelines, command encoders,
workgroups, matmul/softmax/attention kernels, device limits, buffer sharding.
Do: WebGPU matmul → CPU parity → linear forward → linear backward → attention
scores → softmax → attention output → optimizer. Start with matmul only.
Concepts: `ml-webgpu`

### Phase 9 — Evaluation & safety
Goal: know what the model is actually doing.
Learn: validation loss, held-out prompts, qualitative evaluation, memorization
testing, style similarity, hallucination, copying risk, data leakage.
Do: a suite producing base / prompt-only / LoRA / LoRA+retrieval outputs.
See `evaluation.md`.
Concepts: `ml-evaluation`

---

## 12-week schedule

| Weeks | Build | Milestone |
| ----- | ----- | --------- |
| 1–2   | bigram LM, byte tokenizer, cross entropy, sampling | Generate text from a tiny LM |
| 3–4   | GPT block, attention, MLP, training loop, checkpointing, sampling | 0.8M TinyGPT overfits 10 KB |
| 5–6   | val loss, loss chart, fixed eval prompts, grad clipping, checkpoint reload, dataset manifest | Reliable, reproducible training harness |
| 7–8   | `LoRALinear`, adapter injection, frozen base, adapter checkpoint, base-vs-LoRA comparison | Tiny base model adapts to a small corpus |
| 9–10  | Worker training shell, WASM backend, WASM-SIMD build, OPFS checkpointing, capability panel | Tiny model trains in browser, UI not frozen |
| 11–12 | WebGPU matmul, CPU/WebGPU parity, linear-forward acceleration | One WebGPU kernel correct and faster |

---

## Traps to avoid

1. **Starting with the browser.** Right order: PyTorch reference → WASM
   reference → WebGPU acceleration.
2. **Too large a model.** First targets: 0.8M from scratch; 5M–15M frozen base
   for LoRA. Not 100M+.
3. **Confusing style with intelligence.** LoRA learns tone, format, phrasing,
   argument rhythm, vocabulary — not truth, judgment, or reliable reasoning.
4. **Not testing memorization.** Always test: training prefix → generated
   continuation. If it copies exact passages, reduce training or improve data.
5. **Skipping baselines.** Always compare base / few-shot / LoRA /
   LoRA+retrieval. If LoRA does not beat prompting, the adapter wasn't worth it.

---

## Final build order

```
1.  Byte-level 0.8M TinyGPT in Python
2.  Train from scratch on tiny text
3.  Save / load checkpoints
4.  Pretrain a 5M–15M base model outside the browser
5.  Add LoRA on q_proj / v_proj
6.  Fine-tune on clean blog-style examples
7.  Evaluate base vs prompt vs LoRA vs retrieval
8.  Port inference/training loop to a browser Worker
9.  Add the WASM backend
10. Add WebGPU matmul — only after correctness
```

## Project deliverables

1. Python TinyGPT reference
2. Python LoRA fine-tuning reference
3. Clean blog/style dataset builder
4. Browser TinyGPT demo
5. WASM backend
6. Adapter checkpoint format
7. Evaluation harness
8. WebGPU matmul prototype
9. Learning notes explaining every component
