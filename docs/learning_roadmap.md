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
External sources:
- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) (chs 1-4: vectors, matmul)
- [3Blue1Brown — Backpropagation](https://www.3blue1brown.com/lessons/backpropagation)
- [Karpathy — Spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) (builds a tiny autograd by hand)
Concepts: `ml-math`, `ml-gradient-descent`, `ml-backprop`, `ml-softmax-xent`, `ml-adamw`

### Phase 2 — Language modeling basics
Goal: understand next-token prediction.
Learn: tokenization, byte/char-level tokens, context windows, causal masking,
sampling, temperature, top-k, perplexity.
Do: build a bigram model and a char-level MLP; generate text; measure train/val loss.
Key idea: a language model is not "answering" — it is predicting the next token.
External sources:
- [3Blue1Brown — But what is a GPT?](https://www.3blue1brown.com/lessons/gpt) (25 min big-picture)
- [Karpathy — makemore series](https://github.com/karpathy/makemore) (bigram → char-level MLP → transformer, notebook-by-notebook)
- GPT-1/2/3 papers: [Radford et al. 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Brown et al. 2020](https://arxiv.org/abs/2005.14165)
Concepts: `ml-tokenization`, `ml-language-modeling`, `ml-sampling`

### Phase 3 — Transformer / GPT internals
Goal: understand the architecture.
Learn: embeddings, position embeddings, self-attention, causal attention,
multi-head attention, layernorm, residuals, MLP blocks, GELU, logits.
Do: build one transformer block, then a full TinyGPT; train 0.8M on a small
corpus; overfit 10 KB; generate samples.
External sources:
- [Attention is All You Need — Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) (the original transformer)
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/) (Vaswani paper, code-annotated)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (the visual canon)
- [Karpathy — Let's build GPT, from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [RoPE — Su et al. 2021](https://arxiv.org/abs/2104.09864) (rotary position embeddings — what modern models use instead of learned positions)
- Implemented at: `python_ref/model.py`, `wasm/src/attention.cpp`, `webgpu/attention_fa2.wgsl`.
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
External sources:
- [LoRA — Hu et al. 2021](https://arxiv.org/abs/2106.09685) (the original)
- [QLoRA — Dettmers et al. 2023](https://arxiv.org/abs/2305.14314) (int4 base + fp16 LoRA)
- [DoRA — Liu et al. 2024](https://arxiv.org/abs/2402.09353) (magnitude + direction decomposition)
- Implemented at: `python_ref/lora.py`; mechanics + maths in [`lora_guide.md`](lora_guide.md).
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
External sources:
- [MDN — WebAssembly concepts](https://developer.mozilla.org/en-US/docs/WebAssembly/Concepts)
- [MDN — Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers)
- [MDN — OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
- [emscripten.org](https://emscripten.org/) (pthreads + SAB build flags)
- Implemented at: `browser/src/worker.ts`, `wasm/src/`, [`browser_notes.md`](browser_notes.md).
Concepts: `ml-browser-runtime`

### Phase 8 — WebGPU
Goal: accelerate tensor-heavy operations.
Learn: WGSL, GPU buffers, bind groups, compute pipelines, command encoders,
workgroups, matmul/softmax/attention kernels, device limits, buffer sharding.
Do: WebGPU matmul → CPU parity → linear forward → linear backward → attention
scores → softmax → attention output → optimizer. Start with matmul only.
External sources:
- [WebGPU Fundamentals](https://webgpufundamentals.org/) (compute articles)
- [WebGPU spec (W3C)](https://www.w3.org/TR/webgpu/)
- [WGSL spec](https://www.w3.org/TR/WGSL/)
- [FlashAttention — Dao et al. 2022](https://arxiv.org/abs/2205.14135) and [FlashAttention-2 — Dao 2023](https://arxiv.org/abs/2307.08691) (the online-softmax + recomputation trick we use in WGSL)
- Implemented at: `webgpu/matmul_blocked.wgsl`, `webgpu/attention_fa2.wgsl`; perf walk-through in [`perf_quest.md`](perf_quest.md).
Concepts: `ml-webgpu`

### Phase 9 — Evaluation & safety
Goal: know what the model is actually doing.
Learn: validation loss, held-out prompts, qualitative evaluation, memorization
testing, style similarity, hallucination, copying risk, data leakage.
Do: a suite producing base / prompt-only / LoRA / LoRA+retrieval outputs.
See [`validation_report.md`](validation_report.md) (evaluation-and-safety appendix).
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
