# Evaluation & safety

Phase 9 — and ongoing. Know what the model is *actually* doing.

---

## 1. Required correctness tests

The full table is in `../tests/README.md`. Summary:

| Test                | Purpose                                      |
| ------------------- | -------------------------------------------- |
| Tokenizer roundtrip | bytes → text → bytes is lossless             |
| Shape tests         | every layer returns the expected shape       |
| Loss sanity         | random model loss near `ln(256) ≈ 5.54`      |
| Tiny overfit        | model overfits 1–10 KB repeated text         |
| Gradient check      | finite-difference check on a tiny layer      |
| PyTorch parity      | ported forward matches the PyTorch reference |
| Checkpoint reload   | same loss after save + reload                |
| Sampling fixed seed | deterministic generation for a fixed seed    |
| Browser refresh     | run resumes after a page reload              |

**The most important test:** can it overfit a tiny repeated dataset? If not, do
not scale — the model, backprop, or data pipeline is broken.

---

## 2. The evaluation matrix (LoRA)

Always run four comparisons:

```
A. Base model only
B. Base model + prompt examples (few-shot)
C. Base model + LoRA
D. Base model + LoRA + retrieval
```

Typical outcome:

| Setup            | Behaviour                       |
| ---------------- | ------------------------------- |
| Base only        | generic                         |
| Few-shot prompt  | immediate style improvement     |
| LoRA             | stronger tone/style adaptation  |
| Retrieval + LoRA | best practical quality          |

LoRA teaches **style**. Retrieval supplies **context/facts**.

> If LoRA does not beat few-shot prompting, the adapter was not worth training.

---

## 3. Memorization test

Tiny models memorize; LoRA adapters can memorize too.

```
Take the first 30–50 tokens of a training example.
Ask the model to continue.
Check whether it reproduces the rest verbatim.
```

If it copies too much: train fewer steps, lower rank, lower learning rate, add
dropout, deduplicate data, increase dataset size, avoid repeated text.

---

## 4. Qualitative questions

For a style-adaptation project, ask:

- Does it sound more like the corpus?
- Does it copy exact training text?
- Does it hallucinate facts?
- Does retrieval improve grounding?
- Does LoRA actually beat few-shot prompting?

---

## 5. Backend parity (Phase 4–5)

- **WASM vs PyTorch** — the WASM forward must match the Python reference within
  tolerance before you trust WASM training.
- **WebGPU vs WASM** — each WebGPU kernel must match the WASM kernel within
  tolerance before it joins the pipeline. Start with matmul.

---

## 6. Safety notes

- Style ≠ intelligence. LoRA on blogs learns tone, format, phrasing, argument
  rhythm, vocabulary — not truth, judgment, current beliefs, or reliable
  reasoning.
- Watch for data leakage and copying risk, especially with tiny models.
- Do not redistribute adapters trained on a living author's writing.

---

## Deliverable

A small evaluation suite producing: base output, prompt-only output, LoRA
output, and LoRA + retrieval output — for the same held-out prompts.
