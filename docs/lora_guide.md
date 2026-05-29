# LoRA guide — fine-tuning the tiny model

Phase 3. LoRA adapts an already-trained, frozen base model by training small
low-rank matrices inside selected linear layers.

Exact configs live in `configs/lora.json`.

---

## 1. What LoRA does

Original linear layer:

```
y = xW
```

LoRA layer:

```
y = xW + scale * xAB

W     = frozen base weight
A, B  = trainable low-rank matrices
rank  = r
scale = alpha / r
```

LoRA freezes the pretrained weights and injects trainable low-rank matrices,
greatly reducing the number of trainable parameters.

---

## 2. Why LoRA fits this project

Full fine-tuning updates every weight; LoRA updates only small adapters. In the
browser, optimizer state and gradients dominate memory, so this matters:

| Base model | Full trainable | LoRA trainable |
| ---------: | -------------: | -------------: |
|         5M |             5M |      ~50K–500K |
|        15M |            15M |       ~100K–1M |
|        30M |            30M |       ~250K–2M |

---

## 3. Recommended target

Use a custom pretrained base first:

```
5M–15M params, byte-level or small BPE tokenizer, context length 256,
trained outside the browser, loaded into the browser frozen.
```

Do **not** start with a 100M+ model — that turns a learning project into a
systems-pain project.

---

## 4. Target modules

Start with `q_proj`, `v_proj`. Then expand to `q_proj, k_proj, v_proj, o_proj`.
Then maybe `mlp_up, mlp_down`. Do not LoRA every module on day one — you overfit
faster and make debugging harder.

First run (`configs/lora.json` → `first_run`):

```json
{ "rank": 4, "alpha": 8, "dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "learning_rate": 0.0001, "batch_size": 4,
  "context_length": 256, "steps": 500 }
```

More capacity: `rank 8, alpha 16, target ["q_proj","v_proj","o_proj"], steps 1000`.

---

## 5. Implementation

For a linear layer:

```
x: [B, T, d_in]
W: [d_in, d_out]  frozen
A: [d_in, r]      trainable
B: [r, d_out]     trainable
```

Forward:

```
base = x @ W
lora = (x @ A) @ B * (alpha / r)
y    = base + lora
```

Initialisation: `A = small random values`, `B = zeros`. With `B = 0` the LoRA
contribution is 0 at step 0, so the model initially behaves exactly like the
base model.

---

## 6. The important backprop detail

Freezing `W` does **not** mean stopping gradients through the layer. You still
backpropagate through frozen layers so lower LoRA layers can learn.

For a LoRA linear, with upstream gradient `dy`:

```
dB = (xA)^T @ dy * scale
dA = x^T @ (dy @ B^T) * scale
dx = dy @ W^T + scale * dy @ B^T @ A^T
dW = not computed
```

A common beginner bug is accidentally blocking gradients through frozen layers.

---

## 7. Fine-tuning data format

Do not dump raw text — build task-style examples. (See also `data/README.md`.)

**A. Continuation** — learns voice, rhythm, structure, terminology.

```
### Title:   {blog_title}
### Prefix:  {first_part_of_paragraph}
### Continue: {next_part}
```

**B. Rewrite** — more practical, but needs paired examples.

```
### Draft:   {generic_draft}
### Rewrite in the target style: {styled_version}
```

You can synthesise drafts by simplifying original paragraphs, then training the
model to reconstruct the richer style.

**C. Post/title** — headline style, summary behaviour.

```
### Blog excerpt: {first_500_tokens}
### Title:        {actual_title}
```

**D. Q&A** — useful, but dangerous if you expect factual correctness. Small
models often produce style without truth.

---

## 8. Dataset sizes

| Examples    | Expected result                    |
| ----------: | ---------------------------------- |
|       10–30 | Use prompting, not LoRA            |
|      50–100 | Weak style signal                  |
|   300–1,000 | Useful learning experiment         |
| 1,000–5,000 | Stronger style adaptation          |
|      5,000+ | Better, but memorization risk rises |

Aim for **300–1,000 clean examples**. One consistent author with 200 clean
posts beats 2,000 mixed noisy posts.

---

## 9. Training loop

Same as normal training, except: base weights frozen, only LoRA weights
trainable, optimizer sees only LoRA params, checkpoint saves only adapter state.

```python
base = load_base_model()
freeze(base)
inject_lora(base, target_modules=["q_proj", "v_proj"], rank=4, alpha=8)
optimizer = AdamW(lora_parameters(base), lr=1e-4)

for step in range(max_steps):
    x, y = get_batch()
    logits = base.forward(x)
    loss = cross_entropy(logits, y)
    zero_grad(lora_params)
    loss.backward()
    clip_grad_norm(lora_params, 1.0)
    optimizer.step()
    if step % eval_interval == 0:       evaluate()
    if step % checkpoint_interval == 0: save_adapter()
```

---

## 10. Adapter checkpoint format

Never save the full base model each time. See `../checkpoints/README.md` for the
exact JSON. Save: adapter weights + adapter optimizer state, base model id/hash,
tokenizer id/hash, training config, dataset manifest/hash, loss history, step.

Result: **one base model, many small adapters** — the right architecture.

> For public-blog learning experiments, do not redistribute adapters trained on
> a living author's writing.

---

## 11. Evaluation & memorization

LoRA teaches style; retrieval supplies facts. Always compare base / few-shot /
LoRA / LoRA+retrieval, and always run the memorization test. Full detail in
[`validation_report.md`](validation_report.md) (evaluation-and-safety appendix).

---

## References

- LoRA paper: https://arxiv.org/abs/2106.09685
- Hugging Face PEFT — LoRA: https://huggingface.co/docs/peft/package_reference/lora
