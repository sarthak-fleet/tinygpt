# Session 8 — Training mechanics: the actual loop

> Closes the gap from Session 2's abstract "step downhill" to the real
> 12-hour wsd training run you watched produce huge-base-v1. What does a
> training step actually look like, what knobs matter, what fails, why.

## Where we are

Session 2 covered gradient descent in the abstract: pick parameters,
compute loss, take a step. Sessions 3–7 covered what the model IS, what
data goes in, what paradigms exist. But the actual machinery of "trained
for 200,000 steps over 12 hours with batches of 8 and a wsd schedule
ending at loss 4.16" — we never opened that box.

This session is that box.

---

## The loop in pseudocode

```python
for step in range(total_steps):
    # 1. Get a batch
    batch = get_next_batch(data_loader, batch_size)

    # 2. Forward pass — model predictions
    predictions = model(batch.inputs)

    # 3. Compute loss
    loss = loss_fn(predictions, batch.labels)

    # 4. Backward pass — compute gradients
    loss.backward()

    # 5. Clip gradients (safety)
    clip_grad_norm(model.parameters(), max_norm=1.0)

    # 6. Update parameters
    optimizer.step()

    # 7. Zero gradients for the next step
    optimizer.zero_grad()

    # 8. Adjust learning rate per the schedule
    lr_scheduler.step()

    # 9. Periodically: log, validate, save
    if step % log_interval == 0:
        log_metrics(loss, lr, step)
    if step % eval_interval == 0:
        validate(model, val_data)
    if step % save_interval == 0:
        save_checkpoint(model)
```

Every line has its own knobs. The rest of this session unpacks them.

---

## Batches — the unit of training

A "batch" is the chunk of data we look at to compute one gradient.

### Why not single examples?

Single-sample gradient descent (one batch = one sample) is pure
**stochastic gradient descent**. Two problems:
1. Gradients are very noisy (one sample doesn't represent the data).
2. Hardware (GPUs) doesn't run efficiently — they want parallelism.

### Why not the whole dataset?

Whole-dataset gradient descent (one batch = entire data) is **full-batch
GD**. Two problems:
1. Doesn't fit in memory for any real-world dataset.
2. Noise can be a feature (covered in journal Entry 4 — flat minima).

### Mini-batch: the practical middle

Pick a batch size that fits in memory, computes gradients with manageable
noise, and saturates GPU compute. Typical: 16–512 samples for small/mid
models, thousands for frontier models.

### Gradient accumulation

What if you want a large effective batch but can't fit it in memory?
Accumulate gradients over multiple "micro-batches":

```python
optimizer.zero_grad()
for accum_step in range(accum_steps):
    batch = get_next_batch(micro_batch_size)
    loss = forward(batch) / accum_steps  # divide!
    loss.backward()                       # accumulates
optimizer.step()                          # one update from many micro-batches
```

Effective batch = `micro_batch × accum_steps`. Memory cost stays at
micro_batch.

**huge-base-v1's setup:** batch=8, accum=2 → effective batch = 16 samples
× 256 ctx = **4,096 tokens per step.** Small by modern standards
(frontier runs use 1M+ tokens per step).

---

## Steps, epochs, and what "training duration" means

- **Step (iteration)** = one gradient update, one pass through the loop.
- **Epoch** = one full pass through the training dataset.

For huge-base-v1: 200,000 steps × 4,096 tokens/step = **820M tokens
processed**. Dataset was 241MB ≈ 410M tokens → run did **~2 epochs**.

In modern LLM training, "epoch" is less meaningful than "tokens seen"
because corpora are enormous. Llama-3 trained on 15T tokens, never
repeating — less than 1 epoch in the traditional sense.

For SMALL training runs and fine-tuning, 3–5 epochs is typical and
"epoch" matters.

Three interchangeable ways to set duration:
1. By steps: "run 200K steps and stop"
2. By tokens: "stop after 410M tokens"
3. By epochs: "do 3 passes"

All convertible given batch size and dataset size.

---

## Learning rate — the master knob

If you tune ONE thing, it's the learning rate. The multiplier on every
gradient update.

- **Too high** → loss diverges or oscillates wildly
- **Too low** → training crawls, barely moves
- **Just right** → steady descent

Starting values by task:
- Pretrain transformers: `1e-4 to 1e-3`
- Fine-tune transformers: `1e-5 to 1e-4` (smaller, model is already
  trained; large updates would wreck what's learned)
- LoRA: `1e-4 to 5e-4` (smaller adapter, can take larger LR)
- RLHF/DPO: `5e-6 to 1e-5` (very small, careful)

**huge-base-v1:** `max_lr=3e-4`, `min_lr=3e-5`. Standard transformer
pretrain endpoints (10× decay).

---

## Learning rate schedules

The LR doesn't stay constant. It changes during training following a
"schedule."

### Constant

LR stays fixed. Simple. Rarely optimal for production.

### Cosine decay

Starts high, follows a cosine curve down to a minimum. Smooth landing.

```
   lr
    │\___
    │    \___
    │        \___
    │             \___
    └─────────────────► step
```

Standard for fine-tuning. Locks the LR trajectory in from step 1.

### Cosine with warmup

Most common production pretrain schedule until ~2024:

```
   lr
    │   ___
    │  /   \___
    │ /        \___
    │/             \___
    └──────────────────► step
       warmup    cosine decay
```

### WSD (Warmup-Stable-Decay)

What huge-base-v1 uses. The modern preferred schedule:

```
   lr
    │   ___________
    │  /           \
    │ /             \
    │/               \____
    └─────────────────────► step
       warmup  stable    decay
       (~1%)  (~90%)    (~10%)
```

Three phases:
- **Warmup** (first 0.5–1% of steps): LR ramps from near-zero to max_lr.
- **Stable** (middle 80–90%): LR stays at max_lr.
- **Decay** (last 5–10%): LR drops from max_lr to min_lr.

**Advantage over cosine:** you can stop anytime during the stable phase
and decay to get a usable checkpoint. Cosine forces you to commit to the
total step count up front. WSD = continue-training-friendly.

**huge-base-v1's wsd parameters:** warmup=1000 (0.5%), stable=180K
(implicit middle 90%), decay=20K (last 10%).

### What you saw in the actual loss curve

The mid-training bump in huge-base-v1 (4.58 → 4.81 around step 80K) was
the stable phase wobbling. The high constant LR has enough noise to
cause loss oscillation — but the model is still learning, just noisy.

The clean drop from 160K → 200K (4.33 → 4.16) was the decay phase doing
its job: as LR shrinks, the optimizer settles into a fine-grained
minimum.

**WSD doing exactly what it's designed to do.** No bug.

---

## Warmup — why it's there

Why ramp the LR up gradually instead of starting at max?

Early in training, parameters are randomly initialized. A high LR on
random parameters produces ENORMOUS gradient updates → wild oscillation
or divergence within ~100 steps. Warmup lets the parameters "settle"
under small updates before the optimizer is allowed to take big steps.

Without warmup: probable loss spike or NaN within the first few hundred
steps for transformers.

With warmup: steady descent from step 1.

Typical: 0.5–5% of total steps. **huge-base-v1: 0.5%** (1K of 200K).

---

## Gradient clipping

Sometimes gradients explode (very large values). Caused by:
- Bad initialization (rare today, well-studied)
- Outlier data (a particularly weird sample)
- Numerical instability (fp16 issues)
- LR too high transiently

If you take a step with an exploded gradient, parameters jump wildly →
loss spikes → training might not recover.

**Gradient clipping** caps the gradient magnitude. If `||gradient|| >
threshold`, scale all gradients down by `threshold / ||gradient||`.

```python
total_norm = compute_gradient_norm(model.parameters())
if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in model.parameters():
        p.grad *= scale
```

Standard value: `max_norm = 1.0`. **huge-base-v1** used `grad_clip = 1.0`,
TinyGPT default.

This is cheap insurance. Almost every modern training run uses it.

---

## Validation — knowing when to stop

You can't tell from training loss alone whether the model is overfitting.
Training loss just keeps dropping (the model can memorize). You need a
**held-out validation set** the model doesn't train on.

Periodically evaluate on val:

```python
if step % eval_interval == 0:
    model.eval()
    val_loss = compute_loss(model, val_data)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_best_checkpoint(model)
    model.train()
```

Healthy training: train and val loss both decrease, gap stays small.
Overfitting: train loss keeps dropping, val loss stalls or rises.

**huge-base-v1's final gap:** train 4.16 / val 4.32 = ~4% relative.
Healthy. No overfitting.

If val loss starts rising, **stop training and use the best checkpoint
so far** — this is called "early stopping."

---

## Common failure modes

### Loss spike

```
loss
 │       \
 │        \    /\
 │         \  /  \
 │          \/    \____
 └──────────────────────► step
              ↑
           spike
```

Sudden jump up. Often resolves on its own. Causes:
- Bad batch (outlier data)
- Numerical glitch
- LR temporarily too high

When to worry: if it doesn't resolve, or repeats. Then look at data + LR
+ warmup. TinyGPT logs a `spike` field per step if loss > 3× moving avg;
huge-base-v1 had ZERO spikes in 200K steps.

### Divergence

```
loss
 │              ___
 │             /
 │            /
 │  \_______/
 └────────────────► step
              ↑
         diverged
```

Loss climbs and doesn't come back. Catastrophic. Causes:
- LR too high
- No gradient clipping
- Numerical instability
- Catastrophic forgetting in fine-tuning

Fix: lower LR, enable grad clipping, check data, restart from last good
checkpoint.

### NaN (Not a Number)

Loss becomes `NaN`. Training is ruined. Causes:
- Numerical overflow (fp16 issue)
- Division by zero somewhere
- Bad data with NaN values

Fix: switch to fp32 or bf16, check data, restart.

### Plateau

Loss stops decreasing for many steps. Either:
- **Converged** — great, stop training.
- **Stuck** — try lower LR, more data, different architecture.

---

## Precision — fp32 vs fp16 vs bf16

"Precision" = how many bits each number uses.

| Type | Bits | Range | Best for |
|------|------|-------|----------|
| **fp32** | 32 | huge range, precise | full precision; slow but stable |
| **fp16** | 16 | small range | inference; training risky (overflow) |
| **bf16** | 16 | huge range, less precise | training (range matters more) |
| **fp8** | 8 | tiny range | emerging for inference; risky for training |

**Mixed precision** training: use bf16 for matmul (fast), fp32 for
accumulators and master weights (stable). Standard for modern training.

**huge-base-v1** trained in fp32 — conservative choice, slow but stable.
A modern run would use bf16 mixed precision, ~2× faster on M5 Pro at the
same final quality.

---

## Connect back to huge-base-v1

The values from your actual training run, decoded:

| Knob | Value | Meaning |
|------|-------|---------|
| `batch` | 8 | 8 samples per micro-batch |
| `accum_steps` | 2 | gradient accumulation factor |
| effective batch | 16 | 16 samples × 256 ctx = 4,096 tokens/step |
| `ctx` | 256 | sequence length per sample |
| `total_steps` | 200,000 | iterations |
| total tokens | ~820M | 200K × 4,096 |
| dataset size | ~410M tokens | ~2 epochs |
| `lr_schedule` | wsd | warmup-stable-decay |
| `warmup` | 1,000 | first 0.5% of steps |
| `decay_steps` | 20,000 | last 10% of steps |
| `max_lr` | 3e-4 | standard transformer pretrain peak |
| `min_lr` | 3e-5 | 10× smaller end target |
| `grad_clip` | 1.0 | safety against explosions |
| precision | fp32 | conservative choice |
| wall time | 12 hours | M5 Pro |
| step_per_s | 4.6 | end-of-stable phase |

You can now read this entire table and explain each value.

---

## Self-check

Don't peek:

1. **Why does the loss curve bump in the middle of the wsd stable
   phase?**
2. **You're fine-tuning a pretrained model.** Should you use a smaller
   or larger LR than pretraining? Why?
3. **What's the practical effect of `grad-clip 1.0`?**
4. **You see NaN in your loss at step 500.** What's the first thing
   you'd check?
5. **Trap question:** if mixed-precision (bf16) is ~2× faster and
   "good enough," why does huge-base-v1 still use fp32?

---

## Where this connects

- Closes the gap from Session 2 (abstract gradient descent) to the
  production run from the model dive.
- The training loop is the same skeleton whether you're pretraining,
  fine-tuning, distilling, or doing DPO. Different data + loss
  function + which params get updated. The mechanism is identical.
- Session 9 (production optimizations, journal Entry 14) is "the knobs
  on this loop, plus the architecture-level ones."
- The wsd schedule's "stop in stable phase, decay later" property is
  what makes TinyGPT's `tinygpt train --resume` workflow work cleanly:
  you can pause training, do other things, resume, and the LR is at the
  right point to keep going.
