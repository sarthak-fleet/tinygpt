# Session 5 — Scaling: why bigger models know more (and what scaling DOESN'T cover)

> Closes the question from the model dive: *what would have to change for
> the 22M model to know that Paris is the capital of France?* The first
> half of the answer is "be bigger." The second half is "and a bunch of
> other things scaling laws don't cover."

## Where we are

You just watched a 22M-param model learn English structure but flunk
basic factual recall. The natural question: *what changes at 100M, at
1B, at 100B?* And: *is bigger the only knob?*

The honest two-part answer:

1. **Yes, bigger reliably helps**, in a clean way the field has measured
   carefully. This is the scaling-laws story.
2. **But bigger isn't the ONLY knob.** Data quality, post-training,
   distillation, inference-time compute — these can move you a lot
   without changing pretraining scale. Scaling laws don't describe them.

This session covers both halves.

---

# Part 1 — The scaling-laws story

## The observation

Train many language models of different sizes, plot final pretraining
loss against parameter count:

```
loss
 ▲
 │  ●  22M (your model)
 │
 │      ●  124M  (GPT-2 small)
 │
 │           ●  350M
 │                 ●  1.5B  (GPT-2 XL)
 │                       ●  7B  (Llama-7B class)
 │                             ●  70B
 │                                   ●  175B  (GPT-3)
 │                                         ●  ...
 └─────────────────────────────────────────────► params
```

Each dot = a separately trained model. Bigger reliably → lower loss. Not
a guess; measured across thousands of trained models.

## The shape: a power law

On a log-log plot (both axes scaled by 10×), the curve becomes a
near-perfect straight line — a **power law**:

```
log(loss)
 ▲
 │ \
 │  \
 │   \    ← straight line on log-log
 │    \     = power law
 │     \
 │      \
 └──────────── log(params)
```

The practical translation:

> Each time you 10× the params, loss drops by a fixed FRACTION (not a
> fixed amount).

So 22M → 220M reduces loss by the same fraction that 220M → 2.2B does.
The absolute improvement shrinks each step, but every 10× still buys
something.

## The three knobs (pretraining edition)

- **More params** — bigger model. Limited by GPU memory and training
  time.
- **More data** — more tokens to learn from. Limited by what data exists.
- **More steps** — more passes through the same data. Limited by
  diminishing returns once the model has internalized the patterns.

You can't crank just one. Huge model on tiny data → overfits (memorizes
training set, fails on new data). Tiny model on huge data → plateaus
(no capacity to keep absorbing).

## Chinchilla's finding (DeepMind, 2022)

The famous result: **for a given compute budget, the loss-minimizing
recipe is roughly 20 tokens of training data per parameter.**

Plug in huge-base-v1:

```
410M tokens / 22M params = 18.6 tokens/param
```

Near Chinchilla-optimal. The training was about as good as a 22M model
can get on that compute budget.

### Chinchilla is COMPUTE-optimal, not GLOBALLY optimal

Important caveat the original Chinchilla framing buried. The 20:1 rule
minimizes loss *for a fixed training compute budget*. It does NOT say
"a small model + more data is wasted."

**Modern practice deliberately overtrains small models:**

- Llama-3-8B: ~15 trillion tokens of training. Ratio ≈ 1,875 tokens/param
  — almost 100× past Chinchilla.
- Phi-3-mini: similar over-trained regime.

Why? Inference cost. A bigger Chinchilla-optimal model would be smarter
per training dollar, but you pay its inference cost every single time
you serve it, forever. A smaller over-trained model is "expensive to
train per quality point" but "cheap to run per quality point." Total
lifetime cost is lower.

If you trained your 22M base on 10× more data, it would NOT hit a wall.
You'd get a meaningfully better 22M model. The trade-off is 10× the
training time for diminishing-but-real quality gains. Whether that's
worth it depends on how often you'll run the model.

See journal Entry 10 for the longer version.

## What scaling buys (capability ladder)

| Size | What it can do | Examples |
|------|----------------|----------|
| ~10M | grammatical English, no facts | your `huge-base-v1` |
| ~100M | coherent paragraphs, weak facts | GPT-2 small |
| ~1B | multi-step reasoning, common facts | GPT-2 XL |
| ~10B | meaningful knowledge, instruction following | Llama-7B, Mistral-7B |
| ~100B | strong reasoning, broad world knowledge | GPT-3 era |
| ~1T+ | frontier; near human-expert in many domains | GPT-4, Claude |

A startling sub-pattern: certain capabilities (multi-step reasoning,
in-context learning, instruction following) appear to **emerge
suddenly** at specific sizes — barely possible below a threshold,
reliable above. Called **emergent capabilities**. Phenomenon is real but
some apparent emergence is an artifact of how the capability is
measured.

## Where the laws bend

- **Returns diminish.** Each 10× costs 10× compute but reduces loss by a
  fixed fraction. Eventually a billion dollars buys you 1% loss reduction.
- **Data runs out.** Internet is finite. Frontier labs have started
  hitting the wall of "we've trained on most of the high-quality text
  that exists."
- **Quality bites harder at the top.** A 1B model on clean data beats a
  1B on dirty by more than a 100M on clean beats a 100M on dirty.
- **Some failure modes don't scale away.** Hallucination, calibration,
  certain reasoning gaps — frontier models still have them.

---

# Part 2 — What scaling DOESN'T cover

The scaling-laws story is foundational but partial. It describes
**pretraining inputs → pretraining loss**. It does NOT describe all routes
to capability. Other knobs that matter, in some cases more than scaling:

## Data quality

Phi-3 (Microsoft, 2024) demonstrated that a ~4B model trained on
**textbook-quality, carefully curated data** beats much larger models on
reasoning. The paper title was literally *"Textbooks Are All You Need."*

At a fixed pretraining compute: better data >> more bad data. This
crossed a threshold in 2024 where "quality" became visible as a separate
axis from raw scale.

TinyGPT already ships a quality-classifier (`tinygpt
train-quality-classifier` + `tinygpt quality-filter`) for exactly this
kind of curation.

## Data mixture and curation

What fraction is code? Math? Conversational? Documentation? — directly
shapes what the model is good at. FineWeb-Edu (the data huge-base-v1
trained on) is itself a curated *educational subset* of broader FineWeb.

A model trained on "10% code" knows more about code than one trained on
"0% code" even at identical scale.

## Architectural choices

SwiGLU vs GELU, RoPE vs learned positional, RMSNorm vs LayerNorm,
grouped-query attention vs full multi-head — each is a few percent loss
improvement. Compound across choices for ~10–20% total. Doesn't change
the scaling-laws curve's slope; shifts the whole curve down.

## Post-training (the big omission)

This is HUGE and frequently underweighted in scaling-laws discussions.
The same pretrained base + different post-training stages = wildly
different apparent capability:

- **SFT (Supervised Fine-Tuning).** Takes the base, trains on
  `(prompt, ideal answer)` pairs. Makes the model usable as an assistant.
  No new knowledge added; the model is just taught the shape of
  "respond helpfully."
- **RLHF / DPO.** Refines outputs to match human preferences. Same
  underlying knowledge, different communication style. Turns GPT-3 into
  ChatGPT.
- **Distillation.** Transfers capability from a bigger teacher model into
  a smaller student. The student "inherits" more than its scale would
  predict. This is the TinyGPT thesis: a 22M specialist distilled from
  a 70B teacher beats a 22M generalist by a lot.
- **Tool use / RAG.** Extends the model's reach without changing weights
  — external memory and computation. A 22M model + good retrieval often
  beats a 70B model with no external memory for factual tasks.

## Inference-time compute

A scaling axis the field discovered AFTER GPT-4. Chain-of-thought,
self-consistency, search, structured reasoning chains (o1, o3). "Think
longer at inference" buys quality that you'd otherwise have to buy from
bigger pretraining.

Cheap small model + lots of inference reasoning can match expensive big
model + one-shot answer. The trade-off is latency.

---

## What this means for huge-base-v1 (and the TinyGPT thesis)

Two practical conclusions:

1. **The base won't get materially better by training it more on the same
   data.** Scaling laws give the structural ceiling, and you're near it
   for this size + data combination.

2. **But the *practical* capability ceiling is way higher than the
   pretraining-laws picture suggests** — IF you stack the right stages
   after pretraining:
   - Specialist SFT on a narrow task
   - Distillation from a bigger teacher
   - Constrained generation for structured output
   - Tool use / RAG for tasks that need external memory

This is exactly the TinyGPT thesis. Scaling laws say "your small base is
structurally limited"; the post-training story says "but the *useful*
ceiling is way higher than that suggests, when stages compose right."

Both are true. The pretraining-laws story isn't wrong; it's just not the
whole story.

---

## Self-check

Don't peek:

1. **If you doubled the training run from 200K to 400K steps** (same data,
   same model), how much would loss drop?
2. **If you took the same 22M model and trained it on 10× the data**,
   would that help? Be careful — "Chinchilla optimum" is a trap here.
3. **You have a 1B model that knows world facts but is slow. You have your
   22M base.** Path to a fast SQL-generating assistant?
4. **Trap question:** if scaling laws are so reliable, why didn't OpenAI
   just train GPT-3 directly in 2018 instead of building GPT-1 and GPT-2
   first?

---

## Where this connects

- The "small specialist on small base" play in TinyGPT's product roadmap
  is enabled by **post-training axes scaling laws don't cover**. Without
  that route, a 22M model would have a hard ceiling far below practical
  usefulness. With it, "narrow but sharp" is feasible.
- Journal Entry 8 covers why loss numbers aren't comparable across
  vocab sizes (the prerequisite for reading pretraining-loss curves
  correctly).
- Journal Entry 10 covers the Chinchilla compute-optimal vs
  over-trained-small-model distinction in detail.
