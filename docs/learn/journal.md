# Learning journal

A running log of questions, hunches, "wait, why does anyone do it this way?"
moments, and tangents that come up during the curriculum sessions.

**Discipline:** capture without committing. Most entries will turn out to be
rediscoveries of known ideas or open questions the curriculum will address
later. Occasionally something will graduate into a real investigation —
when it does, we do a prior-art pass before chasing it.

**Why a journal alongside the polished curriculum:** the polished
`session-XX-*.md` files are the *explanation*. This journal is the *thought
process that produced the explanation*. Anyone using this curriculum later
sees both — and the journal is what makes it pedagogically distinctive.
Most textbooks hide the questions.

---

## 2026-06-06 — Entry 1: "Is this just linear regression?"

**During Session 1.** After landing the "neuron = line being fit to data"
idea, the natural realization:

> *Is this what linear regression is, essentially?*

**Yes.** A single neuron with no activation function = exactly linear
regression. Same equation, same fitting procedure, same idea.

Statisticians (Gauss, Legendre) were doing this in the early 1800s. The
"neural" in neural network is mostly marketing inheritance from 1950s
cognitive science — researchers loosely analogized the multiply-and-add
operation to brain neurons "weighing" inputs.

So the honest framing of the whole field:

```
            stack many of them
            ↓ + non-linearities
LINEAR  ───────────────────────────→  NEURAL
REGRESSION                            NETWORK
(1800s, Gauss)                        (1950s+)

Same fitting equation. Same gradient descent. Just composed in more
flexible ways.
```

The boundary between "statistics" and "deep learning" is way blurrier than
the marketing makes it sound.

**Open question to revisit:** at what point does a neural network stop
being equivalent to linear regression and become something genuinely
more powerful?

*Spoiler (to be unpacked in a later session): when you stack layers AND
put a non-linear function between them. Two linear layers without a
non-linearity in between collapse algebraically into one linear layer —
so depth alone buys nothing. The "neural" part is really the non-linear
bumps.*

---

## 2026-06-06 — Entry 2: clarification surfaced by Session 1 self-check

When asked "what does it mean to learn `m` and `b`?", the first instinct
was:

> "Trying different values of x and y until we figure out what m and b are."

The wobble: `x` and `y` don't change. They come from data, fixed forever.
What changes is `m` and `b`.

This distinction (inputs/outputs vs parameters) is one of the most
load-bearing ideas in ML — and it's the kind of thing the curriculum
should make stupidly explicit, since it tripped a thoughtful first-time
reader. Session 1 was revised to include an explicit "what's fixed vs
what changes" sidebar before this entry was filed.

**Takeaway for the curriculum:** any time a concept has "two kinds of
variables doing different jobs," call it out with a table. Don't trust
prose to make the distinction stick.

---

## 2026-06-06 — Entry 3: Why squared, not absolute (or |x|³)?

**During Session 2.** Question that came up naturally after introducing
MSE: why square the error? Why not just take the absolute value? And —
sharper follow-up — why not `|x|³`?

### `|x|` is a real option

Absolute-error loss exists and is used. Called **MAE** (mean absolute
error) or **L1 loss**. The reasons we default to squared (MSE / L2):

1. **Smoothness.** `x²` has a defined slope everywhere. `|x|` has a kink
   at zero where the slope jumps from −1 to +1 discontinuously. Gradient
   descent doesn't love kinks.
2. **Closed-form solution for linear models.** For `y = mx + b` with
   MSE, you can solve algebraically for the best `(m, b)` — Gauss did it
   in 1805 (the "normal equation"). For MAE, no such closed form.
3. **Matches Gaussian noise.** See below.

### `|x|³` is mathematically valid but pointless

Sharper follow-up — if smoothness is the issue with `|x|`, why not `|x|³`?

Math check: `|x|³ = (x²)^{3/2}`. Derivative is `3x·|x|`, which is `0` at
`x = 0`. So `|x|³` IS smooth (twice continuously differentiable). Good
catch — smoothness alone isn't a reason to reject it.

Why nobody uses it anyway:

- **More outlier-sensitive than `x²`.** `|x|³` grows cubically, `x²`
  quadratically. An error of 100 contributes 1,000,000 under `|x|³` vs
  10,000 under `x²`. If you were worried about outliers under MSE, you'd
  be much more worried under cube.
- **No closed-form solution.** The 1805 trick doesn't extend.
- **No natural noise interpretation.** What probability distribution does
  `|x|³` correspond to? Some exotic generalized-normal with no name and no
  intuition.
- **Strictly worse than `x²`** without any compensating advantage.

### The deep reason `x²` is special: Gaussian noise

`x²` isn't privileged because of smoothness. Many functions are smooth.
It's privileged because of its connection to the **Gaussian** (bell-curve,
"normal") distribution.

Take real measurements of anything — baby weights, house prices, sensor
readings. Plot how often each value occurs. The shape is almost always
bell-shaped: clustered near a mean, tailing off in both directions.

**Why?** The **Central Limit Theorem**: when many small independent
random influences add up, the sum tends to a Gaussian — regardless of
what each influence looks like individually. A baby's weight =
genetics + diet + sleep + a thousand other tiny factors. The sum is
Gaussian.

**The MSE connection:** if you model your data as "true line + Gaussian
noise," then minimizing MSE gives the *maximum-likelihood* line — the
single most-probable line consistent with your data. One line of algebra
proves this (Gaussian density's log is `−(x−μ)² / 2σ²` — squared
distance pops out).

So MSE isn't arbitrary. It's the *natural* loss for the *natural* noise
model in nature.

If your noise *isn't* Gaussian — heavy-tailed, lots of rare outliers —
MSE becomes inappropriate. Then MAE (which corresponds to Laplace noise,
heavier tails) is better. **Huber** loss is a hybrid: squared near zero
(smooth, MSE-like), absolute far out (outlier-robust, MAE-like).

| Loss | Punishes outliers | Smooth? | Closed-form? | Implicit noise model |
| ---- | ----------------- | ------- | ------------ | -------------------- |
| MSE (L2)  | a lot         | yes     | yes (linear) | Gaussian             |
| MAE (L1)  | proportionally| no      | no           | Laplace              |
| Huber     | tunably       | yes     | no           | mixture              |

**Open question to revisit:** when scaling up to deep networks, does the
choice of loss still matter as much, or do non-linearities and
overparameterization dominate? (Spoiler from memory: it matters a lot for
the early-training dynamics and for specific tasks; less for "final
accuracy" on well-behaved problems.)

---

## 2026-06-06 — Entry 4: Mini-batch noise — bug or feature?

**During Session 2.** When mini-batches were introduced as "we don't
compute the gradient on all data, just a slice," the natural worry:
*different batches will push toward different gradients. Won't that cause
chaos?*

### They do disagree. By design.

Each batch's gradient is a **noisy estimate** of the true full-data
gradient. Imagine 100 batches drawn from the same dataset:

- Batch 1 says go `[+2, +3]`
- Batch 2 says go `[+1, +4]`
- Batch 3 says go `[+3, +2]`
- ...

If you AVERAGED all 100, you'd get back exactly the true full-data
gradient. So each batch is "the truth + random noise."

In practice we use one batch per step instead of averaging. Over many
steps, the random noise averages out; the systematic downhill direction
wins. We take many tiny noisy steps instead of fewer accurate ones. We
end up in the same valley.

### The twist: noise is a feature

Surprising result: even with **infinite compute** (so full-batch GD is
free), you'd usually still want noise. Reasons:

1. **Flatter minima generalize better.** SGD's noise tends to find broad,
   flat valleys in the loss landscape. Flat valleys → models that
   generalize to new data. Full-batch tends to find sharp narrow minima
   that perform well on the training set but worse on new data. This is
   the **flat minimum hypothesis** — observed empirically, theoretical
   foundations active research.

2. **Escape from saddle points.** High-dimensional loss landscapes are
   full of saddles where gradient descent slows to a crawl. Noise kicks
   you off.

3. **Implicit regularization.** SGD's noise behaves mathematically like
   an extra regularization term that prevents overfitting. You'd have to
   add this manually with full-batch.

There's an active line of research on this — search "implicit bias of
SGD" or "edge of stability" for current papers.

### Pre-selected chunks vs random?

Real concept and people have studied it. The default is random for good
reasons, but task-specific schemes exist:

- **Random** — each batch is an unbiased estimate of the true gradient.
  Default for almost everything. Simple, has theoretical guarantees.
- **Curriculum learning** — order from easy to hard. Sometimes helps,
  especially for RL and code generation. Often doesn't.
- **Stratified sampling** — ensure each batch represents every class.
  Common in classification with imbalanced classes (1% positive, 99%
  negative).
- **Hard example mining** — deliberately oversample examples the model
  currently gets wrong. Used in object detection.

Random wins by default because any fixed scheme is biased toward whatever
pattern you chose, and "representative" requires you to define
representativeness — another modeling decision with its own biases.

**Open question to revisit:** the "flat minimum hypothesis" is folk
wisdom that's been challenged by some recent theoretical work. How
robust is it really? Worth a deeper dive when we cover regularization.

---

## 2026-06-06 — Entry 5: Why is ML research still in flux?

**During Session 3.** Question that came up: *why are research still
pending on these things?* (depth, why neural nets generalize, etc.)

### Honest framing

ML is currently an **empirical science**, not a deductive one. The state
of the field rhymes with physics around 1900: lots of working theory for
specific things, no overarching framework that predicts what will work
next. Most progress is empirical — try it, see if it works, generalize
from observation.

### What's actually open

These are not closed problems:

- **Generalization theory.** Classical learning theory says: more
  parameters than data → overfit catastrophically. Reality: modern LLMs
  have orders of magnitude more parameters than training data and
  generalize fine. The classical framework is wrong; nobody knows what
  the correct framework looks like. Search "double descent,"
  "benign overfitting," "implicit bias of SGD."
- **Mechanistic interpretability.** What does each layer actually
  compute? What do individual neurons mean? Anthropic, DeepMind,
  EleutherAI all have teams attacking this. Early progress (Sparse
  Autoencoders, induction heads) but the bulk is open.
- **Scaling laws.** How does loss / capability scale with parameters,
  data, compute? Chinchilla and OpenAI scaling papers gave power-law
  predictions that hold across orders of magnitude — but we don't know
  why they hold, or when they'll break.
- **Sample efficiency.** Humans learn from orders of magnitude less data.
  Why? What's the right inductive bias?
- **Why attention works.** We use it because it works. No proof that it
  *has* to.
- **Loss landscape geometry.** Why doesn't gradient descent get stuck in
  bad local minima in high dimensions? "Loss landscape is mostly
  saddles" is the working hypothesis but not theorem.

### The takeaway

We have a hammer that works (SGD on big transformer + lots of data). We
don't fully understand why. Most "research" is either incremental
improvements to that hammer, or trying to understand why it works.

The field moves faster than any deductive science because it accepts
empirical demonstration as sufficient — wait for a proof and you lose to
someone who skipped that step.

---

## 2026-06-06 — Entry 6: Are people working on non-linearities specifically?

**During Session 3.** Companion question: *and I assume people are
working on non-linearities constantly?*

### Yes, but it's a mature subfield

Activation functions are still being researched, but the wins are small
and incremental. The timeline:

| Year | Function | Why it mattered |
| ---- | -------- | --------------- |
| 1958 | step function | original perceptron — too harsh, no gradients |
| ~1989 | sigmoid / tanh | smooth, differentiable, dominant for decades |
| 2010 | ReLU (Glorot, Bengio) | huge leap — solved vanishing gradients in deep nets |
| 2016 | GELU (Hendrycks) | smoother ReLU, became transformer default |
| 2017 | Swish / SiLU (Ramachandran et al) | smooth self-gated, marginal improvement |
| 2020 | SwiGLU (Shazeer) | gated variant, now default in Llama / Mistral / Qwen |

### Why gains have shrunk

After ReLU's leap, every subsequent activation function delivers a few
percent improvement at most. The field has converged on a small family
(ReLU, GELU, SwiGLU, occasional experiments).

Where the order-of-magnitude wins live now:

- **Architecture changes** (attention variants, Mixture-of-Experts, sparse
  attention, MLA, state-space models like Mamba)
- **Training procedures** (RLHF, DPO, constitutional AI, RLAIF)
- **Data quality and curation** (filtered web corpora, synthetic data,
  textbook-quality data)
- **Scaling laws and compute optimization**

### The takeaway

Activation functions still matter and people still propose new ones, but
they're no longer where the order-of-magnitude leverage is. People still
publish on them — there's "polishing the hammer" research and "designing
a new tool" research, and activations are now the former.

If you wanted to publish on activations today, the bar is high: you'd
need to show meaningful improvement across a broad set of architectures
and tasks, and the existing functions are already very good.

---

## 2026-06-06 — Entry 7: The taxonomy of ML approaches (and where transformers fit)

**During Session 3 model dive.** Question: "is there also tree regression?
RL? learning by doing? Transformers don't use all of them — what are the
other kinds of training and why aren't transformers used for them?"

The premise conflates three orthogonal axes. Untangling:

### Axis 1 — Training signal (what you learn FROM)

| Paradigm | Signal looks like | Example |
| -------- | ----------------- | ------- |
| **Supervised** | (input, correct answer) pairs | spam classification with labels |
| **Self-supervised** | label derived from input | next-token prediction — the label IS the next word in your data |
| **Unsupervised** | no labels | clustering, dimensionality reduction |
| **Reinforcement** | reward signal from environment | game-playing agent, RLHF |
| **Semi-supervised** | mix of labeled + unlabeled | rare in modern practice |

### Axis 2 — Model family (what shape the function takes)

| Family | Shape | Best for |
| ------ | ----- | -------- |
| Linear regression / GLMs | line/plane | tabular numeric prediction |
| Decision trees, gradient-boosted trees (XGBoost, LightGBM) | tree of threshold splits | tabular data — **often beats neural nets here** |
| SVMs | margin-maximizing boundary | classical classification, small data |
| k-NN | "copy the nearest examples' answers" | recommenders, image lookup |
| Bayesian methods | full probability distributions | calibrated uncertainty matters |
| Neural networks (CNN, RNN, Transformer) | stacked linear+non-linear | high-dim data with lots of examples |

### Axis 3 — Parameter update algorithm

| Algorithm | Used for |
| --------- | -------- |
| Gradient descent / SGD / Adam | anything differentiable |
| Closed-form (normal equation) | linear regression with MSE |
| Tree-building (CART, ID3, gradient boosting) | decision trees |
| EM (Expectation Maximization) | mixture models, HMMs |
| Policy gradient / Q-learning / Actor-Critic | RL |
| MCMC | Bayesian inference |

### These axes are mostly independent

The big realization: you can mix and match.

- Transformer + supervised + SGD = standard fine-tuning
- Transformer + self-supervised + SGD = **pretraining** (next-token prediction, what `huge-base-v1.tinygpt` does)
- Transformer + RL + policy gradient = **RLHF** (the second stage that turned GPT-3 into ChatGPT)
- Tree + supervised + boosting = XGBoost (often beats neural nets on tabular tasks)
- Neural net + RL = AlphaGo, AlphaStar

### Correcting the premise

"Transformers don't use RL" is wrong. **RLHF is exactly transformer + RL.**
ChatGPT, Claude, Llama-Chat all use it. Transformer = model family; RL =
training signal; they compose.

### When transformers are NOT the right tool

- **Tabular data, no sequence structure** → tree ensembles win.
- **Tiny datasets (hundreds, not millions)** → simpler models with stronger
  inductive biases (Bayesian, SVMs) often beat NNs.
- **Calibrated probability matters more than raw accuracy** → Bayesian.

Transformer's superpower is *sequential, high-dimensional data with lots of
examples* — text, audio, video, DNA. For everything else, classical
methods are often better.

**Curriculum implication:** this whole taxonomy is its own session waiting
to be written. For now, the journal entry serves as the placeholder.

---

## 2026-06-07 — Entry 8: Loss numbers aren't comparable across setups

**During Session 5.** The user observed: small browser-trained models had
*lower* loss than the 22M huge-base-v1. If bigger is supposed to be
better, what's going on?

### The instinct is right, but the conclusion is wrong

Loss isn't an absolute quality measure. It's tied to the prediction task
(specifically: vocab size and bytes-per-token), and the same "quality"
shows up as very different absolute numbers in different setups.

### Why the absolute numbers diverge

**Cross-entropy loss has a different ceiling at each vocab size.** A
model that uniformly guesses across the vocab achieves loss `log(vocab)`:

| Setup | vocab | random-baseline loss |
| ----- | ----- | -------------------- |
| Browser byte-level | 256 | log(256) ≈ **5.55** |
| huge-base-v1 (SmolLM2 BPE) | 49,152 | log(49152) ≈ **10.80** |

So a byte model with loss 2.5 went from random (5.55) to 2.5 — about 55%
of the way to perfect. huge-base-v1 with loss 4.16 went from random
(10.80) to 4.16 — about 61% of the way. **The bigger model is actually
further from random** even though its absolute loss number is bigger.

### Bits per byte: the apples-to-apples metric

To compare across vocab sizes, normalize:

```
bpb = loss / log(2) / (avg bytes per token)
```

Byte models: 1 byte per token. BPE models (English): typically ~4 bytes
per token.

| Model | Loss | bpb (approx) |
| ----- | ---- | ------------ |
| Byte browser model | 2.5 | 2.5 / 0.693 / 1 ≈ **3.6 bpb** |
| huge-base-v1 (BPE) | 4.16 | 4.16 / 0.693 / 4 ≈ **1.5 bpb** |
| Frontier LLM | (varies) | ~**0.6 bpb** |

Same metric, both setups normalized: the BPE model is ~2.4× more
efficient at modeling English than the byte model. The "smaller raw
number" on the byte model was an illusion of the different task.

### When can you compare loss directly?

Only when the prediction task is identical: same vocab, same tokenizer,
same dataset, same eval procedure. Within those constraints, raw loss is
fine. Across them, normalize to bpb or use a held-out benchmark.

### Takeaway for the curriculum

Future Session on training mechanics should call this out explicitly: a
loss number alone is roughly meaningless. A loss curve at a fixed setup
is meaningful (relative motion within one task). Comparison across
setups requires bpb or held-out benchmark scores.

---

## 2026-06-07 — Entry 9: LLMs as decoders for "blind spots" in unknown texts

**During Session 5.** Question: *"if I have a large alien language text
with some blind spots, can LLMs help decode the blind spots?"*

### Yes — that's exactly what masked language modeling does

Train an LM on the available alien-language data (self-supervised: the
labels are just other parts of the same text). The model learns the
language's patterns — which tokens commonly follow which, what sentence
shapes exist, what the local grammar looks like. Then for a blind spot,
ask the model "given the surrounding context, what's the most likely
fill?" It returns a probability distribution over candidates.

### Real precedent: Ithaca (DeepMind, 2022)

A transformer trained on ancient Greek inscriptions, fielded for
**restoring damaged text in actual archaeological fragments** and
attributing inscriptions to time and place. Same mechanism the user
described: train on intact context, infer over missing pieces. Worked
because language has redundancy; a model that internalized the
redundancy can fill local gaps better than human experts in many cases.

### Constraints for true alien text

- **Need substantial intact corpus.** Roughly 1 MB minimum, ideally
  10 MB+, of clean unbroken passages. Below that, the model can't
  internalize the patterns.
- **Blind spots must be local.** Filling a few missing tokens between
  intact context works well. Filling entire pages without surrounding
  context doesn't (and the further "out" you extrapolate, the worse
  the model gets).
- **No way to validate.** With known languages you can grade predictions
  against ground truth. For true alien text, you only get the model's
  guess with confidence — and the confidence may be miscalibrated on
  out-of-distribution material.
- **Will produce *something* even when it shouldn't.** Models almost
  never refuse to predict; you must hold outputs as "best guess given
  patterns" not "the answer."

### A subtler use: structural inspection

Even without filling blanks, training an LM on the alien corpus and
inspecting WHAT IT LEARNS (which tokens cluster, which sequences are
common, where the entropy spikes) reveals language structure. This is
**computational decipherment** — a real subfield in linguistics that's
been quietly LLM-augmented in the last few years.

### Open question to revisit

How does this work for languages with VERY different statistical
structure from English? (E.g., agglutinative languages with very long
words, or pictographic systems.) The Ithaca result is on a Indo-European
language with copious cognate data. The story might be more brittle for
truly novel statistical structures.

---

## 2026-06-07 — Entry 10: Chinchilla is compute-optimal, not globally optimal

**During Session 5 self-check.** User answered "if you trained the same
22M model on 10× the data, would that help?" with: *"nope, we are at
optimal ratio, if more it will get overtrained based on research."*

That's a common misconception worth pinning down: **Chinchilla's
"20 tokens per param" rule is COMPUTE-OPTIMAL, not GLOBALLY OPTIMAL**.

### Two different "optima"

| Concept | What it asks | Answer |
| ------- | ------------ | ------ |
| **Compute-optimal** (Chinchilla) | given a fixed compute budget, what params/tokens split minimizes loss? | ≈ 20 tokens / param |
| **Global-optimal for a target model size** | given that I want a *small* model, how much data should I train it on? | as much as you can afford |

The original Chinchilla paper (Hoffmann et al., 2022) framed itself in
terms of compute-optimal training. Reading it as "training small models
past 20:1 is wasted" is wrong — the paper just doesn't address that
question directly.

### What modern practice actually does

Frontier labs deliberately over-train small models:

- **Llama-3-8B**: ~15 trillion tokens of training. Ratio ≈ 15T / 8B ≈
  1,875 tokens/param — almost 100× past Chinchilla.
- **Phi-3-mini**: similar over-trained regime, with the extra twist of
  textbook-quality curated data.
- **Llama-3.1, Llama-3.3, Qwen 2.5, Mistral Small**: all trained well
  past Chinchilla.

### Why over-train?

**Inference cost.** A bigger Chinchilla-optimal model is smarter per
training dollar, but you pay its inference cost every time you serve
it, forever. A smaller over-trained model is "more expensive to train
per quality point" but "much cheaper to RUN per quality point."

Math sketch: a model deployed to serve N inferences over its lifetime
has total cost ≈ `training_compute + N × inference_compute_per_call`.
For high-N regimes (consumer products, agents that call the model
millions of times), inference dominates. Over-training trades training
compute (one-time) for inference compute (per-call, forever).

### The Phi-3 corollary

Combining over-training with carefully curated textbook-quality data
breaks the scaling-laws curve. Phi-3-mini (3.8B params) matches much
larger models on reasoning benchmarks because:

1. It's over-trained (much more data than Chinchilla would allocate)
2. The data is hand-curated for educational value (Phi's "textbooks")

Both moves happen *outside* the standard scaling-laws framework.
Scaling laws describe "more compute → less loss"; Phi-3 shows "better
data + over-training → orders of magnitude better practical capability."

### Implications for huge-base-v1

If you trained huge-base-v1 (22M params) on 4.1B tokens (10× more, ~120
hours wall time), you would NOT hit a wall. You'd get a meaningfully
better 22M model — just at the cost of spending 10× the training time.
Whether that's worth it depends on how often you'll run it: high
throughput → yes; low throughput → no.

For TinyGPT specifically: the product framing is "Mac platform for
building specialists for your specific tasks." Those specialists will be
hit many times (chat, completions, etc.) — so over-training the base
might be a real lever to pull, distinct from "make the base bigger."

### Open question to revisit

The over-training regime has been popular for ~18 months at the time of
writing. Where does the curve actually flatten for small models trained
WAY past Chinchilla — 1000× the optimum, say? The Phi/Llama-3 regime
suggests there's still meaningful improvement at 100×; nobody (publicly)
knows where it stops.

---

## 2026-06-07 — Entry 11: Embedding is dictionary, body is reader

**During Session 6.** User asked: *"if 57% of huge-base-v1's parameters
are in the embedding, what's the remaining 43% storing that's MORE
valuable than the embedding?"* Plus a related model: *"embeddings are
like chains of text, and the model tries to fit input words into those
chains."*

Both are excellent questions and both deserve a careful answer because
the misconceptions are common.

### Embeddings are NOT chains

Each token has ONE vector. For huge-base-v1, that's 256 floats. Total
storage for the whole embedding = vocab × d_model = 12.5M floats.

The vector is a fixed POINT in a high-dimensional concept space, not a
chain or tree of related concepts. Similar concepts (`cat`, `dog`)
cluster near each other in this space; distant ones (`cat`,
`differential equations`) are far apart. But there's no linked list, no
graph, no chain.

When the model receives an input, it looks up the vectors for each
token in the input and feeds them into the transformer body. There's no
"walking through chains."

### The body is what does the actual reasoning

The 12 transformer blocks (~9.5M params, 43% of total) handle three
distinct jobs that embeddings can't:

1. **Attention.** For each token, computes "which OTHER tokens in the
   sequence matter for predicting the next one?" This is where contextual
   meaning is resolved. The word `bank` has the same embedding vector
   regardless of context. But when surrounded by `river`, attention
   draws connections to `river` and produces a "river-bank" interpretation
   for the next prediction. When surrounded by `money`, it draws
   connections to `money` and produces "financial-bank" interpretation.
   Same embedding; different runtime computation.
2. **MLPs (feedforward).** Combines features from many positions to
   derive higher-order patterns. "These two phrases together imply this
   concept." This is where pattern recognition and combinatorial reasoning
   live.
3. **Layer norms + residuals.** Plumbing that keeps gradients flowing
   and keeps representations well-scaled across depth.

### The dictionary-reader analogy

- **Embedding = dictionary.** Each word has a fixed definition. Looking
  up a word gives you its meaning in isolation.
- **Body = reader.** Combines word meanings sequentially, resolves
  ambiguities from context, derives sentence-level and discourse-level
  meaning, predicts what comes next.

You need both. But:
- Huge dictionary + tiny reader = can recall lots of word definitions but
  can't compose them into thoughts.
- Smaller dictionary + bigger reader = can do deep reasoning on a
  slightly less rich vocabulary.

For language MODEL capability, the body is more valuable per parameter
than the embedding past a certain point. This is why huge-base-v1's 57%
embedding share is suboptimal — too much budget on the dictionary, too
little on the reader.

### Practical consequence

For a fixed total parameter budget:
- Larger vocab → larger dictionary → smaller reader → less compute-capable
- Smaller vocab → smaller dictionary → larger reader → more compute-capable
- The optimum depends on the data and task, but for small models on
  general English text, ~16K–30K vocab is typically a better balance than
  49K.

### Open question to revisit

For multimodal models (vision + language), the embedding handles a much
larger "vocabulary" (image patches, audio frames). Does the dictionary
vs reader trade-off still apply, or does the multimodal regime have
different optima? The recent MM-LLM literature is just beginning to
publish on this.

---

## 2026-06-07 — Entry 12: Memorization vs generalization (the Harry Potter test)

**Surfaced during Session 6 experiments.** User asked: *"if I train the
model on, let's say, Harry Potter text, won't it be able to predict the
story from any given sentence?"*

Yes — and that's the point of overfitting / memorization. Worth
unpacking because it's the cleanest concrete case for understanding
when a model is "learning" vs "memorizing."

### Capacity vs data ratio drives the answer

- Harry Potter complete series ≈ 1.1M words ≈ 1.5M tokens
- huge-base-v1: 22M params
- Ratio: 22M params ÷ 1.5M tokens ≈ **15 params per token**

Inverse Chinchilla: instead of 20 tokens per param, this is ~0.07 tokens
per param. The model has MASSIVELY more capacity than data. The result is
overfitting: the model has plenty of room to memorize the exact text.

### What "predict the story" would actually look like

- Give it a UNIQUE early sentence (`"Mr. and Mrs. Dursley, of number
  four, Privet Drive..."`) → very high chance of near-verbatim
  continuation of the original passage.
- Give it a COMMON sentence (`"He said."`) → could continue in many
  directions; might mix passages.
- With temperature = 0 (greedy) → deterministic, often near-perfect
  recall.
- With temperature > 0 → recognizable Harry Potter style but with
  variations.

This is exactly what the existing `data/checkpoints/huge-shakespeare-5000-loss1.22.tinygpt`
demonstrates: huge trained on Shakespeare for 5000 steps achieved loss
1.22 because the model essentially memorized large chunks of Shakespeare.
The same recipe on Harry Potter would produce a "Harry Potter
parrot."

### When this is feature vs bug

**Feature:** for specialists, memorization of the specific domain is
exactly what you want. A code completion model SHOULD remember common
idioms; a SQL specialist SHOULD memorize syntax patterns; a customer
support specialist SHOULD memorize the company's product details.

**Bug:** for general-purpose LMs, memorization is a generalization
failure. The model can't help you with NEW Harry-Potter-style stories;
it can only recite existing passages. Also legally fraught (this is the
core of the OpenAI / NYT / authors copyright lawsuits — GPT-4 has been
shown to reproduce copyrighted text verbatim).

### Connection to the curriculum

This is the *Q5 trap* from Session 2 made concrete. "Loss = 0 means
perfect predictions" sounds great, but if it's predicting verbatim from
memorized training data, that's not "the model learned the structure" —
it's "the model became a compressed copy of the training set."

The diagnostic: train/val gap. If training loss is near 0 but validation
loss (on held-out passages) is high, you have memorization without
generalization. If both are low, you have actual learning.

### Open question to revisit

What's the minimum model size that still meaningfully *generalizes* from
a small corpus rather than memorizing? Phi-3's "textbook quality" data
trick may partly work because curated data forces patterns the model has
to learn rather than memorize. But this isn't well-characterized
academically yet.

---

## 2026-06-07 — Entry 13: Temperature ≠ creativity (variety vs capability)

**During Session 7.** User asked: *"why can't I just increase the
temperature to make the model creative?"*

Worth pinning down because temperature-as-creativity-dial is a common
beginner misconception.

### What temperature does

Reshapes the next-token probability distribution before sampling:
- T = 0 → greedy, deterministic, "boring but coherent"
- T = 1 → natural distribution
- T > 1 → flatten distribution, sample more from low-probability tokens
- T → ∞ → uniform sampling, garbage

### What's in the tail (which T amplifies)

- ✓ Novel combinations the model considered but didn't favor
- ✓ Less obvious continuations  
- ✗ Errors, hallucinations
- ✗ Off-topic tokens
- ✗ Grammar mistakes
- ✗ Incoherent transitions

Temperature can't distinguish these. It amplifies ALL of the tail. High
T = high variance: occasionally brilliant, often garbage.

### Variety vs creativity

| Aspect | Variety (T-controllable) | Creativity (capability-controlled) |
| ------ | ------------------------ | --------------------------------- |
| What | different outputs each sample | novel BUT coherent combinations |
| Mechanism | sampling reshape | underlying distribution shape |
| Knob | temperature | base model + training data |
| Hard limit | model's existing distribution | capacity to generate quality rare options |

**Crucial insight:** a model can only "be creative" within its training
distribution. Temperature lets you SEE the diversity that's already
there; it doesn't ADD diversity that isn't.

### Distribution vs capability framing

Think of the model as a search engine over its learned distribution:
- Small/weak model = small index. High T = same boring stuff with errors
  mixed in.
- Large/strong model = rich index. Moderate T = genuine variety because
  many plausible-and-good options exist in the distribution.

**Temperature is a band-aid for an underpowered model, not a substitute
for capability.**

### Top-k and top-p (the related knobs)

Usually paired with temperature in modern sampling:
- **top-k**: consider only the k most likely tokens, then sample. Cuts
  worst tail.
- **top-p (nucleus)**: consider tokens until cumulative probability ≥ p,
  then sample. Adaptive version of top-k.

These let you raise T for variety within plausible options while clipping
the obvious garbage. Modern systems typically combine moderate T (≈0.8)
+ top-p (≈0.9). Best of both worlds.

### Practical advice for character/NPC work

- T ≈ 0.7–0.9 = sweet spot for character work. Some variety, mostly stays
  in character.
- T = 0 → NPCs robotic, identical responses.
- T > 1.2 → NPCs unpredictable AND incoherent. Often break character.

**If a character feels boring, the fix is more diverse training data + a
better base + moderate T.** NOT crank T higher on a weak base.

---

## 2026-06-07 — Entry 14: Optimization audit — TinyGPT vs Kaiju + modern small models

**Triggered by HuggingFace research for the character recipe.** User
asked: "how can we use the different optimisations [from Character.AI
Kaiju + modern small models] in our own model training regimen?"

Short answer: **TinyGPT already ships virtually all of them.** The gap
is that the default `huge` preset deliberately doesn't enable them —
it's a vanilla GPT-2 teaching baseline. Modernizing the preset would be
a free quality bump.

### Full audit table

| Optimization | Where it comes from | TinyGPT status | Used by `huge` preset? |
| ------------ | ------------------- | -------------- | ---------------------- |
| **MQA** (Multi-Query Attention) | Kaiju | ✅ shipped (`nKvHeads: 1`) | ❌ (uses full MHA) |
| **GQA** (Grouped Query Attention) | Llama-2+ | ✅ shipped | ❌ (uses full MHA) |
| **Sliding window attention** | Mistral, Kaiju (5:1 ratio) | ✅ shipped (`--sliding-window N`) | ❌ |
| **Cross-layer KV sharing (YOCO)** | Kaiju, Sun et al. 2024 | ✅ shipped (`--yoco`, `CrossAttention.swift`) | ❌ |
| **INT8/INT4 quantization (inference)** | Standard | ✅ shipped (`MLXNN.quantize`) | n/a (inference) |
| **KV cache quantization (KIVI)** | KIVI paper | ✅ shipped (`--kv-quantize`) | n/a (inference) |
| **QAT** (Quantization-Aware Training) | Kaiju | ✅ shipped (`--qat`) | optional |
| **RMSNorm** | Llama+ | ✅ shipped (`cfg.useRMSNorm`) | ❌ (uses LayerNorm) |
| **SwiGLU MLP** | Llama+, PaLM | ✅ shipped (`cfg.useSwiGLU`) | ❌ (uses GELU) |
| **RoPE positional encoding** | Llama+ | ✅ shipped (`cfg.useRoPE`) | ❌ (uses learned positional) |
| **Embedding RMSNorm** | Recent 2025 finding | ✅ shipped (`cfg.useEmbeddingRMSNorm`) | ❌ |
| **DeepNorm residual scaling** | Microsoft 2022 | ✅ shipped (`cfg.useDeepNorm`) | ❌ |
| **Layer-wise LR decay** | Standard fine-tune trick | ✅ shipped (`cfg.lrLayerDecay`) | optional |
| **Cosine warmup schedule** | Standard | ✅ shipped (`--lr-schedule cosine`) | optional |
| **WSD schedule** | MiniCPM | ✅ shipped (`--lr-schedule wsd`) | ✅ (huge uses this) |
| **BPE dropout** | Provilkov 2019 | ✅ shipped (`BPEDropout.swift`) | optional |
| **Knowledge distillation** | Hinton et al. | ✅ shipped (`tinygpt distill`) | n/a |
| **DPO / SimPO / KTO / ORPO** | RLHF alternatives | ✅ shipped (`tinygpt dpo` with flags) | n/a |
| **LoRA + DoRA + VeRA + LoftQ + AdaLoRA + RsLoRA + PISSA + LoRA-FA + LayerDrop** | various | ✅ shipped (entire bundle) | n/a |
| **Multi-Token Prediction (MTP)** | Gloeckle 2024, DeepSeek-V3 | ✅ shipped (`cfg.mtpHorizons > 1`) | ❌ |
| **MoE (dense routing)** | Switch Transformer | ✅ shipped (`cfg.nExperts > 1`) | ❌ |
| **NEFTune** (noisy embeddings) | Jain 2024 | ✅ shipped (`--neftune-alpha`) | n/a (SFT-time) |
| **Curriculum-quality classifier** | FineWeb-Edu approach | ✅ shipped (`tinygpt train-quality-classifier`) | n/a (data) |
| **Speculative decoding (Medusa, EAGLE-2)** | Standard | ✅ shipped (`--draft`, `--heads`) | n/a (inference) |
| **Synthetic data generation** | Phi-3 textbook approach | ✅ shipped (`tinygpt synthesize`) | n/a |
| **ALiBi positional** | Press 2022 | ✅ shipped (alt to RoPE) | ❌ |
| **Tied embeddings** | Standard | ✅ shipped | ✅ (huge uses this) |
| **Gradient checkpointing** | Standard | ✅ shipped | optional |
| **Squinch gradient compression** | Kaiju proprietary | ❌ not shipped | n/a (multi-GPU only; mostly irrelevant for Mac local) |
| **Sparse MoE kernels** | DeepSeek | ⬜ TODO (dense routing works; sparse compute is on backlog) | n/a |
| **Token-preserving trajectory recorder** | Poolside | ⬜ TODO (B22 in PLAN.md) | n/a (RL feature) |

### Headline finding

**~95% of modern small-model + Kaiju optimizations are already shipped
in TinyGPT.** The remaining gaps are either niche (Squinch is for
multi-GPU communication, mostly irrelevant on single-Mac), niche-2-2
(sparse MoE kernels — dense routing works for training, sparse compute
is a future engineering project), or non-architectural (B22 trajectory
recorder is RL infrastructure, not a base-training optimization).

### The real opportunity: modernize the `huge` preset

The huge preset (`ModelConfig.huge` in `ModelConfig.swift`) currently
uses GPT-2-vanilla architecture:

- LayerNorm (not RMSNorm)
- GELU MLP (not SwiGLU)
- Learned positional embeddings (not RoPE)
- Full MHA (not GQA/MQA)
- No YOCO

**This is a teaching choice** — the preset is deliberately "what a
clean GPT-2 looks like" so the curriculum can introduce modernizations
one at a time later. But for production-quality small models, the
`huge` preset is leaving free quality on the table.

A modernized `huge-v2` preset turning these on by default would:
- Reduce loss by approximately 5-15% (free win from SwiGLU + RoPE +
  RMSNorm based on published ablations)
- Reduce inference memory ~30% (free win from GQA + YOCO)
- Train slightly slower per step but converge in fewer total steps

Worth a small PRD. Flagged in `factory-character-specialist-recipe.md`
as out-of-scope but mentioned.

### Curriculum implication

A future **Session 8 — production training optimizations** could walk
through this audit, explaining what each optimization does and why
modern small models use most of them. The journal entry is the
short-form reference; the session would be the long-form teaching
version. Adding to the curriculum backlog.
