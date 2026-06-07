# Session 4 — The taxonomy of ML approaches (and where transformers fit)

> *Closes the open question from Session 3's model dive:* "is there tree
> regression? RL? learning by doing? Transformers don't use all of them, so
> what are the other kinds and why aren't transformers used for those?"
>
> Short answer: the premise mixes three orthogonal axes. Untangling them
> shows transformers ARE used across most paradigms, just not all model
> families. This session pulls the axes apart and then dives deep on
> transformers specifically.

## The confusion: "ML approach" mixes three different things

When people say "supervised learning," "neural net," "gradient descent,"
"RL," "tree regression" — they're naming things from **three different
axes** that are mostly independent and freely combine. The single biggest
clarifying move:

```
TRAINING SIGNAL × MODEL FAMILY × UPDATE ALGORITHM
   (what you      (what shape    (how parameters
    learn from)    the function     change)
                   takes)
```

Every actual ML system picks one from each axis. Most "X vs Y" debates
people have are really about ONE axis while the other two are held
constant.

---

## Axis 1 — Training signal (what you learn FROM)

| Paradigm | Signal looks like | Canonical example |
| -------- | ----------------- | ----------------- |
| **Supervised** | `(input, correct answer)` pairs | spam classification with labeled emails |
| **Self-supervised** | label derived from input itself | next-token prediction — label IS the next word |
| **Unsupervised** | no labels | k-means clustering, PCA, autoencoders |
| **Reinforcement** | reward signal from environment | game-playing agent, RLHF |
| **Semi-supervised** | mix of labeled + unlabeled | rare in modern practice — replaced by SSL pretraining + supervised fine-tune |

**Self-supervised is the big shift of the last decade.** Modern LLMs are
self-supervised: the "label" for predicting the next word is just the next
word that actually came in the training corpus. No human had to label
anything; the data labels itself. This is what made LLM-scale pretraining
possible — you can use the entire internet as training data without
labeling effort.

**Reinforcement learning** has a different shape: instead of "here's what
you should have said," the signal is "here's how good what you did was."
The agent acts, the environment gives a reward, the agent learns to take
actions that lead to more reward over time. Classical RL: game playing,
robotics. Modern application: RLHF — the second training stage that turns
a raw pretrained LLM into ChatGPT.

---

## Axis 2 — Model family (what shape the function takes)

| Family | Function shape | Best for |
| ------ | -------------- | -------- |
| **Linear regression / GLMs** | line / plane | tabular numeric prediction with small data |
| **Decision trees** | tree of yes/no splits on features | small tabular |
| **Random forests** | average of many trees | medium tabular |
| **Gradient-boosted trees (XGBoost, LightGBM)** | sequence of trees each correcting prior errors | **tabular data — often beats neural nets** |
| **SVMs** | margin-maximizing boundary | classical classification, small data |
| **k-NN** | "copy the nearest training examples" | recommenders, image lookup |
| **Bayesian methods** | full probability distributions | when calibrated uncertainty matters |
| **Neural networks: CNN** | shared-weight convolutions | images |
| **Neural networks: RNN / LSTM** | shared-weight recurrence | sequences (replaced by transformers) |
| **Neural networks: Transformer** | shared-weight self-attention | sequences, the modern default for language/audio/video |

**Trees vs neural nets** is the most underrated debate. On tabular data
(rows × columns of features, like a CSV), gradient-boosted trees often
beat transformers. The reason: trees have strong inductive biases that
match how tabular data actually works — they handle missing values,
categorical features, and non-monotonic relationships natively, without
needing millions of examples to learn those tricks. Most Kaggle wins on
tabular problems are XGBoost or LightGBM, not transformers.

**Transformers vs everything else** for sequential, high-dimensional data:
basically no contest now. Language, audio, video, DNA — transformer
variants have eaten the field.

---

## Axis 3 — Parameter update algorithm (how the parameters change)

| Algorithm | Idea | Used for |
| --------- | ---- | -------- |
| **Gradient descent / SGD / Adam** | step against the gradient of loss | anything differentiable (= almost all NNs) |
| **Closed-form (normal equation)** | solve algebraically for the optimum | linear regression with MSE — 1805 math |
| **Tree-building (CART, ID3)** | greedily pick best split at each node | decision trees |
| **Gradient boosting** | fit each new tree to residual errors of the prior ensemble | GBT / XGBoost |
| **EM (Expectation-Maximization)** | iteratively fill in missing labels and refit | Gaussian mixtures, HMMs |
| **MCMC (Markov Chain Monte Carlo)** | sample from a probability distribution | Bayesian posterior inference |
| **Policy gradient / Q-learning / Actor-Critic** | step parameters to maximize expected reward | RL |

The axes are **mostly independent**: you can pair any signal with any
family with any update algorithm, as long as the math works out. The
"works out" caveat is what differentiable models care about — gradient
descent needs the model to be differentiable, which is why neural nets and
SGD pair so naturally.

---

## The matrix — what combinations are common

| Combination | Name people use | Example system |
| ----------- | ---------------- | -------------- |
| transformer + self-supervised + SGD | **pretraining** | GPT-4 base, your `huge-base-v1.tinygpt` |
| transformer + supervised + SGD | **SFT (supervised fine-tuning)** | InstructGPT phase 1 |
| transformer + RL + policy gradient | **RLHF / RLAIF** | ChatGPT, Claude, Llama-Chat |
| neural net + RL + various | **deep RL** | AlphaGo, AlphaStar, OpenAI Five |
| GBT + supervised + boosting | **XGBoost** | most tabular Kaggle wins |
| linear + supervised + closed-form | **ordinary least squares** | most statistical regression |
| Bayesian + supervised + MCMC | **Bayesian regression** | calibrated uncertainty |
| autoencoder + self-supervised + SGD | **representation learning** | SimCLR, MoCo, MAE |

So the claim "transformers don't use RL" was wrong. The combination
**transformer + RL** is exactly RLHF, used in every modern frontier model.
The model family stays transformer; the *training signal* changes from
self-supervised next-token to reward-from-human-preference.

---

# Deep dive: transformers across the matrix

The transformer is one model family, but it shows up in several training
configurations. Worth knowing what each is for.

## Pretraining (self-supervised, SGD)

The base layer of every modern LLM. The training task: given the first N
tokens of a document, predict the (N+1)-th token. Loss is cross-entropy.

Why it works: predicting the next word forces the model to internalize a
huge amount of knowledge about how language works, what facts are true,
what reasonable arguments look like, etc. There's no other way to predict
"the capital of France is ____" correctly without knowing the capital of
France.

Your `huge-base-v1.tinygpt` is this. 200,000 SGD steps over ~410M tokens
of FineWeb-Edu. Loss went from 11.37 (uniform random) to 4.16 (good
language model). At the end, the model knows English grammar, common
factual associations, and basic reasoning patterns — purely from
predicting the next token.

## Encoder-only (BERT-style, masked language modeling)

Different self-supervised task: randomly hide some tokens in a sentence
and predict the hidden ones from BOTH sides of context. The transformer
sees the whole sentence at once, not just the past.

Use case: classification, retrieval, embedding. The encoder produces a
rich representation of the sentence as a whole. Not used for generation —
it can't produce text token by token because it sees the future.

Examples: BERT, RoBERTa, MiniLM. Smaller than modern LLMs. Used heavily
inside search engines and retrieval pipelines.

## Decoder-only (GPT-style, causal language modeling)

What your model is. The transformer is masked so each token only sees
tokens BEFORE it — never the future. This is what makes generation
possible: you can produce one token at a time, and the model isn't
"cheating" by peeking ahead.

This is the modern default. GPT-2/3/4, Claude, Llama, Mistral, Qwen,
DeepSeek, Phi — all decoder-only.

## Encoder-decoder (T5-style)

The encoder reads the input fully; the decoder generates the output
token-by-token, attending to the encoder's representation. Used for
sequence-to-sequence tasks where input and output are clearly distinct:
translation (English → French), summarization (article → summary), code
generation (description → code).

Examples: T5, BART, FLAN-T5. Mostly replaced by decoder-only models with
prompting, but still common in production systems where the input/output
boundary is clean.

## Supervised fine-tuning (SFT)

Take a pretrained base model and continue training on `(prompt, ideal
response)` pairs collected from humans. Same SGD on cross-entropy loss,
but now the "label" is a human-written answer instead of "the next token
that randomly came in web text."

This is what gives the model a consistent assistant persona. After
pretraining, the model knows everything but doesn't know it's supposed to
be helpful; SFT teaches the pattern of "user asks → assistant answers
helpfully and concisely."

In TinyGPT terms: `tinygpt sft` is this. Most fine-tunes use LoRA so they
only update a small fraction of parameters — see `Lora.swift`.

## RLHF / RLAIF

Take the SFT'd model and continue training with reinforcement learning. A
reward model (itself a transformer, trained on human preference pairs)
scores the model's outputs; policy gradient nudges the model's parameters
to produce higher-scoring outputs.

Why this stage exists: after SFT the model is helpful but still has lots
of failure modes — preferring shorter responses when longer are better,
saying things that sound right but aren't, repeating itself. RLHF pushes
the model toward outputs humans actually prefer in side-by-side
comparisons.

This is the same transformer architecture, the same parameter set —
trained with a different signal (reward vs cross-entropy) and a different
algorithm (policy gradient vs vanilla SGD).

---

# Why transformers won

Two reasons people usually give:

1. **Attention can model any-to-any token interaction in one layer.** RNNs
   propagate information sequentially; the gradient has to flow through
   many timesteps to connect a token to one 200 positions earlier.
   Transformers attend directly. Long-range dependencies are O(1) hops
   instead of O(length).

2. **Transformers parallelize on GPU.** RNNs are inherently sequential —
   you can't compute step 100 until step 99 is done. Transformers process
   all tokens in parallel. This is the practical reason that scaling
   transformers up is feasible while scaling RNNs up wasn't.

There's also a sneakier reason: **transformers' inductive bias is
weaker** than RNNs or CNNs, which sounds bad but is actually good when
you have enough data. CNNs assume spatial locality matters; RNNs assume
temporal locality matters. Transformers assume neither. With small data
this hurts (RNNs and CNNs win on tasks like small-image classification).
With internet-scale data, it helps — the model learns whatever locality
structure actually exists rather than having one baked in.

---

# Where transformers are NOT the right tool

| Scenario | Better tool | Why |
| -------- | ----------- | --- |
| Tabular data (rows × columns) | gradient-boosted trees (XGBoost) | trees have right inductive bias for tabular structure |
| < 1000 training examples | linear regression, SVMs, Bayesian methods | NNs need lots of data; simpler models with strong priors win |
| Need calibrated probability | Bayesian methods, Gaussian processes | NN softmax probabilities are notoriously overconfident |
| Strict latency / tiny device | distilled small NN, decision trees | transformers are heavy; classical methods are fast |
| Tasks needing exact rules / guarantees | symbolic systems, constraint solvers | NNs are probabilistic, can be wrong |

The transformer's specific edge — *sequential high-dimensional data with
internet-scale training corpora* — happens to match the most economically
important AI problems of the last decade (language, vision, audio,
multimodal). That doesn't make it the right tool for everything.

---

## Self-check

Don't peek:

1. **Name a system that combines transformer + RL.** What's the training
   signal there, and how is it different from pretraining?
2. **You're a data scientist with 5,000 rows of structured customer data
   and need to predict churn.** What model family would you reach for
   first, and why?
3. **Why is "self-supervised" a misleadingly named paradigm?** What's the
   honest description of where the labels come from?
4. **Trap question:** if transformers can theoretically model anything (a
   universal approximator), and we have infinite compute, do we still need
   trees, SVMs, Bayesian methods? Why or why not?

The trap question is the interesting one. Think about it before peeking
at the answer below.

---

## Trap answer (for #4)

Yes, we'd still need them. Three reasons:

1. **Inductive bias matters when data is limited.** Even with infinite
   compute, you don't have infinite data for every task. Simpler models
   with stronger priors generalize better from small datasets. Universal
   approximation doesn't mean "best from small data."

2. **Calibrated uncertainty.** Neural nets are notoriously bad at "I don't
   know." Bayesian methods natively express uncertainty. For high-stakes
   decisions (medical, financial), the uncertainty estimate matters more
   than the point prediction.

3. **Interpretability.** A decision tree's reasoning is "if income >
   50K and credit_score < 600, then deny" — directly readable. A
   transformer's reasoning is buried in 22M floats. For regulated
   industries (insurance, lending), interpretability is mandatory.

Universal approximation is a statement about *what's expressible*, not
about *what's learnable from finite data, with calibrated confidence, in
a human-checkable way.* Those other axes still matter, and other model
families still win on them.

---

## Where this connects

- Pretraining of TinyGPT (`huge-base-v1.tinygpt`) = transformer +
  self-supervised + SGD. The "easiest" combination conceptually, but the
  one that produced the biggest practical leap of the last decade.
- LoRA, SFT, RLHF, distillation — all in TinyGPT under
  `native-mac/Sources/TinyGPT/SFT.swift` and the various `Lora*.swift`,
  `Distill.swift` files. Each is a different point in the 3-axis matrix.
- The journal Entry 7 has the same axes table for quick reference outside
  the curriculum's session order.
