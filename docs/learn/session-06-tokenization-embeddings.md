# Session 6 — Tokenization + embeddings: how text becomes numbers

> The bridge between text (symbols) and neural networks (numbers). The
> first layer of every modern language model, and the one that quietly
> consumes 57% of huge-base-v1's parameters.

## Where we are

Sessions 1–5 covered what a neural net IS and what it learns. But we've
been hand-waving over a key question: **the model is a function on
numbers. Where do the numbers come from when the input is text?**

This session covers that bridge — tokenization (text → integer IDs) and
embedding (integer IDs → vectors of floats) — and why these choices
matter more than most beginners are told.

---

## The problem

Neural networks are functions on numbers. They take numbers in, do
arithmetic (matmul, ReLU, etc.), produce numbers out.

But text is symbols: characters, words, punctuation. `"the cat sat on
the mat"` is not numbers. Before any neural net can process it, we need
to turn it into numbers.

---

## Naive approach 1: character-level

Assign each character a number (e.g., ASCII).

```
'a' = 97   'b' = 98   'c' = 99   ...
"cat" = [99, 97, 116]
```

**Why it's not great:** sequences get very long (one token per
character), inference is proportionally slow, network learns from very
fine-grained signals.

---

## Naive approach 2: word-level

Assign each word a number. Build vocabulary from a corpus.

```
"the" = 1   "cat" = 2   "sat" = 3   ...
"the cat sat" = [1, 2, 3]
```

**Why it's not great:** vocabulary explodes (English has 1M+ word
forms); can't handle out-of-vocabulary words; subword structure lost
(`running` and `ran` look unrelated).

---

## Naive approach 3: byte-level

Each byte of UTF-8 is a token. Vocabulary fixed at 256.

```
"café" → [99, 97, 102, 195, 169]   ("é" is 2 UTF-8 bytes)
```

**Better than character, worse than something smarter:**
- Vocabulary fixed at 256 — never grows, never has `<UNK>`.
- Handles any language, any character.
- But sequences still long (one token per byte).
- This is `byte-tinygpt-v0` in the browser. Fine for teaching;
  inefficient at scale.

---

## The modern answer: BPE (Byte Pair Encoding)

> Start with bytes. Find the most common adjacent pair. Merge it into a
> new token. Repeat until you have the vocabulary size you want.

### Tiny worked example

Corpus:

```
"low lower lowest"
```

**Step 0** — characters:

```
l o w _ l o w e r _ l o w e s t
```

**Step 1** — `l o` appears 3 times. Merge to `Lo`:

```
Lo w _ Lo w e r _ Lo w e s t
```

**Step 2** — `Lo w` appears 3 times. Merge to `Low`:

```
Low _ Low e r _ Low e s t
```

**Step 3** — `Low e` appears 2 times. Merge to `Lowe`:

```
Low _ Lowe r _ Lowe s t
```

After enough merges, common patterns become single tokens:

```
low _ lower _ lowest    ← 3 tokens instead of starting 16
```

The vocabulary contains: every original byte (no `<UNK>`) PLUS all
learned merges. For SmolLM2's BPE: 49,152 tokens = 256 bytes + 48,896
learned merges.

---

## What BPE buys

- Common words → single tokens.
- Rare/novel words → graceful degradation to subwords/bytes.
- Subword structure preserved (`run + ning`).
- Tunable vocab size: 8K → 200K depending on model size and language.

---

## Same text, three tokenizers

`"Hello, world! 42 ML"`:

| Tokenizer | Tokens | Count |
|-----------|--------|-------|
| Byte-level (vocab 256) | `H · e · l · l · o · , · _ · w · o · r · l · d · ! · _ · 4 · 2 · _ · M · L` | **19** |
| SmolLM2 BPE (vocab 49K) | `Hello · , · _world · ! · _ · 42 · _ML` | **7** |
| GPT-4 BPE (vocab 100K) | similar to SmolLM2 | **~7** |

BPE is ~3× more compact than byte-level on English.

---

## From tokens to vectors: the embedding layer

Tokenization gives integer IDs. The neural network needs vectors of
floats. The **embedding layer** is a lookup table:

```
Token ID 42 ("cat")
    ↓ lookup in embedding table (row 42)
[0.12, -0.45, 0.78, 0.21, ..., 0.31]    ← 256 floats
    ↓ flows into the transformer
```

The embedding table is a matrix:

```
shape: (vocab_size, d_model) = (49152, 256) = 12,582,912 parameters
```

That's where the 12.5M comes from — and why it's 57% of huge-base-v1's
total parameter budget.

---

## What embeddings actually learn

Vectors are random at initialization. Training updates them along with
everything else. They end up encoding meaning.

**Tokens with similar meanings get similar vectors.** The classic
word2vec result (Mikolov et al., 2013):

```
vec("king")   - vec("man")    + vec("woman") ≈ vec("queen")
vec("Paris")  - vec("France") + vec("Italy") ≈ vec("Rome")
vec("walked") - vec("walk")   + vec("run")   ≈ vec("ran")
```

The embedding space encodes semantic and syntactic relationships as
*directions*. Not by design — it emerges from gradient descent on
"predict the next word."

---

## Embedding stores identity. The body stores reasoning.

A common confusion: thinking of embeddings as "chains" or "trees" of
related concepts. They're not.

> **Each token has ONE vector. That's it.**

The vector is a fixed POINT in a high-dimensional concept space.
Similar concepts cluster near each other; distant concepts are far
apart. But there are no chains, no linked lists, no graph structure
stored in the embedding.

The "chains" of reasoning — what concept relates to what, given THIS
input — are built at runtime by the BODY (attention + MLPs). The
embedding is dictionary; the body is reader. See journal Entry 11 for
the longer version of this distinction.

---

## Output side: tied vs untied embeddings

The final step predicts probabilities over the vocabulary, requiring a
matrix of shape `(d_model, vocab_size)`.

| Choice | Shape | Extra params | Quality |
|--------|-------|--------------|---------|
| Untied | separate output matrix | +12.5M for huge | slightly better |
| **Tied** (huge-base-v1) | reuse input embedding transposed | 0 | slightly worse |

Tying saves 12.5M params at small quality cost. Most small models tie;
many large models don't.

---

## Vocab size as a design trade-off

| Vocab size | Tokens per text | Embedding params (d=256) | Notes |
|------------|-----------------|--------------------------|-------|
| 256 (byte) | many (~5× more) | 65,536 | inefficient inference |
| 8,192 | few (~1.5× more) | 2,097,152 | balanced for small models |
| 49,152 (SmolLM2) | baseline | 12,582,912 | huge-base-v1 |
| 100,000 (GPT-4) | slightly fewer | 25,600,000 | medium-big models |
| 200,000+ | fewer for non-English | 51,200,000+ | multilingual frontier |

The trade-off: bigger vocab → fewer tokens per text but bigger embedding;
smaller vocab → more tokens per text but more body capacity at the same
total params.

**For huge-base-v1**, SmolLM2's 49K vocab is structurally too big.
16K–30K would give the body more capacity. The 57% embedding share is a
real architectural imbalance — one of the "could have been better" items.

---

## Common pitfalls

- **Numbers tokenize inconsistently.** "1234" might be one token, two,
  or three — making arithmetic reasoning surprisingly hard for LLMs.
- **Whitespace is sticky.** `" python"` vs `"python"` are different tokens.
- **Non-English handled poorly by English-trained BPE.** Chinese
  characters become 2-3 tokens; English words become 1.
- **Code has special needs.** Indentation, brackets, keywords get
  special treatment in code-trained BPEs (StarCoder, Code Llama).
- **Tokenizer mismatches break models.** Train on tokenizer A, infer
  with tokenizer B → garbage. Vectors are indexed by token ID.

---

## How `d_model` is decided (and where the body's capacity comes from)

Designer choice. Common considerations:

- **Param budget.** `total_params ≈ vocab × d_model + n_layers × const × d_model²`
  (rough). For a fixed total, d_model trades against layers.
- **Width vs depth.** Bigger d_model → richer per-token representations;
  bigger n_layers → more compositional reasoning. Empirically the
  optimum is in the middle.
- **Hardware constraints.** d_model must be efficient for matmul (often
  a multiple of 64 or 128).
- **Rough heuristic.** d_model often grows roughly with the cube root of
  parameter count.

huge-base-v1: d_model=256, n_layers=12. Narrow for its depth. Deliberate
small-teaching choice; "real" small models would use 384 or 512.

---

## Self-check

Don't peek:

1. **Why is byte-level tokenization "free of unknown tokens"?**
2. **Why does the embedding layer use so many parameters in
   huge-base-v1?** Give the math.
3. **If you wanted to make huge-base-v1 better at the SAME parameter
   budget, what's one change you could make to the tokenizer?**
4. **Trap question:** if BPE produces fewer tokens per text, doesn't
   that mean the model has LESS information to work with per sentence?
   Why isn't this a problem?

---

## Where this connects

- The 57% embedding share of huge-base-v1 (Session model dive) is a
  direct consequence of choosing SmolLM2's 49K vocab for a 22M model.
- Embedding-vs-body capacity trade-off appears in every model design —
  see journal Entry 11.
- Next: training mechanics (the actual loop with batches, epochs,
  schedules) — or open.
