# How an LLM actually works (the matmul-first explanation)

**Audience:** anyone who understands "tokens are character chunks, model predicts next token" but wants to see *where matrix multiplication comes in* and *why everything is built around it.*

**Premise:** modern LLMs are basically "a giant pile of matrices that, multiplied in sequence with text input, output sensible text." This doc traces every matmul in a forward pass.

## The 60-second version

| Step | What happens | Matmuls |
|---|---|---|
| Token IDs → vectors | Look up each token in an `embedding` matrix (one-hot × matrix) | 1 |
| Self-attention (per layer) | Project to Q/K/V; compute attention; weighted sum; output projection | 6 |
| Feed-forward (per layer) | Expand → non-linearity → contract | 2 |
| Final projection | Last token's vector × embedding-matrixᵀ → vocabulary scores | 1 |
| Softmax → next token | Element-wise; not a matmul | 0 |

For a 12-layer Huge model: **~108 matmuls in the body + ~1 input lookup + 1 output projection = ~110 matmuls per next-token prediction.**

## The full trace

### 0. Token IDs become vectors (one matmul)

The model can't work with token IDs (`42`, `1337`) directly — those are just labels. So:

```
token_id 42  →  embedding[42]  →  a vector of d_model floats
                                   (for Huge: 256 floats)
```

`embedding` is a learned matrix of shape `[vocab_size × d_model]`. Looking up row 42 is mathematically `one_hot(42) · embedding` — a matrix multiplication. Matmul #1.

After this, your token "hello" is a 256-number vector. The model now thinks in continuous space.

### 1. Positions added (not a matmul)

Each position in the sequence gets a positional encoding (rotary / ALiBi / learned — varies). Added element-wise to the token embedding. No matmul here.

After Steps 0+1 you have a `[sequence_length × d_model]` matrix where each row is "the meaning of the token at this position."

### 2-N: transformer blocks (the heavy lifters)

Each block has two sub-layers:

#### Sub-layer A: self-attention (6 matmuls)

```
X is [seq_len × d_model]

Q = X · W_Q       ← matmul: project to "query" space
K = X · W_K       ← matmul: project to "key" space
V = X · W_V       ← matmul: project to "value" space

scores = Q · Kᵀ   ← matmul: every query meets every key
                    shape: [seq_len × seq_len]

softmax(scores) → attention probabilities

output = softmax_scores · V    ← matmul: weighted sum of values
final  = output · W_O          ← matmul: project back to d_model
```

`Q · Kᵀ` is what makes attention work — it's how every token "looks at" every other token and computes how relevant they are for understanding the current one.

#### Sub-layer B: feed-forward MLP (2 matmuls)

```
intermediate = X · W_1     ← matmul: expand d_model → d_mlp (4× usually)
intermediate = GELU(intermediate)   ← non-linearity (element-wise)
output = intermediate · W_2 ← matmul: contract d_mlp → d_model
```

Two matmuls + one non-linearity. The non-linearity is what makes deep networks more powerful than a single matmul — otherwise stacking layers would mathematically collapse to one big matmul.

### N+1: output projection (one matmul)

After all blocks, you have a `[seq_len × d_model]` matrix. Take the last row (the last token's representation after seeing all context):

```
logits = last_row · embedding_matrixᵀ
         ([d_model]) · ([d_model × vocab_size]) → [vocab_size]
```

This is the embedding matrix reused (transposed!). The same matrix that converted token IDs to vectors converts the final hidden state back to vocabulary scores. This **weight tying** saves parameters.

### N+2: softmax + sample (not a matmul)

```
probabilities = softmax(logits)
next_token = argmax(probabilities)  OR  sample(probabilities)
```

Element-wise math. Pick the next token.

## Why matrix multiplication specifically?

Three compounding reasons:

| Why | Detail |
|---|---|
| **Simplest learnable transformation** | `y = W·x` has one free parameter set (W). Easy to differentiate, easy to optimize via gradient descent. |
| **Composes well** | Two stacked matmuls = mathematically one bigger matmul. To get expressive power, you sprinkle non-linearities (GELU, softmax) between them. Non-linearities give the *capacity* to learn; matmuls do the *work*. |
| **GPU hardware optimization** | GPUs and TPUs are matmul machines. H100 ≈ 10¹⁵ matmul ops/sec. Building the architecture out of matmuls uses ~95% of hardware's theoretical capability. |

## Where the model's "knowledge" lives

The model's weight matrices ARE the model:

- `embedding_matrix` — what each token "means" as a vector
- `W_Q`, `W_K`, `W_V`, `W_O` (per layer) — what aspects of meaning to compare, how
- `W_1`, `W_2` (per layer) — how to transform each token's representation given context

When you **train**, you learn the values in these matrices via backpropagation.
When you **fine-tune**, you nudge them slightly toward your task.
When you **distill**, you copy what a bigger model's matrices have learned into smaller matrices.
When you **quantize**, you compress the float values from 32 → 4 bits.
When you **prune**, you zero out matrix entries that don't matter.
When you **edit** (MEMIT), you surgically modify specific rows of specific matrices.

A "22M parameter model" means **22 million floats stored in these matrices.** That's the entire model.

## Why size and bandwidth matter

**Why bigger models are smarter:** more matrix entries → more capacity to fit nuanced patterns. A 7B model has 318× the "shapeable surface" of a 22M model. But the cost is proportional in both memory and compute.

**Why memory bandwidth matters at inference:** to generate ONE token, you must read the ENTIRE model through compute. For a 7B model at bf16 = 14 GB read per token. Max tok/s = `bandwidth / model_size`. (See the inference-performance section of `docs/sessions/2026-06-06-mac-specialist-platform.md` for hardware-specific numbers.)

**Why architecture matters:** transformer vs Mamba vs MoE = "what shape are the matrices, in what order do we multiply them, where do we put the non-linearities." Different choices have different speed/memory/quality tradeoffs.

## Mental model summary

> An LLM is a stack of ~10-100 layers. Each layer is "multiply by a matrix, apply a non-linearity, multiply by another matrix." Tokens are pushed through the stack as vectors. Each layer's matrix is *learned* by showing the model billions of training examples and adjusting until it predicts the next token well.
>
> When you talk to an LLM, every token you see is the result of one full pass through the stack — about 110 matmuls per token for a small model, thousands per token for a frontier model. The matrices are the model. Everything else (positional encoding, layer norms, residual connections, attention masks) is plumbing to make the matmul-stack stable and expressive enough to learn language.

## Further reading

- 3blue1brown's "Neural Networks" series on YouTube — visual intuition for matmul
- The Annotated Transformer (Harvard NLP) — every line of attention code annotated
- Karpathy's "Let's build GPT" — implements all this from scratch in PyTorch
- `native-mac/Sources/TinyGPTModel/TransformerBlock.swift` — our actual Swift implementation; ~515 lines covering Q/K/V projection, attention, MLP exactly as described above

## Related TinyGPT docs

- `docs/sessions/2026-06-06-mac-specialist-platform.md` — strategy doc; covers memory bandwidth math + tokenization frontier
- `docs/learn.md` — broader learning roadmap (if it exists)
