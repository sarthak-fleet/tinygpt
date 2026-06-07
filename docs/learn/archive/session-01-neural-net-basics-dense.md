# Session 1 — What's a neural net (the building block)

> Read time: ~25 minutes. Companion to the live teaching session.

A neural net is, at the bottom, a function:

```
y = f(x; W, b)
```

You give it an input `x` (numbers), it gives you an output `y` (numbers).
The parameters `W` and `b` are the things that get learned. The whole
field is about: **what shape of function should `f` be, how do we choose
`W` and `b` to make `y` useful.**

Let's build up from a single neuron.

## 1. The single neuron — linear regression in disguise

The simplest possible "neural net":

```
y = w · x + b
```

`x` is a single number, `w` is a single number (the weight), `b` is a
single number (the bias). This is just `y = mx + b` from middle school.
Slope-intercept form.

If you have data points `(x_i, y_i)` and you want a line through them,
this is what you fit. **A neuron with one input is linear regression.**

### Vector version

If `x` is a vector of D numbers (say, the pixels of an image, or the
features of a row in your spreadsheet):

```
y = w · x + b
  = w_1·x_1 + w_2·x_2 + ... + w_D·x_D + b
```

This is still a single number out, but now from D inputs. The "weights"
`w` are now a vector with D entries. This is **a single neuron with D
inputs**.

Concrete tiny example: predict house price from (sqft, bedrooms, age):
```
price ≈ 200 · sqft + 50000 · bedrooms + (-1000) · age + 30000
```
The weights `[200, 50000, -1000]` tell you "each sqft adds $200, each
bedroom adds $50K, each year of age subtracts $1K." `30000` is the base
price (the bias).

### Geometric meaning

The neuron divides D-dimensional space with a (D-1)-dimensional
**hyperplane**. Inputs on one side give positive output, the other
side gives negative. Just like a line on a 2D plot divides the plane.

That's all a neuron is geometrically: **a tilted plane through space.**

## 2. A whole layer — many neurons in parallel

If you want M outputs from D inputs, stack M neurons:

```
y_1 = w_1 · x + b_1
y_2 = w_2 · x + b_2
...
y_M = w_M · x + b_M
```

Each row of weights is a separate neuron. This is exactly **matrix
multiplication**:

```
y = W · x + b
```

Where `W` is an M×D matrix, `x` is a D-vector, `y` is an M-vector,
`b` is an M-vector.

**That's why matmul shows up everywhere in ML.** A "layer" is just
"compute multiple neurons in parallel," which means "multiply by a
matrix." Every linear layer in PyTorch (`nn.Linear(D, M)`) is one
matmul + one add.

### In TinyGPT code

Open `native-mac/Sources/TinyGPTModel/TransformerBlock.swift`. You'll
see `MLX.matmul(...)` calls. Every single one of those is "compute a
layer of neurons in parallel." Once you internalize "matmul = a layer,"
the rest of the code is just composition.

## 3. The limit of one layer — why we need more

Stack two linear layers:

```
h = W_1 · x + b_1
y = W_2 · h + b_2
```

Substitute:

```
y = W_2 · (W_1 · x + b_1) + b_2
  = (W_2 · W_1) · x + (W_2 · b_1 + b_2)
```

Two layers collapse to ONE matrix. `W_2·W_1` is just another matrix.

**So stacking linear layers buys you nothing.** This is why we need
non-linearities.

## 4. Non-linearities — the "neural" part

Between layers, we apply a **non-linear function** element-wise. The
classics:

| Function | Formula | What it does |
|---|---|---|
| **ReLU** | `max(0, x)` | Zero out negatives. Cheap, ubiquitous. |
| **Sigmoid** | `1 / (1 + e^{-x})` | Squashes to (0, 1). Old-school. |
| **Tanh** | hyperbolic tan | Squashes to (-1, 1). |
| **GELU** | `x · Φ(x)` (Gaussian CDF) | Smoother ReLU. **The transformer default.** |
| **SwiGLU** | gated variant | What Llama/Qwen/Mistral use today. |

With a non-linearity between layers, the network can represent functions
linear layers cannot. The classic example:

### The XOR problem

Try to separate these four points with a single line:

```
(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0
```

No single line works — the two "1" points are diagonal, you can't draw
a line that puts them together. **One layer cannot learn XOR.**

But two layers + a non-linearity can. The hidden layer learns to
project the points into a space where they ARE linearly separable.
That's the trick: **layers + non-linearities let the network reshape
its input space into one where the task is easy.**

This is the entire reason "deep" networks beat shallow ones. Each layer
reshapes the input a little more.

## 5. The forward pass — putting it together

A 3-layer net (input → hidden1 → hidden2 → output):

```
h_1 = activation(W_1 · x + b_1)
h_2 = activation(W_2 · h_1 + b_2)
y   = W_3 · h_2 + b_3      # last layer often has no activation
```

That's it. **The whole "AI" is just this chain of matmul → non-linearity
→ matmul → non-linearity → ...** Transformers, vision models, every
neural net you've heard of. They differ in *what shape of matrices* and
*how they're composed*, but the building block is universal.

## 6. Parameters — what gets learned

For a layer `y = W·x + b` with D inputs and M outputs:
- `W` has D × M numbers
- `b` has M numbers
- Total: D·M + M parameters

A 3-layer net with 1000 inputs and 500-unit hidden layers:
- Layer 1: 1000 × 500 + 500 = 500,500
- Layer 2: 500 × 500 + 500 = 250,500
- Layer 3: 500 × 10 + 10 = 5,010 (10 outputs)
- **Total: 756,010 parameters**

Each parameter is one float (typically 32 or 16 bits). Modern LLMs have
billions of parameters. Your Pace planner today: ~30 billion. The
TinyGPT distill target: ~600M-1.5B.

## 7. What "training" means (a preview)

We have a dataset of `(x_i, y_i)` pairs (inputs and the right answers).
We want the network's prediction `f(x_i; W, b)` to be close to `y_i`.

Define **loss** as a number that's small when predictions are good,
large when they're bad. Classic: mean squared error.

```
loss = average over i of (f(x_i; W, b) - y_i)^2
```

Training = adjust `W` and `b` to make loss smaller. **That's it.** The
"how" is gradient descent, which is Session 2.

## 8. Where we'll go next

| Session | What we'll cover |
|---|---|
| 2 (next) | Gradients + backprop: HOW we adjust W to reduce loss |
| 3 | The actual training loop with batches, epochs, learning rate |
| 4 | Transformers: a specific architecture for sequences |

## 9. Code anchor — see this in TinyGPT

The simplest neural-net forward pass in TinyGPT is the LM head — the
final layer that projects from the transformer's hidden state to
vocabulary scores. Find it in `TinyGPTModel.swift`:

```swift
// rough shape: hidden state [batch, seq_len, d_model] →
//               logits [batch, seq_len, vocab_size]
let logits = MLX.matmul(hidden, embeddingMatrix.transposed())
```

That single line is exactly the building block from Section 2:
`y = W · x` (no bias because we tie weights to the embedding matrix —
more on that in Session 5).

## 10. Test your intuition

Without scrolling up:

1. Why doesn't stacking linear layers give us more power?
2. What's the role of ReLU / GELU?
3. If a layer takes 1024 inputs and outputs 4096 numbers, how many
   parameters does it have?
4. In what shape is the parameters dimension of the matrix? `(in, out)`
   or `(out, in)`? (Hint: think about which dimension dot-products with x.)

Answers at end.

---

### Answers

1. Composing two linear maps gives another linear map. Without
   non-linearity, depth ≠ expressivity.
2. They let the network reshape its input space non-linearly. Without
   them, the network is one big matrix.
3. 1024 × 4096 + 4096 = 4,198,400 (4.2M)
4. `W` is shape `(out, in)` if you think of `y = W·x` with x as a
   column vector. PyTorch/MLX often store it `(out, in)` and call this
   "out_features × in_features." The math is the same.
