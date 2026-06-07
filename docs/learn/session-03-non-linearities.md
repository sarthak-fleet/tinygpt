# Session 3 — What makes a neural net more than linear regression

> Closes the open question from Session 1's tangent: at what point does a
> neural network stop being equivalent to linear regression?

## Where we are

- A neuron = a line: `y = wx + b`
- We can find good `w, b` via gradient descent
- We can measure "good" via loss

What can this setup **not** do, and how do we fix it?

---

## The limit: one line can only learn linear relationships

Imagine data that's flat-then-rising:

```
y
 |         *
 |        *
 |       *
 |      *
 |---**
 |
 +---------- x
```

Can a single line fit this? No. Any line is rising, flat, or falling all the
way through — never "flat then rising."

Or an arch (parabola):

```
y
 |    *
 |  *   *
 | *     *
 +-------- x
```

A single line through this is going to look terrible no matter what `(w, b)`
you pick.

**Conclusion:** lines only capture relationships where y changes at a
*constant rate* as x changes. Real data often doesn't.

---

## Attempt 1: stack two lines (doesn't work)

If one line is limited, try two:

```
h = w₁·x + b₁     (line 1)
y = w₂·h + b₂     (line 2)
```

The first transforms x → h. The second transforms h → y. Substitute:

```
y = w₂·(w₁·x + b₁) + b₂
  = (w₂·w₁)·x + (w₂·b₁ + b₂)
```

Let `W = w₂·w₁` and `B = w₂·b₁ + b₂`. Then `y = W·x + B`. **Still a line.**

Two lines stacked collapse to one line. With different parameter values, but
mathematically identical to a single line. **Stacking lines buys nothing.**

This generalizes: no matter how many lines you stack — 2, 10, 1000 — they
collapse to one line. "Deep linear networks" are a mathematical curiosity,
not a useful architecture.

---

## The trick: insert a non-linear kink between the lines

Fix: insert a non-linear function *between* the two lines. Then the algebra
can't simplify them away.

```
h  = w₁·x + b₁                (line 1)
h' = NON-LINEAR FUNCTION(h)   ← the new ingredient
y  = w₂·h' + b₂               (line 2)
```

The non-linear step survives substitution. The two lines stay distinct, and
the total function can take shapes neither line could make alone.

---

## The simplest non-linearity: ReLU

The modern default is **ReLU** — Rectified Linear Unit:

```
ReLU(z) = max(0, z)
```

If z positive, output z. If z negative, output 0. That's it.

```
ReLU(z)
  |       /
  |      /
  |     /
  |    /
  |___/____ z
       0
```

Two half-lines glued at zero. The *kink* at zero is exactly the
non-linearity we needed.

---

## One ReLU neuron = a hockey stick

```
y = ReLU(w·x + b)
```

The inside `w·x + b` is a line — positive for some range of x, negative for
the rest. ReLU zeros out the negative part.

Result: a hockey stick.

```
y
 |        /
 |       /
 |      /
 |     /
 |____/____ x
      ↑
   kink at x = −b/w
```

Already something a pure line couldn't do. The kink lets the model say "y
stays at zero until threshold, then rises."

---

## Many ReLU neurons summed = any shape

Now sum the outputs of many ReLU neurons, each with different `w, b`. Each
contributes a hockey stick at its own kink position and slope. The sum is a
piecewise-linear function:

```
y
 |       ____
 |      /    \
 |     /      \
 |    /        \___
 |   /             \
 |__/               \___ x
```

With enough hockey sticks at the right positions, you can approximate *any*
function. Smooth curves, sharp peaks, multi-bumps — all as a sum of straight
pieces.

This is the **Universal Approximation Theorem** (Cybenko 1989, Hornik 1991):

> A neural network with one hidden layer and enough neurons + a non-linear
> activation can approximate any continuous function to any desired
> accuracy.

**In words: non-linearity + enough neurons = unlimited flexibility.**
Without the non-linearity, you're stuck with lines forever. With it, you can
fit anything.

This is *the* answer to Session 1's tangent. A single neuron without
activation = linear regression. Add a non-linearity → strictly more
powerful. The boundary between "statistics" and "deep learning" is exactly
**the non-linear kink.**

---

## Honest definition of a neural network

> A neural network is a chain of linear-then-non-linear operations:
>
> ```
> h₁ = ReLU(w₁·x + b₁)
> h₂ = ReLU(w₂·h₁ + b₂)
> y  = w₃·h₂ + b₃         (last layer often has no activation)
> ```

Each (linear + non-linear) pair is a **layer**. Count of layers = **depth**.

---

## Why depth, not just width?

UAT says one hidden layer + enough neurons can approximate any function. So
why use 100 layers instead of 1 huge wide one?

**Honest answer: depth is empirically more parameter-efficient.**
Approximating a hard function with 1 layer might need millions of neurons.
With 4 layers, thousands.

Intuition (not theorem): each layer learns features built on the previous
layer's features. Layer 1 finds edges/slopes. Layer 2 combines them into
corners/curves. Layer 3 combines into textures/parts. Hierarchy lets each
layer specialize.

UAT is proven; the "depth helps" claim is mountains of empirical evidence,
no closed-form proof. See journal Entry 5 on the empirical-vs-theoretical
state of ML.

---

## When NOT to use a deep net

This is half the skill. If your data is genuinely linear:

- Linear regression is the right tool. Closed-form solution exists.
- A deep ReLU network is strictly worse: more parameters → harder to
  optimize, more risk of overfitting, no benefit.

The general principle: **match the model's complexity to the data's
complexity.** Deep networks shine on rich data (images, language, audio)
where the underlying patterns are non-linear and hierarchical. They're
overkill for tabular data with smooth linear-ish relationships, where
classical models (linear regression, gradient-boosted trees) often beat
them.

---

## Other non-linearities (named, not taught)

ReLU is the modern default. You'll see others:

| Name | Formula | Shape | Notes |
|------|---------|-------|-------|
| **ReLU** | `max(0, z)` | hockey stick | dominant; transformer FFN default |
| **Sigmoid** | `1 / (1 + e⁻ᶻ)` | S-curve, squashes to (0, 1) | mostly retired (gradient vanishing) |
| **Tanh** | hyperbolic tangent | S-curve, squashes to (−1, 1) | retired for same reason |
| **GELU** | smoother ReLU | rounded hockey stick | original transformer paper |
| **SwiGLU** | gated variant | more complex | Llama / Mistral / Qwen default |

For now, **ReLU is the canonical non-linearity** — internalize it, others
are variations on a theme. See journal Entry 6 on the activation function
subfield's history.

---

## Self-check

Don't peek:

1. Why does stacking linear layers without a non-linearity buy you nothing?
2. What does a single ReLU neuron's output look like?
3. If your data is perfectly linear, would a 10-layer ReLU network beat a
   single-layer linear regression? Why or why not?
4. UAT says one hidden layer + enough neurons can fit any function. Why do
   real models use 100+ layers?

---

## Where this connects

- Closes the **linear regression vs neural net** question from Session 1's
  tangent + journal Entry 1.
- Sets up future sessions on:
  - **Multiple inputs / outputs** — same idea, vectors and matrices.
  - **Specific architectures** — CNN, RNN, Transformer — each is a way of
    arranging the linear-then-non-linear pattern with structural
    constraints (e.g., shared weights, attention).
  - **Why depth helps in practice** — the empirical evidence and current
    theoretical attempts.
