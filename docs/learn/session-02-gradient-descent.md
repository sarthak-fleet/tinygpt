# Session 2 — How the search for `m` and `b` actually works

> Where we left off: a line `y = mx + b` has two parameters. Given data, we
> want to find good values. Session 1 waved at "search" — this session opens
> the box.

## The setup, restated

We have:

- **Data:** `(x, y)` pairs. Fixed.
- **Model shape:** `y = mx + b`. Fixed shape.
- **Parameters:** `m` and `b`. Currently unknown. To be found.

Before we can search for "good" values, we need to define **what good means.**
The whole search machinery is built on this one definition.

---

## Step 1 — Measuring how wrong (loss)

Pick some random guess: `m = 0, b = 0`. The model now predicts `y = 0` for
every input. Compare to the 5 houses from Session 1:

```
sqft   | predicted | actual | error (pred − actual)
-------|-----------|--------|----------------------
1000   |     0     |  250   |   −250
1500   |     0     |  350   |   −350
2000   |     0     |  420   |   −420
2500   |     0     |  510   |   −510
3000   |     0     |  600   |   −600
```

**Why we can't just sum errors:** positives and negatives would cancel, and a
bad fit could look fine. So we **square** each error first:

- Squaring kills the sign (negative² = positive)
- Squaring punishes big errors more — error of 100 contributes 10,000; error
  of 10 contributes 100. 10× bigger error → 100× more punishment.

Average over all N data points:

```
loss = (1/N) · Σ (predicted − actual)²
```

This is **mean squared error** (MSE). One number that summarizes "how wrong
is this guess `(m, b)`?"

- Loss = 0 → predictions perfect
- Loss = huge → predictions awful
- Goal of training: **make loss small.**

> **Sidebar: is squaring the only choice?** No. See journal Entry 3 for the
> L1 vs L2 vs Huber landscape, why MSE is the default rather than the only
> option, and the Gaussian-noise reason it's privileged.

> **Sidebar: loss numbers are NOT comparable across setups.** A common
> trap. For language modeling specifically, cross-entropy loss has a
> ceiling of `log(vocab_size)` — the random-guessing baseline. A
> byte-level model (vocab=256) has a ceiling of 5.55; a BPE-49K model
> has a ceiling of 10.80. The same "quality" of model produces very
> different absolute numbers in different setups. Never compare two
> loss numbers unless they're from the same vocab, same data, and same
> tokenization. To compare across setups, normalize to **bits per byte
> (bpb)**: `bpb = loss / log(2) / avg_bytes_per_token`. See journal
> Entry 8 for a worked example.

---

## Step 2 — The landscape view (the key insight)

The move that unlocks everything: **loss is itself a function.** A function
of what?

Of `m` and `b`. Pick different `(m, b)`, get different loss.

```
loss = loss(m, b)
```

So we can imagine a landscape:

- two horizontal axes: `m` and `b`
- vertical axis: loss

For a line + MSE, that landscape is a perfect **bowl**:

```
   loss
    |
    |  \\           //
    |    \\        //
    |      \\     //         (b axis going into the page)
    |        \\__//
    |         \__/  ← lowest point = best (m, b)
    +---------------- m
```

Exactly one lowest point. That lowest point is the `(m, b)` that fits the
data best. **Finding it is the entire job of training.**

We want the **minimum**, not the maximum. Going to the highest point would
be gradient *ascent*, used in reinforcement learning to maximize rewards.
Sign matters — get it backward and the model gets steadily worse.

---

## Step 3 — Going downhill, blindfolded

We don't have a map of the whole landscape. If we did, we'd just look up the
lowest point. What we CAN do is evaluate loss at our current spot, and ask:
*if I nudge `m` slightly, does loss go up or down? Same for `b`?*

The "nudge test":

- Current: `(m=0, b=0)`. Loss = 27.67 (say).
- Try `m = 0.01` (keep `b=0`). New loss = 27.44. Dropped, so **increasing
  m reduces loss.** We want to move m up.
- Try `b = 0.01` (keep `m=0`). New loss = 27.57. Also dropped. Move b up.

The information "which way is downhill, how steeply, for each parameter" is
called the **gradient**.

> **Calculus sidebar (skippable on first read):** the nudge test is exactly
> what a derivative measures — *"rate of change of loss as I move m a tiny
> bit."* For MSE on a line, there's a clean formula for the rate, so you
> don't have to actually nudge. We'll use the formula in the worked example
> below, but the *meaning* is just "which way is downhill, and how steeply."
> If you ever forget what the formula is doing, picture nudging.

---

## Step 4 — Take a step

Once we know which way is downhill, we step that way:

```
m_new = m_old − (step size) × (slope of loss in m direction)
b_new = b_old − (step size) × (slope of loss in b direction)
```

**Why minus?** The slope (gradient) points *uphill*. We want to go opposite
— downhill — so we subtract.

**Step size** is called the **learning rate**, written `η` (eta).

| Learning rate | Effect                                                       |
| ------------- | ------------------------------------------------------------ |
| Too small     | Crawls. Takes forever to reach the bottom.                   |
| Too big       | Overshoots. Bounces around the bowl. Sometimes diverges.     |
| Just right    | Smooth descent to the bottom.                                |

There's no formula for the perfect learning rate. People try 0.1, 0.01,
0.001 and see what works. One of the few genuinely empirical knobs in ML.

---

## Step 5 — Repeat

The full algorithm:

```
1. start with random (m, b)
2. repeat for many iterations:
     a. predict y for each data point using current (m, b)
     b. compute loss
     c. compute gradient (which way is downhill, how steeply)
     d. take a small step downhill
3. stop when loss stops shrinking
```

That's **gradient descent.** The whole thing. Every model trained today —
GPT-4, Stable Diffusion, AlphaFold — uses some flavor of this loop. The
complexity comes from how you compute the gradient for very deep models
(backpropagation, future session) and from data plumbing. The core idea is
exactly these five steps.

---

## Worked example with small numbers

Data: `(1, 3), (2, 5), (3, 7)`. The true line is `y = 2x + 1` — you don't
know that; we're going to recover it.

Start: `m = 0, b = 0`. Learning rate `η = 0.1`.

### Iteration 1

**Predict:**

```
x=1 → 0·1 + 0 = 0    (actual 3, error −3)
x=2 → 0·2 + 0 = 0    (actual 5, error −5)
x=3 → 0·3 + 0 = 0    (actual 7, error −7)
```

**Loss:**

```
loss = (1/3) · [9 + 25 + 49] = 27.67
```

**Gradient** (using the calculus shortcut for MSE on a line):

```
slope_m = (2/N) · Σ (predicted − actual) · x
        = (2/3) · [(−3)(1) + (−5)(2) + (−7)(3)]
        = (2/3) · (−34)
        = −22.67

slope_b = (2/N) · Σ (predicted − actual)
        = (2/3) · (−15)
        = −10
```

Both slopes negative → downhill is the positive direction → increase both
`m` and `b`.

**Step:**

```
m_new = 0 − 0.1 × (−22.67) = +2.27
b_new = 0 − 0.1 × (−10)    = +1.0
```

After ONE iteration: `(m, b) = (2.27, 1.0)`. True answer: `(2, 1)`. Nearly
there from one step on a tiny problem.

In a real run we'd iterate hundreds of thousands of times on millions of
data points, and loss would steadily shrink toward the minimum.

---

## What's NOT here yet

Named without teaching, so you know what's coming:

- **Mini-batches (stochastic gradient descent).** For huge datasets, we
  don't compute loss over ALL data each step — we use a random slice. The
  noise this introduces is actually *helpful*, not just tolerable. See
  journal Entry 4 for why, and for the question of curriculum/stratified
  vs random selection.
- **Smarter optimizers.** Adam, momentum, RMSProp. Gradient descent with
  bookkeeping. → future session.
- **What if the landscape isn't a perfect bowl?** Deep networks have bumpy
  landscapes — multiple valleys, saddles, plateaus. Gradient descent might
  land in a "pretty good" valley instead of the global best. Empirically,
  this rarely bites in practice (mysterious but true). → future session.
- **How is the gradient computed for a deep network?** Chain rule, scaled
  up. Called **backpropagation.** → future session.

---

## Self-check

Don't peek. Answer in your own words:

1. **What is "loss"? What is it a function of?**
2. **Why do we square the errors instead of just adding them up?**
3. **What does the gradient tell us — both the sign and the magnitude?**
4. **What happens if the learning rate is too big? Too small?**
5. **Trap question:** if loss is exactly 0, what does that tell you about
   the model? Could that ever be a bad thing?

The last one is a setup for a future session (overfitting). The answer: if
real noisy data fits with zero loss, the model has likely *memorized* rather
than *learned the pattern.* See "where this connects" below.

---

## Where this connects

- Q5's trap connects to **overfitting** — central problem in ML. The whole
  subfield of *regularization* is techniques for preventing it. Future
  session.
- The "loss is a function of parameters" framing is what makes everything
  scale: for a 7B-parameter LLM, loss is a function of 7 billion variables.
  Same idea as our 2-parameter line. Same gradient descent. Just very
  high-dimensional.
- Next: **Session 3** — what makes a neural net *more* than linear
  regression. Stacking + non-linearities. The question your Session 1
  tangent surfaced.
