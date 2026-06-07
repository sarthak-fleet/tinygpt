# Session 1 — From a line to a learned line

> **Where we're starting:** you know basic 2D math (lines, equations) and what
> a function is. That's it. Everything else is built here.
>
> Companion to the live teaching session. Read time: ~15 minutes.

## What we're standing on

- A **function** = a rule that takes an input and gives back an output. Same
  input → same output, every time.
- A **line** = `y = mx + b`. Slope `m`, intercept `b`. Give me `x`, I give
  you `y`.

That's enough to build everything from.

---

## The flip

In math class, you usually see questions like:

> *"Given y = 2x + 3, what's y when x = 5?"*

You're **told** the line — `m` and `b` are handed to you. You plug in
`x = 5`, get `y = 13`. Done.

But that's not how the world works. In real life, you almost never know
the line in advance. You see **data** — examples of inputs paired with
outputs — and you have to figure out the rule yourself.

Concrete example. You collect 5 houses:

```
sqft   |  price ($K)
-------|------------
1000   |  250
1500   |  350
2000   |  420
2500   |  510
3000   |  600
```

You strongly suspect: *price increases roughly linearly with sqft.* So
you guess the rule has the shape:

```
price = m · sqft + b
```

There IS some `m` and some `b` that makes this approximately true. You
don't know what they are yet. Plot the data:

```
price
 600 |                       •
 500 |                  •
 400 |             •
 300 |        •
 250 |   •
     +-------------------------- sqft
       1000  1500  2000  2500  3000
```

By eye, you can almost *see* the line going through those points. Maybe
`m ≈ 0.18`, `b ≈ 75`. Doesn't matter — the point is **a good line
exists**, and the job is to find it.

That job — finding the right `m` and `b` from data — is **learning**.

---

## Same equation, two completely different questions

This is the conceptual centerpiece. Same equation `y = mx + b`. Two
opposite ways to use it:

|                    | Math class                    | Machine learning                       |
| ------------------ | ----------------------------- | -------------------------------------- |
| **What you have:** | `m`, `b`, and one `x`         | A pile of `(x, y)` pairs               |
| **What you want:** | The `y` that matches that `x` | Good values of `m` and `b`             |
| **The work:**      | Plug in. Compute.             | Search. Try values. Adjust. Try again. |

Same equation. Different question.

Everything else in this field — every neural net, every LLM, every image
model — is a fancier version of this exact flip. The function gets more
complicated. The data gets more complicated. But the question stays the
same: *given examples, find the rule.*

---

## Sidebar: what's "fixed" vs what "changes" during learning

This trips everyone up the first time. The four symbols in
`y = mx + b` play two very different roles:

| Symbol | Role          | Where it comes from   | Does it change during training? |
| ------ | ------------- | --------------------- | ------------------------------- |
| `x`    | input         | given by the data     | **no** — frozen                 |
| `y`    | output        | given by the data     | **no** — frozen                 |
| `m`    | **parameter** | started as a guess    | **yes** — search variable       |
| `b`    | **parameter** | started as a guess    | **yes** — search variable       |

Learning = the `(x, y)` pairs sit still, and we move `m` and `b` around
until predictions land near the actual y's.

When someone says "a 7-billion parameter model," that 7B is the count of
`m`-and-`b`-like things. The inputs aren't part of "the model." **The
model IS its parameters.**

---

## So what's a neural network?

The simplest possible neural network is `y = mx + b`. The smallest unit
(called a **neuron**) is literally a line. We only change two things,
both cosmetic:

| Math-class name | ML name    | Same thing?    |
| --------------- | ---------- | -------------- |
| slope `m`       | weight `w` | yes, identical |
| intercept `b`   | bias `b`   | yes, identical |

So `y = mx + b` and `y = wx + b` are the same equation. ML people just
renamed it.

Why "neuron"? Mostly historical baggage — 1950s researchers loosely
analogized multiply-and-add to brain cells "weighing" incoming signals.
**Don't take the brain metaphor seriously.** A neuron is multiply-and-add.
That's it.

When you hear "neural network," picture **a line being fit to points.**
Everything more elaborate is built by stacking and twisting this base case.

---

## What's NOT in this picture yet

Deliberately left out — named without teaching:

- **More than one input** (price depends on sqft AND bedrooms): just
  `price = w₁·sqft + w₂·bedrooms + b`. Still a "line" in higher
  dimensions. → future session.
- **More than one output**: multiple lines in parallel. → future session.
- **The data isn't actually linear** (price grows fast, then plateaus):
  one line can't capture this, which is why "non-linearities" exist.
  → future session.
- **How do you actually search for `m` and `b`?** Gradient descent.
  → next session.

---

## Self-check

Don't peek. Answer in your own words:

1. What does it mean to "learn" `m` and `b`?
2. If I hand you `y = 3x + 1`, is that a trained model or untrained? Why?
3. If you had just ONE data point — `(sqft=1000, price=250)` — could you
   find `m` and `b`? Why or why not?

Question 3 has a punchline: **with N parameters, you generally need at
least N data points to pin them down.** Modern LLMs have hundreds of
billions of parameters — and accordingly need oceans of data. That's the
data race.

---

## Where this connects

- Single neuron = **linear regression** (200 years old; Gauss & Legendre,
  early 1800s). The "neural network" framing is mostly historical
  branding on top of "stack many linear regressions with non-linear bumps
  between them." See `journal.md` Entry 1.
- The dense, vectors-and-matrices version of this material lives in
  `archive/session-01-neural-net-basics-dense.md` — same ideas, faster
  pace, assumes more math fluency.
- Next: **Session 2** — how the search for `m` and `b` actually works
  (gradient descent, done properly).
