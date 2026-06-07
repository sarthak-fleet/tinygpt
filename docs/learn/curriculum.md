# TinyGPT learning curriculum — ground up

A 7-session curriculum that takes you from "what's a neural net" to
"I can read any modern ML paper without hand-waving."

Designed for someone building TinyGPT (the factory) + Pace (the
specialist customer) — every concept ties back to actual code in this
repo or in clickyLocal.

## Sessions

| # | Title | Status | File |
|---|---|---|---|
| 1 | **From a line to a learned line** | drafted (2026-06-06) | `session-01-neural-net-basics.md` |
| 2 | **How the search for `m` and `b` actually works (gradient descent)** | drafted (2026-06-06) | `session-02-gradient-descent.md` |
| 3 | **What makes a neural net more than linear regression (non-linearities)** | drafted (2026-06-06) | `session-03-non-linearities.md` |
| 4 | **The taxonomy of ML approaches (and where transformers fit)** | drafted (2026-06-06) | `session-04-ml-paradigms.md` |
| 5 | **Scaling: why bigger models know more (and what scaling doesn't cover)** | drafted (2026-06-07) | `session-05-scaling.md` |
| 6 | **Tokenization + embeddings: how text becomes numbers** | drafted (2026-06-07) | `session-06-tokenization-embeddings.md` |
| 7 | **How models learn behavior (supervised, imitation, reinforcement)** | drafted (2026-06-07) | `session-07-behavior-learning.md` |
| 8 | **Training mechanics: the actual loop (batches, epochs, schedules, failure modes)** | drafted (2026-06-07) | `session-08-training-mechanics.md` |
| 9 | Production training optimizations (MQA, sliding window, YOCO, QAT, SwiGLU, RoPE, RMSNorm — audit + when to use each) | backlog (audit in `journal.md` Entry 14) | — |
| 4 | The transformer | not-started | — |
| 5 | Tokenization, embeddings, output head | not-started | — |
| 6 | Pretraining + post-training (SFT, LoRA, QLoRA, DPO, distillation) | not-started | — |
| 7 | Inference + interpretability (KV cache, spec dec, quant, SAE, constrained) | not-started | — |

## Pedagogy

- **Math when it matters, words when math is overkill.** Won't dodge
  equations but won't drown you.
- **Tie every concept to TinyGPT code.** Grep-able.
- **Worked examples with tiny numbers.** 4-dim attention head with
  concrete vectors beats hand-waving.
- **Honest about controversy.** Where literature disagrees, says so.
- **You can stop / redirect / push back.** Your fluency is the goal.

## Where you'll be at session 7

- Read papers like Mistral 7B tech report and understand every claim
- Look at a loss curve and predict whether more training helps
- Choose SFT / LoRA / distillation / DPO with reasons for the specific case
- Debug "why is my model bad" with architectural intuition
- Make ML hires + read ML resumes critically

## Companion resources (linked, not duplicated)

- `docs/learn/journal.md` — running log of questions, hunches, and
  tangents from the live sessions; co-primary artifact with the polished
  session files
- `docs/learn/archive/session-01-neural-net-basics-dense.md` — older,
  vectors-and-matrices-from-the-start version of Session 1; kept as a
  faster-paced reference for after the gentler arc lands
- `docs/learn/llm-mechanics-fundamentals.md` — earlier matmul-first
  primer, kept as a high-level reference
- `docs/learn/external-references.md` — curated external articles (LLM
  internals, Apple Neural Engine, distillation, philosophical takes)
  with one-line "why this matters to TinyGPT" notes
- Karpathy "Zero to Hero" YouTube series — most directly aligned external
  series; we cover the same ground but tied to your code
- 3blue1brown Neural Networks series — visual intuition; ~3 hrs total
- The Annotated Transformer (Harvard NLP) — line-by-line of the
  original 2017 paper with running PyTorch
