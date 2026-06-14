---
title: Learn
description: TinyGPT's learning corpus — a ground-up curriculum from "what's a neural net" to modern training mechanics, plus reference + research notes.
---

# TinyGPT learning corpus

A reading map for the `docs/learn/` directory. Three reading paths depending on what you want.

**Start here:** [Mac-local AI mastery map](./mac-mastery-map.md) — the living agenda: everything buildable on a Mac, what's already covered, and the single-machine ↔ distributed boundary. The spine for "learn everything + build everything buildable on this Mac."

## I want to learn ML from scratch

Read these in order — the curriculum is designed as a single arc from
basic math to modern transformer training. Each session is self-contained
but builds on the last.

- [Curriculum overview](./curriculum.md) — the 8-session plan, what each session covers, what you need before starting.
- [Session 1 — From a line to a learned line](./session-01-neural-net-basics.md)
- [Session 2 — Gradient descent](./session-02-gradient-descent.md)
- [Session 3 — Non-linearities](./session-03-non-linearities.md)
- [Session 4 — ML paradigms](./session-04-ml-paradigms.md)
- [Session 5 — Scaling](./session-05-scaling.md)
- [Session 6 — Tokenization + embeddings](./session-06-tokenization-embeddings.md)
- [Session 7 — Behavior learning](./session-07-behavior-learning.md)
- [Session 8 — Training mechanics](./session-08-training-mechanics.md)

## I want the modern-LLM mechanics reference

These document the architectural + algorithmic choices that show up in
current LLMs. They're for someone who knows the basics and wants the
"why" behind specific designs (RoPE, GQA, MoE, etc.).

- [LLM mechanics fundamentals](./llm-mechanics-fundamentals.md) — RoPE, GQA, attention variants, MoE, expert routing

**Interview-grade topic maps** (what / why-it-matters-here / external source / repo anchor — for senior/staff prep):
- [Speech & systems topics](./speech-and-systems-topics.md) — voice-pipeline latency, WER, speech-to-speech, fine-tune debugging, feature selection, queues vs websockets, FSDP2
- [Advanced LLM training & post-training](./advanced-llm-training.md) — ZeRO/FSDP2 depth, precision (bf16/fp8), gradient checkpointing, MoE training, data curation, RLHF/DPO/GRPO/reward-modeling/distillation
- [Advanced LLM inference & serving](./advanced-llm-inference.md) — roofline, KV cache + paging, batching, speculative decoding, quantization, FlashAttention, attention variants, long context, serving architecture
- [Advanced architecture, RAG/agents, eval & system design](./advanced-ml-systems-eval.md) — modern decoder block, attention-as-matmuls whiteboard, RAG, agents, LLM-as-judge, perplexity, contamination, ML system-design rounds, classic-ML depth
- [Qwen3-VL mRoPE + DeepStack](./qwen3-vl-mrope-deepstack.md) — vision-language attention specifics; relevant to Pace's VLM pillar
- [App Intents comparison](./app-intents-comparison.md) — how Pace's action surface relates to macOS App Intents
- [Agent context as a memory hierarchy](./agent-context-hierarchy.md) — the L1/L2/L3 framing for agent context engineering, and the steals it produced (eval failure triage, E9 prompt-tiering A/B, B26 deferred tools)
- [Castform's RL fine-tune platform — what we stole](./castform-rl-finetune.md) — composite reward functions (B28), trace-driven data synthesis (B29), reasoning-depth classification (B30)
- [Tool-calling: how close can a Mac-local small model get to frontier?](./tool-calling-frontier-parity.md) — broken-eval → frontier-validated BFCL gate, the honest size curve, the distillation result (1.7B avg 56→76), and where RL takes over
- [Small-model tool-calling: the SOTA playbook (what others do)](./small-model-tool-calling-playbook.md) — survey of data synthesis, SFT tricks (function masking), RL (ToolRL graded reward, DAPO), eval traps, and on-device serving, with a prioritized steal list
- [Model vs agent — what's actually different](./model-vs-agent.md) — the architectural distinction; "what makes a model into an agent" beyond tools + loop; mapped onto this repo's layers
- [Competitive landscape (2026)](./competitive-landscape.md) — map of fine-tune + eval + interp players, the Mac-first whitespace, and the consolidation signal
- [External references](./external-references.md) — papers, blog posts, code-base reading list

## I want session-specific decisions + project state

Captured-in-the-moment notes from real training sessions and decision points.

- [Eval matrix (2026-06-08)](./eval-matrix-2026-06-08.md) — every Pace LoRA's score against fm-fixtures-v2 (annotated 2026-06-09: v8 baseline non-reproducible mystery)
- [Eval methodology (2026-06-08)](./eval-methodology-2026-06-08.md) — the gate finding that the v1 fixture set was broken; how v2 fixes it
- [ANE research notes](./ane-research/) — M5/M6/M7/M8 bisects, ANE precision drift, chunked inference findings
- [Learning journal](./journal.md) — running log of questions, hunches, tangents while doing the work

## Conventions

- Each session has a one-sentence "where we're starting" up top — read that first.
- Annotated diagrams (in text) prefer ASCII over images so they're readable in git diffs and on every renderer.
- "Why this matters to THIS project" is called out where applicable — the curriculum isn't generic, it's anchored to tinygpt's choices.
- Annotated retroactively when an earlier claim turns out wrong (see eval-matrix's 2026-06-09 addendum for an example).

## Related

- Top-level `docs/PLAN.md` — long-term project roadmap (not learning material)
- Top-level `docs/prds/` — per-feature PRDs (not learning material)
- Memory entries (private to my agent context) capture decisions + doctrine separately
