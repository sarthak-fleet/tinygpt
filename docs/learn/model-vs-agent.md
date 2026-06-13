---
title: Model vs agent — what's actually different
description: The architectural distinction between a fine-tuned model and the agent that runs it, and what "making a good model into an agent" actually requires. Mapped onto this repo.
---

# Model vs agent — what's actually different

A recurring question worth pinning: what's the difference between a
*fine-tuned model* and an *agent* that uses it? Is an agent just a
model with tools and an infinite loop?

Tools + loop is necessary but not sufficient. The honest version
has more pieces.

## The model is one component; the agent is the system

A **model** is a parameterized function: input tokens → output tokens.
It has *capabilities* (what it can do when prompted right) but no
autonomy, no loop, no memory beyond its context window. Every call
is stateless.

An **agent** is a *system* built around a model. The model is the
brain; the agent is everything else that lets the brain *do something*
in the world over time.

The minimum extra pieces:

| Piece | Role | Where it lives in this repo |
|---|---|---|
| **Tools** | Read + write surfaces the model can call (search, run code, click a button, write a file) | `Sources/TinyGPT/Agent.swift` tool registry; pace's `Action.*` schema |
| **The loop** | Turn-taking orchestrator: model proposes → tool runs → result feeds back → repeat | `Sources/TinyGPTServe/AgentLoop.swift` |
| **State / memory** | What persists across turns beyond the model's context window — scratchpad, RAG hits, episodic memory, task goal | Currently: chat history + KV prompt-cache. Beyond that: deferred (a known gap) |
| **Goal management** | What the agent is trying to do; decomposition for long horizons | Implicit in the prompt today; explicit task-graph is a future feature |
| **Error recovery** | What happens when a tool fails or the model goes off-rails | Cloud-escalate fallback (regex today; B5 trained signal next) |
| **Safety / scoping** | What the agent IS and ISN'T allowed to do; destructive-action gating | Pace's `confirm_destructive` intent + the H4 rule in `pace-system-prompt-v11.txt` |

You can have a model without an agent (every call to `tinygpt sample`
qualifies). You cannot have an agent without a model.

## What makes a "good" agent — beyond the bare minimum

The "tools + loop" minimum runs *an* agent. Making it a *good* agent
is mostly orchestration discipline, not model capability:

1. **Tool design.** Tool names + descriptions + arg schemas are the
   model's interface to the world. Bad tool surfaces produce bad
   agents regardless of model quality. This is why
   `factory-pace-planner-v6_1.md` spends most of its words on the
   action registry's wording, not on the model.

2. **Context discipline (the L1/L2/L3 framing).** Per
   `docs/learn/agent-context-hierarchy.md`: tier the available
   capabilities so the always-resident prompt stays small. A model
   degrading under context bloat looks like an agent that "got
   worse" — but the model is fine; the context is misallocated.

3. **Verifiable rewards / observable progress.** A good agent
   knows when it's done. Tool calls that return success/failure,
   eval suites that score the trajectory, B22 trajectory recorder
   that captures `input_ids`/`output_ids`/`rewards` — these are
   what let the loop terminate honestly instead of running forever.

4. **Loop bounds.** Step limits, sandbox CPU/RAM, sampling-budget
   caps. Agents without bounds are bugs waiting. B23
   agent-eval-protocol formalizes this for evaluation; the same
   bounds apply at runtime.

5. **Tool-call efficiency as a metric, not an afterthought.**
   B28 composite reward includes `tool_call_efficiency` as a
   first-class scoring dimension. An agent that solves the task
   but calls 50 tools when 5 would do is a bad agent.

## How fine-tuning *helps* the agent

A fine-tuned model is a *better substrate* for the agent — same
loop, same tools, but the substrate behaves more reliably. Useful
fine-tuning targets, in rough priority:

- **Tool-call faithfulness.** Don't hallucinate tools that don't
  exist. Don't malform args. Pace's planner specialist (A1) is
  exactly this fine-tune.
- **Output format adherence.** The wrapping JSON / grammar stays
  stable under load. Constrained-decoding (FSM masking) is the
  belt; the fine-tune is the suspenders.
- **Long-context coherence.** The agent's accumulated scratchpad
  doesn't degrade reasoning. SFT on multi-turn trajectories (B22 →
  B29) trains for this directly.
- **Self-correction.** The model recognizes it took a wrong turn
  and backs out. Hard to train explicitly; falls out of RLVR
  (5.1) when rewards punish dead-ends.
- **Defer-to-cloud signal.** The model knows when it shouldn't
  answer locally (B5). The cleanest "agent boundary" feature: the
  trained signal makes escalation honest instead of regex-driven.

## What fine-tuning does NOT fix

- **Bad tool design.** No amount of fine-tuning saves an agent
  whose tools have ambiguous names.
- **Missing memory.** A model can't remember across sessions
  unless you give it a memory store. Fine-tuning teaches it to
  *use* memory; it doesn't provide memory.
- **Loop bugs.** If the orchestrator double-counts tool results
  or drops error returns, the agent breaks regardless of model
  quality.
- **Insufficient scoping.** Pace's `confirm_destructive` exists
  because a brilliant tool-caller fine-tuned to be helpful will
  cheerfully empty your trash. The intent-level scoping is
  agent-side policy, not model behavior.

## The TinyGPT framing

This repo treats the two layers explicitly:

- **Model layer** — `tinygpt train`, `sft`, `dpo`, `distill`,
  `peft variants`, `quantize`. Substrate quality.
- **Agent layer** — `tinygpt serve` (the runtime that hosts the
  substrate), `tinygpt agent` (the loop that drives it), the
  tool registry, the system prompt, the constraint grammars, the
  cloud-escalate policy. System quality.

The two layers ship and improve independently. A1 specialist (the
fine-tune) and B6 Factory Mac app (the agent UX surface) are
*both* doing agent work, but at different layers. B22-B23-B28-B29-B30
form the feedback loop from agent traces back into model
training — closing the substrate-improvement cycle that real
products depend on.

## Sources

- [Lilian Weng — LLM Powered Autonomous Agents (2023)](https://lilianweng.github.io/posts/2023-06-23-agent/) —
  the canonical agent-architecture overview (planning + memory +
  tool use).
- [ReAct, Yao et al. 2022](https://arxiv.org/abs/2210.03629) —
  the reasoning + acting interleaving pattern that most agent
  loops still derive from.
- [BabyAGI](https://github.com/yoheinakajima/babyagi) — the
  simplest "model + tools + loop" agent reference. Worth reading
  the README to see what *not* to ship as production.
- `docs/learn/agent-context-hierarchy.md` (this repo) — the L1/L2/L3
  framing that bridges the model and agent layers.
