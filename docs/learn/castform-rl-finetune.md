---
title: Castform's RL fine-tune platform — what we stole
description: Patterns lifted from castform.com (RL fine-tune SaaS) into TinyGPT's specialist-training surface. Sibling page to docs/learn/agent-context-hierarchy.md.
---

# Castform RL fine-tune — what we stole

**Source:** [castform.com](https://castform.com/) — RL fine-tune SaaS
that lets engineers train open-source models on proprietary data + a
user-defined reward function. Visited 2026-06-13; surface inspected
via product copy + the public Python SDK sketches.

**Why look at them:** Castform's product thesis ("4B fine-tuned beats
GPT-5.4 on the narrow task at 1.0× the cost") is identical to
TinyGPT's Mac-specialist thesis. The interesting question isn't
*whether* small models can win — it's *which engineering primitives*
make training them practical. Castform exposes four that are worth
borrowing.

This page is a steal-map (same format as
[agent-context-hierarchy.md](agent-context-hierarchy.md)): one
section per pattern, what it gives us, where it goes in the PRDs.

## The four steals

### 1. Composite reward functions — filed as B28

Castform: a reward is a *composition of named dimensions* —
`correctness + conciseness + citation + tool_call_efficiency` — each
independently scorable, then aggregated via weights into the
training signal. The dashboard shows the per-dimension breakdown
per rollout so you can see *which* axis is driving the gradient.

Where we are: DPO has a single implicit reward from
`log_π_pol − log_π_ref`. ES has a single scalar (negative loss).
There's no abstraction for "this reward is *actually* four things
in a trenchcoat", which is the practical shape every real
specialist eventually needs.

Filed as **B28 composite-reward-framework** in
[PLAN.md](../PLAN.md) — a `Reward` struct with named dimensions,
weighted aggregation, per-dimension logging. Usable from DPO (as
the chosen/rejected score), GRPO (5.1), and ES.

**The base struct ships in this PR**
([`CompositeReward.swift`](../../native-mac/Sources/TinyGPTModel/CompositeReward.swift)).
The training-loop integrations are the rest of B28.

### 2. Trace-driven dataset synthesis — filed as B29

Castform: pull from production agent traces (Braintrust, Langfuse,
LangSmith) and RAG corpora (Turbopuffer, Pinecone, Chroma, Postgres).
Auto-filter via dedup + tool-echo drop + LLM-judge pivot at
configurable thresholds (0.6–0.9). The output is a training-ready
JSONL the user never had to hand-label.

Where we are: B22 (token-preserving trajectory recorder) ships the
*substrate* — every `.atraj` file carries `input_ids`, `output_ids`,
tool calls, rewards. But there's no consumer that turns the
substrate into SFT/DPO training data.

Filed as **B29 trace-to-training-data** in [PLAN.md](../PLAN.md) —
the bridge between B22 and A1. Reads `.atraj` files, runs the
existing `tinygpt dedupe`, `tinygpt judge` (E7) shims for filtering,
emits training JSONL.

### 3. Multi-hop reasoning classification — filed as B30

Castform: classify training prompts by reasoning depth
(single-hop, multi-hop, comparison) so the training mix is
balanced — too much single-hop and the model never learns to chain.
Too much multi-hop on a base that hasn't shipped basic capability
yet and it diverges.

Where we are: no classifier; balance is per-corpus by hand. Works
for narrow specialists; falls over for the agent-trace data B29
will produce, which is intrinsically mixed-depth.

Filed as **B30 prompt-reasoning-classifier** — small classifier
head trained on a labeled subset, scores any prompt into
{single-hop, multi-hop, comparison, other}. Drops into B29's
filtering pipeline + the leaderboard's per-category breakdown.

### 4. Pluggable `BaseEnv` interface — not filed (out of scope)

Castform: users subclass `BaseEnv` with `async run_tool()` +
`compute_reward()`. The platform handles rollouts + the training
loop. This is a UX choice for a SaaS surface.

Where we are: TinyGPT is a CLI + Swift library, not a SaaS. Users
write a Swift `EvalCompare.Row` row or a Python eval harness; the
"plug in your env" abstraction is already there, just spelled in
two languages.

Deliberately **not** filed. The B6 Mac app (Factory tab) is the
right place for a friendlier env-config UX, and B6 already covers
that — adding a `BaseEnv` Python class layer is fake reuse since
the underlying mechanism is the same.

## What we deliberately did not steal

- **The dashboard.** Castform's training dashboard (avg reward per
  step, response length, solve rate, max reward per prompt) is
  covered by **C10 train-run-dashboard** + **B23 agent-eval-protocol**
  — already filed. Castform's specific axes are listed in B28's
  acceptance criteria so the train viewer adds those columns when
  composite reward is on.
- **The pay-per-compute UX.** Castform is a SaaS; we're a Mac CLI.
  Not transferable.
- **External observability tool ingestion** (Braintrust / Langfuse /
  LangSmith). B29 reads our own `.atraj` files. External-tool
  ingest is a V2 if users actually have agent traces in those
  systems they want to import; until then, premature.

## The addition this analysis surfaced (our note)

Castform's "solve rate (pass@k)" metric is the per-prompt version
of what B23 protocol does *across* the eval suite. Combined: every
leaderboard row reports `mean ± σ` (B23) AND the per-prompt
`pass@k` distribution under the recorded budget. That's a
deliverable for B23's V2 — not blocking now, but the data is free
once K-pass rollouts are running.
