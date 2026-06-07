---
name: Distribution cookbook pages — smolagents + Pydantic AI + Continue.dev
status: shipped-docs-2026-06-06-needs-demo-benchmarks
owner: unassigned (parallel-agent task — content / docs, no code)
created: 2026-06-06
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (Strategic Decision 3 — agent-framework distribution > editor integrations)
---

# PRD — three cookbook pages for high-intent distribution

## Goal

Ship three short, concrete, copy-paste-friendly cookbook pages that
demonstrate TinyGPT specialists slotting into popular consumer
frameworks. Target audiences already running local models and already
caring about tool-call / structured-output / per-repo specialization.

Lower-effort than building integrations; higher-leverage than another
"vs Ollama" comparison post.

## Why now

Wave 2 landscape research (2026-06-06) surfaced:

- **smolagents (HF)** + **Pydantic AI** users are exactly our target
  audience: they already run local models, already care about
  tool-calling and structured outputs. Zero acquisition friction.
- **Continue.dev / Aider** users already have working LLMs but no
  per-repo specialist option. A "tinygpt specialize --repo ." recipe +
  Continue config snippet is a credible Tabnine alternative.
- Cookbook pages cost ~1 day each. Three of them ship in a week and reach
  more high-intent users than any technical PRD.

## Scope — in

### Cookbook 1 — `docs/recipes/cookbook-smolagents.md`

A walkthrough that takes a smolagents user from "I have HF smolagents
working with Llama 3" to "I'm running a TinyGPT specialist as the
backing model for my agent."

Outline:
1. Why a specialist? (one paragraph — tool-call accuracy + structured
   output reliability + local speed)
2. Train the specialist: minimal `tinygpt distill` invocation against
   hermes-fc data
3. Spin up `tinygpt serve` with the trained specialist
4. Point smolagents at the OpenAI-compat endpoint (`OPENAI_BASE_URL` env
   var or equivalent)
5. Run a smolagents demo (use the `tool_calling_agent` example)
6. Benchmark: specialist vs. general Llama 3 on the same task

Total target length: ~300 lines including code snippets. Ship a runnable
example dir at `examples/smolagents-tinygpt/` with a single `run.sh`.

### Cookbook 2 — `docs/recipes/cookbook-pydantic-ai.md`

Same shape, different framework. Pydantic AI's value prop is
type-safe / structured output via Pydantic models. Our constrained
decoding (via ConstrainedGen.swift, already shipped) makes us
exceptionally well-suited to back this.

Outline:
1. Why a specialist for Pydantic AI? (structured-output reliability is
   the value prop both sides sell)
2. Train a specialist with constrained-decoding awareness (the recipe is
   distill + grammar config)
3. Point Pydantic AI at `tinygpt serve` (it speaks OpenAI-compat)
4. Demo: a Pydantic AI agent producing JSON that matches a schema, with
   compliance ≥ teacher
5. Bench: structured-output compliance rate of specialist vs general

Example dir at `examples/pydantic-ai-tinygpt/`.

### Cookbook 3 — `docs/recipes/cookbook-personal-code-specialist.md`

The "Tabnine alternative for individuals" demo. Different audience from
the first two — this targets individual developers wanting their own
per-repo code specialist.

Outline:
1. The thesis: per-org code specialists (Tabnine Enterprise) are an
   enterprise-tier cloud product. Per-individual / per-repo specialists
   are an underserved gap.
2. `tinygpt specialize --repo .` — produces a fine-tuned specialist on
   the user's code (idea: this could be a shortcut alias that wraps
   `extract` + `distill` for a single repo)
3. Plug into Continue.dev (config snippet)
4. Plug into Aider (config snippet)
5. Demo: 30-second screencast of "trained on my repo's patterns,
   completes them better than vanilla Qwen2.5-Coder"
6. Honest caveat: this isn't going to magically be smarter than GPT-4 at
   general code — it's domain-specialized at your stack's patterns.

Example dir at `examples/repo-specialist-cli/` with a `run.sh` that does
the full flow against the local TinyGPT repo as a self-referential demo.

### Cross-cookbook conventions

- Each starts with a 5-line "what you'll get" + a "what this is NOT"
- Each ends with "honest limitations" — what doesn't work yet
- Each has a single `run.sh` + a screen recording (or screenshot
  sequence)
- Each cross-links to the others under "see also"

## Scope — out

- LangGraph cookbook (lower-priority audience overlap; backlog)
- AutoGen cookbook (smaller audience)
- CrewAI cookbook (smaller audience)
- Editor integrations beyond Continue/Aider (Cursor, Codeium are cloud
  products; not in our lane)
- Marketing material / blog posts / Twitter threads — those are content
  marketing, not technical cookbooks; separate work

## Acceptance criteria

1. Each cookbook page is ≤ 300 lines, follows the shape above
2. Each `examples/<framework>-tinygpt/` directory has a runnable
   `run.sh` that works from a clean clone
3. Each cookbook has a benchmark / bench output section (real numbers, not
   marketing claims)
4. Each cookbook crosses to the strategy doc's positioning statement it
   demonstrates
5. Three example dirs, three docs, three runnable smoke flows
6. `docs/recipes/README.md` (if missing, create) lists all three with
   one-line summaries

## File paths

| Action | Path |
|---|---|
| **create** | `docs/recipes/cookbook-smolagents.md` |
| **create** | `docs/recipes/cookbook-pydantic-ai.md` |
| **create** | `docs/recipes/cookbook-personal-code-specialist.md` |
| **create** | `examples/smolagents-tinygpt/run.sh` + supporting files |
| **create** | `examples/pydantic-ai-tinygpt/run.sh` + supporting files |
| **create** | `examples/repo-specialist-cli/run.sh` + supporting files |
| **modify or create** | `docs/recipes/README.md` — index entry per cookbook |
| **don't touch** | Source code (`native-mac/Sources/`), eval pipeline, `docs/PLAN.md`, `HANDOFF.md`, `Package.swift` |

## Inputs the agent has

| Resource | Location |
|---|---|
| Distillation recipe template | `docs/recipes/distillation-fc.md` |
| smolagents docs | https://github.com/huggingface/smolagents |
| Pydantic AI docs | https://ai.pydantic.dev/ |
| Continue.dev model config docs | https://docs.continue.dev/customize/models |
| Aider LLM connection docs | https://aider.chat/docs/llms.html |
| Our serve OpenAI-compat surface | `Sources/TinyGPTServe/Serve.swift` |
| Existing cookbook prior art | `docs/recipes/b25-scaledown.md`, `docs/recipes/distillation-fc.md` |

## Estimated effort

**~3-4 days for all three cookbooks.**

- 1 day: smolagents cookbook + working example
- 1 day: Pydantic AI cookbook + example
- 1 day: code-specialist cookbook + screencast + example
- 0.5 day: cross-linking + recipes README + polish

## Coordination

PR description must include:
1. Screenshots / screencasts of all three demos
2. Benchmark numbers for each (claim → measurement)
3. Confirmation that all three `run.sh` work from clean clone

Maintainer merges, posts each cookbook to the relevant community
(smolagents Discord / HF forums / Pydantic AI repo / Continue/Aider
GitHub discussions) as the distribution step.

## Why these three (and not others)

| Framework | Audience size | Audience intent | Effort |
|---|---|---|---|
| smolagents | Mid (HF audience) | **Very high** — already running local models with tool calls | 1 day |
| Pydantic AI | Mid + growing | **High** — structured-output users care about reliability | 1 day |
| Continue/Aider | Large dev audience | High for "better completions" buyers | 1 day (per code-specialist) |
| LangGraph | Largest | Mid — many use cloud LLMs, fewer care about local | (backlog) |
| AutoGen / CrewAI | Mid | Lower — more enterprise / less individual | (backlog) |

The three chosen all have the property: **users are already self-selected
to care about local-model quality.** That's where TinyGPT lands cleanly.

## Distribution flow (after merge)

1. Post the smolagents cookbook in HF community forum + tag the
   smolagents maintainers
2. File a "TinyGPT integration" example PR against Pydantic AI's repo
3. Drop the code-specialist cookbook in Continue.dev's discussions + r/LocalLLaMA
4. Track inbound (GitHub stars, issues, traffic) — informs which
   audience converts best for future investment

## Honest caveat

These are distribution / content artifacts, not code. They land downstream
of having: working distillation flow (✅), working serve (✅), working
constrained decoding (✅), per-language specialist proof-of-concept
(needs the A1 specialist to actually land). If A1 doesn't ship clean
results, the cookbooks are pitching vapor. **Sequence: A1 lands → write
cookbooks**, not the other way.
