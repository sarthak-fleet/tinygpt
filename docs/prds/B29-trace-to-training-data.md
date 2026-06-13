---
name: B29 trace-to-training-data pipeline
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B29)
parent_learn: docs/learn/castform-rl-finetune.md (Steal #2)
related_prds: B22-trajectory-recorder.md (the substrate this consumes),
              A1-first-specialist-tool-caller.md (the eventual consumer of the output JSONL),
              E7-judge-shim.md (the LLM-judge filter step)
---

# PRD — Turn `.atraj` rollouts into SFT/DPO training JSONL

## Goal

`tinygpt traces-to-data <atraj-dir> --task <task> --out <data.jsonl>`
reads every `.atraj` file in a directory and emits training-ready
JSONL. Filters out dedupe + tool-echo noise + LLM-pivot-judge low-
quality rollouts. The user gets a clean SFT corpus from production
agent traces without hand-labeling.

Lifted from Castform's dataset-synthesis pattern
(`docs/learn/castform-rl-finetune.md` §2). They synthesize from
external observability tools (Braintrust / Langfuse / LangSmith);
we synthesize from our own `.atraj` files first, external ingest
is V2.

## Why now

- B22 ships `.atraj` per-rollout files. Without B29, those files
  are write-only diagnostics — no automated consumer.
- A1 specialist training (and B5 cloud-escalate, B1 second
  specialist, …) all want trace-derived SFT data. B29 is the
  shared producer.
- All filtering primitives already ship: `tinygpt dedupe` (exact +
  MinHash), `tinygpt judge` (E7) for pivot filtering.

## Scope — in

- `Sources/TinyGPT/TracesToData.swift` — orchestrator
  - Read every `.atraj` (or `.atraj.gz`) under the input dir
  - Extract per-turn (user-prompt, assistant-response, tool-calls,
    tool-results) tuples — the SFT shape
  - Run shipped filters in this order:
    1. **Dedup** — exact-match drop via `tinygpt dedupe` (shared
       library) on the prompt
    2. **Tool-echo drop** — discard turns where the assistant's
       response only echoes a tool result (regex + heuristic)
    3. **LLM pivot judge** — `tinygpt judge` (E7) scores each turn
       1–10; filter at a configurable threshold (default 6, range
       0.6× max-score so it matches Castform's 0.6–0.9)
- Output shape: ChatML-style JSONL (`{messages: [...], task: "...",
  source_atraj: "...", judge_score: …}`).
- Two modes: `--mode sft` (default — chat continuations) and
  `--mode dpo` (build chosen/rejected pairs from the trace's
  reward field, if present).
- `--task <name>` is a free-form label that becomes the row's
  `task` field; used by downstream B30 reasoning classifier and
  data-mixer (B21).

## Scope — out

- **External observability tool ingest** (Braintrust / Langfuse /
  LangSmith). V2 — premature until users actually ask. Our `.atraj`
  files are the V1 substrate.
- **RAG corpus seeding** — Castform also generates from RAG indices
  (Turbopuffer / Pinecone / …). Out of scope; user provides the
  source documents per their existing pipelines.
- **Auto-tuning of the pivot threshold.** V1 manual via flag.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/TracesToData.swift` | new — orchestrator |
| `Sources/TinyGPT/TinyGPT.swift` | `case "traces-to-data"` |
| `Sources/TinyGPTModel/AtrajReader.swift` | small extension — iterate dir, yield trajectories |
| `evals/traces-to-data-smoke.sh` | new — 5-trajectory fixture → expected output count |
| `docs/recipes/from-traces.md` | new — recipe |
| `docs/PLAN.md` | B29 ⬜ → ✅ on ship |

## Don't touch

- `tinygpt dedupe`, `tinygpt judge` — consume them as library
  functions, no API changes.
- The `.atraj` format — frozen by B22.

## Acceptance criteria

- [ ] `tinygpt traces-to-data /tmp/atrajs --task tool-call
  --judge-model qwen3-9b --judge-threshold 0.7 --out
  /tmp/tool-call-sft.jsonl` runs end-to-end on a 100-trajectory
  directory.
- [ ] Output rows are valid ChatML JSONL; spot-check 5 rows match
  the source trajectories' content.
- [ ] Dedup actually drops near-duplicates (MinHash, threshold
  configurable).
- [ ] Pivot-judge filters out the lowest-scoring tail per the
  threshold.
- [ ] Smoke test passes in CI.

## Reference patterns

- `Sources/TinyGPT/Dedupe.swift` — the dedup pipeline.
- `Sources/TinyGPT/JudgeShim.swift` — the E7 judge.
- B22 trajectory format spec — the input shape.
- [Castform's filtering copy](https://castform.com/) — cite, don't
  reimplement their flow exactly.

## Open questions

- Whether the DPO-pair construction in `--mode dpo` requires a
  reward signal in every trajectory (filter out unrewarded ones)
  or builds pairs by judge-score margin (works on unrewarded too).
  **Recommendation:** ship both; `--dpo-source {reward,judge}` flag.
