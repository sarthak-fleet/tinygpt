---
name: B2/B2b/B3/B7 mini-router family (bundled)
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B2, B2b, B3, B7)
related_prds: factory-pace-fast-router-100ms.md, factory-planner-v7-tools-in-prompt.md
              (the existing planner-v7 + fast-router PRDs are the building blocks
              this PRD ties together with real BFCL data and a bake-off)
---

# PRD — Mini-router on real BFCL data + bake-off + FSM-injection (B2-B3-B7)

## Goal

Bundle the four router items from PLAN.md Tier B into one coordinated
PRD because they are sequentially dependent on each other:

| Item | What | Sequence |
|---|---|---|
| **B2** | Train + eval the existing mini-router on real BFCL data (today it trains on synthetic seeds) | 1 |
| **B2b** | Bake-off — classifier-head router vs the pure-GPT-with-FSM alternative | 2 |
| **B3** | If B2b says classifier wins: wire FSM constraint-injection from the router's prediction | 3 |
| **B7** | A higher-level "specialist routing model" that picks A1 vs B1 vs B8 vs cloud | 4 |

Each is small. Together they answer the architectural question: "is
the per-task classifier head worth the deviation from one
unified-model + grammar?"

## Why now

- A1 ships a specialist; B2 needs that specialist to score against.
- BFCL data is on disk (D2 + A4 done); the mini-router's BFCL-on-real-
  data path is just wiring the training data.
- Bake-off is the gating question. Without B2b's result we don't
  know whether to keep iterating on the classifier-head approach
  or to fall back to pure-FSM.

## Scope — in

**B2 (mini-router on real BFCL data):**
- `scripts/recipes/b2-router-train.sh` — pulls BFCL train split,
  reformats as `{query, tool}` pairs, trains the mini-router via
  `tinygpt train-extractor`.
- Eval the trained router via `tinygpt eval-bfcl` with the router
  in front (intent → tool name → constrained generation).

**B2b (bake-off):**
- Wire the alternative: `tinygpt serve --tools <catalog> --grammar
  bfcl-tool.gbnf` end-to-end (no router head; FSM only).
- `evals/router-bakeoff.sh` runs both configurations against the
  same BFCL eval; reports per-category + average + per-sample
  latency.
- Decision criterion: classifier-head wins if it beats pure-FSM
  by ≥ 3pp average AND per-sample latency is within 1.2× pure-FSM.
- The losing approach is parked, not deleted — kept on disk as the
  reference comparison for future router experiments.

**B3 (FSM constraint-injection):**
- If classifier wins, `Sources/TinyGPTServe/RouterFsmInject.swift` —
  the router's top-1 tool name selects which sub-grammar the FSM
  uses for the rest of the response.
- If FSM wins, skip B3; the unified grammar already does this.

**B7 (specialist routing):**
- Once ≥ 2 specialists ship (A1 + B1), train a 2nd-level router that
  picks among them based on intent. Reuses the mini-router infra,
  different label set.
- Requires B22 trajectory recorder for the training data.

## Scope — out

- **Production multi-specialist routing in serve** — that's
  downstream; this PRD validates the architecture, doesn't ship
  the runtime.
- **Cost-aware routing** (cheaper specialist for easy queries,
  expensive for hard). V1 routes by accuracy; cost-aware is V2.

## Files to touch

| File | Change |
|---|---|
| `scripts/recipes/b2-router-train.sh` | new |
| `evals/router-bakeoff.sh` | new |
| `Sources/TinyGPTServe/RouterFsmInject.swift` | new (only if B2b winner = classifier) |
| `Sources/TinyGPT/TrainExtractor.swift` | extend if needed for real-BFCL data shape |
| `docs/specialists/b2-router-bakeoff.md` | new — the decision document |
| `docs/PLAN.md` | flip B2, B2b, B3, B7 statuses based on outcome |

## Acceptance criteria

- [ ] **B2**: trained mini-router on BFCL train split; eval accuracy
  ≥ 70% on the BFCL test split.
- [ ] **B2b**: side-by-side numbers published; decision recorded
  in `b2-router-bakeoff.md` with reasoning.
- [ ] **B3**: only if B2b winner = classifier — FSM-injected router
  ships and gets a BFCL row in the leaderboard.
- [ ] **B7**: post-A1+B1, the specialist router ships and routes
  with ≥ 80% accuracy on a held-out "which specialist should answer
  this?" test set.

## Reference patterns

- `factory-pace-fast-router-100ms.md` — existing fast-router PRD,
  contains the V0 architecture.
- `factory-planner-v7-tools-in-prompt.md` — the unified-grammar
  alternative we're comparing against.
- `Sources/TinyGPT/TrainExtractor.swift` — the existing trainer
  (B2 = extend its data path; B7 = extend its label space).

## Open questions

- Whether B2b uses a fixed sampling seed (deterministic bake-off)
  or repeated K=3 (per B23 protocol). **Recommendation:** K=3.
  Bake-off conclusions deserve the rigor.
