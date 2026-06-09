# Pace planner v11 — unhappy-path training data PRD

**Date**: 2026-06-09
**Status**: SPEC — execution conditional on v10 result missing one or more dimensions of the [v11 ship gate](pace-planner-v11-ship-gate.md)
**Owner**: tinygpt repo

---

## Goal

Produce ~450 training rows that teach a Qwen3-0.6B planner to emit three new intent classes:
- `out_of_scope` (refusal): ~150 rows
- `clarify` (ask back): ~150 rows
- `confirm_destructive` (safety gate): ~150 rows

These rows compose with the existing v10 corpus (404 happy-path rows) to form the v11 training set of ~850 rows total.

## Why this dataset has to be built carefully

Past Pace versions over-eagerly emit AX.press for every prompt because:
1. The training distribution only contains AX.press / AX.setValue happy-path examples
2. The grammar / schema doesn't allow non-action outputs
3. The teacher used for v10 multiplication reflexively produced AX.press regardless of input

Adding 450 rows of unhappy-path examples **fixes (1)**, an action-registry update **fixes (2)**, and the data-generation recipe in this PRD **fixes (3)**.

---

## Action-registry extension (v10 → v10.5)

`grammars/v10-actions/registry.json` adds three new top-level intent values alongside `action`:

```json
{
  "intent": {
    "type": "string",
    "enum": ["action", "out_of_scope", "clarify", "confirm_destructive"]
  }
}
```

Per-intent payload shapes:

| Intent | Payload | Example |
|---|---|---|
| `action` | `{name, args}` (existing v10 shape) | `{name: "Mail.draft", args: {...}}` |
| `out_of_scope` | `{reason}` — short user-facing explanation | `{reason: "Weather requires a cloud query"}` |
| `clarify` | `{question, options?}` — what to ask the user | `{question: "Which file do you mean?", options: ["Q4 report", "Q3 report"]}` |
| `confirm_destructive` | `{action, target}` — planned action + thing being destroyed | `{action: "Mail.deleteAll", target: "all emails in Inbox"}` |

Grammar-constrained decoding at inference uses this extended registry. The constraint mathematically prevents the model from emitting an action when it's supposed to refuse — so even if the model internally guesses, the structural output is forced correct.

---

## Data-generation recipe — three-stage hybrid

### Stage 1 — Hand-curated seeds (1.5h human, 0 compute)

50 high-quality examples per class. Hand-written by the engineer.

**Why hand-curate seeds and not full corpus?** Seeds anchor the teacher's behavior. With 50 anchor examples per class showing exactly the shape we want, the teacher's output distribution shifts toward our target. Without anchors, the teacher generates "what it thinks an LLM would emit" — which is biased toward the most common shapes in its pretraining (which is AX.press-shaped, because Mac voice assistants are rare in pretraining).

Per-class seed shape:
- **OOS seeds**: Pull from `pace/evals/fm-fixtures-oos/` (30 fixtures already exist). Augment 20 more from adjacent categories: jokes, philosophical questions, requests for the model to take an action that requires multiple unsupported integrations.
- **Clarify seeds**: Pull from `pace/evals/fm-fixtures-ambig/` (20 fixtures exist). Augment 30 more with: missing recipients, missing dates, ambiguous app references when multiple apps could match.
- **Destructive seeds**: Pull from `pace/evals/fm-fixtures-destructive/` (10 fixtures exist). Augment 40 more with: financial actions ("send $500 to John"), comms with permanence ("post to Twitter and don't delete"), file moves to System folders.

### Stage 2 — Teacher amplification (3h compute, ~30min human supervision)

Use **Qwen3-14B-MLX-4bit with thinking ENABLED**. Local, no API cost.

Anti-mistakes from the last multiplier run:
- **Thinking ON** — last time we set `/no_think` for speed, which killed the teacher's ability to reason about scope and intent. Reasoning tokens are free at offline data-gen time.
- **Best-of-N=4** — generate 4 candidates per seed, judge picks top 1-2. Filters out the AX.press-default failure mode.
- **System prompt with explicit scope list** — tell the teacher exactly what Pace's 12 actions are and what falls outside them. The teacher knows what "out of scope" looks like in general; we just need to give it our specific scope.
- **Single-seed expansion**, not corpus mixing — each generation call has ONE seed in context, plus the scope spec, plus the target intent class. No room for the teacher to drift toward AX.press because the prompt explicitly says "produce an out_of_scope refusal."

Per-class expansion plan:

| Class | Hand seeds | Variations/seed | Pre-judge candidates | Target post-judge |
|---|---|---|---|---|
| `out_of_scope` | 50 | 4 | 200 | ~150 |
| `clarify` | 50 | 4 | 200 | ~150 |
| `confirm_destructive` | 50 | 4 | 200 | ~150 |

### Stage 3 — Judge filter (~1h compute)

For each candidate, run a critic prompt against the same Qwen3-14B teacher (different sampling temperature):

```
Critic prompt:
You are checking a Pace planner training example. Pace is a Mac voice
assistant. The training row claims the user input belongs to the
"<intent_class>" category. Rate this example 1-5:

5 = perfect example of the category, schema-valid, would be useful training data
4 = good example, minor issues
3 = ambiguous category fit, could go either way
2 = wrong category but related
1 = bad example, would hurt training

Input prompt: <prompt>
Generated response: <response>
Expected intent: <intent_class>

Return ONLY: SCORE: <1-5>
```

Keep candidates rated 4 or 5. Reject 1-3.

Expected pass rate ~50-70% per category based on Tulu-3 / Llama-3.1 paper baselines. With 200 candidates × 12-class slot × ~60% pass = ~120-140 per class, slightly under our 150 target. If short, hand-write the remainder (cheap last-mile).

---

## Quality controls

- **Diversity check**: after building the 450-row corpus, embed each row with the existing qualified Qwen3-Embedding-0.6B. Compute pairwise cosine similarity within each class. Reject the closest 10% of pairs (avoids near-duplicates that bloat with no diversity).
- **Schema-validity check**: every generated row passes through grammar-constrained-decoding validation before adding to the dataset. Schema-invalid rows are dropped (not silently fixed) — schema errors are signal that the generation was confused.
- **Distribution audit**: at the end, plot the prompt-length histogram, action-distribution histogram, vocabulary-coverage. Verify no class collapsed to a single canonical phrasing.

---

## Output

```
~/.cache/tinygpt/datasets/pace-v11-unhappy.jsonl
  - one JSONL per row, shape: {messages: [{system, user, assistant}]}
  - assistant = {spokenText, intent, payload}
  - ~450 rows total, balanced across the 3 classes

~/.cache/tinygpt/datasets/pace-v11-merged.jsonl
  - v10 (404) + v11-unhappy (~450) = ~850 rows
  - shuffle preserved, no class label leakage
```

Merge order: existing v10 rows tagged with `intent: action` for consistency, then concatenated with the new classes. Grammar at training time uses the extended registry so the model learns the full 4-intent vocabulary.

---

## Time + cost

| Phase | Human | Compute | Wall |
|---|---|---|---|
| 1. Hand-curate 150 seeds | 1.5h | 0 | 1.5h |
| 2. Teacher amplification (600 candidates total) | 30min supervision | ~2.5h | 2.5h |
| 3. Judge filter | 0 | ~30min | 30min |
| 4. Diversity + schema audit | 30min | ~5min | 30min |
| 5. Merge with v10 corpus | 5min | 0 | 5min |
| **Total** | **~2.5h human** | **~3h compute** | **~5h wall** |

Less than the original "4h hand-curate" estimate AND scales to 1000+ rows if v11 needs more.

---

## Failure modes the judge filter cannot catch

- **The teacher itself has the wrong notion of Pace's scope**. Mitigated by the scope spec in the system prompt — but if Qwen3-14B "thinks" sending money via Mail.draft is in scope (it isn't, no payment integration), the judge will let it through. Mitigation: include 5-10 explicit *negative* examples per class in the system prompt.
- **Mode collapse on a single refusal phrasing**. Without diversity guard the model may emit "I can't do that on Mac" 100 times. Mitigated by Stage 4 audit.
- **Adversarial seeds we didn't think of**. If users say things our seed set doesn't cover, v11 won't refuse them properly. Mitigated by holdout test before training: 30 "wild" prompts hand-written separately, scored against the trained v11 to surface gaps.

---

## Activation

This corpus is built **only if** v10 cascade results (in flight 2026-06-09) miss the [v11 ship gate](pace-planner-v11-ship-gate.md) on dimensions 3, 4, or 6. If v10 happens to clear all six dimensions (unlikely but possible), no v11 needed and this PRD shelves.

If activated, the recipe runs against the same teacher endpoint (Qwen3-14B-MLX-4bit at port 1234) that produced the v10 multiplier — with the four corrections listed in Stage 2.

---

## Why not just hand-curate

Time is comparable (~2.5h vs 4h), but the hybrid:
1. Scales — if v11 needs 1500 rows instead of 450, the same recipe applies; hand-curation does not scale
2. Captures phrasing diversity the engineer wouldn't think of
3. Doubles as quality control — the judge step is reusable for any future data generation

The seed-anchored approach is roughly how Anthropic, OpenAI, and Tulu generate alignment data. We are not inventing a new methodology; we are correcting last week's mistake (no thinking, no judge, no anchored seeds) by following the canonical recipe.

---

## What this PRD does NOT cover

- The training run itself (separate from data generation) — covered by an eventual v11-training-recipe doc
- Grammar update implementation — separate small change to `grammars/v10-actions/registry.json` plus the constrained-decoding code path
- Eval — already locked in [pace-planner-v11-ship-gate.md](pace-planner-v11-ship-gate.md)
