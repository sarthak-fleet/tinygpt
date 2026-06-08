# Eval methodology — the gate finding

**Date**: 2026-06-08
**Status**: confirmed, the gate task #270 has its evidence
**TL;DR**: fm-fixtures are passable in full (19/19) by a rule-based
endpoint that runs zero model inference. Every Pace LoRA we trained
scores ≤ this baseline. The fixtures test framework, not model.

## What we ran

`scripts/fake_pace.py` — a 300-line Python script using ONLY:
- regex over the user prompt to detect intent (click / type / scroll /
  key combo / open app / identity probe / QA)
- substring matching against element labels for click targets
- a JSON-shape wrapper and free-text-mode toggle that mirror Pace's
  v6-label system prompt rules

No tokenizer beyond `re`. No model. No neural net. No LoRA. No SFT.

## Results

| System | fm-fixtures | Notes |
|---|---|---|
| Pace v3 LoRA | (shipped) | one of the early Pace LoRAs |
| Pace v5 LoRA | **17/19** | best-scoring LoRA on this eval |
| Pace v6 LoRA | **14/19** | label-based architecture; treated as regression |
| Pace v6.1 SFT | **10/8/2/0** | four-way collapse over hyperparameter sweeps |
| **FakePace (rule-based, ZERO model)** | **19/19** | trivial Python script |

The rule-based baseline is the **ceiling** of what fm-fixtures can
distinguish. Every LoRA we shipped lives below it.

## Implications

1. **Pace v5's "17/19" was not measuring model capability.** It was
   measuring the model's compliance with the JSON grammar + the
   label-lookup convention. A regex endpoint matches it.
2. **v6's "regression" to 14/19 was the model unlearning the format**,
   not regressing at any task. The grammar mask had to fall back to
   default behavior for 5 fixtures.
3. **v6.1's four-way SFT collapse (10 → 8 → 2 → 0)** was the model
   finding *new* formats that broke the eval's regex assumptions —
   not a model that became worse at planning. We were optimizing
   toward an eval that doesn't reward planning improvements.
4. **The "factory beats teacher" framing is unsupported.** The Pace
   specialist may or may not beat the Qwen3-30B-A3B teacher at
   planning; fm-fixtures cannot tell us either way.

## What an honest eval would look like

We need fixtures where **a rule-based endpoint fails** and **a real
planner succeeds**. Candidate axes:

### Axis 1 — Semantic disambiguation that requires world knowledge

```
USER: open the app i use to write code
ELEMENT: [0] dock_icon|..|Mail
ELEMENT: [1] dock_icon|..|Xcode
ELEMENT: [2] dock_icon|..|Safari
ELEMENT: [3] dock_icon|..|Spotify
EXPECT_CLICK_ID: 1   # the model must KNOW Xcode is a code editor
```

A regex over "code" wouldn't match any of those labels. The model
has to bring the world knowledge that Xcode is an IDE.

### Axis 2 — Multi-element reasoning ("the most expensive one")

```
USER: click the cheapest plan
ELEMENT: [0] button|..|Free plan|$0/mo
ELEMENT: [1] button|..|Pro plan|$15/mo
ELEMENT: [2] button|..|Enterprise plan|$99/mo
EXPECT_CLICK_ID: 0
```

A regex doesn't know to read the `text` field for prices, parse them,
and pick the minimum.

### Axis 3 — Contextual reference resolution

```
USER: click the same button as last time
HISTORY: [previous turn shows user clicked "Save Draft"]
ELEMENT: [0] button|..|save draft
ELEMENT: [1] button|..|discard
EXPECT_CLICK_ID: 0
```

Requires the model to track conversational history. Regex can't.

### Axis 4 — Open-ended generation quality

```
USER: summarize what's on my screen
ELEMENT: [0..N] (a complex screen)
EVAL: human or LLM judge rates summary on 1-5 scale
```

No fixed regex assertion. The judge evaluates the response itself —
relevance, accuracy, brevity, tone.

### Axis 5 — Behavioral diversity tests

50 paraphrases of "save this file" — the model should produce 50
correct responses with appropriate variation in spoken text, not the
same template every time. A regex endpoint produces one template.

## The eval gate is now built

`scripts/fake_pace.py` is the gate. To compare any LoRA against the
real-model-contribution standard:

1. Run the LoRA via `tinygpt serve` against fm-fixtures →
   `lora_score / N`
2. The FakePace baseline depends on the fixture set:
   - v1 (`clickyLocal/evals/fm-fixtures`): **19/19** — useless for
     measuring model contribution; only format compliance
   - v2 (`clickyLocal/evals/fm-fixtures-v2`): **1/15** (6.7%) — the
     genuine baseline for model-required tasks
3. **model_contribution = lora_score(v2) − 6.7%**. ≥85% on v2 is
   the bar for "this model adds real capability."

## v2 baseline shipped (2026-06-08)

`clickyLocal/evals/fm-fixtures-v2/` — 15 fixtures across three axes:

- **Semantic disambiguation (7)**: "open the app i use to write code"
  → Xcode. Element labels don't contain function words; only the
  model's world knowledge does.
- **Multi-element reasoning (6)**: "click the cheapest plan" requires
  parsing $0/$15/$99 in the element `text` field and picking the
  minimum.
- **Abstract reference resolution (2)**: "click the one for sending
  money" → Transfer. User names a goal; model maps to action.

FakePace result on v2: **1/15 (6.7%)** — the single pass is a lucky
first-match-wins on `reason-cheapest-plan` (element 0 happened to
be the answer). The eval is calibrated correctly to require real
model capability.

PR: <https://github.com/sarthakagrawal927/clicky/pull/new/eval/fm-fixtures-v2>

## What we should do next

1. **Don't train new LoRAs against fm-fixtures.** Until new fixtures
   that distinguish model from framework exist, training is theater.
2. **Build the new fixture set.** ~20-30 fixtures covering axes 1-5.
   Most can be hand-written; behavioral diversity (axis 5) can use
   LLM-paraphrase synthesis.
3. **Re-baseline FakePace against the new fixtures.** Target: FakePace
   ≤ 50%, a real model ≥ 85%. The delta IS the moat.
4. **Re-measure v5 / v6 / v6.1 against the new fixtures** to see
   whether any of them have residual capability that the old eval
   masked.

## Stakes

This finding makes most of today's earlier work — ANE M8 Swift
orchestrator, M9 spec dec experiment, v6.1 augmented SFT attempts —
either irrelevant or premature. The artifacts are still real; they
just don't answer the question we thought they answered.

The single most valuable hour of this session was building this
baseline. Knowing the eval is broken is worth more than any number
of speed improvements measured against it.
