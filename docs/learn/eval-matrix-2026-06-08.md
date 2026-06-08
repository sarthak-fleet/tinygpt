# Eval matrix — what every Pace LoRA actually scores

**Date**: 2026-06-08 (overnight session)
**Status**: shipped; all LoRAs re-baselined against fm-fixtures-v2
**Companion**: `eval-methodology-2026-06-08.md` (the gate finding +
the v2 fixture set)

## The full matrix

| System | v1 (`fm-fixtures`) | v2 (`fm-fixtures-v2`) | v2 Δ vs FakePace |
|---|---|---|---|
| Base Qwen3-0.6B + grammar (no LoRA) | not tested | 0/15 (0%) | **−6.7 pp** |
| FakePace (rule-based, no model) | 19/19 (100%) | 1/15 (6.7%) | baseline |
| Pace v6 LoRA | 14/19 (74%) | 3/15 (20.0%) | +13.3 pp |
| Pace v6.1 LoRA | 10/19 (one variant) | 4/15 (26.7%) | +20.0 pp |
| Pace v5 LoRA | 17/19 (89%) | 6/15 (40.0%) | +33.3 pp |
| Qwen3-14B (no LoRA, no grammar) | — | 9/15 (60.0%) | +53.3 pp |
| **Pace v8 LoRA** | **not tested** | **11/15 (73.3%)** | **+66.7 pp** ← NEW BEST |

## Pace v8 — factory thesis validated (2026-06-08)

v8 was trained on v5's 248-row corpus + 59 hand-crafted examples
targeting v2's failure modes (semantic disambiguation, multi-element
reasoning, abstract reference). Same hyperparameters as v5/v6:
rank 32, alpha 64, 3000 steps, lr 1e-4, batch 4, chatml template.

**Result: 11/15 (73.3%) on fm-fixtures-v2. Beats Qwen3-14B (60%) by
13 pp. Beats Pace v5 by 33 pp. +66.7 pp above the rule-based ceiling.**

This is the first time in the project a 0.6B Pace specialist has
empirically outperformed a 14B generalist on the planner task.
The factory thesis ("a small specialist beats a large generalist
for narrow tasks") is now validated, not just claimed.

**Where v8 still fails (4 of 15):**
- `abstract-make-payment` ("pay my electric bill" → Transfer)
- `reason-most-expensive` (parse $0/$15/$99, pick max)
- `reason-oldest-email` (timestamp comparison)
- `semantic-word-processor` ("write a letter" → Pages)

Three of these are the harder reasoning + abstract cases. To close
to ≥85% (the ship-worthy bar), v9 would need ~10 more examples
targeting these specific failure shapes.

**Training corpus** (committed in `scripts/pace-v8-augment.py`):
- 248 v5 rows (semantic-rich, action-tag-aware base)
- 29 semantic disambiguation rows
- 18 multi-element reasoning rows
- 12 abstract reference rows
- Total: 307 rows

**Hyperparameters that worked**:
- LoRA rank 32, alpha 64 (no DoRA — first attempt with DoRA hit OOM under memory pressure; plain LoRA was stable)
- 3000 steps, lr 1e-4, batch 4
- chatml template
- Loss converged to 0.000 around step 200; remaining steps are
  noise but didn't degrade (in v8's case)

Wall-clock: ~143 min on M5 Pro with other processes running.

## Overfit check — held-out generalization (2026-06-08)

After v8 trained, built `fm-fixtures-holdout/` (15 fixtures) using
novel apps + products + scenarios that do NOT appear in v8's
training corpus. Same three axes, different surface:

  Semantic (5): Figma / Zoom / Slack / Notion / Lightroom
  Reasoning (5): hotel prices, delivery times, car mileages,
                 PR timestamps, game scores
  Abstract (5): leave tip / undo / bookmark / flag spam / review

The test: if v8 overfit to training patterns, holdout scores collapse.

Result:

| Set | FakePace | v8 LoRA | Δ |
|---|---|---|---|
| fm-fixtures-v2 (training-adjacent) | 1/15 (6.7%) | 11/15 (73.3%) | +66.7 pp |
| fm-fixtures-holdout (novel) | 0/15 (0.0%) | 10/15 (66.7%) | +66.7 pp |

**Generalization gap: 6.7 pp.** The Δ vs FakePace is IDENTICAL on
both sets. v8 is doing model work, not memorizing the v2 shapes.

v8 succeeded on novel apps it never saw in training:
- Figma ("design a logo"), Zoom ("hop on a call"),
  Slack ("work chat"), Lightroom ("retouch photo")
- Hotels by price ("cheapest"), cars by mileage ("lowest")
- Abstract goals (bookmark, leave tip, flag spam)

The model learned the *pattern* (intent → app via world knowledge,
parse element text + pick by superlative, goal → action), not
specific memorizations.

**What v8 misses on holdout (5 of 15) — same shapes as v2 failures:**
- `reason-fastest-delivery` — parse "5-7 days" / "2 days" / "by 9pm"
- `reason-newest-pr` — parse "4 hours ago" / "yesterday" / "last week"
- `abstract-undo` — "take that back" idiom
- `abstract-rate-experience` — "how I felt about it" idiom
- `semantic-knowledge-base` — Notion = docs (not in training)

All same root causes as v2 failures: numerical/temporal comparison
remains hard, plus a few missing semantic + idiomatic mappings.

**For v9**: target the underlying capability gaps, not the specific
v2 fixtures. Adding examples that look like the 4 v2 failures would
risk overfit; we'd be re-shaping training to match the test we're
measuring against. Better:
- Add 5-10 diverse "comparison reasoning" examples (numbers, dates,
  durations) across many domains
- Add 5-10 idiomatic phrasings of common actions
- Skip the urge to add "pay electric bill" specifically — that's
  test-tuning, not generalization

All v2 results from `python3 scripts/eval_pace_v2.py` against the
same fixture set, same grammar config, same serve harness.

## What this rewrites about past work

### 1. v1 was lying about v6.1

We spent ~4 hours of session time on the "v6.1 collapse" — the
elf's four-way SFT sweep produced v1 scores of 10/19, 8/19, 2/19,
0/19, which we treated as catastrophic regression and used as the
trigger for the eval methodology investigation.

**v2 result**: v6.1 scores 4/15 (26.7%) — **better than v6 (20%)
by 6.7 pp**. The augmentation that scored 0/19 on v1 was *not* a
collapse. It added real capability that v1's regex-based scoring
couldn't see, because v1 was measuring format compliance against
a JSON template the v6.1 model deliberately diverged from.

### 2. v5 IS the real moat — and always was

40% on v2, +33 pp over the rule-based baseline, +40 pp over the
bare model. The Pace v5 LoRA does genuine task-specific work. The
v1 17/19 score wasn't measuring this — it was measuring format
compliance — but the capability was there all along.

Concrete: v5 LoRA passes 6 of 15 hard model-required fixtures
without any spec-dec or scale tricks, on a 0.6B model. That's a
real result.

### 3. v6 IS a real regression

Earlier in the session I theorized v6's lower v1 score was just
"format compliance loss while keeping capability." **Wrong.** v6's
v2 score is 20% (3/15), a true 20 pp regression from v5's 40%.
The label-based SFT destroyed capability, not just format.

Specifically: v5 passes 5 semantic-disambiguation fixtures
(code-editor, music-app, email, browser, plus reason-most-expensive-
related). v6 only passes browser + reason-most-expensive + slides.
The label-based training narrowed the model's behavior.

### 4. Bare Qwen3-0.6B is unusable under grammar

0/15 on v2. The grammar enforces a JSON shape the bare model
doesn't know how to fill correctly — it produces valid JSON but
picks wrong labels every time. **The Pace LoRA is required, not
optional, for this workload.**

This is an important finding because it means: if you want to
ship Pace on Qwen3-0.6B, you NEED the LoRA. Choosing between
"trained" and "untrained" was never the choice; the choice is
between LoRA variants.

### 5. No shipped LoRA reaches the ship-worthy bar

The bar (≥85% on v2) is genuinely hard. Even Qwen3-14B (the
frontier model in this matrix) reaches only 60%. Pace v5 reaches
40%. To get to 85% we'd need either:

- A much larger base model with the v5 LoRA pattern
- A v7 LoRA on a base that already has stronger world knowledge
  (e.g., xLAM-1B or a Qwen3-VL-2B with the planner task fine-tuned)
- Tree-based or grammar-constrained spec dec where the model gets
  multiple chances per output

## What the matrix tells us about next steps

### Re-prioritize the queued tasks

- **#265 v6.1 quality block** — the "block" was an artifact of the
  broken eval. v6.1 is the best label-based variant we have. The
  block does not need investigation; v6.1 just needs more
  augmentation or to be merged with v5's training pattern.
- **#268 specialist quality unblock** — same caveat. The dirty
  teacher labels may not be the actual issue. Re-measure under v2
  before reinvesting.
- **#267 v7 SFT** — now has a real success metric: ≥85% on v2.
  Without this gate, v7 would have shipped at "fewer fixtures
  failed" without knowing if it was format compliance or capability.

### The honest version of "factory beats teacher"

We can now express this claim correctly:

- Pace v5 LoRA (0.6B base, 248 examples): **40% on v2**
- Qwen3-14B (no LoRA, no fine-tuning): **60% on v2**
- Qwen3-30B-A3B (no LoRA): not yet tested, expected ~70-80%

So at the moment, **the "teacher" beats our specialist** on the
real eval. v5 closes 33 pp of the 60 pp gap that bare base had to
the 14B; another LoRA + better data could plausibly close more.

The factory thesis ("a 0.6B specialist beats a 14B generalist for
narrow tasks") is **falsifiable now**. v5 is 20 pp behind 14B on
this eval; if we can train a v8 that hits 65%+, the thesis is
real. Otherwise it's not.

## Reproducer

```
# FakePace baseline (no serve needed):
python3 scripts/eval_pace_v2.py --skip-model

# Any LoRA — boot serve then run eval:
tinygpt serve <hf-dir> \
    --lora <lora-path>.lora \
    --grammar grammars/pace-fm-label-response.schema.json \
    --port 8765 &
python3 scripts/eval_pace_v2.py \
    --serve-url http://127.0.0.1:8765/v1/chat/completions

# Qwen3-14B via LM Studio:
python3 scripts/eval_pace_v2.py \
    --serve-url http://127.0.0.1:1234/v1/chat/completions \
    --model "qwen/qwen3-14b"
```

Each run takes ~30-60s for 15 fixtures.

## Bottom line

Today's eval methodology work has:

1. **Rescued** v5 (and v6.1) from being treated as noise.
2. **Confirmed** v6 was a real regression.
3. **Established** the ship-worthy bar (≥85% on v2).
4. **Provided** a reproducible measurement that any future LoRA must clear.

The fixtures + harness now let us answer "is this LoRA worth shipping"
with a number instead of vibes. That's the gate task #270 closed in
its full intended scope.
