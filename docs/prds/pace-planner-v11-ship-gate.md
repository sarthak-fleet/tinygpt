# Pace planner v11 — ship gate

**Signed**: 2026-06-09 (Sarthak + assistant pair)
**Status**: SHIP-GATE LOCKED — do not edit thresholds without a new dated entry below
**Replaces**: implicit gating from v8/v9 era (the 73% phantom)

---

## Why this document exists

v1 through v10 of the Pace planner cycled because we trained → measured → argued whether the version was better. No fixed bar, no fixed eval, no immutable record of pass criteria. The v8 "73.3% non-compose" number — celebrated, baked into landing copy — turned out to be **non-reproducible** in 2026-06-08 re-evaluation (actual: 33.3%). Several training sessions were spent chasing a metric that wasn't stable.

This document fixes the bar so v11 cannot accidentally inherit the same problem.

**The rule:** if v11 does not clear every threshold below against the exact eval suite committed today, **it does not ship as the default Pace planner.** No partial credit. No retrofit. No "but BFCL is up so let's call it a win."

---

## North Star recap

Pace planner contribution to the formula:

```
result = (speed × accuracy) / cost
```

A planner that hits 60% accuracy at 80ms TTFW is worth more than one at 65% accuracy at 200ms TTFW. Cost = bundle MB. v11 must not regress speed_score or cost_score vs v9.

---

## The six dimensions

v11 has to be good across **all six** of these — not just the happy path. v9's known failure mode was over-eager AX.press on out-of-scope / ambiguous / destructive prompts. v11 closes that hole.

| # | Dimension | What the model must do |
|---|---|---|
| 1 | **Happy path — selection + binding** | Pick the right action AND fill correct parameters |
| 2 | **Happy path — tool-call BFCL** | Same on a wider 12-action surface in BFCL-AST format |
| 3 | **Abstention (out of scope)** | Emit `intent: out_of_scope` when no Mac action satisfies the request |
| 4 | **Disambiguation** | Emit `intent: clarify` with a `question` field when context is insufficient |
| 5 | **Schema validity** | Output must be grammar-conformant 100% of the time (constrained decoding handles this; trained behavior should not undermine it) |
| 6 | **Safety** | Emit `intent: confirm_destructive` for destructive verbs; never directly fire |

Deferred (not in v11 bar): multi-turn state tracking, screen-grounding (deferred to VLM specialist).

---

## The bar — immutable thresholds

| Dim | Eval suite (committed today) | Threshold |
|---|---|---|
| 1 | `fm-fixtures-v2` (16 fixtures, Pace evals dir) | **≥ 60%** (v9 baseline: 33.3%) |
| 2 | `bfcl-pace-12` subset (built as part of #311) | **≥ 40% AST exact-match** |
| 3 | `fm-fixtures-oos` (30 new prompts, #310) | **≥ 80% correct refusal** |
| 4 | `fm-fixtures-ambig` (20 new prompts, #310) | **≥ 50% correct ask-back** |
| 5 | All emitted JSON across suites 1–4 | **≥ 95% AST-valid** |
| 6 | `fm-fixtures-destructive` (10 new prompts, #310) | **≥ 90% confirm-emission** |

Additionally — **non-regression gates**:

| | Threshold |
|---|---|
| `fm-fixtures-compose` (v9 = 70%) | **≥ 65%** (v11 must not break compose) |
| TTFW p95 warm (v9 = ~119ms) | **≤ 140ms** |
| Bundle size delta vs v9 | **≤ +20MB** |

If any non-regression gate fails, v11 fails the ship gate too. No "the new dimensions worked so we'll eat the latency regression" — Pace doctrine is latency is sacred.

---

## What "ship" means

If v11 clears the gate:
1. v11 becomes the default `--lora` path served by `tinygpt serve`
2. `pace/serve` in Pace's deployment bundle swaps to v11 weights
3. v9 weights stay in the registry for one release as a known-good fallback
4. Landing-page numbers (Codevetter draft, `pace/docs/landing/v1-draft.md`) update with v11 results — and only with v11 results

## What "fail" means

If v11 misses any threshold:
1. v9 stays as the production planner
2. v11 becomes a **research artifact** — kept, named `v11-research-2026-06-XX`, scored, documented in `docs/learn/v11-postmortem-2026-06-XX.md`
3. **We do not retrain v11 with different data and call it "v11.1"** — that is the trap. Instead, a fresh planner version (v12) gets its own ship-gate document with its own immutable suite, learning from v11's failure modes.
4. The eval suites stay. Every future planner version scores against the same six dimensions or doesn't ship.

---

## What is explicitly NOT allowed

- **Retroactively changing thresholds**. If v11 scores 58% on dim 1, that is not 60%. Period.
- **Replacing the eval suite mid-cycle**. The suites are frozen at the commit attached to this doc.
- **Cherry-picking dimensions**. "v11 hit 5 of 6, let's ship" → no. Ship gate is six AND non-regression.
- **Adding new dimensions to lower the bar**. If we discover a 7th dimension v11 is great at, that's a v12 input, not a v11 saver.
- **Re-running the eval to get a better number**. One run, one commit, done.

---

## Probability ledger (committed today, before any v11 training)

Per-dimension P(meets threshold), based on prior-version performance + technique-stack literature:

| Dim | P(meets threshold) | Mechanism that produces it |
|---|---|---|
| 1 Happy path | 70% | Thinking-teacher distillation + rejection sampling on 240+ rows |
| 2 BFCL-12 | 50% | Same training data + grammar constraint at inference |
| 3 Abstention | 75% | 150 hand-curated `out_of_scope` rows + rules pre-filter |
| 4 Disambiguation | 60% | 150 hand-curated `clarify` rows |
| 5 Schema | 95% | Grammar constraint (mathematically forced) |
| 6 Safety | 80% | 150 hand-curated `confirm_destructive` rows + rules pre-filter |
| **Joint (all 6 pass)** | **~45-50%** | Multiplicative composition |
| Joint (with preflight ablation gate) | **~70%** | Ablation eliminates the bad-data case |

**Honest call**: this is a coin flip raw, ~70% after preflight. Not a guaranteed ship. The conviction comes from the bounded downside — the eval suite, the hand-curated corpus, and the rules + grammar layer ship regardless of whether v11 itself ships.

---

## Decision authority

- **Eval execution**: any contributor can run the suites
- **Eval result interpretation**: must use the score CLI (`scripts/score_formula.py`) — no eyeballing
- **Ship decision**: Sarthak signs by appending a dated line to the bottom of this document

---

## How v11 is built (referenced for clarity, full plan elsewhere)

This doc is a gate, not a build plan. The build plan lives in:
- `#310` — eval suite construction (60 new prompts + score v9)
- `#311` — BFCL-12 gate eval
- `#291` — v10 finish + eval (decides whether v11 is even needed)
- `#312` (to be created) — v11 preflight ablation (200-row v10.5 test)
- `#313` (to be created) — v11 full training + eval

If v10 happens to clear this gate, v11 doesn't get built. The gate is what matters, not the version number.

---

## Append-only signature log

```
2026-06-09  Sarthak  Locked thresholds. v9 baselines: 33% v2, 70% compose, ~119ms TTFW.
                     v11 conviction: ~50% raw, ~70% after preflight ablation.
```
