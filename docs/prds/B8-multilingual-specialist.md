---
name: B8 multilingual specialist (Indic-focus first)
status: not-started (blocked-by A7)
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B8)
related_prds: A1-first-specialist-tool-caller.md (A1's recipe template),
              docs/research/indic_evals.md (eval surface)
---

# PRD — Multilingual specialist on top of Sarvam-Edge / Airavata base

## Goal

Train a multilingual SFT/LoRA specialist on a Sarvam-Edge or Airavata
base (both are Indic-LLM bases optimized for English + 10+ Indian
languages) for a Mac-runnable Indic-capable agent. Ship gate: beats
the 0-shot base on the MILU eval (`tinygpt eval-indic`) by ≥ 3pp
average across at least 3 Indian languages.

Different from A1/B1: A1/B1 are domain specialists; B8 is a
*language-coverage* specialist. Same recipe shape; different
acceptance axis.

## Why now

- Indic evals are wired (`tinygpt eval-indic`, smoke-validated per
  PLAN.md). The eval surface exists; the *trained-for-Indic*
  specialist doesn't.
- The Indian-language LLM ecosystem (Airavata, OpenHathi, Sarvam)
  has shipped solid open bases. Specializing on top of them is
  the cheapest way to deliver a usable Indic agent without a
  multi-month pretrain.
- Distinct user need from the English specialists: a Mac-running
  Hindi/Tamil/Telugu agent has no good open competitor at the
  ≤ 4B size class.

## Scope — in

- **Base candidates:** Sarvam-Edge (newest), Airavata (well-tested),
  or OpenHathi as fallback. V1 picks one based on Mac MLX
  compatibility; document in `decision_log.md`.
- **Training data:** Indic SFT corpora (Aya from CohereForAI,
  IndicSUPERB, MILU's training split). All open.
- **Recipe:** mirror A1's recipe shape (`scripts/recipes/b8-indic.sh`).
- **Eval:** existing `tinygpt eval-indic` extended to report per-
  language scores in MILU + IndicGenBench-XQuAD.
- **Ship gate:** average across Hindi/Tamil/Telugu MILU ≥ base
  + 3pp under B23 K=3 protocol.

## Scope — out

- **Full multilingual** (50+ languages, mBERT-style). V1 is
  Indic-focused.
- **Cross-lingual transfer experiments.** Train and eval same
  language family for V1.
- **TTS / Indic speech.** Distinct PRD if needed (consumes 5.6).

## Files to touch

| File | Change |
|---|---|
| `scripts/recipes/b8-indic.sh` | new — recipe |
| `Sources/TinyGPT/EvalIndic.swift` | already exists; add per-language breakdown if missing |
| `docs/specialists/b8-indic.md` | new — brief |
| `docs/PLAN.md` | B8 ⬜ → ✅ on ship |

## Acceptance criteria

- [ ] Hindi MILU ≥ base + 3pp, Tamil + Telugu similarly.
- [ ] No regression on English BFCL ≥ -2pp (specialist shouldn't
  break English capability).
- [ ] B8 row appears on the SLM leaderboard alongside A1 / B1.

## Reference patterns

- A1's recipe.
- [Airavata paper](https://arxiv.org/abs/2401.15006) — base
  characteristics + Indic SFT recipe references.
- `docs/research/indic_evals.md` — eval landscape.

## Open questions

- Sarvam-Edge vs Airavata. **Recommendation:** pick at training-
  time based on MLX compatibility check; both are valid V1 bases.
