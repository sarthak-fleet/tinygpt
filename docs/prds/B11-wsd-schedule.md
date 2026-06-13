---
name: B11 WSD (warmup-stable-decay) schedule
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B11)
related_prds: B10-quality-classifier.md, B12 (loss-spike recovery, partly covered by adam-state-persistence.md)
---

# PRD — WSD learning-rate schedule (warmup → stable → decay)

## Goal

Add `--lr-schedule wsd` to `tinygpt train` (`tinygpt finetune` / `sft` /
`distill` get it for free since they share the optimizer wrapper) with
three flags: `--warmup N` (steps to peak LR), `--stable-frac F`
(fraction of remaining steps held at peak), `--decay-shape {1-sqrt,
cosine,linear}` (decay curve from peak to 0 across the remainder).

Replaces cosine-warmup as the curated default for new pretrain runs.

## Why now

- WSD has become the small-model pretrain default since SmolLM ([HF blog,
  2024](https://huggingface.co/blog/smollm)) and MiniCPM ([Hu et al.
  2024](https://arxiv.org/abs/2404.06395)). The case: the *decay* phase
  IS the annealing knob — instead of running a fresh "annealing" pass
  on top of a separately-tuned cosine, one schedule does both, and the
  stable phase is checkpoint-mergeable (you can branch a run from the
  end of the stable phase into N different decay targets — different
  domains, different lengths — without redoing the warmup).
- Existing cosine has no equivalent of "branch the run into a
  domain-adapt decay" — every continued-pretrain re-warmups from a
  flat starting LR or pays a stability tax.
- Roughly half a day's work: the curve is already half-implemented in
  `Optimizers.swift`'s cosine path; WSD swaps the body of the schedule
  function and adds two flags.

## Scope — in

- `Sources/TinyGPTModel/LRSchedule.swift` (likely already exists; if
  not, factor cosine + linear into one file) gains a `WSD` case.
- `--lr-schedule wsd --warmup 500 --stable-frac 0.7 --decay-shape 1-sqrt`
  is the curated default the README example uses. `1-sqrt` (MiniCPM)
  decays slower than `cosine` early and faster late — empirically the
  best of the three.
- `tinygpt train --resume <ckpt> --lr-schedule wsd --warmup 0 ...` —
  the killer use case: resume from the end of a stable-phase checkpoint
  into a decay-shaped fine-tune.
- One sentence in `docs/training_guide.md` flipping the curated default.
- One smoke test in `evals/wsd-curve-smoke.swift` asserting the
  emitted LR-by-step matches the closed-form expression to ε=1e-6.

## Scope — out

- **Adaptive stable-fraction** (e.g. extend stable until grad-norm
  starts trending up). Nice in theory but adds a 200-line controller
  for a 1% gain at our scale; defer.
- **Multi-stage WSD** (warmup → stable_1 → decay_1 → warmup_2 →
  stable_2 → ...) — the µP-style continued-pretrain pattern. Skip
  until B21 micro-AutoMixer is shipped and we're actually re-mixing.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/LRSchedule.swift` | add `.wsd(warmup, stableFrac, decayShape)` |
| `Sources/TinyGPT/Train.swift` | parse `--stable-frac` + `--decay-shape`; pass to schedule constructor |
| `Sources/TinyGPT/TinyGPT.swift` | NO change — `--lr-schedule` is already in `Train.swift` |
| `evals/wsd-curve-smoke.swift` | new — closed-form LR check at warmup, mid-stable, mid-decay, end |
| `docs/training_guide.md` | curated default flip |

## Acceptance criteria

- [ ] `tinygpt train --lr-schedule wsd --warmup 500 --stable-frac 0.7
  --decay-shape 1-sqrt --steps 5000 ...` runs and the in-loop LR matches
  the closed-form WSD curve at every checkpoint print.
- [ ] On a 22M shakespeare pretrain over 5K steps, val PPL at the end
  is ≤ within 1% of the cosine baseline (this is the floor check —
  WSD shouldn't underperform).
- [ ] Resume-from-stable-checkpoint pattern works: pretrain 5K steps
  with `--stable-frac 0.9` (so the run ends mid-stable), then
  `--resume <ckpt> --lr-schedule wsd --warmup 0 --stable-frac 0.0
  --decay-shape 1-sqrt --steps 2000` decays cleanly to 0.
- [ ] `docs/training_guide.md` curated invocation block updated.

## Reference patterns

- `Sources/TinyGPTModel/Optimizers.swift` — the existing cosine
  schedule lives here; same struct add-a-case shape.
- [MiniCPM paper](https://arxiv.org/abs/2404.06395), §3.2 — the
  closed-form `1-sqrt` decay equation. Don't redocument it; cite.
- [SmolLM blog](https://huggingface.co/blog/smollm) — the WSD-vs-cosine
  pretrain ablation table. Reference for the "WSD ≥ cosine at fixed
  budget" claim.

## Open questions

- Whether the V1 default `--decay-shape` should be `1-sqrt` (MiniCPM)
  or `cosine` (SmolLM uses cosine-shaped decay). **Recommendation:**
  ship with `1-sqrt` per the published ablation. Add a one-line
  decision log in `docs/decision_log.md`.
