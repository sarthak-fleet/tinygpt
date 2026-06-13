---
name: B15 layer-wise LR decay for SFT
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B15)
related_prds: B11-wsd-schedule.md (sibling — LR-schedule family)
---

# PRD — Layer-wise LR decay (`--llrd γ`) for `tinygpt sft`

## Goal

Add `--llrd γ` to `tinygpt sft` (and the DPO family for symmetry).
For γ < 1, each transformer block's LR is multiplied by γ^(depth_from_top),
so the embedding + lower blocks see exponentially smaller updates than
the LM head + upper blocks. The standard "freeze the foundation, finetune
the head" trick from ULMFiT ([Howard & Ruder 2018](https://arxiv.org/abs/1801.06146))
turned into a continuous knob.

Default γ = 0.95 (mild decay) when the flag is omitted with a value;
γ = 1.0 (today's behavior) when the flag isn't passed at all so existing
recipes don't drift.

## Why now

- We already ship `cfg.lrLayerDecay` for *pretrain* (verified 2026-06-02
  in the third-audit pass). It is NOT wired into `tinygpt sft`'s
  optimizer construction — finetune runs use a flat LR across all
  parameter groups. Mismatched.
- SFT on small + tiny corpora overfits the upper layers and degrades
  the embedding's broader features unless you either freeze the lower
  layers (loses 1-2% quality) or scale their LR down. LLRD is the
  smooth middle option.
- Half-day add — the `Optimizers.swift` group-LR scaffolding already
  exists for the pretrain path; the work is exposing it in the SFT
  CLI + adapter loop.

## Scope — in

- `--llrd γ` flag on `tinygpt sft`, `dpo`, `finetune`. Range check:
  0.5 ≤ γ ≤ 1.0; γ outside that prints a usage error.
- Per-block LR computed as `base_lr × γ^(n_layers - 1 - layer_idx)` so
  the topmost transformer block gets `base_lr`, and the embedding +
  block 0 get `base_lr × γ^(n_layers - 1)`.
- LM head + final norm always at `base_lr` (γ doesn't decay them — they
  *are* the head).
- One row in the SFT run banner: `LR layer-decay: γ=0.95 → block_0
  LR=2.3e-6, head LR=3e-5`.

## Scope — out

- **Per-group LR (group different parameter families differently)** —
  that's an orthogonal knob; defer.
- **Adaptive γ** (estimate γ from gradient norms per block) — research-
  grade; defer.
- **LLRD for pretrain** — already shipped via `cfg.lrLayerDecay`; this
  PRD is only the SFT-side surfacing.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/Optimizers.swift` | factor `groupedLRs(γ, layers)` helper (likely already exists for pretrain — re-use, don't duplicate) |
| `Sources/TinyGPT/SFT.swift` | parse `--llrd`, pass to optimizer construction, log resolved per-group LRs in banner |
| `Sources/TinyGPT/DPO.swift` | same flag, same plumbing |
| `Tests/TinyGPTModelTests/LLRDTests.swift` | new — assert per-block LR matches γ^k closed-form |

## Acceptance criteria

- [ ] `tinygpt sft --llrd 0.95 ...` runs and reports per-block LR
  in the banner.
- [ ] On a fixed shakespeare → alpaca SFT recipe, val loss at the
  end of training is ≤ the no-LLRD baseline (this is the
  no-regression floor; the real win is on larger bases).
- [ ] Held-out OOD eval (a domain *unlike* alpaca, e.g. code) shows
  ≤ 1% PPL regression vs no-LLRD, where no-LLRD typically regresses
  ~3-5%. This is the "preserve foundation features" payoff.
- [ ] LLRDTests.swift passes — verifies group LR computation matches
  γ^k at every depth for n_layers ∈ {4, 12, 24}.

## Reference patterns

- `Sources/TinyGPT/Train.swift` — already constructs parameter groups
  for `cfg.lrLayerDecay`. Lift the helper into a module-level fn so
  both Train and SFT can use it.
- [ULMFiT, Howard & Ruder 2018](https://arxiv.org/abs/1801.06146) —
  the original LLRD recipe (they used "slanted triangular LR" but
  the per-layer decay is the same concept).
- Modern adoption: BERT, GPT-3 finetuning notebooks; not novel,
  just unsurfaced in our SFT CLI.

## Open questions

- Whether to default γ=1.0 (off) or γ=0.95 (on) when `--llrd` isn't
  passed. **Recommendation:** default OFF for back-compat; the
  recipe-doc and `--llrd 0.95` curated invocation become the
  recommended path. Decision logged in `docs/decision_log.md`.
