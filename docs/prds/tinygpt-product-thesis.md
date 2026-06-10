# tinygpt product thesis — the embedded local-AI runtime for Mac apps

Status: positioning doc (2026-06-10). Written as the specialist track closes
(v11 = final 0.6B planner run, verdict pending) and focus shifts to
fine-tuning/distilling larger open models. This doc makes the thesis explicit
so the large-model phase starts from it, not from habit.

## The thesis in one line

**tinygpt's durable asset is the runtime, not the factory. Product = the
embedded local-AI runtime for Mac apps. Pace is its first proof.**

## Evidence (why the factory isn't the product)

- v1–v11: ~11 planner versions taught one lesson — the 0.6B ceiling
  (~60% fm-fixtures-v2) is the model, not the pipeline. Zero-shot Qwen3-14B
  matches it with no training; Pace's runtime planner is already
  qwen3-30b-a3b.
- The month's real wins were all runtime-side: ANE chain, TTFW 119ms,
  grammar-constrained serving, int8/fp16-IO quantization, streaming partial
  JSON. None came from training.
- Specialist training only paid when it was *adaptation* of a strong base
  (LoRA/DoRA on Qwen3) — never from-scratch.

## What tinygpt uniquely has (the moat inventory)

1. **ANE end-to-end LLM decode with LoRA support** (M8: 28-block stateful
   CoreML chain, fp32-compute/fp16-state, int8 per-block weights, numerics
   gate). Nobody else ships this — anemll has no LoRA path, Apple's stack is
   closed.
2. **Grammar-constrained Swift serving** — JSON-Schema/GBNF FSM masking at
   119ms warm TTFW with streaming partial JSON. mlx-lm is Python; Ollama/LM
   Studio are apps, not embeddable libraries.
3. **On-device adaptation loop** — LoRA/DoRA train → bake → serve → eval →
   quantize, all base-model-agnostic, zero Python at runtime.

The competitive gap: Apple Foundation Models framework is closed, small, and
uncustomizable; the MLX ecosystem is Python-first. Embeddable + customizable
+ Apple-silicon-native is the empty quadrant tinygpt occupies.

## What tinygpt is NOT (scope discipline)

- Not a from-scratch training framework (learning corpus only — keep for the
  HN story, don't maintain as product surface; SAE/MEMIT/interp likewise).
- Not a Python library or cloud anything.
- Not a model zoo — leverage open weights; custom only where the Mac/ANE/UX
  moat lives (per the leverage-first rule).

## How the roadmap serves the thesis

| Phase | Thesis role |
|---|---|
| 1. v11 specialist close-out | Bounded: one run, ship or fail. Either way the track closes and the adaptation pipeline is proven on a real gate. |
| 2. VLM + Pace needs | Voice loop airtight first (WhisperKit → planner → executor under 100ms doctrine) — the make-or-break demo. VLM A/B picks the port target; the port itself extends the runtime (M8 pattern → vision tower). |
| 3. Fine-tune/distill larger models | tinygpt's new job: adapt + serve 4–14B open weights brilliantly on Apple silicon. QLoRA (LoRA-on-quantized-base) is the hardware unlock on 48 GB; 30B-A3B stays teacher-only. |

Decision rule carried forward from the specialist era: **before any
fine-tune, measure the zero-shot base on the same gate.** Train only when
zero-shot demonstrably falls short. (This rule would have saved most of
v1–v10.)

## Risks

- **Apple commoditization** — Foundation Models + App Intents could absorb
  the easy 80%. Defense: stay at the customizable/controllable end (custom
  adapters, grammar control, multi-model orchestration) Apple won't expose.
- **Solo vs ecosystem velocity** — the thesis only holds with ruthless
  leverage: port primitives (e.g. the int8 handoff port), never rebuild
  ecosystems (tinygrad/anemll rejections were correct calls).
- **Runtime without a customer is a library nobody asked for** — Pace must
  stay the forcing function; ship runtime features only when Pace (or the
  eval gates) pulls them.

## v11 verdict

_Placeholder — fill when the 2026-06-10 pipeline run lands:_
- Dim1–6 scores vs ship gate: TBD
- Ship / fail: TBD
- Either way: specialist track closes, phase 2 begins.

## Related

- `PLAN.md` — canonical shipped/skipped/TODO
- `docs/prds/pace-planner-v11-ship-gate.md` — the gate this thesis inherits
- `docs/prds/quantized-inference-swift.md` — Phase 2 (native packed load / QLoRA) now centerpiece
- `docs/prds/vlm-ab-uivenus-vs-qwen3vl.md` — phase 2 decision gate
- Memory: `feedback-focus-finetune-distill-large`, `feedback-leverage-first`,
  `feedback-tinygpt-north-star`
