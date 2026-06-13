---
name: C3 DoRA on-disk adapter format
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier C (C3)
related_prds: factory-dora-serialization.md (incomplete sibling — covers the protobuf-vs-safetensors choice)
---

# PRD — DoRA on-disk adapter format

## Goal

DoRA already trains in-session under `tinygpt sft` (default PEFT
variant since 2026-05; `PeftVariants.swift`). It does NOT yet persist
to disk in a reusable format — the adapter lives only in the running
process's parameters, so finishing an `sft` run with DoRA produces no
shippable artifact. Ship the on-disk format + reader so DoRA reaches
parity with LoRA / LoRA+ / VeRA (all of which already roundtrip).

## Why now

- This is the highest-friction PEFT path: DoRA is the default and the
  one users hit first, and right now you can't save what you trained.
  Diagnosed in `factory-dora-serialization.md` but not closed out.
- LoRA's adapter format is well-defined (`LoraAdapterReader.swift`).
  DoRA needs the same shape plus the per-row magnitude vector `m`
  that distinguishes DoRA from LoRA. Three small fields, well-bounded.
- Specialists ship LoRA-style adapters to consumers; DoRA's slight
  quality edge over LoRA goes away if we can't deliver the artifact.

## Scope — in

- Extend the existing `.lora` adapter format with two optional fields:
  - `dora_magnitude: [Float]` per-row magnitude vector (one per Linear
    in the adapted set)
  - `peft_variant: "lora" | "dora" | "lora+" | ...` discriminator string
  Backwards-compat: a `.lora` without `peft_variant` defaults to `lora`,
  so existing adapters still load.
- `LoraAdapterReader.read(...)` learns to apply the magnitude when
  `peft_variant == "dora"`. New name kept for backcompat: `PeftAdapter`
  as a module-level alias may come later, but not in this PRD.
- `tinygpt sft --peft dora ... --out adapter.lora` actually writes a
  loadable adapter at the end of training. Today it prints "DoRA
  adapter not yet serializable" — that warning gets deleted.
- `tinygpt compare base.tinygpt + adapter.lora vs base.tinygpt` (the
  existing compare CLI) gives a real per-eval delta now that adapters
  are reloadable.

## Scope — out

- **Brand-new file extension** (`.dora`). Same envelope, discriminator
  field — keeps the registry small. (`factory-dora-serialization.md`
  considered this and rejected; honor.)
- **GGUF-style adapter export** — that's an interop story for HF
  consumers, not us today.
- **Adapter merging** (DoRA-on-DoRA stacking) — niche, defer.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/LoraAdapterReader.swift` | add `dora_magnitude` field; apply when `peft_variant == "dora"` |
| `Sources/TinyGPTModel/LoraAdapterWriter.swift` | write magnitude when source is DoRA |
| `Sources/TinyGPTModel/PeftVariants.swift` | expose the per-Linear magnitude tensor + the `peft_variant` tag for the writer |
| `Sources/TinyGPT/SFT.swift` | delete the "not yet serializable" warning; on `--out`, write the DoRA adapter |
| `Tests/TinyGPTModelTests/DoRARoundtripTests.swift` | new — train 50 steps DoRA on shakespeare, save, reload, assert output token-equal at T=0 vs in-memory adapter |
| `docs/lora_guide.md` | one paragraph: DoRA now persists; format is `.lora` with `peft_variant: dora` |

## Don't touch

- LoRA / LoRA+ / VeRA save paths — extend `LoraAdapterReader/Writer`
  without forking. The magnitude field is optional, so other variants
  are unaffected.

## Acceptance criteria

- [ ] `tinygpt sft --peft dora --out /tmp/a.lora ...` produces a file
  with `peft_variant: "dora"` and a non-empty `dora_magnitude` block.
- [ ] `tinygpt sample <base> --lora /tmp/a.lora` loads, runs, and
  produces *token-identical* output at T=0 to in-memory DoRA after the
  same SFT step count.
- [ ] DoRARoundtripTests.swift passes — save → reload → forward
  matches step-end logits to ε=1e-5.
- [ ] `tinygpt compare base.tinygpt + a.lora vs base.tinygpt` on a 200-
  prompt eval shows a non-zero delta consistent with the in-memory DoRA's
  effect.

## Reference patterns

- `Sources/TinyGPTModel/LoraAdapterReader.swift` — direct template;
  the safetensors-style binary envelope.
- `Sources/TinyGPTModel/PeftVariants.swift` — `dora` is in-session
  today, so the parameter shapes are settled; just need to expose
  `m` for serialization.
- `factory-dora-serialization.md` — note the prior reasoning. Don't
  redo the format-choice argument; reference.

## Open questions

- Whether magnitude should be stored as fp16 or fp32. **Recommendation:**
  fp32 since it's one row per Linear (tiny — kilobytes for a 24-layer
  model), and DoRA quality is sensitive to magnitude precision per the
  original paper.
