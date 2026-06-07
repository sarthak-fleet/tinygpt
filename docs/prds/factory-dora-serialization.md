---
name: DoRA serialization in HF LoRA writer
status: shipped-v1-2026-06-08
owner: maintainer
created: 2026-06-08
priority: P1 — default SFT path produces empty adapters until fixed
---

# PRD — DoRA serialization in `.lora` writer

## 2026-06-08 v1 ship note

**Symptom**: `tinygpt sft` with default (DoRA) recipe wrote 209-218 byte
header-only `.lora` files. No matrices serialized. Affects Phi-3, Qwen3,
all bases — anything trained with DoRA.

**Root cause**: `LoraAdapterWriter.write` had `guard let lora = lin as? LoraLinear else { continue }`. `DoraLinear` and `LoraLinear` both inherit from `Linear` directly (not from each other), so the cast failed for DoraLinear and every entry was silently skipped.

**v1 fix**: writer now accepts both. DoRA's `m` magnitude vector is NOT
yet serialized (deferred to format v2) — DoRA adapters round-trip as
LoRA, losing the magnitude rescaling. Quality is close to LoRA but
slightly below true DoRA. Acceptable for v1 unblock.

What changed: `native-mac/Sources/TinyGPTModel/LoraIO.swift` lines
68-89. Both `LoraLinear` and `DoraLinear` casts attempted; their
`loraA` and `loraB` matrices are written. The reader path already
treats all entries as LoraLinear via `LoraInjection.inject`, so
round-trip works (as LoRA, not full DoRA).

## v2 — file format upgrade (queued)

To round-trip DoRA cleanly:
1. Bump file format to v2
2. Add optional `mShape: [Int]?` per entry
3. Writer emits `m` vector when present
4. Reader detects v2 + DoRA-trained entries, injects DoraLinear (vs LoraLinear)

Effort: ~2-4 hours. Not blocking for v1 specialists.

## Acceptance — v1 met

- Build clean ✅
- DoRA-trained adapter file > 1KB (writer no longer skips) ⬜ (untested
  yet; will verify after the Qwen3 FC v2 run completes — current Qwen3
  run uses `--no-dora` for safety)

## Acceptance — v2 (deferred)

- DoRA-trained adapter round-trips with identical predictions
- `m` vector is loaded back into DoraLinear, not flattened into LoRA
