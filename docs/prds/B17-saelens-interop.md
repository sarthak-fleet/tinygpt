---
name: B17 SAE Lens interop / Neuronpedia format export
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B17)
related_prds: B13-interp-on-checkpoints.md, B19-group-sae.md
---

# PRD — Export TinyGPT SAEs in SAELens / Neuronpedia format

## Goal

Add `tinygpt sae export --format saelens <input.sae> --out <dir>`
that converts our shipped `.sae` sidecar format to the
[SAELens](https://github.com/decoderesearch/SAELens) on-disk format
([Neuronpedia](https://www.neuronpedia.org/) ingest target). One-way
export only — we keep our format internally; SAELens / Neuronpedia
get a consumable artifact.

This is the cheapest of the three interop options floated in PLAN B17:
- (a) keep ours, no interop — current state
- (b) port to SAELens — 1-2 weeks, throws out our shipped code
- (c) export only — 1-2 days, no internal disruption

(c) is what this PRD does.

## Why now

- Our SAE infra exists and works (`tinygpt sae`, `sae-explore`).
  Whether to interop with the rest of the field is a UX question, not
  a science question.
- Neuronpedia is the public storefront for SAE feature dictionaries —
  if we ever want our findings cited beyond our docs, that's the
  publication target.
- The format gap is just JSON metadata + safetensors weights with a
  specific naming convention. No re-training, no quality risk.

## Scope — in

- `Sources/TinyGPT/SaeExport.swift` — new subcommand. Reads a `.sae`
  file, writes a SAELens-compatible directory:
  - `cfg.json` (architecture + training config — fields documented in
    SAELens README)
  - `sae_weights.safetensors` (encoder + decoder)
  - `sparsity.safetensors` (per-feature activation rate, computed
    from a calibration corpus or extracted from `.sae` if we logged it)
  - `README.md` per Neuronpedia's required schema
- Reverse direction (SAELens → `.sae`) is **out** for this PRD —
  defer until we have a reason to consume their dictionaries.
- One round-trip test: export a known `.sae`, load via SAELens'
  Python API in a Python subprocess (no Python source in our tree),
  assert encoder + decoder weights byte-equal after re-normalization
  to the same convention.

## Scope — out

- **Inbound import** (SAELens → tinygpt). Defer.
- **Cross-format feature explorer** — keep our `sae-explore`; the
  exported SAEs live on Neuronpedia for cross-tool exploration.
- **Auto-publish to Neuronpedia.** Their API exists but ingestion
  goes through their PR queue; we'd write the artifact and the human
  uploads.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/SaeExport.swift` | new — converter |
| `Sources/TinyGPT/TinyGPT.swift` | `case "sae-export"` (split from `sae` to keep the original tight) |
| `Sources/TinyGPTModel/SaeReader.swift` | expose private `.sae` parsing fields needed for export (sparsity stats etc.) |
| `Tests/TinyGPTTests/SaeExportRoundtripTest.swift` | new — see above |
| `docs/interpretability.md` | "Publishing to SAELens / Neuronpedia" section |

## Acceptance criteria

- [ ] `tinygpt sae export --format saelens existing.sae --out /tmp/exp/`
  produces `cfg.json`, `sae_weights.safetensors`, `sparsity.safetensors`,
  `README.md`.
- [ ] SAELens' loader (called via a 5-line Python subprocess in CI)
  loads the export without error.
- [ ] Roundtrip test passes — exported encoder/decoder weights are
  numerically identical (modulo SAELens's column-vs-row convention)
  to the originals.
- [ ] `docs/interpretability.md` has a worked example pointing at a
  fixture export.

## Reference patterns

- [SAELens repo](https://github.com/decoderesearch/SAELens), folder
  `sae_lens/sae.py` — the on-disk format definition. Cite, don't
  re-document.
- `Sources/TinyGPT/HfLoad.swift`'s HF-tensors writer — the
  safetensors-writing template; we already produce `.safetensors`
  for HF-compatible model exports.
- `Sources/TinyGPTModel/SafetensorsWriter.swift` — shared utility,
  reuse.

## Open questions

- Whether to emit per-feature human-readable IDs (the SAELens
  convention) or pure numeric ids. **Recommendation:** numeric only;
  feature naming is the user's job, not a converter's.
