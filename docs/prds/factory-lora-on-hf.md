---
name: LoRA application for TinyGPTModelHF
status: shipped-2026-06-08
owner: maintainer
created: 2026-06-08
priority: P0 — blocks inference-testing every HF-trained specialist
---

# PRD — LoRA application for `TinyGPTModelHF`

## 2026-06-08 ship note

The factory primitive (`LoraAdapterHFReader.apply`) was already shipped
in `LoraHF.swift`. What was missing was the **CLI surface** — no command
exposed it. Added `--lora <path.lora>` flag to:

- `tinygpt hf-load --sample` — loads HF base, applies LoRA, samples
- `tinygpt serve` — loads HF base, applies LoRA, serves OpenAI-compat

Both routes through `LoraAdapterReader.read(URL(...))` to parse the
adapter file, then dispatch to `LoraAdapterReader.apply` (for .tinygpt
bases) or `LoraAdapterHFReader.apply` (for HF dirs). Same `.lora` format
either way.

Files changed:
- `native-mac/Sources/TinyGPT/HFLoad.swift` — parse `--lora`, apply
  before sampling, doc string updated
- `native-mac/Sources/TinyGPTServe/Serve.swift` — parse `--lora`,
  pass through to `Server.boot`, apply after model load, switch on
  `load.model` (AnyModel) to call the right reader, doc string updated

Build clean. Help text verified shows both flags.

End-to-end specialist arc is now unblocked:
1. `tinygpt sft <hf-dir> --data ... --no-dora --out spec.lora` (train)
2. `tinygpt hf-load <hf-dir> --lora spec.lora --sample --prompt "..."` (validate)
3. `tinygpt serve <hf-dir> --lora spec.lora --port 8765` (use)

## Problem

`LoraAdapterReader.apply` only operates on `TinyGPTModel` (from-scratch).
HF-loaded bases use `TinyGPTModelHF` — a separate class with different
naming conventions (e.g. `layers.0.self_attn.q_proj` vs
`blocks.0.attn.q_proj`).

Result: we can TRAIN LoRA adapters on any HF base (Qwen3, Phi-3, Llama)
via `tinygpt sft`, but there's **no path to inference-test them**:
- `tinygpt sample` requires `.tinygpt` files only
- `tinygpt hf-load --sample` doesn't accept `--lora`
- `tinygpt serve` doesn't accept `--lora`

This blocks the entire "specialist arc" — you can't validate what you trained.

## Scope — in

### 1. `TinyGPTModelHF.applyLoRA(adapter:)` method

Mirror of `LoraAdapterReader.apply` for the HF model class. Walk
`model.blocks`, find each Linear, inject `LoraLinear` wrappers, load
saved A/B matrices.

The adapter file uses HF-style naming (`layers.0.self_attn.q_proj`)
already — written by `LoraAdapterHFWriter`. Reader just needs to
parse the same names back.

### 2. `tinygpt hf-load --sample --lora <path>` support

Add `--lora` flag to `hf-load` command. After base load, apply adapter
before sampling. Verify generation reflects the LoRA's training data.

### 3. `tinygpt serve --lora <path>` support

Add `--lora` flag to `serve`. Same pattern: load base, apply LoRA, then
serve via HTTP. Critical for using specialists in production (Pace, KB).

### 4. Round-trip smoke

A 1-step SFT + save + load-via-hf-load --sample on the same base should
not crash. Output of the LoRA-applied model should differ from the
unadapted base on the SFT data distribution.

## Scope — out (v2)

- Multi-LoRA composition (loading 2+ adapters and mixing weights). This
  exists for `sample` on `.tinygpt` bases; extend to HF later.
- LoRA training resume from a saved adapter (continue training where
  you left off).
- DoRA `m` vector round-trip (covered in separate DoRA-v2 PRD).

## Acceptance criteria

1. `tinygpt hf-load <qwen3-dir> --lora qwen3-fc-v2.lora --sample --prompt "test"` runs without crash, produces output different from the no-LoRA path.
2. `tinygpt serve <qwen3-dir> --lora qwen3-fc-v2.lora --port 8765` serves the LoRA-adapted model; `curl localhost:8765/v1/chat/completions` returns responses.
3. Phi-3 + LoRA + sample regression-clean.
4. Build clean.

## Files likely involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTModel/LoraHF.swift` | Add `LoraAdapterHFReader.apply(_:to:)` mirroring the from-scratch one |
| `native-mac/Sources/TinyGPT/HFLoad.swift` | Add `--lora` flag + apply call |
| `native-mac/Sources/TinyGPTServe/Serve.swift` | Add `--lora` flag + apply call before serving |

## Estimated effort

**~2-3 hours.** The pattern from `LoraAdapterReader.apply` directly
translates; mostly an mlx_swift NestedDictionary plumbing exercise.

## Why P0

This is the **completing-the-circle** PRD: SFT works, but the trained
adapter is unusable without this. Pace specialist + KB embedder + every
future specialist arc depends on it.
