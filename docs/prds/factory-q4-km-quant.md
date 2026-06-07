---
name: Q4_K_M quantization — the missing GGUF quant level
status: shipped-covered-by-gguf-reader-2026-06-07
owner: unassigned
created: 2026-06-07
priority: P2 — small but valuable
---

# PRD — Q4_K_M quant in `tinygpt gguf-extract`

## 2026-06-07 resolution

No new tensor enum was needed. GGUF stores tensor block types such as `Q4_K`
and `Q6_K`; `Q4_K_M` is a llama.cpp quantization preset that commonly mixes
those block types by tensor. `GGUFReader.loadTensor` already supports
`Q4_K`, `Q5_K`, `Q6_K`, and `Q8_K`, and `gguf-extract` materializes every
supported tensor to `model.safetensors`.

This pass only corrected stale `gguf-inspect` help text that still claimed
K-quants printed as unsupported.

## Goal

Add Q4_K_M quantization to the existing GGUF quantization path. Q4_K_M
is the de-facto "best Q4" level — used as default in llama.cpp + Ollama
for most models. We ship Q5_K, Q6_K, Q8_K today but not Q4_K_M, which
is the most commonly-requested level for "smallest Mac-shippable."

## Scope — in

### CLI

```
tinygpt gguf-extract --bits q4_k_m --in model.gguf --out weights/
```

Add `q4_k_m` to the existing `--bits` enum. Implementation mirrors the
existing Q5_K path; main work is:
1. Reading Q4_K_M block format (32 weights/block, super-blocks of 8)
2. Dequant routine: 4-bit + super-block scale + per-block scale
3. Roundtrip test

### Acceptance

1. Convert a real Q4_K_M GGUF (any HF model, e.g. Qwen3-0.6B-Q4_K_M)
2. Sample from it via `tinygpt sample` and confirm reasonable output
3. File size matches expected ~4.3 bits/weight average
4. Sha256 round-trip identical for the dequantized weights

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPTModel/GGUFReader.swift` — Q4_K_M block |
| **modify** | `native-mac/Sources/TinyGPT/GGUFExtract.swift` — flag |
| **don't touch** | Other quants (Q5_K etc.); already shipped |

## Estimated effort

**~1-2 days.** Algorithm well-documented in llama.cpp source.

## Source

- llama.cpp Q4_K block format: `ggml-quants.c`
- Format reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
