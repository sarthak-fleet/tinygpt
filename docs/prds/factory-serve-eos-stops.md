---
name: serve EOS stop tokens — stop on <|im_end|>, <|endoftext|>, etc.
status: shipped-2026-06-07
owner: unassigned (parallel-agent task — Swift)
created: 2026-06-08
priority: P1 — Pace v2/v3 LoRAs hit this; trailing garbage breaks JSON parsers
---

# PRD — serve stops at native EOS tokens

## 2026-06-07 ship note

`tinygpt serve` now detects EOS/chat-end token ids from HF tokenizer files,
adds common chat sentinels such as `<|im_end|>` when they tokenize to a single
id, and stops before emitting those tokens in both streaming and non-streaming
decode loops.

Shipped:
- tokenizer-config and tokenizer-json EOS detection
- `--no-eos-stop` opt-out
- request-body `stop_token_ids: [int]` additive override
- EOS checks in `generate` and `generateStreaming`

## Problem

Pace v2/v3 LoRAs (and most modern chat LoRAs) emit `<|im_end|>` at the
natural end of their response. `tinygpt serve` doesn't honor this — it
keeps generating until `max_tokens`, producing trailing garbage like:

```
{"spokenText":"hi", "pointAtElementId":-1, "clickElementId":-1}<|im_end|>
<|im_end|>
<|im_end|>
asdfasdf
```

This breaks Pace's JSON parser AND wastes ~hundreds of tokens of
inference cost per request.

## Scope — in

### 1. Read EOS tokens from tokenizer config

HF tokenizers expose `eos_token_id` (and sometimes a list of additional
EOS tokens via `added_tokens_decoder` for chat templates: `<|im_end|>`,
`<|endoftext|>`, `<|eot_id|>`, etc.).

Auto-detect on serve startup:
- Read `tokenizer_config.json` for `eos_token`
- For Qwen3 + similar: also `<|im_end|>` is the conversation end
- Build a Set<Int> of stop token ids

### 2. Wire into decode loop

In `Serve.swift` `generate` / `generateStreaming` (both rewritten in
the recent KV-cached refactor), after sampling each token:

```swift
let id = Int(nextId.item(Int32.self))
if eosTokenIds.contains(id) { return /* finished */ }
```

Apply to both paths. Preserve client-supplied `stop` strings as-is.

### 3. Override + opt-out

- Honor request body `stop_token_ids: [int]` to add more
- Honor `--no-eos-stop` CLI flag to disable (for testing / debugging)

## Acceptance

1. Pace v3 LoRA response stops at the first `<|im_end|>` token —
   response is clean JSON, no trailing garbage
2. SSE streaming variant also stops at EOS (not just emits the token
   then continues)
3. Regression check: bare Qwen3 + simple "hi" → still produces meaningful
   output, doesn't truncate prematurely
4. Build clean; serve tests pass

## Files involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTServe/Serve.swift` | Detect EOS at startup; check per-token in decode loop |
| `native-mac/Sources/TinyGPTModel/HFTokenizer.swift` (or similar) | Expose `eosTokenIds: Set<Int>` if not already |

## Estimated effort

**~1-2 hours.** Mostly tokenizer-config parsing + a one-line stop check
in each of the two decode paths.

## Why P1

Without this, every Pace LoRA response has trailing garbage. The grammar
work (`factory-serve-grammar.md`) doesn't fix this — grammar enforces
format, not stopping. Both are needed for Pace integration to work.
