---
name: tinygpt serve --grammar — constrained decoding in HTTP path
status: shipped-v0-2026-06-07-json-schema-plus-pace-gbnf-subset
owner: unassigned (parallel-agent task — Swift + MLX)
created: 2026-06-08
priority: P0 — locks structured-output specialists to valid format always
---

# PRD — `tinygpt serve --grammar <gbnf>`

## 2026-06-07 ship note

`tinygpt serve` now accepts `--grammar <path>` and per-request
`"grammar": "<inline grammar>"` on OpenAI chat/completion requests.

Shipped:
- server-level grammar loaded at boot
- per-request grammar override
- grammar-aware sampler wired into both non-streaming and SSE generation
- JSON Schema support via the existing `JSONSchemaFSM`
- Pace tool-tag GBNF subset support for `grammars/pace-tool-tags.gbnf`

The full llama.cpp GBNF grammar language is not implemented in this pass.
Unknown GBNF now fails loudly instead of silently running unconstrained.

## Goal

Add a `--grammar <path.gbnf>` flag to `tinygpt serve` that constrains
generation to a GBNF (llama.cpp-style) grammar. Every output token is
filtered through a per-position grammar mask so the response is
guaranteed to match the grammar — even if the underlying LoRA wobbles.

Sample CLI already does this via `--json-schema` / GBNF infra
(`ConstrainedGen.swift` + `JSONSchemaFSM.swift` + `LogitsMasker.swift`).
Serve needs the same wiring inside its decode loop.

## Why P0

Pace v3 LoRA emits production `{spokenText, pointAtElementId, clickElementId}`
JSON. With grammar enforcement, this output is **guaranteed valid JSON
matching the schema**. Without it, occasional model wobbles produce
malformed JSON that Pace's parser chokes on → broken specialist
even though the LoRA learned the format ≥98% of the time.

Locks the v3 specialist's output format permanently regardless of
training quality variance.

## Scope — in

### 1. CLI flag

```bash
tinygpt serve <hf-dir> --lora <lora> --grammar grammars/pace-tool-tags.gbnf --port 8765
```

Optional. When absent, serve behaves as today.

### 2. Wiring in Serve.swift

Both decode paths (`generate` non-streaming and `generateStreaming`)
need a grammar-aware sampler. The pattern from `Sample.swift`:

```swift
// At request time, build:
let fsm = JSONSchemaFSM(...)  // or GBNF FSM
let masker = LogitsMasker(vocab: ..., fsm: fsm)

// In decode loop, BEFORE argMax/categorical:
let masked = masker.maskLogits(logits)
let nextId = argMax(masked, axis: -1)
masker.advance(tokenId: id)  // FSM state forward
```

Pass the loaded grammar through `Server` init + into the per-request
`generate` / `generateStreaming` calls.

### 3. Per-request override

Honor an optional `grammar` field in the chat-completion request body
so clients can override the server-level default per call:

```json
{ "model": "tinygpt", "messages": [...], "grammar": "<inline-gbnf>" }
```

Fallback hierarchy: request `grammar` field > server `--grammar` flag > no grammar.

## Scope — out

- JSON Schema in addition to GBNF (sample already handles; defer for v2)
- Streaming-token correctness in the FSM (advance() on each delta)
- Per-conversation grammar state (each request is fresh)

## Acceptance criteria

1. Smoke: `tinygpt serve <hf> --lora pace-planner-v3.lora --grammar grammars/pace-tool-tags.gbnf --port 8765` succeeds
2. POST a chat completion → response body content is valid against the grammar (parseable; matches expected shape)
3. Without grammar: existing behavior unchanged (regression)
4. Build clean; existing serve tests pass

## Files involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTServe/Serve.swift` | Parse `--grammar` flag; thread through Server; wire FSM into decode |
| Use existing `ConstrainedGen.swift` / `JSONSchemaFSM.swift` / `LogitsMasker.swift` from TinyGPTModel | (no new code; reference existing) |
| Reference `Sample.swift` for the exact wiring pattern (line 264-330 area) | (no edit; just look) |

## Reference

- llama.cpp GBNF format: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
- TinyGPT's existing JSON Schema FSM is already correct for token-level masking
- Sample.swift `--json-schema` flag already proves the FSM works end-to-end

## Estimated effort

~2-3 hours focused work. Mostly wiring; the FSM + masker primitives
exist and are correct.
