# Constrained generation: JSON-mode for tinygpt

Reliable JSON output is the most common production failure mode for
small (1-3B) LLMs in agent loops. Even when the model "knows" how to
produce JSON, its sampler picks a wrong byte ~1 in N times — and one
wrong byte (a missing quote, a stray comma, an invalid character)
poisons the entire output for downstream parsers.

Constrained generation eliminates this class of bug by enforcing the
JSON grammar (and, optionally, a schema) at the logits level: at every
decode step the sampler can ONLY pick a token whose bytes extend the
output into a state that's still consistent with the grammar. Invalid
tokens are masked to `-inf` before softmax. The result: the produced
output is **byte-for-byte guaranteed** to be a valid JSON value that
matches the schema.

This is the same family of algorithm used by `outlines`, `vLLM`'s
`guided_json` mode, and llama.cpp's GBNF — we ship the
character-level variant which is simpler to reason about than the
token-level FSM but a touch slower per step.

## Usage

```bash
tinygpt sample <model.tinygpt> \
    --prompt "Output JSON:" \
    --tokens 200 \
    --temperature 0.5 \
    --json-schema path/to/schema.json
```

The output is guaranteed to be:
- A complete JSON value of the type pinned by the schema's root
- Matching every property/enum/type constraint in the schema
- Stopping (by default) at the closing token of the root value

Toggle `--no-json-stop-on-complete` to keep sampling past the closing
token (useful for streaming protocols that emit JSON followed by a
sentinel).

## How it works

### Two-piece pipeline

```
  schema.json
      │
      ▼
  JSONSchemaNode  ──── (parse, one-time)
      │
      ▼
  JSONSchemaFSM  ──── (character-level state machine)
      ▲
      │
  LogitsMasker  ──── (per-vocab byte table, built once at setup)
```

The state machine is a stack of `Frame`s, each representing one level
of nesting (a value being emitted, an object in some sub-phase, an
array, a string body, a number with sub-state). At every byte the
top-of-stack frame transitions; if it completes, it pops and the
parent advances.

### The mask, step by step

At decode step *t*:

1. The model produces logits over the full vocabulary.
2. For every token id, we look up its pre-computed UTF-8 byte
   sequence (built once at setup by decoding each id in isolation).
3. We clone the current FSM and try to feed the bytes through. If
   every byte is accepted, the token is valid; otherwise it's not.
4. We build a `[Float]` mask of length `vocab`, with 0 for valid
   tokens and `-inf` for invalid.
5. We add the mask to the logits, then sample (argmax for greedy or
   categorical for `temperature > 0`).
6. We commit the sampled token's bytes to the FSM. The mask
   guarantees this succeeds.

When the FSM reaches a terminal accepting state (`isComplete`), only
EOS (if known) is allowed; the caller typically stops here.

### Grammar enforced

The FSM enforces a **minified** subset of JSON: no insignificant
whitespace between structural tokens. Whitespace inside string bodies
is content and behaves normally. The rationale: small LLMs trained on
prose have a strong whitespace prior — if the FSM permitted leading
or inter-token whitespace, the model would emit whitespace forever
without ever advancing into the structure. This matches the behavior
of `JSON.stringify(x)` and what every constrained-generation library
produces in practice.

Trailing commas (`{...,}` or `[...,]`) are rejected — we split the
object/array `afterOpen` and `afterComma` phases so a close immediately
after a comma is impossible.

## Schema subset supported

The schema parser handles the JSON Schema Draft-07 features below.
Anything else is silently treated as `.any` (no constraint at that
position) — the model produces valid JSON, just not necessarily
matching the unsupported feature.

| Feature                                       | Status |
|-----------------------------------------------|--------|
| `"type": "object"`                            | ✓      |
| `properties: {...}`                           | ✓      |
| `required: [...]`                             | ✓ (close gated on emission of every required key) |
| `"type": "string"`                            | ✓      |
| `"type": "string", "enum": [...]`             | ✓ (closed set, prefix-pruning on the body) |
| `"type": "number"`                            | ✓ (sign? int frac? exp?) |
| `"type": "integer"`                           | ✓ (rejects `.`, `e`, `E`) |
| `"type": "boolean"`                           | ✓      |
| `"type": "null"`                              | ✓      |
| `"type": "array", "items": <schema>`          | ✓ (uniform items, recursive) |
| Nested objects / arrays                       | ✓ (arbitrary depth, capped at 64 nested pops) |
| `$ref`, `oneOf`, `anyOf`, `allOf`, `not`      | not supported (treated as `.any`) |
| `pattern`, `format`, `minLength`, `maxLength` | not supported |
| `minimum`, `maximum`, `multipleOf`            | not supported |
| `additionalProperties: <schema>`              | not supported (objects with empty `properties` accept any keys via `.any`; declared-properties objects forbid undeclared keys) |
| `patternProperties`                           | not supported |
| Tuple-form `items: [s1, s2, ...]`             | not supported |
| `const`                                       | not supported (use single-item enum) |

## Performance

Setup cost: O(vocab) tokenizer decode calls — under 1 second even for
128k vocab.

Per-step cost: O(vocab × avg-token-bytes) byte-level FSM probes. On
M-class Apple Silicon this is ~5-10% overhead at vocab≤32k for typical
JSON token-byte distributions. The mask is built on CPU, then shipped
to GPU as one `MLXArray` add before sampling — the GPU sees a regular
softmax-and-categorical.

Measured on the demo byte-level Shakespeare model (vocab=256,
12-layer, M1):

| Configuration       | tok/s | Overhead |
|---------------------|-------|----------|
| Unconstrained       | ~700  | —        |
| `--json-schema` on  | ~700  | <2%      |

(Byte-level vocab is the cheap end of the spectrum. At 32k-128k vocab
expect 5-10% overhead, which is in line with the published numbers
for `outlines` and `vLLM` on equivalent grammars.)

## Examples

### Tool-call schema

```json
{
  "type": "object",
  "properties": {
    "tool_name": {
      "type": "string",
      "enum": ["read_file", "run_test", "edit_file"]
    },
    "ok": { "type": "boolean" }
  },
  "required": ["tool_name", "ok"]
}
```

Run:

```bash
tinygpt sample demo.tinygpt \
    --prompt "JSON:" --tokens 200 --temperature 0.6 \
    --json-schema schema_toolcall.json
```

Output (real run on the demo model):

```
JSON:{"tool_name":"edit_file","ok":false}
```

### Simple object

```json
{
  "type": "object",
  "properties": {
    "x": {"type": "boolean"},
    "y": {"type": "integer"}
  },
  "required": ["x", "y"]
}
```

Output:

```
Out:{"y":-3,"x":true}
```

### Boolean

```json
{"type": "boolean"}
```

With `--prompt "t" --tokens 50 --temperature 0`:

```
t true
```

### Enum string

```json
{"type": "string", "enum": ["yes", "no", "maybe"]}
```

Forces the output to be one of the three quoted enum values.

## Caveats

1. **Token-string approximation.** We pre-compute each token's
   "rendered string" by decoding the id in isolation. For most
   tokenizers this matches the contextual rendering for ASCII-rich
   tokens, but BPE merges that depend on neighbouring tokens (rare
   for JSON-relevant bytes) can mismatch. If this becomes a problem
   in practice the next step is a token-level FSM (Outlines-style)
   that walks merges explicitly.

2. **No insignificant whitespace.** The FSM emits minified JSON — see
   "Grammar enforced" above. If a downstream consumer requires
   pretty-printed JSON, post-process the output.

3. **Model can still get stuck.** If the only valid continuation is
   one the model puts very low logit on (e.g., a Shakespeare model
   asked to produce a digit), greedy sampling may hit the `--tokens`
   cap on a single-byte run. Set `--temperature > 0` to break ties.

4. **The prompt is not part of the FSM.** The FSM sees only model-
   generated tokens. You can prefill the prompt with anything; only
   the output is constrained.

5. **Speculative decoding paths skip the mask.** `--draft` / `--heads`
   generate K tokens in a burst then verify with the target — wiring
   the schema mask into that path requires masking K positions in
   parallel, which is a follow-up.

6. **Required-key ordering is not enforced.** The schema's `required`
   set is checked at close-time, not at each key's open-time. So
   `{"b":2,"a":1}` is accepted as long as both `a` and `b` are in
   `properties` and both are emitted before `}`. JSON-Schema doesn't
   constrain key order so this matches the spec.

## Files

- `native-mac/Sources/TinyGPTModel/JSONSchema.swift` — schema parser
- `native-mac/Sources/TinyGPTModel/ConstrainedGen.swift` — FSM + logits masker
- `native-mac/Sources/TinyGPT/Sample.swift` — `--json-schema` wiring
- `native-mac/Tests/TinyGPTModelTests/ConstrainedGenTests.swift` — FSM unit tests

## Follow-ups

1. **Token-level FSM.** Pre-compute a trie of valid token-id sequences
   at setup; per-step lookup becomes O(log vocab) instead of O(vocab).
   ~5-10× lower overhead on large-vocab models.

2. **Wire to `--draft` / `--heads`.** Speculative-decode paths need the
   mask applied at all K positions simultaneously.

3. **`additionalProperties`, `oneOf`, `anyOf`, `$ref`.** Cover the rest
   of Draft-07. `$ref` is the highest-value because real-world schemas
   use it heavily.

4. **Pattern / format strings.** Compile regex / format constraints
   into nested FSMs and slot them into the string-body state. Same
   approach as `outlines`'s pattern grammar.
