---
name: tinygpt synthesize — labeled data from any teacher endpoint
status: shipped-2026-06-07
owner: unassigned (parallel-agent task — CLI + Swift glue)
created: 2026-06-07
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (factory-completeness, the missing primitive)
priority: P0 — blocks every distillation arc
---

# PRD — `tinygpt synthesize`

## Goal

Add a `tinygpt synthesize` subcommand that takes a teacher endpoint
(any OpenAI-compatible server: local LM Studio, our own `tinygpt
serve`, free-AI providers, cloud APIs), a pool of input prompts, and a
schema/grammar, and emits a JSONL of `{input, output}` labels suitable
for distillation training.

**This is the missing factory primitive.** Every distillation arc today
requires hand-rolled Python that spawns the teacher + parses output.
Codifying it as a CLI command makes the factory's first step a single
command.

## Why P0

User goal: "build the factory and ensure that the factory is as fast as
it could be." (2026-06-07 strategy clarification.)

The factory has shape:
```
data → SYNTHESIZE → distill → eval → ship
       ^^^^^^^^^^^^
       missing primitive
```

Every primitive downstream of synthesize works today; synthesize itself
is missing. Filling this gap unblocks all specialist distillation arcs
(Pace planner, KB embedder, etc.).

## Scope — in

### 1. CLI surface

```
tinygpt synthesize \
    --teacher http://localhost:1234/v1 \              # any OpenAI-compat endpoint
    --teacher-model qwen3-30b-a3b \                   # model id served at endpoint
    --inputs inputs.jsonl \                           # one prompt per row
    --input-field prompt \                            # which field to read
    --system "You are a Pace tool dispatcher..." \    # system prompt template (optional)
    --schema schema.json \                            # output JSON schema (optional)
    --grammar grammar.gbnf \                          # OR GBNF grammar (optional)
    --max-tokens 256 \
    --temperature 0.0 \                               # deterministic for training
    --parallel 4 \                                    # concurrent requests
    --rate-limit 30 \                                 # per second (free-tier safe)
    --out labeled.jsonl                               # {input, output, _meta} rows
```

### 2. OpenAI-compat client

Uses the canonical OpenAI client shape: `/v1/chat/completions` POST with
`messages: [...]`. Supports:
- Local servers (LM Studio at :1234, our `tinygpt serve`, Ollama at :11434)
- Cloud (api.openai.com, api.anthropic.com via proxy, api.deepseek.com,
  api.groq.com)
- Reads `OPENAI_API_KEY` or `TEACHER_API_KEY` env var

### 3. Parallel + rate-limited

- `--parallel N`: N concurrent in-flight requests
- `--rate-limit N`: per-second cap (token-bucket; free tiers + local
  servers benefit)
- Resumes from partial output JSONL on rerun (skips rows already in `--out`)

### 4. Schema / grammar enforcement

Two paths for structured output:
- **`--schema schema.json`**: append "respond with JSON matching this
  schema" to the prompt. Validate output is parseable JSON matching the
  schema; drop rows that fail.
- **`--grammar grammar.gbnf`**: if the teacher endpoint supports
  llama.cpp's GBNF (LM Studio does), pass it in the request. Server-side
  constrained decoding ensures every output is valid.

### 5. Output JSONL row shape

```json
{
  "input": "<original prompt field>",
  "output": "<teacher's response>",
  "_meta": {
    "teacher_model": "qwen3-30b-a3b",
    "teacher_endpoint": "http://localhost:1234/v1",
    "timestamp": "2026-06-07T15:23:01Z",
    "tokens_used": 234,
    "latency_ms": 825
  }
}
```

The `_meta` field lets downstream tools filter (e.g., drop slow rows)
or compare distillation quality across teachers.

### 6. Progress + ETA + log

While running, emit to stderr:
```
[synthesize] 1234 / 10000 (12.3%) · 24 req/s · ETA 5m23s · 8 retries · 2 schema-fails
```

## Scope — out (v2)

- Multi-teacher consensus (label rows with N teachers, keep agreed
  outputs)
- Active learning (pick which inputs to label next based on student
  uncertainty)
- Streaming output writes mid-completion (write only on completion;
  simpler + safer for resume)
- Fine-grained retry policies per error class

## Acceptance criteria

1. Smoke against running LM Studio:
   ```
   tinygpt synthesize \
       --teacher http://localhost:1234/v1 \
       --teacher-model qwen3-30b-a3b \
       --inputs sample.jsonl \
       --max-tokens 100 --temperature 0 --parallel 4 \
       --out labeled.jsonl
   ```
   Produces N labeled rows matching N input rows.

2. Smoke against `tinygpt serve` (our own endpoint):
   - Spawn `tinygpt serve data/gallery/code.tinygpt --port 9999`
   - Run `tinygpt synthesize --teacher http://127.0.0.1:9999/v1 ...`
   - Confirms our own endpoint can be a teacher

3. Resume works: kill mid-run with Ctrl-C; rerun with same args;
   continues from partial output without duplicating rows.

4. Rate limiter works: `--rate-limit 5` produces ≤5 req/s observed.

5. Schema validation: `--schema schema.json` with strict JSON output;
   verify malformed teacher responses are dropped + logged.

6. Build passes. Integrates with existing `tinygpt distill` (the JSONL
   output is direct input to `--data` of distill).

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/Synthesize.swift` |
| **modify** | `native-mac/Sources/TinyGPT/TinyGPT.swift` — dispatch entry for `synthesize` |
| **don't touch** | Distill.swift, eval pipeline, app code, `docs/PLAN.md`, `HANDOFF.md` |

## Inputs the agent has

| Resource | Location |
|---|---|
| OpenAI-compat client patterns | `Sources/TinyGPTApp/HFBrowserController.swift` (URLSession + JSON), `Sources/TinyGPTServe/Serve.swift` (server-side shape) |
| Existing async + parallel patterns in Swift | `Sources/TinyGPT/Train.swift` (Task / async / await) |
| JSON Schema validation | use Swift `JSONSerialization` + manual walk (no third-party) |
| GBNF support test target | LM Studio (default port 1234) |
| Test fixture inputs | `~/.cache/tinygpt/datasets/hermes-fc.jsonl` (11K rows, system+user → tool call) |

## Estimated effort

**~2 days focused work.**

- 0.5 day: CLI parsing + arg validation
- 0.5 day: OpenAI-compat HTTP client with parallel + rate-limit
- 0.5 day: JSONL read/write + resume logic
- 0.5 day: schema validation + retry/drop logic

## Why this is the unblock

Once shipped:
- `tinygpt synthesize` + `tinygpt distill` is the complete distillation factory loop
- Pace planner specialist becomes a 3-command recipe
- KB embedder specialist becomes a 3-command recipe (with embedder
  variant of distill)
- Every future specialist arc reuses this primitive

PRIORITY OVER everything else in `docs/prds/`. Until this ships, the
distillation surface is incomplete.
