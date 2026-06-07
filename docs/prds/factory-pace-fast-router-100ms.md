---
name: Pace fast router — under-100ms first tool call
status: shipped-m0-endpoint-2026-06-08
owner: unassigned
created: 2026-06-08
priority: P0 — current schema-constrained LLM serve path is too slow for Pace first-hop UX
depends-on: factory-inference-heatmap.md
unblocks: Pace local first-action routing
---

# PRD — Pace fast router under 100ms

## Problem

The Pace v6.1 planner is now correct on the narrow fm-fixture suite
(**19/19**), but the current serving path is not usable as a fast first hop.
Measured with prompt cache enabled and JSON Schema grammar enabled:

- click-like request: ~3.1s
- key shortcut request: ~1.8s
- QA request: ~2.6s
- steady p50: ~2.6s

The owner requirement is stricter: **route to the next thing in under
100ms**, or it is useless.

The current design cannot reliably hit that target because it performs
autoregressive full-JSON generation with general JSON Schema masking across
the full Qwen3 vocabulary.

## Goal

Build a separate Pace fast-router path that returns the next action or
escalation decision in **<100ms p95 warm** on the target Mac.

The router's job is not to be the whole assistant. Its job is to make the
first decision:

- click a visible target
- type text
- press a key
- scroll
- open an app
- answer a tiny fixed identity/simple response
- escalate to the larger planner/model

## Hard requirement

The fast path must not use full autoregressive JSON generation.

If it emits dozens of tokens through the general `JSONSchemaFSM`, it is the
wrong architecture for this PRD.

## Architecture decision

Use a classifier / scorer pipeline, not a generator.

### Current slow path

```text
system prompt + screen + user
  -> Qwen3-0.6B LoRA
  -> generate full JSON object token by token
  -> JSON Schema mask scans ~152k vocab candidates per token
  -> parse JSON
```

### Fast path

```text
screen labels + user
  -> cheap deterministic features
  -> tiny router/classifier
  -> optional target scorer over visible labels
  -> compact action object
```

Output is a host-side struct, not generated prose:

```json
{"verb":"click","target_id":1,"confidence":0.91}
```

or:

```json
{"verb":"escalate","reason":"knowledge_or_ambiguous","confidence":0.42}
```

## Latency budget

| Stage | Budget |
|---|---:|
| HTTP/app request parse | 2ms |
| screen/user normalization | 3ms |
| deterministic obvious-action pass | 1-5ms |
| router inference | 5-25ms |
| target scoring | 5-25ms |
| response encoding | 1ms |
| safety gates | 1-5ms |
| total p95 warm | <100ms |

This is plausible only if the model does **one classification pass** or a
small number of scorer passes. It is not plausible with token-by-token JSON
generation.

## Router output contract

```swift
struct PaceFastRoute {
    enum Verb: String {
        case click
        case type
        case key
        case scroll
        case openApp
        case answer
        case escalate
    }

    var verb: Verb
    var targetId: Int?          // visible element id, if applicable
    var text: String?           // type/answer payload
    var key: String?            // key shortcut
    var direction: String?      // scroll direction
    var appName: String?        // open app
    var confidence: Float
    var reason: String?
}
```

The app can convert this to the old v6 JSON shape or v7 tool-call shape
downstream. The router itself should stay compact.

## Routing policy

### Execute locally

Only execute without escalation when:

- verb confidence >= threshold
- target confidence >= threshold when target is required
- action is low risk
- no ambiguity among duplicate labels
- no destructive verb

### Escalate

Escalate to the larger planner/model when:

- knowledge answer is non-trivial
- target is ambiguous
- user asks for multi-step work
- user asks for destructive/high-risk action
- confidence is below threshold
- screen labels are poor/missing

This lets the fast router be aggressive on easy actions without pretending
to be a complete assistant.

## Implementation strategy

### M0 — Deterministic fast lane

Before training anything, add a deterministic fast lane for obvious patterns:

- `click/tap/open/press/select <label-ish text>` -> fuzzy match visible labels
- `press command s`, `cmd+s`, `save shortcut` -> `key(cmd+s)`
- `scroll down/up` -> `scroll(direction)`
- `type <text>` -> `type(text)`
- `open <known app>` -> `openApp(appName)`
- identity probes -> `answer("i'm pace")`

Expected latency: <5ms.

This is not a replacement for the router. It catches the highest-confidence
obvious cases and gives us a baseline.

### M1 — Train/use tiny router classifier

Reuse the existing tool-call extractor scaffold:

- `native-mac/Sources/TinyGPTModel/ToolRouterModel.swift`
- `tinygpt extractor-data`
- `tinygpt train-extractor`
- `tinygpt extract`

But train a Pace-specific label set:

```text
click
type
key
scroll
open_app
answer
escalate
```

Input should be compact:

```text
labels:
0 search button
1 save draft button
2 message input
user: click save
```

Do not include the full Pace system prompt.

### M2 — Target scorer

For actions that need a target, score visible labels separately.

Options ranked:

1. deterministic fuzzy scorer for v1
2. tiny cross-encoder classifier over `(user, label)` pairs
3. one-pass model that outputs verb + target index classes

M2 target: <25ms for up to 40 visible labels.

### M3 — Fast endpoint

Add a new endpoint to the existing serve mode:

```bash
tinygpt serve <model.tinygpt | hf-dir> --port 8765
```

The endpoint bypasses model inference and the serial inference queue. The
model path is still required today because `tinygpt serve` boots one shared
server process around a loaded model.

Endpoint:

```text
POST /v1/pace/route
```

Request:

```json
{
  "user": "click save",
  "free_text_mode": false,
  "elements": [
    {"id": 0, "role": "button", "x": 412, "y": 40, "label": "search button", "text": "Search"},
    {"id": 1, "role": "button", "x": 548, "y": 40, "label": "save draft button", "text": "Save Draft"}
  ]
}
```

Response:

```json
{
  "verb": "click",
  "target_id": 1,
  "target_label": "save draft button",
  "x": 548,
  "y": 40,
  "text": null,
  "key": null,
  "direction": null,
  "app_name": null,
  "confidence": 0.93,
  "latency_ms": 0.42,
  "reason": "deterministic_label_match",
  "fallback": false,
  "spoken_text": "opening the save draft button",
  "action_tags": []
}
```

### M4 — Fallback bridge

If response is `escalate`, call the current larger planner path:

- qwen3-30b teacher
- current Qwen3-0.6B JSON planner
- future v7 tools-in-prompt model

The fast router still wins because easy actions never touch the larger path.

## Data plan

Start from existing fm-fixtures plus synthetic variants:

```jsonl
{"user":"click save","labels":["search button","save draft button"],"verb":"click","target_id":1}
{"user":"press command s","labels":["editor pane"],"verb":"key","key":"cmd+s"}
{"user":"what is html","labels":["save button"],"verb":"escalate"}
{"user":"who are you","labels":[],"verb":"answer","text":"i'm pace"}
```

Build:

- 19 fm-fixtures -> gold seed
- generate 30-50 paraphrases per action class
- include ambiguous duplicate labels
- include missing target
- include destructive/high-risk examples routed to `escalate`
- include knowledge examples routed to `escalate`, not local answer,
  except fixed identity probes

## Acceptance

### Latency

Measured with `factory-inference-heatmap.md` tooling:

- deterministic obvious-action route: <10ms p95 warm
- router classifier route: <50ms p95 warm
- full fast-router endpoint including parse/encode: <100ms p95 warm
- no request in the easy-action suite >150ms

### Correctness

On the current fm-fixture suite:

- easy action subset: >=95%
- no destructive/high-risk action executes locally
- ambiguous target cases escalate or clarify
- knowledge questions escalate unless explicitly allowlisted

### Integration

- Pace can call `/v1/pace/route`
- Fast route returns within 100ms for obvious click/key/scroll/type/open
- Larger model fallback still works
- Logs include route reason and confidence

## Why not make Qwen3 faster?

We should still optimize Qwen3 serve, but that is not the path to <100ms
first-hop routing.

The current slow path does:

```text
generated_tokens × 151,936 vocab candidates × FSM checks
```

Even if model forward improves, full JSON generation and general grammar
masking remain the wrong shape for an under-100ms router. The fast path must
avoid generation.

## Files likely involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPT/PaceFastRoute.swift` | new router CLI/logic |
| `native-mac/Sources/TinyGPTServe/Serve.swift` | optional `/v1/pace/route` endpoint |
| `native-mac/Sources/TinyGPTModel/ToolRouterModel.swift` | reuse existing classifier |
| `scripts/pace-router-data.py` | build router training data from fixtures |
| `scripts/pace-router-eval.py` | latency/correctness eval for fast route |
| `docs/prds/factory-inference-heatmap.md` | tracing dependency |

## Open questions

1. Should the first app integration call a new endpoint (`/v1/pace/route`) or
   shell out to `tinygpt pace-route` for simpler iteration?
2. Should identity probes be handled locally or escalated with all QA?
3. What is the maximum visible element count Pace should send to the fast
   router?
4. What confidence threshold feels right for live use: 0.7, 0.8, or 0.9?
5. Should the deterministic lane ship before the trained router? The answer
   should probably be yes because it proves the app path and latency budget.

## Recommended first slice

Build M0 + heatmap first:

1. Add `tinygpt pace-route --fixtures ...` deterministic router.
2. Evaluate it on fm-fixtures and report latency.
3. Add `/v1/pace/route` only after CLI behavior is solid.
4. Train the tiny router only for cases deterministic rules should not own.

This gives Pace a usable under-100ms local first hop quickly while keeping the
model path honest and measurable.

## Implementation note — 2026-06-08 M0 standalone script

Added the first deterministic slice as a standalone Python evaluator:

```bash
python3 scripts/pace-fast-router-m0.py --repeat 200
```

Scope is intentionally narrow: it parses
`/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures/*.txt`, routes
obvious `click`, `type`, `key`, `scroll`, `open_app`, fixed tiny `answer`, and
`escalate` cases without calling TinyGPTServe or any model runtime, and reports
fixture pass/fail plus latency percentiles.

Initial local result:

- correctness: **19/19 fm-fixtures passed**
- latency: **p50 0.0022ms, p95 0.3545ms, max 0.7910ms** over 3,800 route calls
- compile check: `python3 -m py_compile scripts/pace-fast-router-m0.py`

## Implementation note — 2026-06-08 M0 serve endpoint

Added the deterministic router to TinyGPTServe:

- `native-mac/Sources/TinyGPTServe/PaceFastRouter.swift`
- `native-mac/Sources/TinyGPTServe/Serve.swift`
- endpoint: `POST /v1/pace/route`

Request accepts `elements` as either object arrays or fixture-style strings.
`free_text_mode` / `free_text` controls `spoken_text`: execution fixtures can
ask for `[CLICK:x,y]` / `[TYPE:text]` tags, while user-facing mode returns
plain text such as `opening the file menu`. Structured route fields are always
present, so the application should prefer `verb`, `target_id`, coordinates,
`text`, `key`, `direction`, `app_name`, `confidence`, and `reason`.

Local HTTP eval against all 19 fm-fixtures, 20 repeats per fixture:

- correctness: **19/19 fm-fixtures passed**
- server-side route latency: **p50 0.0328ms, p95 0.5109ms, max 0.6370ms**
- local HTTP end-to-end latency: **p50 0.1801ms, p95 0.6271ms, max 7.0527ms**
- samples: 380 endpoint calls
- build check: `swift build -c release --product tinygpt`

## Pause note — what M0 does and does not prove

M0 is not an LLM and should not be presented as one.

The current `/v1/pace/route` path is:

```text
HTTP parse
  -> JSON parse
  -> deterministic intent rules
  -> fuzzy label scoring over visible elements
  -> JSON response
```

It does **not** run Qwen, LoRA, MLX inference, JSON Schema constrained
generation, or any learned router. The `tinygpt serve` process still loads a
model only because the server currently requires a model at boot. The route
endpoint bypasses the inference queue.

This means the sub-ms latency is expected. It is useful, but it is not evidence
that a specialist model can route under 100ms.

### What M0 proves

- Obvious first-hop actions do not need an LLM.
- The old full-JSON schema-constrained generation path is the wrong default
  for every next-tool-call decision.
- The compact route contract is enough to represent click/type/key/scroll/open
  plus escalation.
- A cheap local layer can remove boring traffic from the model path.

### What M0 does not prove

- It does not prove generalization.
- It can be overfit to the 19 fm-fixtures.
- It does not understand UI intent beyond hand-coded patterns.
- It does not replace a specialist model.
- It does not validate the 100ms target for learned routing.

### Product architecture decision

If M0 stays deterministic, the daily-driver implementation should probably live
inside `clickyLocal`, not behind a local HTTP endpoint.

Preferred product shape:

```text
clickyLocal user event
  -> in-process M0 fast router
     -> obvious safe action: execute immediately
     -> uncertain / risky / novel: escalate upward
```

The endpoint remains valuable as a reference implementation and eval harness,
but it is not the ideal production boundary. Removing HTTP avoids socket churn,
server lifecycle dependency, JSON round-trips, and loading a model for a route
that does not use one.

Rust or Go is not the main win here. The router's compute is already tiny; the
bigger win is removing the process/HTTP boundary and calling the router
in-process from the Mac app.

### Cascade framing

The intended design is a confidence cascade:

```text
M0 deterministic router
  -> handles obvious safe actions
  -> escalates gray area

M1 tiny learned router
  -> handles paraphrases, noisy speech, less obvious labels
  -> escalates hard cases

M2 specialist planner
  -> handles multi-step UI intent
  -> escalates broad reasoning

M3 large model
  -> handles real ambiguity, world knowledge, and fallback reasoning
```

Each layer must be able to say "I do not know." Wrong execution is worse than
escalation.

The useful product metric is not just accuracy. Track:

- resolved at each layer
- p50/p95/p99 latency at each layer
- wrong-execute rate
- correct-escalate rate
- whether the next layer fixes what the lower layer could not handle

### Next eval before calling this real

Do not judge M1/M2 against only the 19 known fixtures. Add:

- hidden holdout fixtures with new wording, labels, coordinates, and screens
- adversarial UI cases with duplicates, misleading labels, disabled targets,
  destructive buttons, and nearby alternatives
- noisy ASR transcripts with filler, corrections, partial commands, and
  changed intent
- generalization split by app/screen family
- fallback-weighted scoring: `correct execute + correct escalate - wrong execute`

The next useful question is:

```text
Can a tiny learned router safely catch cases M0 escalates, under 100ms p95?
```

If it cannot beat M0 on hidden messy cases while keeping wrong-execute low, it
should remain behind the cascade rather than becoming the default router.
