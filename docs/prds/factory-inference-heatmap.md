---
name: Inference heatmap — per-request latency trace for LLM calls
status: shipped-v1-2026-06-08-nonstreaming-chat
owner: unassigned
created: 2026-06-08
priority: P0 — required before claiming any Pace fast-router latency number
unblocks: factory-pace-fast-router-100ms.md, Pace in-app planner integration
---

# PRD — Inference heatmap

## Ship note — 2026-06-08

Implemented v1 for non-streaming `/v1/chat/completions`:

- `tinygpt serve --trace-infer --trace-dir <dir>` writes one JSON trace per
  request
- traces include route, model, total time, token counts, prompt-cache hit/miss,
  aggregate spans, and per-generated-token model/constraint/decode timings
- `tinygpt infer-heatmap <trace.json>` renders a terminal heatmap
- `tinygpt infer-heatmap <trace.json> --html <out.html>` writes a static HTML
  report
- the Mac app now has a `Trace` workspace that opens a trace JSON and renders
  the same heatmap breakdown

Measured against the current Pace v6.1 JSON-schema path:

```text
total 3739.5ms · prompt 657 tok · generated 35 tok · cache hit

tokens.constraint          2539.0ms
constraint_init             228.5ms
prompt_cache_load           116.8ms
tokens.model                 18.2ms
prefill_tail                  1.4ms
```

This confirms the 100ms route should not try to optimize full JSON generation
first. The model forward itself was not the dominant cost; general JSON Schema
masking was.

Known v1 limits:

- non-streaming chat path only
- default traces omit raw prompt/output text
- no trace sampling controls yet
- HTML renderer is intentionally static/minimal

## Problem

We need the same visibility for model inference that backend engineers expect
from an HTTP trace:

- network time
- JSON parse time
- tokenization time
- prompt-cache time
- prefill time
- per-token decode time
- constrained decoding / grammar mask time
- output parsing time
- response write time

The current Pace v6.1 adapter is correct on the narrow fm-fixture gate
but is not fast enough for the intended use. A quick measurement of the
current schema-constrained `tinygpt serve` path showed steady-state calls at
roughly **1.8-3.1 seconds**, even with prompt cache enabled. That number is
not actionable until we can see which span dominates each request.

## Goal

Add a first-class inference heatmap tool that records per-request spans and
renders them as both machine-readable JSON and a human-readable heatmap.

The tool must answer:

1. Was this slow because of prompt prefill, generation, grammar masking,
   tokenization, cache load, or HTTP overhead?
2. How many generated tokens did the request produce?
3. How much time did each generated token spend in model forward vs
   constraint masking?
4. Did prompt cache hit or miss?
5. Is the request even eligible for a sub-100ms path?

## Non-goal

This PRD does not optimize inference. It makes the bottleneck visible. The
100ms router work lives in `factory-pace-fast-router-100ms.md`.

## User story

As the owner, when a local model call feels slow, I want a heatmap that shows
where the time went so we stop guessing about whether the blocker is memory,
model compute, grammar masking, tokenization, or app/network overhead.

## Output shape

### JSON trace

```json
{
  "request_id": "pace-20260608-001",
  "model": "Qwen3-0.6B + pace-v6_1-fixture-system-300.lora",
  "route": "serve.chat.completions",
  "total_ms": 2580.0,
  "prompt_tokens": 812,
  "generated_tokens": 48,
  "cache": {
    "enabled": true,
    "hit": true,
    "prefix_tokens": 714,
    "load_ms": 38.2
  },
  "spans": [
    {"name": "http_read", "start_ms": 0.0, "duration_ms": 0.8},
    {"name": "json_parse", "start_ms": 0.8, "duration_ms": 0.4},
    {"name": "tokenize", "start_ms": 1.2, "duration_ms": 3.1},
    {"name": "prompt_cache_load", "start_ms": 4.3, "duration_ms": 38.2},
    {"name": "prefill_tail", "start_ms": 42.5, "duration_ms": 180.0},
    {"name": "decode_total", "start_ms": 222.5, "duration_ms": 2310.0},
    {"name": "output_parse", "start_ms": 2532.5, "duration_ms": 0.6}
  ],
  "tokens": [
    {"i": 0, "model_ms": 12.4, "constraint_ms": 41.0, "decode_ms": 0.2},
    {"i": 1, "model_ms": 12.1, "constraint_ms": 38.8, "decode_ms": 0.1}
  ]
}
```

### CLI heatmap

```text
total 2580.0ms · prompt 812 tok · generated 48 tok · cache hit

http/json/tokenize       4.3ms   ▏
prompt cache load       38.2ms   █
prefill tail           180.0ms   █████
decode model           580.0ms   ███████████████
decode constraint     1720.0ms   ███████████████████████████████████████████
output parse             0.6ms   ▏

slowest token spans:
  tok 04  model 12.8ms  constraint 48.4ms
  tok 17  model 12.5ms  constraint 46.9ms
  tok 31  model 12.1ms  constraint 45.8ms
```

### HTML heatmap

A single static HTML file is enough for v1:

- waterfall bars by span
- stacked aggregate chart by category
- per-token strip showing model vs constraint vs decode
- red/yellow/green thresholds for the 100ms target

No server UI is required for v1.

## Instrumentation points

Add lightweight span timers around:

| Span | File / area |
|---|---|
| `http_read` | `TinyGPTServe/Serve.swift` request read path |
| `json_parse` | OpenAI request JSON parse |
| `render_prompt` | Chat/completion prompt render |
| `tokenize_prompt` | tokenizer encode |
| `prompt_cache_lookup` | cache key and path check |
| `prompt_cache_load` | `KVCache.load` |
| `prompt_cache_save` | `KVCache.saveToDisk` |
| `prefill_full` | uncached `model.forwardCached(prompt)` |
| `prefill_tail` | cached-prefix remaining-token forward |
| `constraint_init` | `ServeConstraint` / `ServeTokenMasker` construction |
| `constraint_mask` | `ServeTokenMasker.mask(for:)` |
| `decode_forward` | per-token `model.forwardCached([token])` |
| `tokenizer_decode` | generated-token decode |
| `stop_check` | stop string / EOS checks |
| `response_write` | HTTP/SSE write |

## CLI/API design

### Serve flags

```bash
tinygpt serve ... \
  --trace-infer \
  --trace-dir /tmp/tinygpt-traces
```

Behavior:

- disabled by default
- when enabled, write one JSON trace per request
- optional `--trace-sample-rate` can come later
- never log request text unless `--trace-include-text` is explicitly set

### Standalone renderer

```bash
tinygpt infer-heatmap /tmp/tinygpt-traces/request.json \
  --html /tmp/request.html
```

Also support:

```bash
tinygpt infer-heatmap /tmp/tinygpt-traces/*.json --summary
```

## Privacy / safety

Default trace files must not include raw user text, raw screen labels, or
generated content. They may include token counts, span names, durations, cache
hit/miss, and grammar kind.

For local debugging, `--trace-include-text` can include redacted prompt/output
snippets, but it must be opt-in.

## Acceptance

1. Run `tinygpt serve` with `--trace-infer --trace-dir /tmp/tinygpt-traces`.
2. Send one Pace v6.1 request.
3. A trace JSON file is written with total request time, span times, token
   counts, cache hit/miss, and per-token model/constraint timing.
4. `tinygpt infer-heatmap <trace.json>` prints a terminal heatmap.
5. `tinygpt infer-heatmap <trace.json> --html /tmp/trace.html` writes a
   browser-openable static report.
6. Trace overhead is less than 5% for a normal request with tracing enabled.
7. Tracing disabled leaves serve behavior and latency unchanged.

## Implementation plan

### M1 — Trace data model

- Add `InferenceTrace`, `InferenceSpan`, `TokenTrace` structs.
- Use monotonic time, not wall-clock `Date`, for durations.
- Add a no-op tracer implementation so instrumentation calls are cheap when
  disabled.

### M2 — Serve instrumentation

- Thread a request-local tracer through chat/completion generation.
- Record aggregate spans first.
- Add per-token spans only inside decode loop.
- Record cache hit/miss and prompt/generated token counts.

### M3 — CLI renderer

- Add `tinygpt infer-heatmap`.
- Read one or more trace JSON files.
- Print terminal bars and summary stats.
- Emit a static HTML report.

### M4 — Pace benchmark recipe

- Add a script or doc recipe that runs the three representative Pace calls:
  click, key, and QA/escalate.
- Record baseline current path numbers.
- Use this before and after the 100ms router work.

## Files likely involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTServe/Serve.swift` | request/decode instrumentation |
| `native-mac/Sources/TinyGPT/InferHeatmap.swift` | new CLI renderer |
| `native-mac/Sources/TinyGPT/TinyGPT.swift` | dispatch `infer-heatmap` |
| `native-mac/Sources/TinyGPTModel/InferenceTrace.swift` | trace structs/helpers |
| `docs/prds/factory-pace-fast-router-100ms.md` | consumer PRD |

## Why this comes before optimization

We measured the current path at seconds, not milliseconds. The suspected
bottleneck is JSON Schema constrained decoding, but the heatmap is how we
prove that and keep ourselves honest after each optimization. The target is
not "faster"; the target is **under 100ms for the first routing decision**.
