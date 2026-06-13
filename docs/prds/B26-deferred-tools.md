---
name: B26 server-side deferred tools
status: scaffolding-shipped-2026-06-13 (BFCL parity gate pending)
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B26)
parent_learn: docs/learn/agent-context-hierarchy.md (Steal #3)
---

# PRD — `tinygpt serve --tool-mode deferred`

## Goal

Today, `tinygpt serve --tools <catalog.json>` injects every tool's full
JSON schema into the system prompt of every request via
`ServeToolsSpec.systemPrompt()` (`DynamicGrammar.swift`). For a catalog
of 20–100 tools that is hundreds of always-resident tokens that the
model pays for on tasks where it never uses any tool.

Ship a deferred-catalog mode in which the system prompt carries only a
one-line-per-tool index and a built-in `get_tool_info(name)` meta-tool.
The model fetches schemas on demand; serve answers the meta-tool itself
and re-prompts so the round-trip is transparent to OpenAI-compatible
clients.

## Why now

- Steal #3 from the Shortcut vertical-agents essay (see
  `docs/learn/agent-context-hierarchy.md`). The essay's "L1 bloat
  biases behavior" claim is concrete here: large catalogs degrade
  unhappy-path performance even when the answer needs no tool at all,
  because resident schema tokens still occupy attention.
- KV prompt-cache (`--prompt-cache-dir`) already amortizes the
  *cost* of a fat L1 across requests. It does NOT rescue *accuracy* —
  attention still reads those tokens — so caching is not a substitute.
- Pace's first cut has only 12 actions, so the win there is small.
  The win compounds at larger catalogs (App Intents, an MCP-style
  generic tool surface, multi-domain specialists).

## Scope — in

- `ServeToolsSpec.compactSystemPrompt()` — emits `name — first-line-of-description`
  per tool plus the `get_tool_info` contract. No schemas.
- `ServeToolsSpec.compactGrammarSpec()` / `compactOutputSchemaJSON()` —
  same envelope as `grammarSpec()`, but the `verb` enum is extended
  with `get_tool_info`.
- `ServeToolsSpec.toolInfo(name:)` — JSON schema lookup for one tool;
  returns nil for unknown names so the caller can emit an error sentinel
  that the model can react to.
- `tinygpt serve --tool-mode {full,deferred}` CLI flag (default
  `full` → exact byte-for-byte parity with today's behavior).
- `/v1/chat/completions` (non-streaming): after `generate()` returns,
  parse the model's JSON; if `verb == "get_tool_info"`, look up the
  schema, append a synthetic `assistant` turn + a `tool` turn carrying
  the schema, and re-call `generate()`. Cap at 3 hops.
- `Server.parseGetToolInfoCall(_:)` static helper — used by the
  interception loop and unit tests.
- Unit tests in `Tests/TinyGPTServeTests/DeferredToolsTests.swift`.

## Scope — out (deliberately deferred)

- **Streaming `/v1/chat/completions` + `/v1/completions` + Ollama
  (`/api/chat`, `/api/generate`):** when `--tool-mode deferred` is on
  *and* `stream: true` is requested, the model is free to emit
  `verb=get_tool_info` (the grammar allows it) but serve does NOT
  intercept — the response streams through unchanged. The OpenAI
  surface contract is that streaming gives token-level deltas;
  buffering the whole response, parsing, re-prompting, and replaying
  a synthetic delta sequence is messier than it's worth for the first
  cut. Documented behavior, not a bug: clients that want deferred
  tools should set `stream: false`.
- **Session-level "schema already fetched" memoization:** each request
  starts cold. If you want to keep the schema resident across turns,
  the client controls that by carrying the `tool` message in its
  next request — which is the normal OpenAI tool-use protocol anyway.
- **Bigger-than-12 tool catalogs end-to-end test.** Pace's catalog is
  small; the framework is generic, but we don't ship a stress test of
  a 200-tool catalog under this PR.

## Why this design

The essay's argument is that good context discipline tiers capabilities
by frequency: hot in L1, warm in L2, cold in L3. Deferred mode is the
L2 layer for tool schemas. The compact index keeps "what tools exist"
in L1 (cheap, always needed) and pushes "what each tool's arguments
look like" to L2 (expensive, sometimes needed).

The decision to keep serve-side interception (rather than expose a
`/v1/tool-info` endpoint that the client calls) was load-bearing:

- The PLAN.md entry explicitly says "any OpenAI-compatible client gets
  it for free since the round-trip is an ordinary tool call." A
  client-side endpoint would require modifying every consumer; serve-side
  interception lets `lm-eval-harness`, `bfcl_eval`, and Continue.dev
  benefit without touching their code.
- Non-streaming chat completions is by far the dominant code path for
  the eval harnesses (BFCL, τ-bench, lm-eval-harness's
  `local-chat-completions`), so the surface-area concession is small.

## BFCL parity ship gate

Deferred mode is OFF by default until BFCL says it doesn't regress.
The gate (not run by this PR; user runs against a loaded model):

```
# baseline
tinygpt serve <model> --tools pace-tools.json &
tinygpt eval-bfcl <model> --out /tmp/bfcl-full.jsonl
pkill -f "tinygpt serve"

# deferred
tinygpt serve <model> --tools pace-tools.json --tool-mode deferred &
tinygpt eval-bfcl <model> --out /tmp/bfcl-deferred.jsonl
pkill -f "tinygpt serve"

tinygpt eval-compare /tmp/bfcl-full.jsonl /tmp/bfcl-deferred.jsonl --by model
```

**Accept** if the deferred BFCL average is within ±2pp of the full
average across the 10 BFCL categories, AND the average number of
`get_tool_info` round-trips per BFCL sample is ≤2.

**Reject** if either condition fails. The likely failure mode is the
model under-using `get_tool_info` because the index entry was misread
as sufficient; mitigations to try in order:

1. Tighten the index format (e.g. include argument names, not just
   purpose).
2. Lower the temperature on the first generation pass.
3. Inject a "you may need to call get_tool_info first" reminder when
   the model emits a `verb=<tool>` without having fetched its schema
   in the current session.

Update `docs/PLAN.md` B26 status with the BFCL delta + decision once
the gate runs.

## Files

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTServe/DynamicGrammar.swift` | `ServeToolMode` enum + `compactSystemPrompt()` + `compactGrammarSpec()` + `compactOutputSchemaJSON()` + `toolInfo(name:)` |
| `native-mac/Sources/TinyGPTServe/Serve.swift` | `--tool-mode` parsing, plumb through `Server.boot`, select compact prompt/grammar in deferred mode, post-`generate()` interception loop in `handleChatCompletions` non-streaming branch, `parseGetToolInfoCall(_:)` static helper |
| `native-mac/Tests/TinyGPTServeTests/DeferredToolsTests.swift` | 5 unit tests (no model needed): `compactSystemPrompt` strips schemas, `compactGrammarSpec` adds `get_tool_info` to the verb enum, `toolInfo` resolves known/unknown names, `parseGetToolInfoCall` recognizes the canonical shape and rejects garbage |
| `docs/PLAN.md` | B26 entry already filed; status update after BFCL gate runs |
| `docs/learn/agent-context-hierarchy.md` | Already linked as the parent learn doc (Steal #3) |

## Out of scope but worth noting later

- `get_tool_info(name)` is the simplest meta-tool. A larger catalog
  invites richer probes: `list_tools(filter=…)` for fuzzy search,
  `tool_examples(name)` for in-context one-shot priming, or a
  pre-grouped `list_tool_groups()` for hierarchical browsing. Don't
  add them until a real catalog needs them.
- The compact-mode index could itself be tiered: top-K most-frequent
  tools get a slightly fatter entry (one-line arg names too), rare
  tools stay name-only. Build only if measurement shows a hot/cold
  split worth the complexity.
