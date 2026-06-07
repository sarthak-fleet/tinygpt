---
name: Planner v7 — tools-in-prompt (generalizable function calling)
status: implementation-v0-2026-06-07-long-sft-and-external-evals-pending
owner: unassigned (parallel-agent task — Swift training + serve + Python eval)
created: 2026-06-08
priority: P1 — unblocks Pace tool churn without retrain; foundation for any multi-app planner consumer
depends-on: v6 ANE arc landing (#263), BFCL eval shipping (#231)
authorized-by: maintainer 2026-06-08 (architectural direction confirmed — implementation gated on owner go-ahead after ANE arc completes)
---

# PRD — Planner v7: tools-in-prompt architecture

## Ship note — 2026-06-07

Owner approved the previously gated v7 work. The first implementation slice
landed:

- `tinygpt serve --tools <tools.json>` injects the tool catalog into
  the ChatML system prompt
- `--tools` also installs a default constrained output schema shaped as
  `{ "verb": <tool-name-enum>, "args": {}, "spoken_text": "" }`
- The larger tools system prompt is included in prompt-cache prefixing
- Added `grammars/pace-system-prompt-v7-tools.txt`
- Added data/eval scaffolding:
  - `scripts/v7-data/normalize-xlam.py`
  - `scripts/v7-data/build-pace-topup.py`
  - `scripts/v7-data/build-heldout.py`
  - `scripts/v7-eval/run-heldout.py`
  - `scripts/v7-eval/run-bfcl.py`
  - `scripts/v7-eval/run-tau-bench.py`
  - `scripts/v7-eval/heldout-tools.jsonl`
- Generated local Pace top-up data at
  `~/.cache/tinygpt/datasets/pace-v7-topup.jsonl`

Known v0 limitation: the current TinyGPT JSON-Schema FSM does not support
`oneOf` / conditional schemas, so v0 constrains the verb enum but does not
yet enforce per-verb argument schemas. That remains the next grammar
milestone before claiming full v7 acceptance. The 6-8 hour xLAM-scale SFT
and external BFCL / tau-bench runs are still pending.

## Why this PRD exists

v6's grammar hardcodes Pace's action set. Adding a new action type
(drag-and-drop, multi-select, keyboard shortcut sequences, novel app
integrations) requires re-training. That's manageable when Pace evolves
slowly, but the moment Pace gains 5+ new tool types in a sprint, or a
second TinyGPT consumer wants its own tool set, the v6 architecture
becomes a bottleneck.

v7 inverts the design: **tools live in the system prompt at inference
time, not in training data**. The model learns the *pattern* of "fill
this schema correctly," not the specific tool list. Adding tools = one
prompt line, no retrain. Matches OpenAI function calling, Anthropic
tool use, and MCP.

This PRD is a draft. The case for shipping v7 NOW vs deferring is
genuinely close (see "Cost-benefit" below). The owner should weigh
deferring until Pace tool churn becomes real pain.

## Goal

Train a Qwen3-0.6B (or xLAM-1B — see leverage choice below) specialist
that:

1. Takes a system prompt containing a list of available tool schemas
2. Takes a user intent (text or transcribed voice)
3. Emits a structured tool call: which tool, with which arguments
4. Generalizes to tools NOT seen during training, given only their
   schemas in the prompt
5. Drops into Pace via `tinygpt serve --tools <tools.json>` — and into
   any future TinyGPT consumer the same way

## Architectural shift — what changes from v6

| Aspect | v6 | v7 |
|---|---|---|
| Tool list | Hardcoded in training | Schema in system prompt at inference |
| Adding new tool | Retrain (~40 min) | Add one line to system prompt |
| Output shape | Fixed JSON schema (`pace-fm-label-response.schema.json`) | Parameterized JSON schema (verb + args) generated at request time |
| Grammar | Static GBNF | Dynamic GBNF generated from tool schemas per request |
| Base model | Qwen3-0.6B | xLAM-1B (already function-call-trained) — see leverage choice |
| Training data | 248 hand-anchored Pace examples | ~10K-60K diverse (intent, tools, call) triples from xLAM dataset + Pace-specific top-up |

## Leverage-first choices (per `[[feedback_leverage_first]]`)

**Base model**: **xLAM-1B-fc-r** (Salesforce, Apache 2.0). Already
trained on function calling at scale; we LoRA-fine-tune for Pace
specifics rather than teaching function-calling from scratch on bare
Qwen3-0.6B. Saves weeks of SFT and lifts the floor.

**Training data**: **xlam-function-calling-60k** (already on the queue
as task #227 — D1 pending). Diverse synthetic function calls. Use as
warmstart. Pace-specific top-up (~500 examples) for the long tail.

**Schema patterns**: **MCP tool specification** + **Anthropic tool
use** + **Apple App Intents**. Don't invent a new schema; align with
the three industry references. (App Intents study is its own PRD —
`factory-app-intents-taxonomy-study.md`.)

**App Intents study landed**: see
`docs/learn/app-intents-comparison.md`. Before v7 SFT lock-in, replace
the draft taxonomy below with the study's recommended set:
`query`, `perform`, `set`, `compose`, `clarify`, `say`,
`query_memory`, `wait`, `open`, `schedule`. Also add a shared
`EntityRef` schema, fold top-level `search` into `query(mode=search)`,
rename `navigate` to `open`, and split broad `act` examples into
`perform` and `set`.

## Action collapse plan — many actions, few verbs

The model should not learn one verb per surface action. Pace can execute a
large and growing action set — click, double click, type, key chord, scroll,
drag, multi-select, open app, open file, set value, search, summarize,
clarify, wait, schedule — but the planner should choose from a small stable
verb set and put the variability inside typed args.

The core rule:

```json
{
  "verb": "perform",
  "args": {
    "action": "click",
    "target": {"type": "ui_element", "label": "save draft button"}
  },
  "spoken_text": "opening save draft"
}
```

`click`, `type`, `key`, `scroll`, `drag`, `multi_select`, and similar UI
operations are **actions under `perform`**, not top-level verbs. This keeps
the model's classification problem small while keeping the executor
expressive.

Use specialized top-level verbs only when they imply a different executor,
latency model, safety policy, or product behavior:

| Top-level verb | Use when | Example |
|---|---|---|
| `perform` | Immediate UI/app action | `{"action":"click","target":EntityRef}` |
| `set` | Changing a named value | `{"target":"brightness","value":"50%"}` |
| `open` | Showing an app, URL, file, or view | `{"target":{"type":"app","label":"Safari"}}` |
| `query` | Reading, searching, summarizing, or describing | `{"source":"current_screen","mode":"describe"}` |
| `compose` | Creating text/content | `{"format":"reply","content":"thanks"}` |
| `clarify` | Missing value, ambiguous target, or confirmation | `{"kind":"choice","choices":[...]}` |
| `say` | Voice-only response with no side effect | `{"text":"i'm pace"}` |
| `query_memory` | User/project memory lookup | `{"query":"preferred editor"}` |
| `wait` | Event/time-coupled pause | `{"condition":"download finishes"}` |
| `schedule` | Future reminder/calendar-like action | `{"when":"tomorrow morning","what":"send the deck"}` |

This is the bridge from v6 to v7:

- v6 legacy tags (`[CLICK:x,y]`, `[TYPE:text]`, `[KEY:cmd+s]`) remain an
  executor compatibility layer.
- v7 supervision should prefer structured calls such as
  `perform(action=type, text=...)`; tags can be rendered downstream for old
  Pace clients.
- Dynamic schemas should enum-constrain the things visible in the current
  request: `verb`, `perform.action`, target labels, key names, scroll
  directions, and available app/file/entity refs.
- Adding a new primitive action should usually mean adding a new
  `perform.action` enum value and executor handler, not adding a new planner
  verb or retraining the whole model.
- Add a top-level verb only after the executor semantics diverge from
  `perform` enough that the model benefits from a separate bucket.

**Eval**: **BFCL** (Berkeley Function Calling Leaderboard, #231) +
**τ-bench** (#232) + held-out-tools test set (our own).

## Verb taxonomy (the architectural decision)

The taxonomy IS the design. Get this wrong and v7 fails to generalize.
Proposed taxonomy:

### 5 core verbs (cover ~90% of intents)

| Verb | Schema sketch | Examples |
|---|---|---|
| `read` | `{source, query?, filter?, format?}` | "what's on my calendar today" → `read(source=calendar, filter={today})`. Replaces all "get_X" tools. |
| `act` | `{target, action, params?}` | "click submit" → `act(target=label:submit, action=click)`. "set brightness to 50" → `act(target=brightness, action=set, params={value:50})`. |
| `compose` | `{format, content, target?}` | "write a tweet about X" → `compose(format=tweet, content=...)`. |
| `ask` | `{question, choices?, intent?}` | Clarification back to the user — model uses this when ambiguous. |
| `say` | `{text, expression?}` | TTS output. |

### 5 specialized verbs (carve out edge cases that don't fit cleanly)

| Verb | Schema sketch | Why separate from core |
|---|---|---|
| `recall` | `{memory_scope, query}` | Long-term memory has different latency/source characteristics than `read` |
| `wait` | `{condition, timeout}` | Time-coupled — needs separate handling from `act` |
| `navigate` | `{target}` | Opens app/url/file — could fold into `act` but explicit verb is clearer for the model |
| `schedule` | `{when, what}` | Time-bound action; distinct from immediate `act` |
| `search` | `{scope, query}` | Stateful, may return many results; distinct from `read` (single source) |

### Why this carving

- **Bounded vocabulary**: model learns 10 verbs deeply, not 500
  endpoints shallowly. Sample-efficient.
- **Apps as values, not tools**: adding Bluesky = enum value under
  `act.target`, not a new tool.
- **Composable**: model can chain (`read` → `compose` → `act`) for
  multi-step task planning.
- **MCP-shaped**: each verb maps cleanly to MCP tool definitions.
- **App Intents-aligned**: Apple's verb taxonomy is similar (read the
  App Intents study before final taxonomy lock).

### Held-out verbs for eval

Train on verbs A/B/C/D, eval on verb E (schema-only in prompt). If
the model calls E correctly with no training examples, generalization
is real. Pick `compose` or `recall` as the held-out — they're the
most semantically distinct from the action-oriented verbs.

## System prompt structure

```
You have these tools available. Use them to fulfill the user's intent.

Tools:
[
  {
    "name": "read",
    "description": "Pull information from a known source.",
    "schema": {
      "source": {"enum": ["calendar", "email", "files", "current_screen"]},
      "query": {"type": "string", "optional": true},
      "filter": {"type": "object", "optional": true}
    }
  },
  ...
]

Emit a JSON object with:
  {
    "verb": "<verb_name>",
    "args": {...},
    "spoken_text": "<what to say to the user, optional>"
  }
```

The full tools list (~10 verbs × schemas) comes to roughly 1500
tokens. Prompt cache (#260, already shipped) means this is parsed
once per process, not per request.

## Grammar generation at request time

v6 uses a static GBNF. v7 needs **dynamic GBNF** generated from the
tools list passed at request time. Implementation:

1. `serve` accepts `--tools <tools.json>` flag (or `tools` field in
   request body for OpenAI-compat)
2. On request, generate a GBNF that enforces:
   - `verb` is one of the tool names
   - `args` matches the schema of the selected verb (mutually
     exclusive: if verb=read, args must match read's schema)
3. Cache the generated GBNF by (tools.json hash) — same tools list
   reuses the same compiled grammar
4. JSON Schema → GBNF is a known transformation (we already do part
   of this for v6); needs extension to handle the verb-conditional
   args.

This is non-trivial but not novel — `llama.cpp`'s grammar generator
already handles JSON Schema. Borrow from there.

## Training data plan

**Stage A — public function-call warmstart**:
- **xlam-function-calling-60k** (Salesforce, Apache 2.0) — diverse
  synthetic function calls. ~60K (intent, tools, call) triples.
- Format-normalize to our verb schema (one-off Python script).

**Stage B — Pace-specific top-up**:
- ~500-1000 hand-crafted (intent, Pace_tools, correct_call) triples
- Cover Pace's actual tool set across all 10 verbs
- Include ambiguity / clarification cases for `ask`
- Include held-out verb's examples (to verify generalization, NOT to
  train on it)

**Stage C — adversarial / generalization examples**:
- (intent, tools_with_one_unfamiliar_verb, correct_call) — model
  should use the unfamiliar verb correctly from schema alone
- (intent, tools_missing_obvious_verb, ask_for_help_call) — model
  should emit `ask` instead of fabricating

**Total**: ~62K training examples. Roughly 4× v6's data set. SFT
budget: ~6-8 hours on M5 Pro with current pipeline.

## Eval plan

Three layers:

1. **BFCL** (#231 — ship this before v7 SFT starts) — objective number
   vs xLAM, GPT-4-mini, Claude Haiku, others.
2. **τ-bench** (#232) — multi-turn agent eval, more realistic than
   one-shot function calling.
3. **Held-out tools (our own)** — 50 (intent, tools_with_held_out_verb,
   gold_call) triples we craft ourselves. The verb is in the prompt
   but NOT in training. Generalization check.

Target: BFCL within 5% of xLAM-7B (we'll be 7× smaller), τ-bench
within 10% of teacher, held-out >70% accuracy.

## Cost-benefit (the "should we even do this" check)

**Cost**: ~2-3 weeks of work (data prep + SFT + grammar generator +
serve wire-in + eval).

**Benefit**: every future Pace tool change costs zero retrain. Every
non-Pace TinyGPT consumer gets a planner that works for their tools.

**When v7 pays off**: when Pace gains 5+ new tool types, OR when a
second consumer shows up, OR when the user wants to expose 3rd-party
MCP servers to Pace dynamically.

**When v7 is premature**: if Pace's tool set stays stable for 3+
months, the ~40 min × N retrains for v6 stays cheaper than 2-3 weeks
of v7 work.

**Recommendation**: hold off on v7 SFT until ANE arc lands and v6 has
been daily-driven for ~2 weeks. If Pace's tool set churns visibly in
that window, ship v7. If it's stable, v6 keeps winning on
specialization-for-known-scope.

## Migration from v6

v7 output schema is a **superset** of v6's. The click executor PRD in
clickyLocal (`docs/prds/click-executor-improvements.md`) already
plans for this — it ships schema-tolerant code that accepts both v6's
single-label output and v7's tool-call output.

When v7 lands:
1. Pace's `LocalPlannerClient` schema decode already handles both
2. Pace's system prompt template gains a `tools:` section
3. The planner endpoint URL stays the same
4. Roll back to v6 if v7 regresses (binary swap)

## Scope — in

- Stage A/B/C training data pipeline
- LoRA SFT on chosen base (xLAM-1B preferred per leverage)
- Dynamic GBNF generator (JSON Schema → grammar)
- `serve --tools <tools.json>` flag
- BFCL + τ-bench + held-out-tools eval
- Migration smoke test (Pace runs against v7 endpoint)

## Scope — out

- Multi-tool calls in one response (model emits ONE call; orchestrator
  chains them)
- Streaming tool calls (single output, streamed token-by-token is fine)
- Tool description in non-English (English schemas only for v1)
- Tool versioning (caller assumes the tools list is current truth)
- Multi-LoRA for v6→v7 hot-swap (separate concern)

## Files involved

**New**:
- `scripts/v7-data/normalize-xlam.py` — convert xlam-60k to our verb schema
- `scripts/v7-data/build-pace-topup.py` — hand-curate Stage B
- `scripts/v7-data/build-heldout.py` — generate Stage C
- `native-mac/Sources/TinyGPTServe/DynamicGrammar.swift` — JSON Schema → GBNF
- `scripts/v7-eval/run-bfcl.py` — wrap BFCL harness for v7 model
- `scripts/v7-eval/run-tau-bench.py` — wrap τ-bench
- `scripts/v7-eval/heldout-tools.jsonl` — our held-out eval set

**Modified**:
- `native-mac/Sources/TinyGPTServe/Serve.swift` — add `--tools` flag,
  per-request grammar generation
- `grammars/pace-system-prompt-v7-tools.txt` — new v7 system prompt template

**Don't touch**:
- v6 LoRA artifacts (parallel ship, v6 keeps running)
- ANE arc (separate moat)
- VLM arc (separate moat)

## Acceptance

1. xlam-60k normalized to our verb schema; format-validated
2. SFT completes; LoRA artifact written
3. `tinygpt serve --tools tools.json` loads and accepts requests
4. BFCL: within 5% of xLAM-7B (subject to held-out result; if too
   ambitious, target xLAM-1B parity)
5. Held-out-tools eval: ≥70% correct call on tools never seen in training
6. Pace smoke: v7 endpoint serves Pace's actual workload without
   regression in fm-fixture pass rate vs v6
7. Round-trip latency: ≤1.5× v6 (the longer system prompt costs some
   prefill; prompt cache mitigates)

## Why this matters — the moat argument

v6 makes Pace specialist-grade. v7 makes the TinyGPT *planner factory*
generalizable. Pace becomes the first consumer; the platform is then
usable for any future Mac-native voice-companion / agent that wants a
small fast planner. That's the difference between "TinyGPT trained one
good model for one app" and "TinyGPT is the way you ship local
planners on Mac."

v7 is what turns a specialist into a platform.
