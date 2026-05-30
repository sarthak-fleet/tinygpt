# Agent runtime

`tinygpt agent <model> --tools tools.json` is the product surface for the
on-device agent SLM factory. The CLI loads a model (browser-trained
`.tinygpt`, HuggingFace dir, or finetuned specialist), reads an
OpenAI-compatible tool schema, and runs a conversation loop where the
model emits structured JSON, the host executes the tools it asks for,
and the result is fed back into the same KV-cached conversation.

This document covers:

- the conversation loop and how KV state is reused across turns
- the tool schema format
- the subprocess execution model and what it does NOT promise
- how to wire a real agent (`read_file` + `run_test` example)
- where it breaks honestly today

The implementation lives in:

- `native-mac/Sources/TinyGPT/Agent.swift`        — CLI handler
- `native-mac/Sources/TinyGPT/AgentLoop.swift`    — conversation loop
- `native-mac/Sources/TinyGPT/ToolSchema.swift`   — JSON tool parser
- `native-mac/Sources/TinyGPT/ToolExecutor.swift` — subprocess backend

## The conversation loop

The agent's KV cache is a single contiguous prefix that grows monotonically
across turns within a session. On startup we prefill it with a system
prompt that lists the available tools and pins the JSON shapes the model
is allowed to emit. The system prompt template is:

```
<|im_start|>system
You are a specialized on-device agent.

You have access to these tools:
<tools>
- read_file(path: string)  Read a file from disk
- run_test(name: string, timeout?: integer)  Run a single test
</tools>

When you need to use a tool, respond with a SINGLE valid JSON object:
{ "tool": "<name>", "arguments": { ... } }

After the tool runs you will receive its output between <tool_result>...</tool_result>.

When you have the final answer, respond with:
{ "answer": "..." }
<|im_end|>
```

On the first launch with `--prompt-cache-dir <dir>` we hash the prompt
plus model identity, save the prefilled KV state to disk, and load it
back on every subsequent launch — recovering the system prompt's
attention state without paying the prefill cost. The cache key is
inherited from the existing `KVCachePersist` infrastructure: SHA-256 over
`(modelName, file fingerprint, prompt text, vocab, layers, dtype tag,
useYOCO)`. Same key → same file; any change invalidates the cache.

Per user turn the loop runs:

1. Append `<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n`
   to the cache. The model's next forward emits logits at the assistant
   marker — that's where generation starts.
2. Sample one token at a time, decoding incrementally. After every
   token we scan the running text for a complete top-level balanced
   `{ ... }` object (string-aware, escape-aware). The moment one
   appears we stop generation — that's the natural turn boundary.
3. Parse the JSON. Two valid shapes:
     - `{ "tool": "<name>", "arguments": { ... } }` → dispatch to the
       executor, format the result as a `<|im_start|>tool` ChatML
       block, append it to the cache, and loop back to (2).
     - `{ "answer": "..." }` → final answer. Return to the caller.
4. If neither shape matches, fall back to treating the raw text as a
   final answer and log it. This degrades cleanly when the base model
   produces malformed output (see "Honest assessment" below).

The loop is bounded by `--max-steps N` (default 8) tool-call rounds
per user turn, and `--max-tokens N` (default 256) per assistant step.

### KV cache mechanics

We use the same `KVCache` class as `tinygpt sample`. The agent loop
runs in `forwardCached` mode for every chunk — system prompt, each user
turn, each tool result. To recover the next-token logits after appending
a chunk (the cache doesn't store logits) we rewind by one token and
re-feed the last token id through `forwardCached`. This produces the
correct next-position logits without storing any extra state per turn.

Pre-allocation (`--no-kv-preallocate` to disable) sizes each layer's
buffer to `cfg.contextLength` up front so per-chunk appends are slice
assignments rather than `concat` allocations. For an agent running 5+
turns on a 2K-context model this drops peak memory by ~30%.

When the cache hits `cfg.contextLength` we stop. The agent will reply
with whatever's been generated so far and surface a warning. The next
user turn will silently drop further chunks — at that point the user
should restart the session. Cache eviction (windowed attention) is a
follow-up.

## Tool schema format

OpenAI-compatible JSON. `tinygpt agent` reads either `{"tools": [...]}`
or a bare `[...]` array. Each entry has a `function` block (the OpenAI
shape) or has the function fields at the top level (some hand-written
schemas do this — we accept both).

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read file contents from disk",
        "parameters": {
          "type": "object",
          "properties": {
            "path": { "type": "string", "description": "Absolute path" }
          },
          "required": ["path"]
        },
        "_exec": "cat \"$path\"",
        "_exec_args": ["path"]
      }
    }
  ]
}
```

Fields:

- `name` — must be a valid identifier (matches `[A-Za-z_][A-Za-z0-9_]*`).
- `description` — included in the system prompt as `name(args)  description`.
- `parameters` — JSON Schema object. We model `properties` (a flat map of
  `name → {type, description}`) and `required` (array of names). Nested
  schemas pass through untouched into the prompt but the executor only
  validates required-name presence.
- `_exec` — bash command template. tinygpt extension; OpenAI ignores it.
  Required for subprocess execution.
- `_exec_args` — explicit ordered list of argument names that `_exec`
  references. If absent, we use the alphabetical order of `properties`.
- `_handler` — reserved for a future Swift callback registry. Today the
  executor errors if `_exec` is also absent.

The system prompt embeds tool descriptions in this form (so the prompt
hash is stable across launches and the schema-internal JSON ordering
doesn't bust the cache):

```
- read_file(path: string)  Read file contents from disk
- run_test(name: string, timeout?: integer)  Run a single test
```

Optional parameters get a `?` suffix on the name. Required parameters
sort first, optional alphabetically after.

## Subprocess execution model

When the model emits `{ "tool": "read_file", "arguments": { "path": "..." } }`
we resolve `read_file` against the schema and run its `_exec` template
through `/bin/bash -c`. Arguments are passed by pre-assigning each as a
shell variable (`path='...'`) before the template, with the value
quoted via single-quote escaping. The template then references them as
`"$path"`.

Specifically:

```bash
path='/etc/passwd'
cat "$path"
```

This avoids string-substituting model output directly into the template,
which would be an easy code-injection sink. The single-quote escape
rewrites embedded `'` as `'\''` (close, escape, reopen).

The child gets a minimal environment: `PATH`, `HOME`, `LANG`. Inherited
environment can leak secrets into a child the user didn't trust.

Timeout per tool defaults to 30s (`--tool-timeout`). On timeout we send
SIGTERM, wait 0.5s, then SIGKILL.

Tool stdout, stderr, and exit code are bundled into a JSON object and
inserted into the conversation as a `<|im_start|>tool` block:

```
<|im_start|>tool
{"tool":"read_file","stdout":"...","stderr":"","exit_code":0}
<|im_end|>
<|im_start|>assistant
```

The model sees the result and can either call another tool or emit a
final `{ "answer": "..." }`.

### Security caveats

Subprocess execution is plainly unsafe under three threats:

1. **Untrusted schemas.** A `_exec` field can run any command the user's
   shell can. Don't load tool schemas you didn't write or audit. We
   reject malformed argument names (`isIdentifier`) so the variable
   name can't itself inject bash, but the template itself is trusted
   code.
2. **Untrusted prompts.** If the model is fooled into calling a
   destructive tool by an adversarial user prompt, the destruction
   happens for real. Limit the toolset to the minimum the agent needs.
3. **Untrusted models.** A backdoored model could emit tool calls the
   user didn't ask for. We don't gate on user confirmation — adding a
   `--confirm-tools` flag that prompts y/N for each call is a sensible
   follow-up.

Mitigations we DO implement:

- Single-quote escaping of every argument value (no `eval` of model text).
- Hermetic child environment (no inherited secrets).
- Per-tool timeout with SIGTERM → SIGKILL.
- Output size cap when feeding back to the model (8 KB stdout/stderr;
  longer outputs get a `…[truncated N bytes]` tail).
- The `--transcript` log captures every tool call and result for
  post-hoc auditing.

For real production deployment, run the agent inside a sandbox
(`sandbox-exec`, `bwrap`, container, or a separate user). The tool
runtime alone is not a sandbox.

## Examples

### Single-shot mode

```
tinygpt agent specialist.tinygpt \
  --tools tools.json \
  --single "Debug the failing test in tests/test_loss.py"
```

Loads the model, prefills the system prompt, runs the loop until the
model emits `{ "answer": "..." }`, prints the answer, exits.

### Interactive REPL

```
tinygpt agent specialist.tinygpt --tools tools.json
```

Drops into a prompt:

```
you> debug the regression in eval.py
agent> ... (tool calls happen here, then a final answer)
```

`:quit` or Ctrl-D to exit.

### JSON event stream

```
tinygpt agent specialist.tinygpt --tools tools.json --json-out
```

Every event is one JSON object per line on stdout. The event types:

```json
{"type":"user","text":"..."}
{"type":"assistant","step":0,"raw":"..."}
{"type":"tool_call","tool":"...","arguments":{...},"stdout":"...","exit_code":0,"duration_sec":0.04}
{"type":"answer","text":"..."}
```

Pair with `--transcript path.jsonl` to also log to disk (the transcript
file format is the same, plus a `tool_result` event).

### A debugger agent

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file from the working tree.",
        "parameters": {
          "type": "object",
          "properties": { "path": { "type": "string" } },
          "required": ["path"]
        },
        "_exec": "cat \"$path\"",
        "_exec_args": ["path"]
      }
    },
    {
      "type": "function",
      "function": {
        "name": "run_test",
        "description": "Run a single pytest by node ID.",
        "parameters": {
          "type": "object",
          "properties": { "node": { "type": "string" } },
          "required": ["node"]
        },
        "_exec": "pytest -xvs \"$node\" 2>&1 | tail -60",
        "_exec_args": ["node"]
      }
    },
    {
      "type": "function",
      "function": {
        "name": "grep_repo",
        "description": "Search the repo for a pattern.",
        "parameters": {
          "type": "object",
          "properties": { "pattern": { "type": "string" } },
          "required": ["pattern"]
        },
        "_exec": "rg --no-heading --color=never \"$pattern\" | head -40",
        "_exec_args": ["pattern"]
      }
    }
  ]
}
```

Usage:

```
tinygpt agent specialist.tinygpt --tools debugger.json \
  --single "tests/test_loss.py::test_xent fails on commit abc123. Find the cause."
```

A well-specialized debugger model with this toolset will:
1. `grep_repo` for `test_xent` to find related code.
2. `read_file` the test and the source file.
3. `run_test` to confirm the failure mode.
4. Emit a final answer describing the bug.

## Persistent prompt caching

The system prompt is identical across launches. Caching its KV state on
disk turns the 200ms-2s prefill into a single mmap + small forward.

```
tinygpt agent specialist.tinygpt --tools tools.json \
  --prompt-cache-dir ~/.cache/tinygpt/kv
```

First launch: prefill + save. Second launch (and every subsequent one):

```
agent: loaded system-prompt cache (192 tokens) — skipping prefill
```

The cache key includes the prompt text and the tool schema (because the
schema is embedded in the prompt), so editing `tools.json` invalidates
the cache. Editing the model file (different size or mtime) also
invalidates. See `Sources/TinyGPTModel/KVCachePersist.swift`.

## Honest assessment

The runtime works end-to-end with the demo byte-level Shakespeare model
(96M params, 256 ctx) — it loads, prefills, generates, the JSON detection
and fallback both fire. But the demo model is dramatically undersized
for the task: the system prompt alone is ~580 bytes, longer than the
model's entire 256-byte context. Even after truncation, an unspecialized
byte-level model trained on Shakespeare will not produce well-formed
JSON. The smoke-test transcript shows the model emitting
`"u\n\nu"` and the loop correctly falling back to "treat the raw text as
a final answer."

This is what the runtime delivers TODAY:

- Loads any `.tinygpt` file or HF model directory through the same
  `ColdStart.loadWithSpinner` path `tinygpt sample` uses.
- Parses OpenAI-compatible tool schemas plus the `_exec` extension.
- Runs the conversation loop, tracks KV state across turns, scans for
  balanced JSON, dispatches tool calls, feeds results back.
- Persists the system prompt KV to disk so launches 2..N skip prefill.
- Logs structured events for eval (`--transcript`) and for piping
  (`--json-out`).
- Sandbox-lite subprocess execution (hermetic env, escaped args,
  timeout, output truncation).

What is NOT going to work without a specialized model:

- Reliable JSON shape. A from-scratch byte-level Shakespeare model has
  never seen a `{` followed by `"tool"` and will not emit one. Even a
  general-purpose chat model needs SFT examples in the
  `{tool, arguments}` / `{answer}` schema to lock in the format. Without
  that, the fallback path triggers every turn and tools are never called.
- Tool selection. The model needs to learn that "read this file" maps
  to `read_file` and not to some made-up tool name. This is the bulk
  of the specialist training data.
- Stopping at the right place. The model has to know that one JSON
  object is one turn. We currently stop on the first balanced `{...}`
  in the decoded stream, which is robust to extra trailing text but
  doesn't help if the model produces a nested object inside an `answer`
  string. Edge cases compound when the model is undertrained.

The next steps for shipping a usable agent are NOT in this runtime —
they're in the specialist training pipeline:

1. Generate SFT data of (system+tool+user → tool-call JSON) pairs.
2. Finetune a base model on that schema with the SFT loop (`tinygpt
   sft`).
3. Evaluate with `--transcript` against a held-out task set.
4. Re-deploy via `tinygpt agent`.

The runtime is intentionally model-agnostic so the same binary works
across the full ladder from a demo model up to a 7B specialist.

## Smoke test reproducer

```
# Build:
cd native-mac
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-agent -configuration Release build

# Echo tool that just returns its input:
cat > /tmp/tools.json << 'EOF'
{ "tools": [{ "type": "function", "function": {
    "name": "echo", "description": "Echo input",
    "parameters": { "type": "object",
      "properties": { "text": { "type": "string" } },
      "required": ["text"]
    },
    "_exec": "printf '%s' \"$text\"", "_exec_args": ["text"]
}}]}
EOF

# Single-shot:
/tmp/tinygpt-agent/Build/Products/Release/tinygpt agent \
  browser/public/demo.tinygpt --tools /tmp/tools.json \
  --single "Use the echo tool" \
  --max-tokens 30 --max-steps 1 \
  --transcript /tmp/agent.jsonl

# Persistent KV cache:
/tmp/tinygpt-agent/Build/Products/Release/tinygpt agent \
  browser/public/demo.tinygpt --tools /tmp/tools.json \
  --single "x" --prompt-cache-dir /tmp/kv
# Second run shows: "loaded system-prompt cache (192 tokens) — skipping prefill"
```

Expected output: the demo model emits gibberish, the JSON parse fails,
the fallback path turns the gibberish into a fake final answer, and the
loop exits cleanly. That is the runtime working correctly — the model
is the bottleneck, not the wiring.
