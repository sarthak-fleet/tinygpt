---
name: tinygpt serve prompt cache — auto-cache system prompt KV across requests
status: shipped-v1-2026-06-07
owner: unassigned (parallel-agent task — Swift)
created: 2026-06-08
priority: P0 — biggest single TTFW win for Pace specialist (20× faster)
---

# PRD — serve auto-caches system prompt prefill

## 2026-06-07 ship note

`tinygpt serve` now accepts `--prompt-cache-dir <dir>` and reuses the existing
`.kvcache` persistence format.

Shipped:
- server-level `--prompt-cache-dir`
- stable chat-prefix caching for leading `system` / `developer` messages
- exact-prompt caching for `/v1/completions`
- cache key scoped by model config plus base/LoRA fingerprint
- cache load, prefix skip, miss rebuild, and metadata sidecar write
- works in both streaming and non-streaming OpenAI decode paths

v1 deliberately avoids guessing over mutable conversation history. For chat,
it caches the stable leading instruction prefix, which is the Pace fast path.

## Problem

Pace sends the **identical system prompt** on every turn. With ~700-token
system prompts, serve currently re-prefills the entire KV cache for that
prompt on every request → ~1000ms wasted before the first generated
token. The Pace specialist achieves nothing on TTFW because of this.

Sample.swift already solved this with `--prompt-cache-dir`. Port the
same pattern to serve.swift.

## Goal

Add `--prompt-cache-dir <dir>` flag to `tinygpt serve`. When set:
1. On each request, hash the messages prefix that's identical to past requests (typically the system message + any unchanged conversation history)
2. If a cached KV file exists for that hash → load it, skip the prefill, jump straight to generating
3. If not → do the full prefill, save the KV cache to `<dir>/<hash>.kv` for next time

**Expected TTFW**: Pace specialist drops from ~1000ms → ~50ms for repeat-prompt sessions (the common case).

## Scope — in

1. **CLI flag**: `--prompt-cache-dir <path>` (default: nil, behaves as today)
2. **Hashing**: SHA256 of (model name + base config + serialized prefix messages)
3. **Storage**: same on-disk format Sample.swift uses (existing `.kv` writer)
4. **Loading**: on request, find longest cached prefix that matches the actual request prefix; load + skip prefill
5. **Auto-write**: after each successful generation, write the prefix KV back if the request was longer than any cached prefix

## Scope — out

- Streaming KV updates (write on every token) — too costly for v1
- Cross-process locking — single-server only, no contention
- Cache eviction — LRU or size cap deferred to v2
- Multi-LoRA-aware caching (different LoRAs → different caches; v1 caches only against base+LoRA combo)

## Acceptance

1. Smoke: `tinygpt serve --lora pace-planner-v6.lora --grammar pace-fm-label-response.schema.json --prompt-cache-dir /tmp/pace-cache --port 8765`
2. First request: cache miss → normal latency (~1000ms TTFW)
3. Second request with same system prompt: cache hit → TTFW drops to <100ms
4. Verify via the `pace-eval-baseline.py` script — second-onward fixture latencies should drop dramatically
5. Build clean, existing serve tests still pass

## Files involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPTServe/Serve.swift` | Parse `--prompt-cache-dir`, thread through Server + decode loop |
| Reference: `native-mac/Sources/TinyGPT/Sample.swift` (lines ~268-340 for the cache-load/cache-save pattern) | (don't edit; just mirror) |
| Reference: `native-mac/Sources/TinyGPTModel/KVCache.swift` (load/save helpers exist) | (don't edit) |

## Estimated effort

**~2-3 hours.** The KV cache load/save primitives exist in KVCache.swift;
Sample.swift's cache lifecycle is the working template. Most of the work
is wiring + the prefix-matching logic.

## Why P0 for the Pace arc

This is the dominant TTFW win. Even a perfect specialist sounds slow to
Pace users if every push-to-talk pays 1000ms before the first spoken
word. Caching turns the repeated-system-prompt case into <100ms TTFW.

Without this, the Pace specialist's quality wins are wasted on latency
that frustrates users.
