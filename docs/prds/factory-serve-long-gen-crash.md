---
name: serve crashes silently during long LoRA-applied generations
status: shipped-2026-06-07
owner: factory
priority: P1 — blocks Pace integration via HTTP (workaround: cap max_tokens ≤200)
---

## Ship note (2026-06-07)

Root cause: serve's `generate` / `generateStreaming` used the uncached,
growing-concat decode loop (`model(cond)` over `concatenated([idx, nextId])`
each step). With LoRA-wrapped Q/K/V projections this leaks MLX graph nodes
per layer per step — at ~500 generated tokens the unified-memory pressure
silently kills the process (no stderr, no stack). Bare-base serve happened
to survive because the per-step graph was small enough; the LoRA path
tipped it over.

Fix: switched both decode paths in `native-mac/Sources/TinyGPTServe/Serve.swift`
to KV-cached decode — prefill once on the bounded prompt, then per-step
`[B,1]` forwards through `model.forwardCached`. Mirrors the `hf-load --sample`
decode loop in `HFLoad.swift` (which had always worked at 500+ tokens with
the same base+LoRA combo). Fresh `KVCache` per HTTP request — no
cross-request sharing.

Side benefit: ~10× faster long-gen latency. The failing 512-token fixture
that used to die after ~79s now completes in ~8s on the same hardware,
matching the O(T) vs O(T²) ratio the KVCache docstring predicts.

Acceptance smoke: 512-token chat completion with the pace-planner-v2
LoRA + Qwen3-0.6B base — survives, returns content, server still alive.
SSE streaming variant survives too (300-token stream, ~19s).

Files changed:
- `native-mac/Sources/TinyGPTServe/Serve.swift` — rewrote `generate` and
  `generateStreaming` to KV-cached decode; replaced left-slice truncation
  with host-side prompt id truncation (left-drop) before prefill.

Unchanged on purpose:
- `scoreLogprobs` — single-pass teacher-forced score, not the bug.
- Sampling / eval / app paths — out of scope for the bug.

# PRD — `tinygpt serve --lora` crashes on long generations

## Repro

```bash
QWEN_DIR=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de...
nohup tinygpt serve "$QWEN_DIR" \
    --lora pace-planner-v2.lora \
    --port 8765 > /tmp/serve.log 2>&1 & disown

# This survives:
curl http://127.0.0.1:8765/v1/chat/completions -d '{"model":"tinygpt",
    "messages":[{"role":"user","content":"hi"}],"max_tokens":20}'

# This crashes serve silently after ~79s:
curl http://127.0.0.1:8765/v1/chat/completions -d '{"model":"tinygpt",
    "messages":[{"role":"system","content":"..."},
                {"role":"user","content":"..."}],
    "max_tokens":512}'
# → "Remote end closed connection without response"
# → process exits with code 0 (no stack trace in stdout/stderr)
```

## Observations

- Crash only happens with `--lora` applied; bare Qwen3-0.6B serve works
- Threshold roughly correlates with total generation length, not just
  max_tokens (200 tokens generated = fine; 500 tokens = dies ~80s in)
- No error output on stderr — likely a SIGSEGV in MLX-Swift's GPU kernel
  or an unhandled fatalError in attention/KV-cache path
- CLI `hf-load --lora --sample --tokens 500` works correctly with same
  base+lora (suggests bug is serve-specific, not LoRA-specific)

## Hypothesis (untested)

`forwardCachedHF` with LoRA-wrapped Q/K/V projections may have a shape
or memory issue at long generations. Specifically:
1. LoRA adds to qProj/vProj output — same shape, so MLX shouldn't care
2. KV cache grows past some boundary (~500 entries × headDim × nKvHeads)
3. Possibly: a Swift array reaches some limit, OR an MLX kernel runs OOM
   on the Mac's unified memory

Sometimes serve is alive after the request errors out; sometimes it
dies. Inconsistent → race condition or GPU OOM.

## Workaround (active)

Cap max_tokens client-side. Pace integration uses `max_tokens ≤ 200`
which is sufficient for voice-companion responses (~1-2 sentences +
one tool tag).

## Fix path (not implemented)

1. Add signal handler / fatalError trap to serve so crashes log a trace
2. Test with progressively longer max_tokens to find exact threshold
3. Test with NON-LoRA Qwen3-0.6B serve to confirm LoRA-specific
4. Check MLX `eval()` calls in serve's decode loop for missing flushes
5. KV cache memory profiling

## Acceptance

- Repro above runs to completion without serve dying
- Pace can daily-drive against `serve --lora` for hours without process death
