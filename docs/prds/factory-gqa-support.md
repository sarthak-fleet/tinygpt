---
name: GQA support in SFT/training path — modern-base unblock
status: shipped-2026-06-07
owner: maintainer
created: 2026-06-07
priority: P0 — blocks every specialist arc on modern bases
---

# PRD — GQA support in TinyGPT SFT

## 2026-06-07 ship note

Root cause was NOT GQA — it was an explicit `head_dim` override in
Qwen3 (1024 hidden / 16 heads but head_dim=128, so Q out=2048 not 1024).
GQA itself was already wired in `TransformerBlock.swift` via `nKvHeads`.

What changed:
- `HuggingFaceConfig.swift`: parses optional `head_dim` field
- `ModelConfig.swift`: added `explicitHeadDim: Int?` field; `headDim` computed
  uses override when set; precondition `dModel % nHeads == 0` relaxed when
  explicit head_dim is given
- `HFModel.swift` `HFConfigConverter`: passes `explicitHeadDim` only when it
  diverges from the canonical ratio (preserves from-scratch + Phi-3/Llama-2 path)
- `HuggingFaceConfig.unsupportedReason()`: dropped the "GQA not wired" false
  positive; only flags when ratios are inconsistent

Also touched:
- `TransformerBlock.swift` `CausalSelfAttention.init`: Q-proj out-dim and
  O-proj in-dim now `nHeads * headDim` (was `dModel`). No-op for canonical
  configs; correct for Qwen3-family. K/V still `nKvHeads * headDim`.

Smoke tests verified (2026-06-07/08):
- Qwen3-0.6B 1-step SFT with default DoRA: completes, loss 13.10, writes
  218 B header-only `.lora` (DoRA pre-existing limit, see below).
- Qwen3-0.6B 1-step SFT with `--no-dora`: completes, loss 11.84, writes
  2.3 MB `.lora` with real LoRA matrices. End-to-end GQA evidence.
- Phi-3-mini-4k-instruct 1-step SFT with `--no-dora`: completes, loss
  19.18, writes 6.3 MB `.lora`. No regression on the MHA path.
- **Qwen3-0.6B 1000-step SFT post-QK-Norm**: loss converged from 12.8
  → 0.044 (300× lower than pre-fix). 8.8 MB adapter.
- **Qwen3-0.6B inference verified**: "The capital of France is" →
  "Paris. The capital of Italy is Rome..." at 81 tok/s.
- **LoRA-applied inference verified**: behavior visibly changes
  (reasons about FC vs geographic continuation).
- `swift build -c release` clean throughout.

Also closes an inference-path gap surfaced during smoke:
`forwardCachedHF` in `KVCacheHF.swift` was the path used by
`hf-load --sample`/`serve` for HF models, and did NOT apply QK-Norm.
Now does. Without this fix, base Qwen3 produced garbage at sample
time even though training was correct.

### Known gaps NOT in scope (now in `factory-qk-norm-support.md`)

- ~~Qwen3 `q_norm` / `k_norm` per-head [128] weights are silently dropped on load~~ — **shipped 2026-06-08** in same session. Step-1 loss on Qwen3-0.6B SFT dropped from 12.8 to 0.68 (19× improvement) — attention was mathematically wrong without QK-Norm.
- **DoRA + HF writer produces header-only `.lora` files.** The default
  `tinygpt sft` recipe uses DoRA. Documented at `Lora.swift:200-203` as
  a pre-existing TODO ("DoRA is currently in-session only"). Phi-3
  exhibits the identical 209 B symptom; not a GQA regression. Use
  `--no-dora` for round-trippable adapters today. Track as a separate
  PRD (`.lora` format v2 + DoRA serialisation).

## Repro

```bash
QWEN_DIR=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/ | head -1)
./native-mac/.build/arm64-apple-macosx/release/tinygpt sft "$QWEN_DIR" \
    --data ~/.cache/tinygpt/datasets/hermes-fc.jsonl \
    --steps 1 --out /tmp/test.lora
```

Result:
```
MLX/ErrorHandler.swift:345: Fatal error:
[reshape] Cannot reshape array of size 4194304 into shape (1,2048,16,64).
```

## Root cause

TinyGPT's attention reshape assumes MHA (multi-head attention with
Q_heads = K_heads = V_heads). Qwen3-0.6B uses GQA (grouped-query
attention) where:
- 16 query heads
- 8 KV heads (or similar — see Qwen3 config)

The reshape tries to split a tensor sized `seq × (Q+K+V combined)` as
if all heads have the same count. Off by a factor that matches the
GQA ratio.

The "2× too big" error (4194304 = 2×2097152) is consistent with this:
the QKV projection produces more channels than the reshape expects.

**Note**: Phi-3-mini-4k-instruct does NOT trip this (it loads + trains
fine via the same code path), suggesting Phi-3 uses a different attention
configuration that happens to match TinyGPT's reshape assumption.

## Scope — in

1. **Detect GQA from config.json**: read `num_attention_heads` and
   `num_key_value_heads` separately. If they differ, treat as GQA.

2. **Update the attention reshape** to use:
   - Q: shape `(B, L, num_q_heads, head_dim)`
   - K: shape `(B, L, num_kv_heads, head_dim)`
   - V: shape `(B, L, num_kv_heads, head_dim)`

3. **Update the attention forward pass** to repeat K/V heads
   `num_q_heads / num_kv_heads` times before the dot-product. This is
   the standard MLX-style pattern (`mx.repeat` along the head axis).

4. **Verify** the SFT path doesn't break for MHA models (Phi-3 still
   trains).

5. **Verify** Qwen3-0.6B SFT now runs without crash on 1-step smoke.

## Acceptance criteria

1. **Repro succeeds**: the smoke command above completes with non-zero
   loss + writes a non-empty LoRA file
2. Phi-3-mini-4k-instruct SFT still works (regression check)
3. TinyGPT's own preset models (Tiny / Small / Huge — all MHA) still
   work (regression check)
4. Build passes; no other tests fail

## Files likely involved

| File | Likely change |
|---|---|
| `native-mac/Sources/TinyGPTModel/HFModelLoader.swift` (or similar) | Parse num_key_value_heads from config |
| `native-mac/Sources/TinyGPTModel/TransformerBlock.swift` | Update attention reshape + repeat K/V heads |
| `native-mac/Sources/TinyGPTModel/Attention.swift` (if exists) | GQA-aware computation |
| `native-mac/Sources/TinyGPT/SFT.swift` | Pass GQA config through, if needed |

## Estimated effort

**~2-4 hours focused.** The change is well-scoped: detect, reshape,
repeat. The risk is whether MLX-Swift's `repeat` / `tile` op is
performant — if not, may need a different approach (e.g., never
materialize the repeated tensor, do the head broadcast in the matmul).

## Why P0

Without this:
- Pace specialist arc (Qwen3-0.6B student) is blocked
- KB embedder arc (any modern base) is blocked
- Factory's "fine-tune any open base on Mac" claim is false
- Phi-3 still works as a fallback, but Qwen3 family is the modern standard

## Don't touch

- Sampling / inference paths (they may have their own attention impl
  that already works — verify by sampling from Qwen3-0.6B, which
  presumably works)
- Eval pipeline
- App UI
- Docs (the PRD update can be done later)
