# MLXFast SDPA & tied-embedding audit (2026-05-30)

Two-item perf + correctness audit, scope-limited to:

1. Every `MLXFast.scaledDotProductAttention` (SDPA) call site, checking that
   we're hitting Apple's fused Metal path (not the unfused fallback).
2. The `tieEmbeddings` config flag — verifying that at build/save/load time
   we actually share the embedding tensor with the LM head (not double-store).

Audit verdict: **both items are CLEAN.** No code changes required. This
document is a positive verification, with the supporting evidence and the
specific gotchas we now know to guard against.

Verified against MLX-Swift checkout under
`/private/tmp/tinygpt-merge/SourcePackages/checkouts/mlx-swift`.

---

## #1 — MLXFast SDPA audit

### Call sites

Every direct call to `MLXFast.scaledDotProductAttention` in the codebase:

| # | File                                    | Line | Mask used     | Where Q,K,V come from                       |
| - | --------------------------------------- | ---- | ------------- | ------------------------------------------- |
| 1 | TransformerBlock.swift (ALiBi)          | 216  | `.array(...)` | proj + reshape + `.transposed(0,2,1,3)`     |
| 2 | TransformerBlock.swift (sliding window) | 221  | `.array(...)` | same                                        |
| 3 | TransformerBlock.swift (plain causal)   | 225  | `.causal`     | same                                        |
| 4 | DifferentialAttention.swift (branch 1)  | 93   | `.causal`     | two Q/K proj + transpose                    |
| 5 | DifferentialAttention.swift (branch 2)  | 96   | `.causal`     | same                                        |
| 6 | CrossAttention.swift (prefill)          | 84   | `.causal`     | local Q proj, external K,V from anchor      |
| 7 | CrossAttention.swift (decode)           | 86   | `.none`       | same                                        |
| 8 | KVCache.swift (cached forward)          | 526  | `.causal`/`.none` | K,V from `cache.keys/.values`           |
| 9 | KVCacheHF.swift (HF cached forward)     | 61   | `.causal`/`.none` | same, GQA-correct nKvHeads reshape      |

`TransformerBlockHF.swift` does NOT call SDPA directly — it routes through
the shared `CausalSelfAttention.computeSDPA` (so its SDPA call IS site #3,
counted once, but exercised by both from-scratch and HF blocks).

### Mask audit — fast-path enum, not bool tensor

The task brief flagged a common slow-path trap: using a manually-built bool
tensor instead of the `.causal` mask enum. Audit result: **every causal call
already uses the enum.** Specifically the `ScaledDotProductAttentionMaskMode`
in MLX-Swift's `Source/MLX/MLXFast.swift:134-161`:

```swift
public enum ScaledDotProductAttentionMaskMode {
    case none
    case array(MLXArray)
    case causal
    public var mode: String { ... case .causal: "causal" ... }
}
```

`.causal` sets the C-API `mode` parameter to the string `"causal"`, which the
C++ dispatcher (`Source/Cmlx/mlx/mlx/fast.cpp:638-645`) decodes into
`do_causal=true`. That flag then propagates to the Metal kernel via the
`ScaledDotProductAttention` primitive and lets the kernel build the triangle
internally — no host-side mask tensor allocated, no `where`/`maximum`
broadcast roundtrip.

The ALiBi and sliding-window paths (#1, #2) DO use `.array(mask)` because
their per-head distance-weighted / window pattern can't be expressed as the
single boolean "j ≤ i" the `.causal` mode hardcodes. That's correct; there's
no enum alternative for these cases.

### Tensor contiguity — Q, K, V are all matrix-contiguous

The fast Metal kernel in
`Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:670-673`
defines the contiguity requirement as:

```cpp
auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
};
```

If `stride[-1] != 1`, the kernel falls back to `contiguous_copy_gpu` —
materialising a transposed copy before launch. Audit result: **all our Q,
K, V satisfy `stride[-1] == 1` already.** Walk through:

  `qProj(x).reshaped([B, T, H, D]).transposed(0, 2, 1, 3)`

  - `qProj(x)` is row-contiguous (output of a Linear); strides `[T*H*D, H*D, D, 1]`.
  - `.reshaped([B, T, H, D])` is a no-op shape rebrand (matches existing
    underlying layout, strides unchanged).
  - `.transposed(0, 2, 1, 3)` swaps axes 1 and 2. New strides:
    `[T*H*D, D, H*D, 1]` — the last dim (D, formerly axis 3) keeps stride 1.

So `stride[-1] == 1` after transpose, and the kernel takes the no-copy
branch. K and V follow identical reshapes; ditto for `KVCache.keys/values`
which materialise via `concatenated(..., axis: 2)` (the concat is along axis 2,
not the last axis, so stride[-1] stays 1).

Cross-attention's externally-supplied K, V come from the anchor's
`forwardCapturingKV` which returns the same `.transposed(0,2,1,3)` shape —
also contiguous on stride[-1].

### Head dim — fast-path eligibility

The fast kernel's `use_fallback` (same file, line 588-637) gates on head dim.
Eligible head dims:

  * vector mode (query length ≤ 8): 64, 96, 128, 256
  * full mode (query length > 8):    64, 80, 128

Our preset configurations:

| Preset    | dModel | nHeads | headDim | Fast-path? |
| --------- | ------ | ------ | ------- | ---------- |
| tiny      | 128    | 4      | 32      | NO         |
| small     | 192    | 6      | 32      | NO         |
| huge      | 256    | 8      | 32      | NO         |
| mega      | 512    | 8      | 64      | YES        |
| behemoth  | 1024   | 16     | 64      | YES        |
| titan     | 1536   | 24     | 64      | YES        |

The smallest fast-path-eligible from-scratch preset is `mega`. tiny/small/huge
all run on the unfused Metal fallback because headDim=32 is outside the
kernel's hardcoded set. This is an MLX limitation, not ours — there's no
configuration knob; we'd need to either resize heads (dModel=512 at H=8 is
the minimum) or upstream a 32-wide kernel into MLX. Either is out of scope
for this audit.

Note: HF-loaded models (Llama 2, Mistral, SmolLM2, etc.) all have headDim
in {64, 80, 96, 128} by design — they are always fast-path eligible.

### Training vs inference

MLX-Swift's SDPA has a subtle but important caveat: during forward+grad
tracing, the metal backend deliberately uses the fallback path. See
`scaled_dot_product_attention.cpp:598-602`:

```cpp
if (is_training) {
    // It's faster for training on Metal to use the unfused SDPA for both
    // forward and backward.
    return true;
}
```

So our `Trainer.step()` workload doesn't benefit from the fused kernel
regardless of what we do at the call site. The fast kernel only fires for
SAMPLE / inference paths (where MLX is NOT inside `eval_grad`). This matches
the expected behaviour and isn't a bug — Apple's fused kernel doesn't have a
fast VJP for the full op, so the unfused decomposition wins overall during
training where you also need gradients.

### Measurement

Built `tinygpt` Release on commit `645c2f4` via:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-mlxfast -configuration Release build
```

Build: **SUCCESS.**

#### demo.tinygpt (headDim=32, slow-path)

  `tinygpt sample browser/public/demo.tinygpt --prompt "ROMEO:" --tokens 50 --temperature 0`

  Run 1: 50 tokens / 0.10s — **483 tok/s** (cold; includes Metal-shader compile)
  Run 2: 50 tokens / 0.08s — **647 tok/s**
  Run 3: 50 tokens / 0.08s — **603 tok/s**
  Run 4: 50 tokens / 0.08s — **603 tok/s**

  This is on the FALLBACK path (headDim=32), so it's a benchmark of the
  current state — no kernel switch available.

#### mega5.tinygpt (headDim=64, fast-path)

Trained 5 steps via `tinygpt train --preset mega --steps 5` so we have a
fast-path-eligible model:

  `tinygpt sample /tmp/mlxaudit/mega5.tinygpt --prompt "hello world" --tokens 50 --temperature 0`

  Run 1: 50 tokens / 0.29s — **175 tok/s** (cold)
  Run 2: 50 tokens / 0.22s — **229 tok/s**
  Run 3: 50 tokens / 0.19s — **261 tok/s**

  Mega is ~6× larger than demo (76M vs 9.6M params), so absolute tok/s is
  lower despite hitting the fast kernel — expected. The point is the call
  shape passes the fast-path predicate.

### What we'd change if we found a real slow-path

Since nothing was on the slow path, no edit was made. For reference, the
fixes the audit would have applied:

  * If a call used `mask: .array(causalMask)` with a manually-built bool
    triangle: switch to `mask: .causal`. Saves a tensor allocation per layer
    per forward and lets the kernel build the triangle internally.

  * If Q, K, or V had non-unit `stride[-1]`: insert `.contiguous()` before
    the call. The kernel would otherwise allocate + copy internally; doing
    it once at the call site is the same cost but visible in profiles.

  * If a call passed an unsupported dtype (none observed; we run fp32 and
    fp16 throughout): cast Q, K, V to the model's runtime dtype before the
    call. The C++ side does this internally (`astype(queries, final_type,
    s)`) but explicit is cleaner.

---

## #2 — Tied embeddings verification

### What "tied" means in our code

`ModelConfig.tieEmbeddings: Bool` (default `true`) controls whether the LM
output head reuses the input embedding matrix (the
"Press & Wolf 2017 weight tying") instead of allocating a separate `lm_head`
Linear.

Implementation lives in three places:

1. **Build** — `TinyGPTModel.init` (line 68-72) and `TinyGPTModelHF.init`
   (line 56-60). When `tieEmbeddings == true`, the `lmHead` `@ModuleInfo` is
   set to `nil`, so no separate Linear weight is allocated.

2. **Forward** — `projectLogits` in `TinyGPTModel.swift:326-332` and the
   tail of `TinyGPTModelHF.callAsFunction` (line 110-113). When `lmHead` is
   `nil`, we call `tokenEmbedding.asLinear(x)` which is MLX-Swift's
   pre-existing helper at `MLXNN/Embedding.swift:47-49`:

   ```swift
   open func asLinear(_ x: MLXArray) -> MLXArray {
       matmul(x, weight.T)
   }
   ```

   `weight.T` is a **transposed VIEW** of the same MLXArray — not a copy.
   No second copy of the embedding matrix is allocated at runtime.

3. **Save / load** — see below.

### At BUILD time — same tensor, not a copy

When `tieEmbeddings == true`:

  * `_lmHead.wrappedValue = nil` — `lmHead` is absent from the model's child
    module tree. Confirmed by:
      - `model.parameters()` doesn't enumerate `lm_head.weight` (MLX-Swift's
        `@ModuleInfo` with Optional reflects nil as "no child").
      - The total parameter count from `tinygpt inspect` matches a model
        without a separate lm_head.

  * At forward, `tokenEmbedding.weight` is the SOLE matrix used for both
    input lookup (via `Embedding.callAsFunction`) and output projection
    (via `asLinear`). One MLXArray, one allocation.

### At SAVE time — no double serialization

`Train.manifestEntries` (Train.swift:801-878) builds the on-disk tensor
list. There is NO `lm_head.weight` entry — only:

```
token_embedding.weight    [vocabSize, dModel]
position_embedding.weight [contextLength, dModel]
...
blocks.{i}.{ln1,attn,ln2,mlp}.{weight,bias}
...
ln_final.{weight,bias}
```

(I read the actual entries Train.swift emits — full list at lines 810-877.)
Empirical confirmation via `tinygpt inspect /tmp/mlxaudit/mega5.tinygpt`:

  * `token_embedding.weight  [256, 512]  131,072` — one and only embedding-shaped tensor in the manifest.
  * No `lm_head.weight` row.
  * Round-trip `tinygpt validate` passes bit-identical.

So a tied model NEVER writes a second copy. **Save path is clean.**

### At LOAD time — reads from the same buffer

`AnyModel.ModelLoader.load` reconstructs `ModelConfig` from the manifest
header. The header schema (`TinyGPTHeader.Config`) does NOT carry a
`tieEmbeddings` field — so on load the config field falls back to the
`ModelConfig` constructor default of `tieEmbeddings: true`.

Concretely, an untied checkpoint **cannot be saved** by the current from-
scratch path because `manifestEntries` never emits `lm_head.weight`. So
"tied at load" is enforced architecturally: there's no path that could load
an untied from-scratch model. The only way to set `tieEmbeddings = false`
is in code, and there's no CLI knob:

  ```
  grep -rn "tieEmbeddings: false" native-mac/Sources/   # empty
  ```

The model built at load time has `_lmHead.wrappedValue = nil` (because
config.tieEmbeddings is true), and the loader's `rewriteLeaves` walks the
existing param tree — so even if a file somehow contained `lm_head.weight`,
that entry would be silently dropped (no matching leaf to overwrite). The
loader uses `verify: [.noUnusedKeys]` which means an extra key would fail
loudly — but the manifest never has one to begin with.

**Load path is clean.**

### HF model side

`TinyGPTModelHF` mirrors the same logic: `lmHead` is Optional and only
allocated when `cfg.tieEmbeddings == false`. The HF safetensors loader
(`HFModelLoader.load`, HFModel.swift:213-386) does an extra paranoia step:

```swift
if !hasLmHead { cfg.tieEmbeddings = true }
```

If the on-disk safetensors don't actually have an `lm_head.weight` tensor,
force-tie the config regardless of what HF's config.json says. This is the
correct behaviour — smaller HF models often omit lm_head implicitly,
relying on the loader to tie. Without this override, the from-scratch
`lm_head` Linear would be left at random init and never overwritten,
producing garbage. The check exists; the behaviour is verified by the
existing HF-load smoke tests (Llama family, SmolLM2).

The opposite case — HF config says tied, but the file ships a redundant
lm_head.weight — is also handled: the load loop at HFModel.swift:349-371
emits an update entry for `lm_head.weight`, but the model has no `lmHead`
child (because tieEmbeddings = true), so MLX-Swift's `update(parameters:)`
path silently ignores it. (The `verify: []` empty array on line 383 is
deliberately permissive on the HF side because HF safetensors carries
extra tensors like rotary inv_freq buffers we don't use.)

### File-size delta

For an HF model with vocab=32000, dModel=2048, tied vs untied:
  * Tied:   1 × 32000 × 2048 × 4 bytes (fp32) = **262 MB**
  * Untied: 2 × 32000 × 2048 × 4 bytes        = **524 MB**

  Delta: ~262 MB at fp32, ~131 MB at fp16/bf16. Real-world Llama 2 7B
  ships untied, so the lm_head is ~262 MB of the disk image — exactly
  the duplication tying eliminates.

For from-scratch with vocab=256 (byte) it's tiny — 256 × 512 × 12 bytes
(fp32 weight + adam m + adam v in the .tinygpt triplet layout) = 1.5 MB.
The mega5.tinygpt file currently lands at 873 MB; if untied were possible,
it'd be ~874 MB. Not enough to matter at this scale. The benefit shows up
at BPE-vocab scale (vocab=32000 → 30000× larger embedding table).

### Was anything broken?

**No.** All three checkpoints — build, save, load — correctly treat tied
as a single shared tensor. The most surprising property is that an UNTIED
from-scratch save can't even be written (manifestEntries doesn't enumerate
lm_head). This isn't a bug — it's a consistent design: from-scratch is
always tied, period. The `tieEmbeddings: Bool` field on ModelConfig is
effectively a no-op for from-scratch models (always true). It MATTERS for
HF-loaded models, where the upstream config.json drives the decision.

---

## Files touched

None. Audit-only changes:

- `docs/perf_audit_mlxfast_tied.md` (this file, new).

The five files the brief listed as candidate targets were all read and
audited; no edits required:

  - `native-mac/Sources/TinyGPTModel/TransformerBlock.swift` — clean
  - `native-mac/Sources/TinyGPTModel/TransformerBlockHF.swift` — clean (delegates)
  - `native-mac/Sources/TinyGPTModel/CrossAttention.swift` — clean
  - `native-mac/Sources/TinyGPTModel/TinyGPTModel.swift` — clean
  - `native-mac/Sources/TinyGPTModel/HFModel.swift` — clean
  - `native-mac/Sources/TinyGPTIO/Manifest.swift` — clean (schema; no lm_head field deliberately)

---

## Build verdict

`xcodebuild -scheme tinygpt -configuration Release` on commit `645c2f4`:

```
** BUILD SUCCEEDED **
```

Sample smoke test passed on both:
  * demo.tinygpt (headDim=32, slow-path fallback) — 603 tok/s
  * /tmp/mlxaudit/mega5.tinygpt (headDim=64, fast-path) — 261 tok/s
  * tinygpt validate confirms .tinygpt round-trip is byte-identical
