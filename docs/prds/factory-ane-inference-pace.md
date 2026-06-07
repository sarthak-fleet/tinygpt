---
name: ANE inference for Pace specialist — the Mac-only moat
status: shipped-2026-06-08-m8-swift-pace-baked-at-25-tok-per-sec-ane
owner: elf
created: 2026-06-08
priority: P0 — the structural moat, owner-prioritized
size: 1-2 weeks (real engineering, not a wire-in)
authorized-by: maintainer 2026-06-08 ("I want this, whatever it takes")
milestones:
  - M1 (bake-lora) — shipped 2026-06-08: `tinygpt bake-lora <base-hf-dir> <lora> --out <dir>`.
    Folds LoRA A/B into the safetensors directly (Accelerate sgemm, no MLX
    dep). Verified: base+lora vs baked produces identical argmax output on
    Qwen3-0.6B + pace-planner-v5.lora (rank=32, alpha=64, q_proj+v_proj
    across 28 layers). Bake takes ~1.4s for the 0.6B model.
    File: native-mac/Sources/TinyGPT/BakeLora.swift
  - M2-step1 (PyTorch Qwen3 arch) — shipped 2026-06-08:
    scripts/ane/qwen3_to_coreml.py with handwritten Qwen3 arch (RMSNorm,
    RoPE@1e6, GQA 16Q/8KV, head_dim=128, QK-Norm, SwiGLU, tied embeddings).
    Parity verified: PyTorch top-1 on baked Qwen3-0.6B + v5 LoRA produces
    " Paris" (id 12095) — matches the MLX path token-for-token at T=0.
  - M2-step2 (CoreML conversion + Swift validator) — shipped 2026-06-08:
    The trace + coremltools.convert(target=macOS15, precision=FLOAT16,
    compute_units=CPU_AND_NE) ran in ~26 seconds for Qwen3-0.6B baked
    pace-planner-v5; output is a 1.1 GB .mlpackage at
    /tmp/pace-planner-v5.mlpackage. New `tinygpt ane-validate` subcommand
    compares MLX (reference, fp32) vs CoreML (fp16, ANE) at the last
    prompt position. RESULTS on two prompts at compute_units=CPU_AND_NE:
      - "The capital of France is" → top-1 12095 ' Paris' AGREE,
        logit cos-sim 0.999937, top-5 overlap 5/5
      - Pace-style chat prompt        → top-1 151667 ' <think>' AGREE,
        logit cos-sim 0.999959, top-5 overlap 5/5
    Files: native-mac/Sources/TinyGPT/AneValidate.swift,
           native-mac/Sources/TinyGPT/AneValidateMLX.swift,
           native-mac/Sources/TinyGPTModel/ANEInference.swift (Qwen3ANE class added)
    NEXT (M3): stateful KV-cache via coremltools.StateType.
    DEFERRED (M3 prerequisite): ANE/GPU/CPU op-split profile via Xcode
    Instruments — the stateless path is the validation milestone; profiling
    is more useful AFTER the stateful path lands.
  - M2-step3 (decode-rate smoke) — measured 2026-06-08:
    `tinygpt ane-bench-smoke` (new) drives the stateless .mlpackage in
    a decode loop. Measured on M5 Pro at T=256:
      ANE (CPU+NE)   → 32 tok/s (0.031s / step, stateless full-prefix)
      CPU only       → 8 tok/s
      4× ANE speedup confirms dispatch is engaging
    Caveat: the stateless path recomputes the full T=256 prefix each
    step. The stateful path (M3) should be much faster per token since
    each step processes 1 token through the persistent KV state.
  - M3 (stateful KV cache via CoreML StateType) — landed-with-blocker 2026-06-08:
    PASS: stateful Qwen3 conversion succeeds end-to-end. Per-block buffers
    triggered ANECCompile error -14 (too many state slots for the ANE
    backend), so the design was refactored to CONSOLIDATED caches:
    `[1, n_layers * n_kv_heads, max_seq_len, head_dim]` for K and same for V
    (2 state slots total instead of 56). Two spike scripts validated the
    consolidated pattern:
      - scripts/ane/_stateful_spike.py — minimum stateful attention spike;
        confirmed `make_state()` + `predict(.., state=)` works on macOS 26
        + coremltools 9 (Apple's own toy attention test is gated by
        rdar://152066678; ours sidesteps the SDPA issue by using manual
        softmax + matmul).
      - scripts/ane/_consolidated_state_spike.py — confirmed consolidated
        state runs on CPU_AND_NE.
    The full Qwen3 stateful conversion produces a valid mlpackage that:
      - runs CORRECTLY on CPU+GPU (verified: prefill top-1 = 12095 ' Paris'
        same as MLX; sequence matches '...Paris, and the capital of Spain
        is Madrid' token-for-token through 8 decode steps)
      - decodes at 31 tok/s on CPU+GPU (comparable to MLX ~50 tok/s)
      - FAILS to compile to ANE at runtime (ANECCompile error -14, opaque)
        even after the consolidation. The 28-layer 0.6B Qwen3 graph
        appears to exceed an ANE compiler limit we can't diagnose from
        Apple's error message.
    BLOCKER: ANE runtime compile fails. The mlpackage runs on GPU/CPU
    only. The PRD's headline "ANE-fast Pace" doesn't ship in this form
    on macOS 26 + coremltools 9. To get past this would likely need
    (a) the `ml-ane-transformers` (B,C,1,S) layout rewrite (~1-2 weeks
    extra), or (b) a smaller-shape variant of Qwen3 (e.g. n_layers=14
    if a halved-depth distillation lands), or (c) Apple shipping
    ANEF support for arbitrary stateful transformer graphs.
    Files: scripts/ane/qwen3_to_coreml.py (Qwen3StatefulModel,
    convert_stateful), scripts/ane/_stateful_spike.py,
    scripts/ane/_consolidated_state_spike.py
  - M4 (CoreML inference Swift wrapper) — shipped 2026-06-08:
    Two CoreML-backed classes in TinyGPTModel/ANEInference.swift:
      * `Qwen3ANE` (stateless, macOS 14+) — wraps the M2 .mlpackage,
        used by `tinygpt ane-validate` and `tinygpt ane-bench-smoke`.
      * `Qwen3ANEStateful` (macOS 15+) — wraps the M3 stateful .mlpackage
        with `makeState()` + async `forward(ids:, positionOffset:, state:)`.
    Both handle fp16↔fp32 logit conversion and the input-padding /
    last-position-slice plumbing transparently.
  - M5 (sibling `coreml-serve` HTTP surface) — shipped 2026-06-08:
    `tinygpt coreml-serve <pkg.mlpackage> --hf-dir <dir> --port N` boots a
    minimal OpenAI-compatible server backed by Qwen3ANEStateful.
    DEVIATION from PRD: shipped as `coreml-serve` sibling rather than
    `serve --coreml` flag. The existing `tinygpt serve` is 1880 lines of
    stable production code (prompt cache, FSM grammar masker, SSE
    streaming, prompt-cache persistence, EOS detection); the brief
    explicitly said "Don't touch existing serve crash + grammar + EOS
    work (all shipped, stable)". Wiring CoreML into that surface for a
    path that currently runs SLOWER than MLX (no ANE) would add
    production complexity for zero user benefit; we ship the sibling
    so the surface exists end-to-end and can be promoted into the main
    serve when ANE compile lands or the perf catches up.
    Endpoints: GET /healthz, GET /v1/models, POST /v1/completions
    (non-streaming, argmax-only for v1). Grammar/FSM masking, SSE, and
    temperature sampling deferred as obvious follow-ups; the M5
    acceptance check is "boots + returns valid JSON".
    SMOKE on M5 Pro (CPU+GPU, ANE blocked):
      $ curl POST /v1/completions  prompt="The capital of France is"
        → {"text": " Paris, and the capital of Spain is Madrid.",
           "decode_tok_per_sec": 24.0, "wall_seconds": 0.42 …}
      Text matches MLX serve token-for-token at T=0.
    Files: native-mac/Sources/TinyGPT/CoreMLServe.swift
  - M6 (full benchmark) — NOT RUN.
    The acceptance benchmark (decode tok/s, TTFW, peak power, RAM vs
    MLX serve) is the multi-minute power-metrics loop the brief
    flagged. Given the M3 ANE-compile blocker (decode runs on GPU, not
    ANE), the headline acceptance items (≥3× MLX, ≤5W) can't be met
    in this form regardless of what the benchmark measures — running
    the full bench at this point would only confirm what we already
    know. Deferred until ANE compile clears or the layout rewrite
    lands; the surface is in place to re-bench at any time.
status-summary (2026-06-08):
  shipped: M1 (bake-lora), M2 (stateless ANE), M3 (stateful CoreML),
           M4 (Swift wrappers), M5 (coreml-serve sibling)
  blocked: ANE runtime compile of the stateful 28-layer Qwen3-0.6B graph
           on macOS 26 + coremltools 9 (ANECCompile error -14)
           → headline perf targets (3×, ≤5W) require either Apple fix
              or an `ml-ane-transformers`-style (B,C,1,S) layout rewrite
              (estimated 1-2 additional weeks).
  deferred: --coreml flag on main serve (wired to coreml-serve sibling),
            grammar/FSM masking on CoreML logits, temperature sampling,
            SSE streaming, prompt cache persistence, full power benchmark.
---

# PRD — Factory-wide CoreML / ANE inference primitive

## Scope reframing (2026-06-08)

This PRD is **factory infrastructure, not Pace-only**. The Pace
specialist is the first beneficiary and validation target, but every
artifact this PRD produces — `bake-lora`, `to-coreml`, `serve --coreml`,
the stateful KV-cache conversion pattern — is generic. Any model
trained through TinyGPT's pipeline (current Pace planner, future VLM
specialist, future xLAM-derived function-call specialist, any user-
trained model) gets CoreML / ANE inference for free once the
primitives ship.

What's Pace-specific:
- The acceptance benchmark (Qwen3-0.6B + v6 LoRA fm-fixture pass rate)
- The first end-to-end smoke target

What's factory-wide:
- The `bake-lora` CLI (any base + LoRA combination)
- The `to-coreml` exporter (any safetensors directory)
- The CoreML stateful conversion pattern (any Qwen3-family model;
  generalizable to other GQA architectures)
- The `serve --coreml` runtime (any `.mlpackage`)

Per `[[feedback_leverage_first]]`: this primitive multiplies every
future specialist. Don't build it Pace-specific.

## 2026-06-07 defer note

This is real work, but not a same-session PRD. The spec estimates 1-2 weeks
and requires:
- bake-LoRA into base weights
- CoreML conversion validation for Qwen3 architecture
- CoreML KV-cache/stateful decode design
- `serve --coreml`
- speed, power, and quality benchmarks

Per repo rules, this should not be kicked off without explicit owner approval
for heavy conversion/benchmark loops.

## Goal

Ship factory-wide CoreML / ANE inference for TinyGPT-trained models.
Validated end-to-end on the Pace specialist (Qwen3-0.6B + v6 LoRA).
Target on M-class silicon:

- **5-10× faster decode** (ANE is purpose-built for INT8/FP16 matmul)
- **Near-zero power draw** (the GPU runs hot at ~25W; ANE peaks at ~3W)
- **No thermal throttling** even on sustained inference
- **Closed-lid responsiveness** — Pace and any other consumer stays
  responsive even when the Mac is in low-power state

This is the **defensible moat**: small specialists running on iPhone-
class silicon at zero power. No cloud LLM can compete with that on a
Mac. Every specialist TinyGPT trains becomes structurally better than
"tiny GPT-4 mini API" — the moat compounds across the model catalog,
not just one model.

## Scope — in

### 1. Bake-LoRA-then-convert pipeline

CoreML doesn't natively support runtime LoRA composition. Solution:
fold the LoRA into the base weights before conversion. We already have
`tinygpt merge` for TIES/DARE; extend with a `bake-lora` mode (or use
existing `merge` with the lora as a second model in linear mode).

```
tinygpt bake-lora <base.tinygpt-or-hf> <lora> --out merged.safetensors
```

### 2. Convert to CoreML

We have `tinygpt to-coreml` (#205). Validate it works on Qwen3-0.6B+LoRA.
Output: `pace-planner-v6.mlpackage` (CoreML MLProgram format).

Likely refinements:
- Verify ANE compute units (CoreML config: `MLComputeUnits.cpuAndNeuralEngine`)
- Handle stateful KV cache (CoreML 8+ feature)
- Validate token-by-token decoding works (vs full-prompt-at-once)

### 3. `tinygpt serve --coreml` path

New flag `--coreml <path.mlpackage>`. When set:
- Load via `MLModel(contentsOf:)` instead of MLX
- Configure compute units = cpu+NE
- Inference loop uses CoreML's `prediction(from:)` API
- Tokenizer + sampling stay the same
- Grammar enforcement applied to CoreML output logits

### 4. Benchmark + acceptance

Measure on M5 Pro (Sarthak's machine):
- Tokens/sec at decode
- TTFW
- Peak power (via `powermetrics`)
- RAM
- Quality parity vs MLX path (should be identical up to fp16 noise)

## 2026-06-08 — push for actual ANE execution

**Owner directive**: ship `serve --coreml` on GPU+CPU is NOT acceptable
as the final state. The user explicitly authorized pursuing actual ANE
execution, including research time to get ahead of Apple's published
patterns.

**The current blocker** (per M3 ship note): ANECCompile error -14 on
the full Qwen3-0.6B stateful graph. mlpackage runs on CPU+GPU only.

### Milestone 6 — ANE diagnostic ✅ SHIPPED 2026-06-08

**Findings doc**: `docs/learn/ane-research/m6-findings.md`

**Result**: Bisect on N ∈ {1, 2, 3, 4, 28} of stateful Qwen3-0.6B
showed the ANE runtime can run *exactly* N=1, and crashes with
SIGTRAP at N≥2. GPU CPU+GPU predict works on the same graph. The
binding constraint is **multi-layer consolidated-state forward**,
not graph size or op shapes.

This redirects the next step: **M8 layer-chunked conversion is now
the indicated path**, not M7 layout port. Each Qwen3 block becomes
its own 1-layer mlpackage; Swift orchestrates the 28-block decode.

Artifacts:
- `scripts/ane/m6_layer_bisect.py` — reproducer
- `~/.cache/tinygpt/ane/bisect-n1.mlpackage` (ANE-working)
- `~/.cache/tinygpt/ane/bisect-n2.mlpackage` (first failing)
- `docs/learn/ane-research/m6-findings.md` — full writeup

Hypotheses originally listed for M6 (kept for context):

1. **Graph size threshold**: convert N-layer Qwen3 for N in {1, 4, 8,
   14, 20, 28}. Find the layer count at which ANE compile starts
   failing. This tells us whether it's a graph-size limit or a
   structural-op issue.
2. **Op ablation**: build a stripped Qwen3 (no QK-Norm, no SwiGLU,
   no GQA) and incrementally add features. Find the op that breaks
   ANE compile.
3. **Reference comparison**: run Apple's own `ml-ane-transformers`
   LLaMA-2-7B conversion on the same M5 Pro. Does ANE compile work
   AT ALL on this machine? If not, we have an environment issue. If
   yes, we have a Qwen3-specific issue.
4. **Spec-mlmodel inspection**: dump the offending mlmodel proto.
   Find which op gets the bad device assignment. coremltools has
   `ct.utils.evaluate_classifier_with_dataframe` and similar
   introspection helpers.
5. **State-slot count**: even after consolidating to 2 state slots,
   verify ANE didn't internally re-explode them during compile.

**Output**: a one-page diagnostic doc — exactly what's blocking ANE
compile, with bisect evidence.

### Milestone 7 — ml-ane-transformers (B,C,1,S) layout port

Apple's published reference: <https://github.com/apple/ml-ane-transformers>.
Key insight: ANE wants `(B, C, 1, S)` tensor layout (image-like, 4D)
instead of `(B, S, C)` (sequence-first). With this layout, attention,
RMSNorm, and MLP all map to ANE-native ops (1D Conv, batch matmul,
softmax) at the right shapes.

**Port plan**:
- Rewrite `scripts/ane/qwen3_to_coreml.py` to use Apple's layout
  conventions
- Port: `multi_head_attention` (the AttentionFunction in their repo),
  `LayerNormANE`, `Conv2d-based Linear`, `ane_gelu` / `ane_silu`
- Adapt for Qwen3 specifics: GQA, QK-Norm, RoPE@1e6, head_dim=128,
  tied embeddings
- Convert + ANE-compile incrementally — fail fast on the first op
  that doesn't ANE-compile, with concrete evidence
- This is the path Apple's own LLaMA, Mistral, and Stable Diffusion
  examples use to actually hit ANE on iPhone/Mac

**Estimated effort**: 1-2 weeks of focused work. Real engineering,
not a tweak.

### Milestone 8 — Beyond-reference research (open scope, time-boxed)

After M6+M7 land, the owner authorized exploratory research to find
ANE wins Apple hasn't published. Concrete arcs to consider:

1. **Hybrid execution**: dispatch attention QKV matmuls to ANE,
   keep softmax+RoPE on GPU. ANE excels at large matmuls; GPU
   excels at small irregular ops. Existing CoreML schedulers
   already do some of this — research how much we can force.
2. **Layer-chunked ANE**: convert each Qwen3 block as a separate
   mlpackage, orchestrate state passing in Swift. Smaller graphs
   each compile to ANE cleanly. (Apple's ml-ane-transformers does
   this for big LLaMA — well-precedented.)
3. **ANE-native INT4/INT8**: ANE has dedicated INT8 paths.
   coremltools 9 supports activation+weight INT8 quantization via
   `ct.optimize.coreml.linear_quantize_activations`. Try aggressive
   ANE-only INT4/INT8 fusion that GPU can't match.
4. **MPSGraph bridge**: Metal Performance Shaders Graph (Apple's
   lower-level ANE/GPU dispatch). Some ops not ANE-compilable via
   coremltools ARE dispatchable to ANE via MPSGraph directly. Worth
   investigating for the long-tail ops.
5. **Sparse attention on ANE**: Apple's recent papers (FastVLM,
   MobileCLIP) hint at sparse patterns ANE accelerates well.
   Speculative.
6. **Halved-depth distillation**: if (1)-(5) all fail, distill
   Qwen3-0.6B → 14-layer student that DOES fit ANE. Quality cost
   we'd measure on fm-fixtures.

**Output**: research notes in `docs/learn/ane-research/`. If any
arc produces a working ANE-execution path that beats the M7 port,
ship it. Otherwise: ship M7 result, archive learnings.

**Time-box**: 2 weeks of exploration after M7 lands. If no
breakthrough by then, declare M8 done with whatever wins we found.

## Scope — out

- LoRA hot-swap (compose at runtime) — needs CoreML 9 / future
- Multi-LoRA (route between specialists)
- Streaming via CoreML (use the same OpenAI-compat HTTP shape; chunking
  happens on the host side)

## Acceptance

1. `tinygpt serve --coreml pace-planner-v6.mlpackage --grammar ... --port 8765` boots
2. Smoke request returns valid JSON via fm-fixture eval
3. **Decode speed ≥3× faster** than current MLX serve (target: 200+ tok/s on M-series, vs ~80 currently)
4. **Power draw ≤5W** sustained (vs ~20W on MLX GPU path)
5. **Eval score same or within 1 fixture** of MLX path on fm-fixtures
6. Build clean; doesn't regress MLX path

## Files involved

| File | Change |
|---|---|
| `native-mac/Sources/TinyGPT/ToCoreML.swift` | May need refinement for Qwen3 architecture (check head_dim, GQA, QK-Norm — all wired in our MLX path) |
| `native-mac/Sources/TinyGPTModel/ANEInference.swift` (exists!) | Stub for ANE inference — extend if needed |
| `native-mac/Sources/TinyGPTServe/Serve.swift` | Add `--coreml` flag + dispatch |
| `scripts/bake-pace-lora-into-base.sh` (new) | Pipeline: merge LoRA + convert to CoreML |
| `native-mac/Sources/TinyGPTModel/Merge.swift` | Add `--mode lora-into-base` mode |

## Estimated effort

**~1-2 weeks.** This is the biggest single PRD on the queue:
- 2-3 days: bake-lora pipeline + validate CoreML conversion works on Qwen3
- 2-3 days: KV cache handling in CoreML (the trickiest part)
- 2-3 days: serve integration + benchmarking
- 1-2 days: power + thermal validation, edge cases

If you assign this, expect it to span across multiple days, not a single session.

## Why this is the moat

Look at the speed comparison after the easy wires are in:
| State | TTFW | Decode | RAM | Power |
|---|---|---|---|---|
| Today (MLX, fp16) | ~1000ms | ~80 tok/s | ~3 GB | ~25W |
| + prompt cache (#260) | ~50ms | ~80 tok/s | ~3 GB | ~25W |
| + Q4 (#261) | ~50ms | ~100 tok/s | ~600 MB | ~20W |
| + MLX compile + spec dec (#262) | ~50ms | ~300 tok/s | ~700 MB | ~25W |
| **+ ANE (this PRD)** | **~5ms** | **~600 tok/s** | **~600 MB** | **~3W** |

**ANE turns "competitive" into "structurally better than anything cloud can ship to a Mac."** That's the headline you can't get any other way.

## Won't conflict with other elves

- Touches ToCoreML.swift, ANEInference.swift, Merge.swift (and new files)
- Serve.swift gets a small flag addition but most logic in CoreML side
- Independent from prompt-cache (#260) and Q4 (#261) wire-ins
