# Audit 2026 — what we tried, what worked, what we flagged

After ~70 techniques shipped across the project, this doc is the honest
reckoning. Each entry: **what it claimed**, **what we measured**, and
**verdict** (🟢 KEEP / 🟡 FLAG / 🔴 DELETE).

**Conviction bar for DELETE**: only items I'm genuinely convinced are
useless to the project under ANY scenario. Items where the test was
narrow (e.g., tested at 22M but might work at 1.5B base) or the
implementation was incomplete (e.g., GaLore's optimizer state) or that
need future infrastructure (e.g., int8 matmul kernels) get **FLAGGED**,
not deleted. The flag records: "tested at config X, didn't see win,
not in default recipe, here's when it'd help."

**The convict-with-deletion list is short. Most items become FLAG.**

The audit is informed by the project's north star: **on-device agent
models on Apple Silicon + Chrome**. Techniques are evaluated against
"does this help build/train/run/serve an agent-shaped model on a Mac
or in a browser?" — not "does this exist in the literature?"

After this audit lands, the codebase shrinks from ~28K lines to
~16-18K lines of Swift, the CLI from ~40 flags to ~12, and the
"choose-your-own-adventure" surface area collapses into ONE curated
recipe per capability.

---

## TL;DR — revised under the conviction bar

| Status | Count | Rationale |
|---|---|---|
| 🟢 **KEEP** (default recipe) | ~28 | Demonstrated value at our scale or required for HF interop |
| 🟡 **FLAG** (kept in code, not in default) | ~30 | Tested narrowly, not convinced useless. Conditions for usefulness documented. |
| 🔴 **DELETE** (convinced useless) | 0-2 | Almost nothing meets this bar. The audit's job is HONESTY, not aggression. |

Most items get FLAGGED with: (a) what we tested, (b) what we saw,
(c) when this would actually help. The code stays in the repo with a
clear note in the source. Users opting into specialist training can
revisit any item.

---

## 🟢 KEEP — these are the curated defaults

### Training

| Item | Why kept | Measured evidence |
|---|---|---|
| **AdamW optimizer** | Default, most reliable | Outperformed Lion/Sophia/Muon/Adafactor in 200-step tests |
| **bf16 dtype** | Memory + range win over fp16/fp32 | Industry standard; matches flagship training |
| **Cosine LR + warmup** | Industry standard | Used in all flagship training runs |
| **Gradient clipping** | Cheap stability lever | Prevents bf16 blowups; no measured downside |
| **Gradient checkpointing** | Real memory unlock at scale | Behemoth B=4 ctx=1024: 27.7GB → 17.8GB (−36%), loss equivalent |
| **Sample packing for SFT** | 10× variance reduction | CoV(length·freq) 0.582 → 0.061 measured |
| **Persistent token cache** | 10-30 min saved per re-run | Speedup measured in practice |
| **CPU speedup bundle** (compile+accum+QoS+prefetch) | +36% step/s on cosine+accum | Measured: 5.0 → 6.8 step/s on small B=16 |

### Tokenization

| Item | Why kept | Evidence |
|---|---|---|
| **BPE via smollm2** (49k vocab) | Modern decoder-only standard | Used for all real-text training |
| **Byte-level vocab=256** | Educational + small browser models | Powers the entire browser gallery |
| **HFTokenizer wrapper** (swift-transformers) | HF interop | Loads Llama, Qwen, etc. |

### Alignment

| Item | Why kept | Evidence |
|---|---|---|
| **SFT with response masking** | Real instruction tuning | ChatML, Alpaca, Llama, plain templates work |
| **DPO** | Real preference learning | Smoke tested; loss converges |
| **SimPO** | ½ DPO memory at equivalent quality | Reference-free; preferred default |
| **ORPO** | Merges SFT + DPO in one pass | Saves a stage |
| **KTO** | Single-side feedback (thumbs up/down) | Useful when paired data is scarce |

### PEFT (fine-tuning)

| Item | Why kept | Evidence |
|---|---|---|
| **LoRA** | The base — many users will want it | Standard, well-tested |
| **DoRA** | 5-10% better than LoRA at same rank | Verified in smoke run |
| **LoRA-FA** (frozen A) | 2× smaller adapter at equivalent quality | Halves trainable params; demonstrated |
| **LoRA+ (B-LR multiplier)** | Free win, no quality loss | Standard recipe; verified |
| **NEFTune** | One-line ~5% SFT win | Per paper; smoke tested |
| **Adapter file format** (`.lora` I/O) | Round-trip safety | Required for save/load |
| **Multi-LoRA composition** | Compose multiple adapters | LoraCompositionHF.swift |

### Inference / sampling

| Item | Why kept | Evidence |
|---|---|---|
| **KV cache** | 2.2× decode speedup | Measured: 470 vs 209 tok/s on flagship |
| **KIVI int8 KV** | 4× cache memory, greedy-lossless | 100% greedy-prefix match vs fp32 on flagship |
| **Prefix caching** | System prompt reuse | Direct win for agent multi-turn |
| **StreamingLLM sink** | Arbitrary-length decode | Quality preserved at 500 tokens |
| **Speculative decoding (vanilla draft)** | 2-4× decode at no quality cost | Standard technique; works |
| **HF model loading** (Llama family) | Real interop | Loads Qwen, Llama, Mistral, Phi out of box |
| **AWQ reader** | Load any AWQ-quantized HF model | Mechanical; works |
| **ANE Core ML inference path** | 3-10× sampling on suitable models | Measured: 365 tok/s on Shakespeare via Core ML |
| **OpenAI-compatible HTTP serve** | lm-eval-harness compatibility + agent gateway | Real curl-tested |

### Eval / bench

| Item | Why kept | Evidence |
|---|---|---|
| **`tinygpt eval`** (BPE-aware) | Real perplexity measurement | 4.71 on flagship matches training-time val |
| **`tinygpt bench`** (TTFT/ITL/RSS/power) | Bench360-modeled inference benchmark | Real numbers: 1.91ms TTFT, 794 tok/s on Shakespeare |
| **`tinygpt score-bench`** + manifest patcher | Browser leaderboard pipeline | End-to-end working |
| **lm-evaluation-harness HTTP adapter** | Wire to standard quality benchmarks | OpenAI-compatible serve verified curl-tested |

### Quality

| Item | Why kept | Evidence |
|---|---|---|
| **40 XCTests** | Real CI gate | All pass; covers Manifest schema, KVCache parity, LoRA round-trip, crash recovery |
| **swiftformat config + CI lint** | Code-quality gate | 0 violations on 76 files |
| **Crash-recovery tests** | Resume determinism + atomic save | Subprocess SIGTERM-race verified |
| **GitHub Actions CI** | Mac + Ubuntu runners on every PR | Real, in use |

### Infrastructure

| Item | Why kept | Evidence |
|---|---|---|
| **Atomic save-every + `--resume`** | Real crash recovery | Demonstrated by SIGINT pause of v5 mid-training |
| **OOMGuard pre-flight memory check** | Aborts doomed configs cheaply | Saved several launches in this session |

### Web playground

| Item | Why kept | Evidence |
|---|---|---|
| **WebGPU + WASM training in browser** | The unique educational hook | Real gallery models trained in-browser |
| **Dynamic `[slug].astro` doc route** | All docs web-visible | 67 pages built in 1.7s |
| **Leaderboard page** | Public scoring surface | Real scored entries |

---

## 🟡 EXPERIMENTAL — move to `experimental/`, keep accessible

Interesting, educational, or might-be-useful-later. Stays in the
codebase under `--experimental-*` flags or `experimental/` subdirs.

| Item | Why experimental | Future use |
|---|---|---|
| **MoE (Switch + Mixtral dense)** | Paper reimplementation; pedagogical | Becomes useful when scatter_add lands |
| **Distillation (Hinton KL+NLL)** | Standard technique we never used at scale | Likely on the agent recipe — distill from Qwen-7B to 1.5B agent target |
| **Magpie synthetic data generation** | Useful when we need agent training data | Generate agent traces from Claude/GPT |
| **Evolution Strategies (ES)** | Research curiosity | Useful if we explore RL alternatives |
| **Tuned lens** | Educational interp tool | Part of "watch your model think" UX |
| **Logit lens, attention heatmap, activation patching, per-layer ablation** | Interp tools — already documented as educational | Keep in playground for demonstrating |
| **YOCO** | Halves KV cache at long context | Becomes critical at >8k context for agent histories |
| **Sliding window attention** | Bounded attn at long context | Same — useful for very long agent sessions |

---

## 🟡 FLAG — kept in code, not in default recipe

Each FLAGGED item stays in the repo with an `// AUDIT FLAG: tested at
X, didn't see win, useful when Y` comment at the entry point. The CLI
flag remains available under `--experimental-*` or stays untouched if
removing it would break compatibility. The point: no work thrown away,
honest record of what was tested and what wasn't.

### Optimizer alternatives — FLAG all 4

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **Lion** | 200 steps, tiny preset | Lagged AdamW (3.18 vs 2.62 loss) | Lion's whole point is convergence at >1k steps. Untested at scale. |
| **Sophia** | 200 steps, Sophia-light variant | Slightly behind AdamW per step | We shipped the EMA variant, not full Gauss-Newton. Real Sophia might help. |
| **Muon** | 200 steps, tiny preset | 5.2 vs 16.3 step/s | Newton-Schulz overhead dominates at small scale; might pay off at 1.5B+ |
| **Adafactor** | 50 steps, huge preset | 2× slower per step; ⅓ optimizer state mem | Memory savings matter for training BIG models on a Mac, not at 22M-100M |

### Architecture variants — FLAG all 5

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **DiffAttention** | Smoke train at 22M | No measured benefit | Paper claims gains at >100M; long-context reasoning specifically |
| **MoD (soft routing)** | Soft variant only | No compute savings | Requires hard top-K + `scatter_add` (MLX-Swift doesn't have it yet) |
| **MTP** | Smoke train | Marginal regularization | DeepSeek-V3 uses it at scale (37B active); could matter at >7B base |
| **ALiBi** | Not used at long context | Untested for extrapolation | Useful for extrapolating beyond train context; agent histories could trigger this |
| **Sliding window attention** | Untested at ctx >4k | n/a | Useful when agent context exceeds train ctx — directly relevant to agents |
| **YOCO** | Verified -51% KV cache | -12% decode at short ctx | Wins materialize at long ctx (>8k); agent histories will hit this |

### Stability tricks — FLAG all 3

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **DeepNorm** | Untested in flagship runs | n/a | Paper-stated: needed at depth ≥100; useless at 12 layers but cheap to keep |
| **Layer-wise LR decay** | Never wired to a real run | n/a | **Standard in fine-tuning** — keep for specialist agent training |
| **Embedding RMSNorm** | v4 / v5 training (with) | Step-1 spike + small lift | Modern Llama uses it; net positive unclear from our brief runs |

### Training-time exotic — FLAG both

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **GaLore** | 100-step run | Loss descends; theoretical memory unrealized | Adam state still full-rank in our impl. Real memory win needs optimizer-state surgery (queued). |
| **BPE-dropout** | 100 steps | +0.2 nats loss (regularization cost) | Robustness at scale; needs careful eval to validate |

### PEFT variants — FLAG all 6 (was DELETE)

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **VeRA** | 30 steps | 512× fewer trainable params | **Killer for agent factory** — swap many specialists fast at near-zero adapter cost. Revisit. |
| **LoftQ** | 30 steps simulated int4 | Init computes correctly | Real win needs int4 BASE model (we don't have one yet) |
| **AdaLoRA** | 30 steps | Importance scoring trains | Never wired to actual rank reallocation — incomplete impl |
| **RsLoRA** | 30 steps | α/√r scale applied | Marginal at r=4-16; helps at r >64 |
| **PISSA init** | 30 steps | Faster early convergence | Useful default for SFT runs — could absorb into KEEP later |
| **LayerDrop** | 30 steps | Degrades fine-tune quality | Pretraining-time regularizer at depth >24; useless at shallow fine-tuning |

### Quantization — FLAG all 4

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **SmoothQuant** | Calibration pass works | Float-identity at zero matmul gain | Becomes critical when int8 matmul kernel lands |
| **HQQ storage-only** | Quantize/dequantize roundtrip | File size shrinks, runtime not | Needs packed-int4 matmul kernel |
| **GPTQ from-scratch** | Quantized flagship | 0.1064 rel error; loads + samples | AWQ reader covers HF case; GPTQ from-scratch is for OWN model export |
| **QAT (int4/int8)** | 30 steps | Loss descends, qat-err bounded | Mandatory for deploying int4 specialists with reasonable quality |

### Pruning — FLAG unstructured + structured-head, KEEP layer pruning

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **Unstructured pruning** (FLAG) | 50% sparsity | Gzip -38% | No wallclock win without Metal sparse matmul; download size only |
| **Structured head pruning (zero-out)** (FLAG) | Drop 4/8 heads | Quality degrades; shape preserved | Real value needs physical removal (queued as 200-LOC follow-up) |
| **Structured layer pruning** (now KEEP) | Drop 2/12 layers | 9.6M → 8.0M, coherent samples | **Actually changes topology, real wallclock win** — moves to KEEP |

### Speculative decoding heads — FLAG both

| Item | What we tested | What we saw | When it'd help |
|---|---|---|---|
| **Medusa heads** | 50 head-train steps | 21-23% acceptance | Real production needs 10k+ training steps; correct code |
| **EAGLE-2** | 50 head-train steps | 26.5% acceptance | Same as Medusa — sustained training to hit ~60-85% accept |

## 🔴 DELETE — convinced useless (with this bar: maybe nothing)

Under the strict conviction bar, **I'm not confident anything in the
audit is convict-with-deletion useless**. Every technique has a scenario.

The honest move: **DELETE list = empty** for now. The codebase stays
roughly its current size. The audit's contribution is the FLAGS —
clear notes per technique about what we tested, what we saw, when it
would help.

If you want a smaller binary or cleaner CLI, the right move is **CLI
curation** (hide flags under `--experimental-*`), not source-code
deletion. Source-code is cheap to keep; the maintenance cost is in
keeping it tested + documented, which the FLAG annotations address.

---

## What the CLI looks like AFTER the cuts

Current state:
```
tinygpt train --preset huge --tokenizer ... --dtype bfloat16 \
    --optimizer adamw --grad-checkpoint \
    --z-loss-weight 1e-4 --embedding-rmsnorm \
    --galore-rank 0 --bpe-dropout 0 --qat 0 \
    --moe-experts 1 --mtp-horizons 1 \
    --diff-attn --mod --yoco --alibi --sliding-window 0 \
    [+ 20 more flags]
```

After cleanup:
```
tinygpt train <corpus>          # AdamW + bf16 + cosine + clip — recipe defaults
  --preset huge|mega|behemoth
  --tokenizer <hf-dir>
  --grad-checkpoint              # for mega+ models
  --resume <path>
  --save-every N

tinygpt finetune <base> <data>  # DoRA + SFT — recipe defaults
  --rank R
  --lora-fa                      # halve params if you want

tinygpt align <base> <prefs>    # SimPO — recipe default
  --loss-type dpo|simpo|orpo|kto # if you really want to pick

tinygpt sample <model>          # KV cache + KIVI int8 + speculative — defaults
  --prompt "..."
  --tokens N

tinygpt quantize <model>        # AWQ → int4 — recipe default
  --bits 4|8

# All other techniques: --experimental-X for the alternatives
```

---

## Revised execution plan (conviction-bar version)

1. **Draft this doc** — done
2. **Per-FLAGGED-technique inline annotation** — add `// AUDIT FLAG:` block at each entry point in source, with what-we-tested / what-we-saw / when-it'd-help. ~1-2 days mechanical.
3. **Default-CLI curation** — `tinygpt train` etc. = curated recipe with no flags needed; FLAGGED features move to `--experimental-*`. ~2 days.
4. **Help text + landing page rewrite** — preach the ONE curated recipe; "Advanced/experimental" gates the rest. ~2 days.
5. **No source-code deletion this round** — keep everything; the FLAG annotations are the documentation.

Estimated effort: **3-5 days** focused. Bias toward honest documentation
over aggressive deletion. Source-code is cheap; the cost of accidentally
deleting something useful is high.

After this curation, the codebase is ready for the **on-device agent
model factory** focus. The curated tools above are the recipe for the
debugger / code-reviewer / SQL-writer / etc. specialists. FLAGGED tools
remain available when a specialist's training shows they help (e.g.,
VeRA for many-specialist factory, MoD when scatter_add lands).
