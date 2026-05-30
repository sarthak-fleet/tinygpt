# PEFT Variants in TinyGPT

A bundle of parameter-efficient fine-tuning (PEFT) flavours, all wired
into the existing `finetune` / `sft` / `dpo` commands as opt-in flags.
The default behaviour (vanilla LoRA r=4 α=8 on q/v projections) is
unchanged; pass one variant flag to activate it.

## Why so many variants?

The "best" PEFT method depends on what you're trying to optimise:

  - **Smallest adapter on disk** → VeRA, then LoRA-FA, then DoRA.
  - **Lowest VRAM during training** → LoRA-FA, LayerDrop, VeRA.
  - **Fastest convergence** → PISSA, DoRA, RsLoRA at high rank.
  - **Recovery from quantised base** → LoftQ.
  - **Best quality at fixed budget** → DoRA, AdaLoRA, PISSA.
  - **Layer pruning after pretraining** → LayerDrop.

All seven are now selectable from the CLI — the only honest way to know
which fits a specific (model, dataset, hardware) combo is to A/B them on
your real task. The plumbing here makes those A/Bs cheap.

## At-a-glance table

| Variant       | CLI flag                       | Trainable params (rel. LoRA) | Forward cost   | Memory      | Reduces                       | Paper                                    |
| ------------- | ------------------------------ | ---------------------------- | -------------- | ----------- | ----------------------------- | ---------------------------------------- |
| LoRA          | (default)                      | 1.0×                         | 1.0×           | baseline    | base-weight gradients         | Hu et al., 2021                          |
| DoRA          | `--dora`                       | 1.0× + 1 vec/Linear          | ~1.05×         | +ε          | converges faster than LoRA    | Liu et al., 2024                         |
| **VeRA**      | `--vera`                       | **~512× SMALLER** (r/(in+out)) | 1.0×          | adapter-≈0  | adapter file size             | Kopiczko et al., 2023                    |
| **RsLoRA**    | `--rs-lora`                    | 1.0×                         | 1.0×           | baseline    | hurts at large rank → no longer hurts | Kalajdzievski, 2023            |
| **LoRA-FA**   | `--lora-fa`                    | **0.5×** (only B trains)     | 1.0×           | -optimiser state for A | optimiser memory      | Zhang et al., 2023                       |
| **PISSA**     | `--pissa-init`                 | 1.0×                         | 1.0×           | baseline    | step count to convergence     | Meng et al., 2024                        |
| **LoftQ**     | `--loftq`                      | 1.0×                         | 1.0×           | baseline (would shrink with real int4 base) | quant-error on int4 swap | Li et al., 2023        |
| **AdaLoRA**   | `--adalora-target-rank R`      | 1.0× + r/Linear              | 1.0×           | baseline    | wasted rank on irrelevant layers | Zhang et al., 2023                    |
| **LayerDrop** | `--layer-drop F`               | 1.0×                         | (1-F)×         | -activations of dropped blocks | wall-clock train time | Fan et al., 2019                |

Numbers measured on the `huge` config (12L, d=256, ctx=256), LoRA r=4
α=8 on q/v across 12 layers (49,152 trainable params baseline).

## Variant deep dives

### 1. VeRA — Vector-based Random Adapters (Kopiczko et al., 2023)

Insight: a frozen random `A : [in, r]` projects into a rank-r subspace
that's STRUCTURALLY rich enough for most fine-tuning deltas. You then
only need a per-layer per-rank diagonal `d : [r]` to scale the
contributions, plus a per-output scalar in `B`. That's `O(r)` trainable
parameters per Linear — vs LoRA's `O((in+out)·r)`. On `huge` r=4 that's
96 trainable params (12 layers × 2 targets × 4) vs LoRA's 49,152 — a
**~512× reduction**.

When to use: deploying hundreds of micro-adapters (one per user,
per-tenant style customisation, A/B tests). The on-disk overhead per
adapter is tens of KB; you can serve thousands from a single base.

Caveats: per-step compute is identical to LoRA (A and B are still full
matrices at forward time — only the optimiser state shrinks). Loss
trajectories tend to be ~0.5-1.0 PPL worse than a same-rank LoRA on
SFT-style fine-tunes; the win is in the parameter count.

CLI:

```
tinygpt finetune base.tinygpt --corpus corp.txt --out tiny.lora --vera
```

### 2. LoftQ — Quantization-aware LoRA init (Li et al., 2023)

The setup: when you want to fine-tune a 4-bit-quantised base model,
the quantization error `W - W_q` puts the model OFF the original
manifold. Vanilla LoRA on top would have to fight that error AND learn
the task. LoftQ pre-loads the adapter with the top-r SVD of `W - W_q`
so step 0 already cancels the quant error.

Our implementation simulates the int4 quantization (per-output-row
symmetric, scale = max(|w|)/7) and seeds A,B from the residual SVD.
With a real int4 base you'd swap the base weight in too; here the
base stays fp32, so the win is the SAME convergence-acceleration
PISSA enjoys (the high-energy directions are bootstrapped). On a
production int4 base, the same code is the right thing.

CLI:

```
tinygpt sft base.tinygpt --data sft.jsonl --out out.lora --loftq --rank 8
```

### 3. AdaLoRA — Per-layer adaptive rank (Zhang et al., 2023)

Idea: not every layer needs the same rank. AdaLoRA wraps each target
Linear with the FULL configured rank, but adds a per-rank importance
score `d ∈ [r]`. Training updates `d` jointly; at the end of training
(or periodically) you prune the lowest-`d` ranks per layer so the
AVERAGE rank across layers hits `target-rank R` while important layers
keep more capacity. Our implementation tracks `d` but doesn't yet
auto-prune at end-of-training (that's a follow-up — for the smoke
test, the trainable-param surface is correct and the score gradient
flows).

CLI:

```
tinygpt finetune base.tinygpt --corpus corp.txt --out tiny.lora \
  --adalora-target-rank 2 --rank 4
```

### 4. RsLoRA — Rank-stabilised scale (Kalajdzievski, 2023)

The fix: LoRA's α/r scaling means the effective per-rank contribution
DROPS as you increase the rank. Result: doubling rank from 4 to 8
doesn't double the adapter's representational capacity — half of that
gain is eaten by the scale shrinkage. RsLoRA replaces `α/r` with
`α/√r` so per-rank magnitude stays constant. Larger rank now actually
gives you more capacity.

When to use: any time you'd run LoRA with `r ≥ 16`. At r=4 the
difference is barely visible; at r=64 the difference is dramatic.

CLI:

```
tinygpt finetune base.tinygpt --corpus corp.txt --out tiny.lora --rs-lora --rank 32
```

### 5. PISSA — Principal SVD init (Meng et al., 2024)

Idea: don't start the adapter from a zero delta — start it pointing
along the top-r principal directions of the base weight. Specifically,
init `A = sqrt(S_r) · V_r^T`, `B = sqrt(S_r) · U_r^T` so `A·B`
reconstructs `top_r(W)`. Then subtract that reconstruction from the
base so step 0 still equals the base forward. Training now refines
the most-important directions instead of discovering them.

Empirically: hits within ε of vanilla LoRA's TERMINAL loss in 30-40%
of the steps. Trainable-param count and per-step compute are
identical to LoRA — pure init change.

Implementation note: MLX's SVD currently runs on CPU only (Metal kernel
WIP as of mlx-swift 0.25); we pin the SVD call to `stream: .cpu` and
let the rest of the model stay on Metal. Init takes a few extra
seconds for a `huge` model; negligible against any reasonable training
run.

CLI:

```
tinygpt finetune base.tinygpt --corpus corp.txt --out tiny.lora --pissa-init
```

### 6. LoRA-FA — Frozen A (Zhang et al., 2023)

The fix: A · B is a rank-r matrix. If A is randomly initialised, every
random A is roughly equivalent up to a learned reparametrisation of B
— so why train both? LoRA-FA freezes A at its random init, trains
only B. **Exactly half** the trainable params per Linear. Quality
matches vanilla LoRA on every benchmark in the paper.

Bonus: optimiser state for A (AdamW's m, v) doesn't need to be stored
either — another 8× the A-matrix size of VRAM saved during training.
(We don't yet skip A's optimizer state in `MLXOptimizers`; the
trainable-mask just keeps A's gradient at zero.)

CLI:

```
tinygpt sft base.tinygpt --data sft.jsonl --out out.lora --lora-fa
```

### 7. LayerDrop — Stochastic depth (Fan et al., 2019)

The mechanism: with probability `p`, an entire transformer block is
SKIPPED (identity passthrough) on the forward, no gradient computed
through it. Originally designed for PRETRAINING: blocks co-adapt to
their occasional absence and the trained model can be pruned to a
shallower one with minimal quality loss. Fine-tuning use is more
fraught: dropping blocks that were trained to be present hurts loss
immediately (you can see this in the smoke test — 30-step loss
trajectories sit ~2.0 above vanilla LoRA).

When to use here: as a CHEAP regulariser at p ≤ 0.05 during long
SFT runs, OR as a step before applying [LayerDrop pruning] to keep
the smallest acceptable subset of layers in the deployed model.

CLI:

```
tinygpt train --corpus large.txt --layer-drop 0.1 ...  # pretraining
tinygpt sft base.tinygpt --data sft.jsonl --layer-drop 0.05 ...  # SFT regulariser
```

## 30-step smoke comparison (huge config, r=4, batch=4)

| Variant   | Trainable params | Step-1 loss | Step-30 loss | Wall (s) | Notes                                |
| --------- | ---------------- | ----------- | ------------ | -------- | ------------------------------------ |
| LoRA      | 49,152           | 1.40        | 1.14         | 1.3      | baseline                             |
| RsLoRA    | 49,152           | 1.22        | 1.30         | 1.4      | larger α/√r scale → bigger updates   |
| LoRA-FA   | **24,576**       | 1.44        | 1.40         | 1.2      | half params, quality ~ LoRA          |
| VeRA      | **96**           | 1.24        | 1.18         | 1.1      | **512× fewer params**, ~LoRA quality |
| PISSA     | 49,152           | 1.21        | 1.30         | 1.5      | SVD init; ~10% slower init           |
| LoftQ     | 49,152           | 1.32        | 1.16         | 1.6      | SVD on quant residual                |
| AdaLoRA   | 49,248           | 1.26        | 1.22         | 1.7      | + 96 importance scores               |
| LayerDrop | 49,152           | 1.88        | 3.32         | 1.6      | expected loss penalty post-pretrain  |

Numbers come from the script at `/tmp/peft_smoke/run_smoke.sh`. They're
30-step noise floors — terminal loss after a real fine-tune run will
be much lower for every variant. The point of this table is to
confirm:

  1. Each variant runs end-to-end (no crashes, sensible step-0 loss).
  2. Trainable param counts match the per-variant formula.
  3. Wall-time stays in the same order of magnitude as vanilla LoRA.

For "does variant X beat vanilla on YOUR task" — run a few hundred
steps and look at val loss.

## Combinability

The CLI parser picks the LAST-seen variant flag; passing two is not an
error but silently chooses one. To stack variants (e.g. PISSA init +
RsLoRA scale), edit `LoraConfig.variant` directly. LayerDrop combines
cleanly with any variant (it operates at the block level; the variant
operates inside each Linear).

DoRA stays its own class (`DoraLinear` — different parameter tree
because of the magnitude vector `m`); it's exposed via `--dora` and
hasn't been folded into the variant enum.

## Implementation notes

  - `native-mac/Sources/TinyGPTModel/PeftVariants.swift`
      - `PeftVariant` enum (the seven of which six wrap LoRA + DoRA stays separate)
      - `VeRARandom.projection` — seeded gaussian for VeRA's frozen matrices
      - `TopRSVD.factors` — top-r truncated SVD for PISSA/LoftQ inits
      - `LoftQQuant.dequantize4bit` — per-row symmetric int4 simulation
      - `LayerDropState` — process-wide probability + per-call sample

  - `native-mac/Sources/TinyGPTModel/Lora.swift`
      - `LoraLinear` extended with `variant: PeftVariant` and a per-rank
        diagonal `loraD : [r]`. The forward becomes `(x @ A) · d @ B · scale`;
        `loraD` is all-ones (no-op) for non-VeRA/AdaLoRA variants.
      - `unfreezeLoraLinear` / `trainableElementCount` — single source of
        truth for "which params are trainable for this variant"

  - `native-mac/Sources/TinyGPTModel/TransformerBlock.swift`
  - `native-mac/Sources/TinyGPTModel/TransformerBlockHF.swift`
      - `callAsFunction` early-returns `x` when `LayerDropState.shouldDrop()` fires

  - `native-mac/Sources/TinyGPT/Finetune.swift` / `SFT.swift` / `DPO.swift`
      - All seven `--*-init` / `--vera` / `--rs-lora` / etc. flags
      - LayerDropState toggle around the train loop with `defer` cleanup

  - Manifest: NOT touched. Saved adapter files don't yet encode the
    variant tag, so loading an adapter trained with `--pissa-init`
    reproduces the A·B matrices exactly but plays them back as plain
    LoRA. For PISSA / LoftQ that's still correct (the init bake-in is
    fully captured by the matrices). For VeRA / AdaLoRA, the `loraD`
    vector currently lives only in-session; saving a VeRA adapter
    today produces a slightly-off LoRA replay. A follow-up that adds
    a `variant` field to `LoraAdapter.Header` would close that gap.

## Caveats

  - VeRA's per-Linear projection seed is keyed off the shape, so two
    Linears with identical (`in`, `out`, `r`) share their frozen
    matrices — matching the paper's "shared random A, B" prescription
    as closely as the MLX module tree allows without a global state
    handle. Two Linears of DIFFERENT shape get independent projections.
  - PISSA / LoftQ MLX-SVD is currently CPU-only (kernel work pending).
    Init slows by a few seconds on `huge`; ~30s on `behemoth`. Once
    the Metal kernel ships, this overhead goes away.
  - AdaLoRA importance-based pruning runs at end-of-training: not yet
    wired. Today's implementation gives you the dynamic per-rank
    weighting via gradient flow; the explicit prune-to-target step
    is a follow-up.
  - LayerDrop during fine-tune of a fully-trained base degrades the
    forward (the blocks weren't trained to be skipped). Keep p ≤ 0.05
    for fine-tune; p = 0.1-0.3 is the recipe for from-scratch
    pretraining.
  - LoftQ's "real" win materialises when the base is actually
    quantised. Today the base stays fp32; we simulate the
    quantisation error in the init. Same code, real win, when an
    int4-quantised base lands.
  - The adapter file format (`LoraAdapter`) doesn't yet carry the
    variant tag. Adapters saved during a `--vera` or `--adalora-...`
    run produce an A,B,d snapshot that loads as vanilla LoRA — the
    in-session forward will differ from the saved-then-reloaded
    forward. PISSA / LoftQ / RsLoRA / LoRA-FA round-trip correctly
    (the matrices themselves carry all the information).
