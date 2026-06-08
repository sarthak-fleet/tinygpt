---
name: VLM M4 — Qwen3-VL implementation plan
status: drafted-2026-06-08-ready-for-elf
owner: unassigned (parallel-agent task — Swift + MLX-Swift)
created: 2026-06-08
priority: P0 — gates the entire VLM specialist arc
depends-on: VLM M1-M3 already shipped + scaffold types in Qwen3VLScaffold.swift
estimated-effort: 1-2 weeks of focused Swift + MLX work
---

# VLM M4 implementation plan — file-by-file

Context already shipped (M1-M3):
- `VisionEncoder.swift` — CLIP-style ViT (works against CLIP-ViT-L-14)
- `CrossModalProjection.swift` — LLaVA-1.5-style 2-layer MLP
- `TinyGPTModelVLM.swift` — front-prepend forward (LLaVA-1.5 convention)
- `Qwen3VLScaffold.swift` — value types for mRoPE, image-token
  replacement, deepstack injection

Decided 2026-06-08 (`factory-vision-m4-architecture-decision.md`):
**Option A — full Qwen3-VL port using UI-Venus-1.5-2B as base.**

## What UI-Venus actually looks like

Inspected `mlx-community/UI-Venus-1.5-2B-6bit` (also available
unquantized as `inclusionAI/UI-Venus-Ground-2B`). Weight tree:

```
language_model.lm_head.{weight, scales, biases}
language_model.model.embed_tokens.{weight, scales, biases}
language_model.model.layers.{0..27}.input_layernorm.weight
language_model.model.layers.{0..27}.self_attn.{q_proj, k_proj, v_proj, o_proj}.{weight, scales, biases}
language_model.model.layers.{0..27}.self_attn.{q_norm, k_norm}.weight
language_model.model.layers.{0..27}.mlp.{gate_proj, up_proj, down_proj}.{weight, scales, biases}
language_model.model.layers.{0..27}.post_attention_layernorm.weight
language_model.model.norm.weight

vision_tower.patch_embed.{...}
vision_tower.pos_embed
vision_tower.blocks.{0..23}.attn.qkv.{weight, bias}      # fused QKV
vision_tower.blocks.{0..23}.attn.proj.{weight, bias}
vision_tower.blocks.{0..23}.mlp.linear_fc{1,2}.{weight, bias}
vision_tower.blocks.{0..23}.norm{1,2}.{weight, bias}

vision_tower.merger.linear_fc{1,2}.{weight, bias}        # the main projector
vision_tower.merger.norm.{weight, bias}
vision_tower.deepstack_merger_list.{0..2}.linear_fc{1,2}.{weight, bias}
vision_tower.deepstack_merger_list.{0..2}.norm.{weight, bias}
```

Three architectural surprises vs LLaVA:

1. **Fused QKV** in vision blocks (one big weight tensor for q+k+v
   concat, not three separate projections). Our existing `CLIPAttention`
   in VisionEncoder.swift has separate q/k/v.

2. **deepstack_merger_list** with 3 entries — Qwen3-VL uses three
   separate projection MLPs to inject vision features at three
   different LLM depths. Per text_config:
   `deepstack_visual_indexes = [5, 11, 17]`.

3. **Quantized weights** (in the MLX version): each `weight` has
   sibling `scales` and `biases` tensors. Either dequant before
   load (like `scripts/ane/dequant_mlx4bit.py`) or use the
   non-quantized HF version.

## M4 implementation breakdown

### M4.1 — HF VLM loader (~2-3 days)

New `native-mac/Sources/TinyGPTModel/HFVLMLoader.swift`:

```swift
public enum HFVLMLoader {
    public static func load(hfDir: URL) async throws -> (TinyGPTModelVLM, Qwen3VLConfig) {
        // 1. Parse config.json. Detect "Qwen3VLForConditionalGeneration".
        // 2. Inspect vision_config:
        //    - depth, hidden_size, patch_size, spatial_merge_size,
        //    - out_hidden_size (must == text_config.hidden_size)
        //    - deepstack_visual_indexes
        // 3. Inspect text_config:
        //    - rope_scaling.mrope_section
        //    - hidden_size, num_hidden_layers, num_attention_heads,
        //      num_key_value_heads, head_dim, intermediate_size,
        //      rope_theta, rms_norm_eps, tie_word_embeddings
        // 4. Walk *.safetensors, route tensors:
        //    - vision_tower.* → Qwen3VLVisionTower
        //    - vision_tower.merger.* → CrossModalProjection (existing)
        //    - vision_tower.deepstack_merger_list.* → [CrossModalProjection] × 3
        //    - language_model.* → Qwen3HFModel (existing path)
        // 5. Return assembled TinyGPTModelVLM.
    }
}
```

Key risk: the vision tower's fused QKV requires a new attention class
distinct from `CLIPAttention`. Solution: add `Qwen3VLVisionAttention`
that unfolds the fused weight into separate q/k/v projections (or
keeps it fused — MLX matmul doesn't care).

### M4.2 — Multimodal RoPE (~1-2 days)

Per Qwen3-VL: positions are 3-tuples `(time, height, width)` for
image tokens, scalar for text tokens. `mrope_section` = `[24, 20, 20]`
means: of the 64 head_dim/2 frequencies, the first 24 are used for
`time`, next 20 for `height`, last 20 for `width`. Interleaved.

Implementation: extend the LLM attention's RoPE step to accept a
`Qwen3VLMRoPEMetadata` (already in scaffold). For text-only prompts,
collapses to standard 1-D RoPE (the three sections all index the
same position). For prompts with images, image-token spans get
distinct (h, w) positions per patch.

New file or extension: `Qwen3VLMRoPE.swift` — single function
`applyMRoPE(q, k, positions: Qwen3VLMRoPEMetadata, cos, sin)`.

### M4.3 — Image-token replacement (~1 day)

Replace the LLaVA "prepend vision tokens" pattern with Qwen3-VL's
inline substitution:

```
text_tokens = [t0, t1, IMG, IMG, ..., IMG, t100, t101, ...]
embeds = [E[t0], E[t1], V[0], V[1], ..., V[N-1], E[t100], ...]
```

Where IMG = `image_token_id = 151655`. The text-side embedding is
overwritten by the projected vision tokens at IMG positions.

The scaffold already has `Qwen3VLImageTokenReplacementPlan` to
validate the alignment. M4.3 is the actual MLXArray scatter.

Forward: `TinyGPTModelVLM.forward(image, tokens)`:
1. Tokenize prompt (yields token IDs with N copies of 151655 where
   the image is supposed to be)
2. Compute embeddings as usual
3. Run vision encoder → patch features
4. Run merger projection → llm-dim vision tokens
5. Scatter vision tokens INTO embeddings at IMG positions

### M4.4 — Deepstack visual injection (~1 day)

After the merger produces the primary vision tokens, ALSO produce
three "deepstack" feature maps (taps at vision ViT layers [5, 11, 17]
of the 24-layer vision tower), each passed through its own
`deepstack_merger_list[i]` projection.

At LLM forward, AT the layers `[5, 11, 17]` of the 28-layer LLM,
the residual stream at image-token positions has the corresponding
deepstack feature ADDED (residual injection, not replacement).

New helper in `TinyGPTModelVLM.swift`:
```swift
func injectDeepstack(hidden: MLXArray, layer_idx: Int, deepstack_features: [MLXArray]) -> MLXArray {
    let depthstack_idx = [5: 0, 11: 1, 17: 2][layer_idx]
    if let i = depthstack_idx {
        // hidden[batch, img_token_positions, :] += deepstack_features[i]
        return hidden + scatter(deepstack_features[i], at: img_token_positions)
    }
    return hidden
}
```

### M4.5 — Parity tests (~1-2 days)

For each subcomponent, parity vs HF PyTorch reference on a tiny
prompt + image:

- `M4.1`: load + dump tensor stats → match HF (mean, std, max)
- `M4.2`: mRoPE for synthetic (h, w) positions → match HF output
- `M4.3`: post-embedding-substitution token stream → match HF
- `M4.4`: deepstack injection at layer 5 → match HF post-injection hidden state
- `M4.5 full`: top-1 logits on `[image, "describe this"]` vs HF

Script template: `scripts/vlm/qwen3vl_parity.py`. Borrow shape from
`scripts/vlm/clip_parity.py` (already shipped for M1).

### M4.6 — Smoke + acceptance (~half day)

`tinygpt qwen3vl-smoke <ui-venus-dir> <image.png>` — loads UI-Venus,
runs `forward(image, "What's on screen?")`, prints top-5 tokens.
Acceptance: top-1 is plausible (not gibberish) AND parity cos_sim
≥ 0.99 against HF PyTorch on the same input.

## Total estimate

| Sub | Effort |
|---|---|
| M4.1 HF VLM loader | 2-3 days |
| M4.2 Multimodal RoPE | 1-2 days |
| M4.3 Image-token replacement | 1 day |
| M4.4 Deepstack injection | 1 day |
| M4.5 Parity tests (per sub + end-to-end) | 1-2 days |
| M4.6 Smoke + acceptance | 0.5 day |
| **Total** | **6-9 days focused** |

## What's out of scope for M4

- VLM-specific tokenizer special handling (vision_start/end tokens)
- Multi-image prompts (single image for v1)
- Video / temporal_patch_size > 1
- Streaming output during VLM generation
- Pace integration handoff (M9-M10)
- Pace VLM SFT training (M5-M7)

These are real M5+ work but separable.

## Where to start

The cleanest first PR for the elf picking this up:

1. Read this doc + `Qwen3VLScaffold.swift` + the existing
   `TinyGPTModelVLM.swift`
2. Decide: dequant UI-Venus-6bit OR download `inclusionAI/UI-Venus-Ground-2B`
   (unquantized) for a clean fp16 reference
3. Write `HFVLMLoader.swift` (M4.1) that LOADS and reports tensor
   stats — no forward pass yet. Smoke: `tinygpt qwen3vl-load <dir>`
   prints "loaded 28 LLM layers + 24 vision blocks + 3 deepstack
   mergers + merger projection."

That alone is a meaningful first-PR; the forward pass can land in M4.2-M4.4.
