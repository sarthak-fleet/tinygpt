# Qwen3-VL mRoPE + DeepStack — math spec from HF reference

Documented 2026-06-08 from `huggingface/transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`. Closes research tasks #277 and #278.

## 1. Multimodal RoPE (mRoPE)

### Shape conventions

For head_dim = 128, freq dimension is head_dim/2 = 64. mrope_section
`[24, 20, 20]` splits the 64 frequencies:
- 24 for temporal (T)
- 20 for height (H)
- 20 for width (W)

Sum: 24 + 20 + 20 = 64 ✓

### Position IDs

Position IDs are 3-dimensional: `(T_pos, H_pos, W_pos)` per token.

For TEXT tokens: all three dimensions = the scalar text position.
```python
if position_ids.ndim == 2:
    position_ids = position_ids[None, ...].expand(3, ...)  # repeat scalar pos
```

For IMAGE tokens: each token carries its own (T, H, W) coords. For a
single 2D image: T=fixed (e.g., 0), H ∈ [0..h_patches), W ∈ [0..w_patches).
For a video: T varies across frames.

### Frequency computation

Standard RoPE inv_freq is computed once. Then expanded for the 3 dims:

```python
inv_freq_expanded = inv_freq[None, None, :, None].expand(3, bs, -1, 1)
position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, seq)
freqs = inv_freq_expanded @ position_ids_expanded  # (3, bs, head_dim//2, seq)
freqs = freqs.transpose(-1, -2)  # (3, bs, seq, head_dim//2)
```

So `freqs[i]` is the frequency tensor for dimension i (T/H/W).

### Interleaved mRoPE — the key step

```python
def apply_interleaved_mrope(self, freqs, mrope_section):
    """Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT]"""
    freqs_t = freqs[0]  # start from T-frequencies
    for dim, offset in enumerate((1, 2), start=1):  # H=1, W=2
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t
```

**Visualizing the interleaved layout for mrope_section=[24, 20, 20]**:

Indices 0..63 of the freq dimension get assigned:
- idx 0:  T_0
- idx 1:  H_0  (offset=1, stride=3, length=60)
- idx 2:  W_0  (offset=2, stride=3, length=60)
- idx 3:  T_1
- idx 4:  H_1
- idx 5:  W_1
- ...
- idx 60: T_20  (within the 24 T-positions: indices 0, 3, 6, ..., 60 = 21 of them; T_21, T_22, T_23 fill the trailing positions 61, 62, 63)
- idx 61: T_21
- idx 62: T_22
- idx 63: T_23

So H gets indices [1, 4, 7, ..., 58] (20 positions, length=60, offset=1, stride=3).
W gets indices [2, 5, 8, ..., 59] (20 positions, length=60, offset=2, stride=3).
T fills the rest: [0, 3, 6, ..., 60, 61, 62, 63].

For text-only prompts, freqs[0] == freqs[1] == freqs[2] (since T=H=W=text_pos),
so the interleaving is a no-op — collapses to standard 1D RoPE.

### Implementation note for MLX-Swift port

The interleaved-write step:
```
freqs_t = freqs[0]  # copy
freqs_t[..., 1:60:3] = freqs[1, ..., 1:60:3]
freqs_t[..., 2:60:3] = freqs[2, ..., 2:60:3]
```

In MLX this is just three array writes. The hard part is computing the
correct (T, H, W) position triple per token (text vs image distinction).

## 2. DeepStack visual injection

### Mechanism: RESIDUAL ADDITION (not replacement)

```python
def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    hidden_states = hidden_states.clone()
    local_this = hidden_states[visual_pos_masks, :] + visual_embeds  # ← residual add
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states
```

Only positions where `visual_pos_masks == True` are augmented. Other
positions pass through unchanged.

### Where injection happens (the corrected understanding)

**IMPORTANT**: `deepstack_visual_indexes = [5, 11, 17]` (in vision_config)
refers to **vision-tower layer indices where features are TAPPED**, NOT
the LLM layers where features are injected.

Vision tower forward (in vision encoder, NOT LLM):
```python
for layer_num, blk in enumerate(self.blocks):  # 24 vision blocks
    hidden_states = blk(...)
    if layer_num in self.deepstack_visual_indexes:  # [5, 11, 17]
        deepstack_feature = self.deepstack_merger_list[
            self.deepstack_visual_indexes.index(layer_num)
        ](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)
# Returns: deepstack_feature_lists = [features_from_layer_5,
#                                      features_from_layer_11,
#                                      features_from_layer_17]
```

LLM decoder forward (in language model):
```python
for layer_idx, layer in enumerate(self.layers):  # 28 LLM blocks
    hidden_states = layer(...)
    if deepstack_visual_embeds is not None \
       and layer_idx in range(len(deepstack_visual_embeds)):  # 0, 1, 2
        hidden_states = self._deepstack_process(
            hidden_states, visual_pos_masks,
            deepstack_visual_embeds[layer_idx]
        )
```

So:
- Vision tower TAPS at layers [5, 11, 17] (3 taps)
- LLM INJECTS at layers [0, 1, 2] (the first 3 layers — N where N = number of taps)
- Tap k from vision → injects at LLM layer k

**The scaffold (Qwen3VLDeepstackPlan) needs correcting**: the injection
LLM-layer indices are [0, 1, 2], NOT [5, 11, 17]. The [5, 11, 17] is
where the vision tower SAMPLES features, separate from where they land
in the LLM.

### Vision-tower merger details

Each tap layer's features pass through a `Qwen3VLVisionPatchMerger`
(the `deepstack_merger_list[i]` MLP, hidden_size → out_hidden_size).
Per the UI-Venus config: `hidden_size=1024 → out_hidden_size=2048`.

These are SEPARATE projections from the main merger (the final
projection that goes onto the embeddings). UI-Venus has 1 main merger
+ 3 deepstack mergers in the vision tower.

### Position specificity

`visual_pos_masks` is a `(batch, seq_len)` bool tensor. True at positions
where image tokens live in the prompt; False at text positions. This
means deepstack injection only modifies the image-region hidden states,
not the text-region ones.

## Implications for M4 implementation

| Sub | What this changes |
|---|---|
| M4.1 (loader) | Load 3 separate `vision_tower.deepstack_merger_list[N]` projections + 1 main `vision_tower.merger` projection. Already documented in factory-vision-m4-impl-plan.md. |
| M4.2 (mRoPE) | Implement `apply_interleaved_mrope` exactly as above. Three position-id tensors for T/H/W. Apply once per attention layer. |
| M4.3 (image tokens) | Compute `visual_pos_masks` at embed time — track which positions in the token stream are image-placeholder substitutions. |
| M4.4 (deepstack) | **Inject at LLM layers [0, 1, 2], NOT [5, 11, 17].** Residual add via `visual_pos_masks`. Three sets of deepstack features, one per first-N LLM layers. |

## Source

[transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) — read 2026-06-08
