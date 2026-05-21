"""
model.py — byte-level TinyGPT model (Phase 1-2).

STATUS: documented stub. No implementation yet — start here for Phase 1.

Defines the ~0.8M-parameter causal language model from
docs/model_guide.md and configs/model.byte-tinygpt-v0.json.

Forward pass:
    token ids
      -> token embedding + position embedding
      -> N pre-LayerNorm transformer blocks
      -> final LayerNorm
      -> logits = x @ token_embedding.T        (tied embeddings)
      -> next-token cross-entropy loss

Components to implement (you only need backprop for these — do NOT write a
general autograd engine first):
    Embedding            token [vocab_size, d_model] + position [context_length, d_model]
    LayerNorm
    CausalSelfAttention  q/k/v projections, scaled dot-product, causal mask, multi-head
    MLP                  Linear(d_model -> 4*d_model), GELU, Linear(4*d_model -> d_model)
    TransformerBlock     pre-LN:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))
    TinyGPT              assembles the above; ties the output head to token embedding

Sanity: a random model's loss should sit near ln(256) ~= 5.54.

Spec:  configs/model.byte-tinygpt-v0.json
Guide: docs/model_guide.md  ("Architecture details", "Output head", "Loss function")

TODO(phase-1): implement the modules above in PyTorch.
TODO(phase-1): assert expected shapes at every layer (see tests/README.md).
TODO(phase-1): support a tie_embeddings flag (output_weight == token_embedding_weight).
"""
