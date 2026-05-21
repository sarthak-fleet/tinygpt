"""
lora.py — LoRA fine-tuning reference (Phase 3).

STATUS: documented stub. No implementation yet. Do Phase 1-2 (model + training)
first; LoRA adapts an already-trained, frozen base model.

LoRA replaces a linear layer  y = xW  with  y = xW + (alpha/r) * (xA)B
    W  [d_in, d_out]  frozen base weight
    A  [d_in, r]      trainable, initialised to small random values
    B  [r, d_out]     trainable, initialised to ZEROS  -> step 0 == base model

Backward (the subtle part — you MUST still backprop through frozen layers so
lower LoRA layers receive gradient):
    dB = (xA)^T @ dy * scale
    dA = x^T @ (dy @ B^T) * scale
    dx = dy @ W^T + scale * dy @ B^T @ A^T
    dW = not computed (W is frozen)

Procedure:
    base = load_base_model(); freeze(base)
    inject_lora(base, target_modules=["q_proj","v_proj"], rank=4, alpha=8)
    optimizer = AdamW(lora_parameters(base), lr=1e-4)
    ... standard training loop, but only LoRA params are trainable ...
    save_adapter()   # adapter weights only — never the full base each time

Adapter checkpoint records: adapter weights + optimizer state, base model id/hash,
tokenizer id/hash, training config, dataset manifest/hash, loss history, step.
Result: one base model, many small swappable adapters.

Spec:  configs/lora.json
Guide: docs/lora_guide.md

TODO(phase-3): implement LoRALinear (forward + the backward above).
TODO(phase-3): inject_lora() over named target modules; freeze base weights.
TODO(phase-3): adapter-only save/load; base-vs-LoRA output comparison.
"""
