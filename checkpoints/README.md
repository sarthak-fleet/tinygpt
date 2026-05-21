# checkpoints/

Saved training state. Contents are gitignored (binary weights are large);
this README is kept.

## From-scratch checkpoint layout

```
checkpoint/
  model_config.json
  training_config.json
  dataset_manifest.json
  trainer_state.json
  weights.f32
  adam_m.f32
  adam_v.f32
  loss_history.json
```

Saving `adam_m` / `adam_v` is what makes a run truly resumable — without the
optimizer state, resume restarts the moments from zero.

## LoRA adapter checkpoint

Never save the full base model each time. Save adapter weights + adapter
optimizer state, plus identity of what they attach to:

```json
{
  "base_model": { "id": "byte-tinygpt-base-15m", "sha256": "..." },
  "adapter":    { "type": "lora", "rank": 4, "alpha": 8,
                  "target_modules": ["q_proj","v_proj"], "params": 240000 },
  "training":   { "learning_rate": 0.0001, "batch_size": 4,
                  "context_length": 256, "step": 500 },
  "dataset":    { "dataset_id": "sha256...", "examples": 700 }
}
```

Result: one base model, many small swappable adapters.

## Browser checkpoints

In the browser, checkpoints live in OPFS or IndexedDB (subject to storage
quota). The "Browser refresh" test in `../tests/README.md` requires a run to
resume after a page reload. See `../docs/browser_notes.md`.
