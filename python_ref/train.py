"""
train.py — training loop for TinyGPT (Phase 1).

STATUS: documented stub. No implementation yet.

Loop (config: configs/training.json):
    for step in range(max_steps):
        x, y  = get_batch("train")
        logits = model(x)
        loss   = cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        if step % eval_interval       == 0: evaluate()
        if step % sample_interval     == 0: sample_text(prompt)
        if step % checkpoint_interval == 0: save_checkpoint()

Optimizer: AdamW, betas (0.9, 0.95), eps 1e-8, weight_decay 0.1.

Debugging expectations:
    random model        -> loss near ln(256) ~= 5.54
    repeated tiny data  -> loss falls fast
    loss does not fall  -> bug in model / backprop / data
    loss becomes NaN    -> learning rate, softmax, grad explosion, or bad init

Spec:  configs/training.json
Guide: docs/model_guide.md  ("Training config", "Training loop")
Tests: tests/README.md      (loss sanity, tiny overfit, gradient check)

TODO(phase-1): build dataset + model + AdamW from the JSON configs.
TODO(phase-1): implement the loop above with eval / sample / checkpoint hooks.
TODO(phase-1): log step, train loss, val loss, tokens/sec; keep runs seed-reproducible.
"""
