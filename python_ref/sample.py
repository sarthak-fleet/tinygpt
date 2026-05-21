"""
sample.py — text generation / sampling from a trained TinyGPT (Phase 1).

STATUS: documented stub. No implementation yet.

Autoregressive decoding:
    start from a prompt (bytes -> token ids)
    repeat:
        logits = model(tokens[-context_length:])
        next   = sample(logits[-1], temperature, top_k)
        append next to tokens
    decode tokens -> bytes -> UTF-8 text

Sampling controls:
    temperature   scales logits before softmax (lower = greedier)
    top_k         restrict sampling to the k most likely tokens
    greedy        argmax (temperature -> 0)

Determinism: with a fixed seed, generation must be reproducible
(see tests/README.md "Sampling fixed seed").

Guide: docs/model_guide.md  ("Loss function") and docs/learning_roadmap.md (Phase 2)

TODO(phase-1): load a checkpoint; implement generate(prompt, max_new_tokens, ...).
TODO(phase-1): implement temperature + top-k sampling with a seeded RNG.
"""
