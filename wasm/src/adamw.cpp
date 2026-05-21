// adamw.cpp — AdamW optimizer step (Phase 4).
//
// STATUS: documented stub. No implementation yet.
//
// Per parameter, given gradient g (config: configs/training.json):
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   p = p - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * p )
//
// betas (0.9, 0.95), eps 1e-8, weight_decay 0.1. Apply grad clipping
// (grad_clip 1.0) BEFORE the step. m and v are persisted in checkpoints.
//
// Guide: docs/model_guide.md ("Training config")
//
// TODO(phase-4): in-place AdamW update over a parameter buffer.
