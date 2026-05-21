// attention.cpp — causal multi-head self-attention, forward + backward (Phase 4).
//
// STATUS: documented stub. No implementation yet.
//
// Per block (shapes: batch B, seq T, d_model C, heads H, head_dim = C / H):
//   q = x @ Wq;  k = x @ Wk;  v = x @ Wv
//   scores = q @ k^T / sqrt(head_dim)
//   scores = causal_mask(scores)          // no attending to future tokens
//   attn   = softmax(scores)
//   out    = attn @ v
//   out    = out @ Wo
//
// Backward propagates through softmax, the masked scores, and all four
// projections (Wq, Wk, Wv, Wo).
//
// Guide: docs/model_guide.md ("Causal self-attention")
//
// TODO(phase-4): forward with causal mask + softmax; full backward.
