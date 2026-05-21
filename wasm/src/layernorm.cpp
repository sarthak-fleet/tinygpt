// layernorm.cpp — LayerNorm, forward + backward (Phase 4).
//
// STATUS: documented stub. No implementation yet.
//
// Pre-LayerNorm is applied before attention and before the MLP in every block:
//   y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Backward must return gradients for x, gamma, and beta.
//
// Guide: docs/model_guide.md ("Transformer block"), docs/browser_notes.md
//
// TODO(phase-4): forward + backward over the last (feature) dimension.
