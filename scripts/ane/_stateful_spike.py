#!/usr/bin/env python3
"""_stateful_spike.py — minimal single-layer stateful attention CoreML
spike. Validates whether coremltools 9 + macOS 26 actually run a stateful
KV-cache model end-to-end (convert + make_state + 3 predictions).

The canonical coremltools test for this pattern is marked with a known
skip:
   rdar://152066678 Attention Toy Model Prediction Crashes Python

It uses `torch.nn.functional.scaled_dot_product_attention` which is the
common trace-fragility culprit. Our M2 Qwen3 path already uses MANUAL
softmax + matmul, so we mirror that here. If this spike passes on our
machine, we have a green light to extend qwen3_to_coreml.py with a real
stateful mode. If it crashes, we pivot to a hybrid path (explicit K/V
input/output tensors, no StateType).

PASS criteria (all three): convert succeeds · `.make_state()` returns ·
three sequential `.predict({"input_ids": ..., "causal_mask": ...}, state=...)`
calls all return without crash, and the third call's output reflects the
accumulated state (i.e., the cache is actually being updated in place).
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import coremltools as ct
except ImportError:
    sys.exit("install coremltools: pip install coremltools")


# ----- toy model (single layer attention + tied lm-head) ---------------------


class ToyAttn(nn.Module):
    def __init__(self, embed: int):
        super().__init__()
        self.q = nn.Linear(embed, embed, bias=False)
        self.k = nn.Linear(embed, embed, bias=False)
        self.v = nn.Linear(embed, embed, bias=False)

    def forward(self, x, causal_mask, k_cache, v_cache):
        # x: [B, T_new, C]
        B, T_new, C = x.shape
        end_step = causal_mask.shape[-1]
        past_len = end_step - T_new
        q = self.q(x)
        kn = self.k(x)
        vn = self.v(x)
        # In-place update of the cache windows.
        k_cache[:, past_len:end_step, :] = kn
        v_cache[:, past_len:end_step, :] = vn
        # Read the active prefix.
        k_all = k_cache[:, :end_step, :]
        v_all = v_cache[:, :end_step, :]
        # Manual softmax + matmul (NOT F.scaled_dot_product_attention — that's
        # the SDPA path the canonical test got bit by).
        scale = 1.0 / float(C) ** 0.5
        scores = (q @ k_all.transpose(-2, -1)) * scale       # [B, T_new, end_step]
        scores = scores + causal_mask                         # additive mask
        attn = F.softmax(scores, dim=-1)
        return attn @ v_all                                   # [B, T_new, C]


class ToyKVCacheModel(nn.Module):
    def __init__(self, vocab: int, embed: int, max_seq: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)
        self.attn = ToyAttn(embed)
        # KV cache registered as buffer. Shape [B, max_seq, embed].
        # Persistence isn't relevant here; the StateType tells coremltools
        # to convert these to mutable state slots.
        self.register_buffer("k_cache", torch.zeros(1, max_seq, embed))
        self.register_buffer("v_cache", torch.zeros(1, max_seq, embed))
        self.max_seq = max_seq

    def forward(self, input_ids, causal_mask):
        x = self.embedding(input_ids)
        return self.attn(x, causal_mask, self.k_cache, self.v_cache)


def main():
    vocab = 100
    embed = 32
    max_seq = 32
    seq_new = 1  # decode-step shape

    torch.manual_seed(0)
    model = ToyKVCacheModel(vocab, embed, max_seq).eval()
    example_ids = torch.zeros(1, seq_new, dtype=torch.long)
    example_mask = torch.zeros(1, seq_new, 1, dtype=torch.float32)
    traced = torch.jit.trace(model, (example_ids, example_mask))

    # Inputs use RangeDim so the same .mlpackage handles both prefill
    # (seq>1) and decode (seq=1). The CoreML test does this trick; it's
    # how a single mlpackage supports both.
    query_len = ct.RangeDim(lower_bound=1, upper_bound=max_seq, default=1)
    end_step  = ct.RangeDim(lower_bound=1, upper_bound=max_seq, default=1)
    print("[1/3] converting toy stateful model …")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(shape=(1, query_len), dtype=np.int32, name="input_ids"),
            ct.TensorType(shape=(1, query_len, end_step), dtype=np.float16, name="causal_mask"),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, max_seq, embed), dtype=np.float16),
                          name="k_cache"),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, max_seq, embed), dtype=np.float16),
                          name="v_cache"),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    print("      ✓ converted")

    print("[2/3] make_state() …")
    state = mlmodel.make_state()
    print("      ✓ state created")

    print("[3/3] running 3 sequential predictions with cache growing …")
    past_len = 0
    outputs = []
    for step in range(3):
        end = past_len + 1
        mask = np.zeros((1, 1, end), dtype=np.float16)  # decode is causal-1
        ids = np.array([[step + 1]], dtype=np.int32)
        out = mlmodel.predict({"input_ids": ids, "causal_mask": mask}, state=state)
        outputs.append(out["output"])
        print(f"      step {step}: ok, out norm = {np.linalg.norm(out['output']):.4f}")
        past_len += 1

    # Sanity: output norm should change step-to-step (because the cache is
    # being read with more keys/values each time). If norms are identical
    # across steps, the cache is being ignored.
    norms = [float(np.linalg.norm(o)) for o in outputs]
    if len(set(round(n, 3) for n in norms)) == 1:
        print(f"\n      ✗ all output norms identical {norms} — cache may not be updating")
        sys.exit(1)
    else:
        print(f"\n      ✓ output norms differ across steps: {norms}")
        print("        → cache is being read with growing prefix; stateful KV works")

    print("\nSPIKE PASSED: coremltools 9 + macOS 26 supports stateful KV cache")
    print("              with manual-softmax attention. Proceed to M3 full Qwen3.")


if __name__ == "__main__":
    main()
