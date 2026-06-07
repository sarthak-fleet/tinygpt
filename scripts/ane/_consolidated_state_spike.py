#!/usr/bin/env python3
"""_consolidated_state_spike.py — variant of the stateful spike where ALL
layers share ONE consolidated K-cache and ONE consolidated V-cache. The
goal is to determine if the ANE compiler rejection on the per-layer
stateful Qwen3 (56 mutable state slots) is a slot-count issue or a more
fundamental incompatibility.

Setup:
  - 4 small attention layers (down from Qwen3's 28 — keeps spike fast)
  - One unified K-cache `[1, n_layers, max_seq, embed]`
  - One unified V-cache `[1, n_layers, max_seq, embed]`
  - Each layer indexes into its own slot via Python-int layer-index

PASS = mlpackage loads + predicts on ComputeUnit.CPU_AND_NE without the
ANECCompile -14 error.
FAIL = same error → ANE doesn't accept StateType graphs for transformer
shapes on this OS; pivot to hybrid serve.
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


# Two layer attention block writing into a shared cache.
class Attn(nn.Module):
    def __init__(self, embed: int, layer_idx: int):
        super().__init__()
        self.q = nn.Linear(embed, embed, bias=False)
        self.k = nn.Linear(embed, embed, bias=False)
        self.v = nn.Linear(embed, embed, bias=False)
        self.layer_idx = layer_idx
        self.embed = embed

    def forward(self, x, causal_mask, k_cache, v_cache, past_len, end_step):
        # x: [B, T_new, C]
        B, T_new, C = x.shape
        q = self.q(x); kn = self.k(x); vn = self.v(x)
        # Update one layer's slice in the unified cache.
        k_cache[:, self.layer_idx, past_len:end_step, :] = kn
        v_cache[:, self.layer_idx, past_len:end_step, :] = vn
        # Read the active prefix.
        k_all = k_cache[:, self.layer_idx, :end_step, :]
        v_all = v_cache[:, self.layer_idx, :end_step, :]
        scale = 1.0 / float(C) ** 0.5
        scores = (q @ k_all.transpose(-2, -1)) * scale
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)
        return attn @ v_all


class ToyConsolidated(nn.Module):
    def __init__(self, vocab: int, embed: int, n_layers: int, max_seq: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)
        self.layers = nn.ModuleList([Attn(embed, i) for i in range(n_layers)])
        # Unified caches — one per "direction" (K and V) across all layers.
        self.register_buffer("k_cache", torch.zeros(1, n_layers, max_seq, embed))
        self.register_buffer("v_cache", torch.zeros(1, n_layers, max_seq, embed))

    def forward(self, input_ids, causal_mask):
        x = self.embedding(input_ids)
        end_step = causal_mask.shape[-1]
        past_len = end_step - x.shape[1]
        for blk in self.layers:
            x = blk(x, causal_mask, self.k_cache, self.v_cache, past_len, end_step)
        return x


def main():
    vocab = 100
    embed = 32
    n_layers = 4
    max_seq = 32

    torch.manual_seed(0)
    model = ToyConsolidated(vocab, embed, n_layers, max_seq).eval()
    ex_ids = torch.zeros(1, 1, dtype=torch.long)
    ex_mask = torch.zeros(1, 1, 1, dtype=torch.float32)
    traced = torch.jit.trace(model, (ex_ids, ex_mask))

    q_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq, default=1)
    e_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq, default=1)
    print("[1/4] converting consolidated-state toy …")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(shape=(1, q_dim), dtype=np.int32, name="input_ids"),
            ct.TensorType(shape=(1, q_dim, e_dim), dtype=np.float16, name="causal_mask"),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, n_layers, max_seq, embed), dtype=np.float16),
                          name="k_cache"),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, n_layers, max_seq, embed), dtype=np.float16),
                          name="v_cache"),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    print("      ✓ converted")
    out_path = "/tmp/consolidated_spike.mlpackage"
    mlmodel.save(out_path)
    print(f"      ✓ saved to {out_path}")

    print("[2/4] reload + make_state on ComputeUnit.CPU_AND_NE …")
    m = ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    try:
        state = m.make_state()
        print("      ✓ state created on CPU+NE")
    except Exception as e:
        print(f"      ✗ make_state failed on CPU+NE: {e}")
        print("      → ANE rejects consolidated-state. Pivoting to hybrid path.")
        sys.exit(2)

    print("[3/4] predict on CPU+NE …")
    ids = np.array([[1]], dtype=np.int32)
    mask = np.zeros((1, 1, 1), dtype=np.float16)
    try:
        out = m.predict({"input_ids": ids, "causal_mask": mask}, state=state)
        print(f"      ✓ predict ok, output shape = {out['output'].shape}")
    except Exception as e:
        print(f"      ✗ predict failed: {e}")
        sys.exit(3)

    print("[4/4] run 3 sequential predictions …")
    past_len = 0
    for step in range(3):
        end = past_len + 1
        ids = np.array([[step + 1]], dtype=np.int32)
        mask = np.zeros((1, 1, end), dtype=np.float16)
        out = m.predict({"input_ids": ids, "causal_mask": mask}, state=state)
        print(f"      step {step}: out norm = {float(np.linalg.norm(out['output'])):.4f}")
        past_len += 1

    print("\nSPIKE PASSED: consolidated-state stateful KV on ANE works.")
    print("              Refactor Qwen3 to use unified K/V tensors per direction.")


if __name__ == "__main__":
    main()
