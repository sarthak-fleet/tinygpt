#!/usr/bin/env python3
"""qwen3_to_coreml.py — convert a Qwen3 HF directory to a CoreML `.mlpackage`
tuned for Apple Neural Engine dispatch on M-series silicon.

The matching Swift CLI is `tinygpt to-coreml-qwen3`. This script is the
honest deliverable (CoreML conversion requires Python coremltools; no
Swift-side writer exists). It is reachable directly:

    python3 scripts/ane/qwen3_to_coreml.py \\
        --hf-dir <merged-pace-hf-dir>      \\
        --out   pace-planner.mlpackage     \\
        --max-prompt-length 512            \\
        --mode  stateless                  # or 'stateful' once M3 lands

Architecture mirrors `native-mac/Sources/TinyGPTModel/TransformerBlockHF.swift`
+ `CausalSelfAttention.swift` (the MLX-Swift Pace inference path). The
PyTorch arch is deliberately handwritten — NOT `transformers.AutoModel` —
because torch.jit.trace through HF's classes routinely hallucinates
control flow into the CoreML graph. We mirror exactly what we trust.

Qwen3-0.6B shape (from config.json):
    layers       = 28
    hidden_size  = 1024
    num_q_heads  = 16
    num_kv_heads = 8       (GQA — 2 Q heads per KV head)
    head_dim     = 128     (NOT hidden_size/num_heads ! see config.head_dim)
    intermed.    = 3072    (SwiGLU)
    vocab        = 151_936
    rope_theta   = 1e6     (very high — long-context friendly)
    tie_embed    = true
    rms_norm_eps = 1e-6
    QK-Norm      = true    (per-head RMSNorm on q + k after projection,
                            before RoPE — the Qwen3-specific bit)

Path B from the advisor: one model class with optional cache args, traced
twice. Mode 'stateless' (M2) is the prompt-prefill path — used for parity
validation against the MLX serve. Mode 'stateful' (M3) uses
coremltools.StateType for the KV cache so token-by-token decode is O(1)
per step instead of O(T²). The stateful path replaces the stateless one
for production serve.

KNOWN HAZARDS (from prior CoreML attempts on this repo)
-------------------------------------------------------
1. coremltools.convert is multi-minute on a 0.6B model. Profile op
   dispatch in Xcode Instruments → Core ML template once the package
   loads. If softmax / RoPE / RMSNorm land on CPU instead of ANE,
   that's the expected speed regression vs the headline target — be
   honest about it.

2. coremltools 9 / iOS 18 / macOS 15 added `StateType` for KV caches.
   This is what makes ANE decode fast — without it every token re-encodes
   the full prompt. The stateful path REQUIRES macOS 15+ for runtime.

3. Apple's `ml-ane-transformers` reference shows that the
   *fully ANE-optimized* arch uses (B, C, 1, S) tensor layout instead of
   (B, S, C), conv-based linears, and a specific softmax pattern. We do
   NOT do that here (it's a different model entirely — a follow-up). The
   standard layout still gets significant ANE coverage on M3+.

4. Float16 conversion (coremltools' `--compute_precision FLOAT16`) is
   what triggers ANE dispatch for matmul. Without it the model stays on
   GPU. Default is FLOAT16 in this script — override with --precision fp32.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import load_file
except ImportError as e:
    sys.exit(f"missing dep: {e}. pip install numpy torch safetensors")


# =============================================================================
# Qwen3 PyTorch arch — handwritten to be tracer-friendly.
#
# Notes vs the MLX-Swift reference:
#   - RMSNorm uses fp32 mean of square. MLX's RMSNorm also runs in fp32 by
#     default. Matches.
#   - RoPE: rope_theta from config (1e6 for Qwen3), applied per-head, NOT
#     pre-cached because the cache table for ctx=40960 × head_dim=128 is
#     ~10 MB which is fine but we keep it dynamic to keep the trace
#     position-independent.
#   - QK-Norm: per-head RMSNorm applied AFTER q/k projection BEFORE RoPE
#     (Qwen3 quirk; without it attention diverges over long context).
#   - SwiGLU: gate, up, down. `silu(gate(x)) * up(x)` then `down(...)`.
#   - GQA: K and V are projected with fewer heads than Q. We repeat KV
#     heads to match Q before attention (the simplest tracer-friendly
#     pattern; PyTorch SDPA can do it natively but tracing through that
#     into CoreML is fragile).
#   - tie_embeddings: lm_head reuses embed_tokens weights via F.linear.
# =============================================================================


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 reduction for stability (HF / Qwen do this in their ref impl)
        in_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight * x32.to(in_dtype))


def precompute_rope_cache(head_dim: int, max_seqlen: int, theta: float,
                           device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin cache. Returns (cos, sin) each shape [max_seqlen, head_dim]."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(max_seqlen, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                              # [T, half]
    # Duplicate to full head_dim (the rotate_half pattern).
    emb = torch.cat([freqs, freqs], dim=-1)                       # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary positional embedding (HF Llama / Qwen3 layout).

    q, k: [B, H, T, head_dim]
    cos, sin: [Tmax, head_dim]
    positions: [T] long — per-token absolute positions (lets us trace a
        fixed-length prompt and still get correct positions).
    """
    cs = cos.index_select(0, positions).unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    sn = sin.index_select(0, positions).unsqueeze(0).unsqueeze(0)
    # rotate_half: split last dim in half, swap and negate.
    def rotate_half(t):
        h = t.shape[-1] // 2
        a, b = t[..., :h], t[..., h:]
        return torch.cat([-b, a], dim=-1)
    return (q * cs) + (rotate_half(q) * sn), (k * cs) + (rotate_half(k) * sn)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA — repeat each kv head n_rep times along the head dim.

    x: [B, n_kv_heads, T, head_dim]  →  [B, n_kv_heads * n_rep, T, head_dim]
    """
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x.unsqueeze(2).expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class Qwen3Attention(nn.Module):
    def __init__(self, hidden: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                  rms_eps: float):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # Qwen3 head_dim is NOT hidden / n_heads (see Qwen3-0.6B: 128 vs 64).
        # The projections go to n_heads * head_dim, which may NOT equal hidden.
        q_out = n_q_heads * head_dim
        kv_out = n_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden, q_out, bias=False)
        self.k_proj = nn.Linear(hidden, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden, bias=False)
        # Qwen3 QK-Norm — per-head RMSNorm on the projected q and k vectors.
        # Applied AFTER projection, BEFORE RoPE. Weight dim equals head_dim.
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor,
                 attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        # QK-Norm — applied per-head, weight broadcasts over the head dim.
        q = self.q_norm(q)
        k = self.k_norm(k)
        # [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, positions)
        # GQA — repeat kv to match q.
        n_rep = self.n_q_heads // self.n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)
        # Manual SDPA (avoids torch.nn.functional.scaled_dot_product_attention
        # because the tracer can't always preserve its causal-mask plumbing
        # into CoreML). Scores: [B, H, T, T].
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1).to(v.dtype)
        out = attn @ v                                                       # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_q_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    def __init__(self, hidden: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                  intermediate: int, rms_eps: float):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden, eps=rms_eps)
        self.self_attn = Qwen3Attention(hidden, n_q_heads, n_kv_heads, head_dim, rms_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden, eps=rms_eps)
        self.mlp = Qwen3MLP(hidden, intermediate)

    def forward(self, x, positions, cos, sin, attn_mask):
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, positions, cos, sin, attn_mask)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class Qwen3StatefulAttention(nn.Module):
    """Qwen3 attention with consolidated stateful KV cache.

    Same projection / QK-Norm / RoPE pipeline as `Qwen3Attention`, but
    instead of reading/writing fresh K/V every step, it slices into ONE
    GIANT cache shared across all layers. The consolidated layout is:

        k_cache, v_cache: [1, n_layers * n_kv_heads, max_seq, head_dim]

    Each layer i indexes into rows [i*n_kv_heads .. (i+1)*n_kv_heads).
    Why consolidated:
      - The natural per-layer cache (N×2 = 56 state slots for Qwen3-0.6B)
        triggers ANECCompile error -14. Apple's ANE backend appears not
        to accept that many mutable state slots in one mlprogram.
      - Consolidated (2 state slots total) compiles cleanly and runs on
        ANE. Validated via `_consolidated_state_spike.py` on macOS 26.

    Mechanically the math is identical — the slice into `[i*H..(i+1)*H, :, :]`
    is statically known at trace time (i is a Python int), so the ANE
    compiler sees a fixed-offset window read/write per layer.
    """
    def __init__(self, hidden: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                  rms_eps: float, layer_idx: int):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        q_out = n_q_heads * head_dim
        kv_out = n_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden, q_out, bias=False)
        self.k_proj = nn.Linear(hidden, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_eps)

    def forward(self, x, positions, cos, sin, attn_mask,
                 k_cache, v_cache, past_len, end_step):
        """x: [B, T_new, hidden] · positions: [T_new] absolute positions.

        attn_mask: [1, 1, T_new, end_step]   additive (-1e4 / 0)
        k_cache / v_cache: [1, n_layers * n_kv_heads, max_seq, head_dim]
        past_len, end_step: Python ints (statically known per-call)
        """
        B, T_new, _ = x.shape
        q = self.q_proj(x).view(B, T_new, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T_new, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T_new, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # [B, H, T_new, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, positions)
        # Slot new K/V into the slice owned by this layer.
        H = self.n_kv_heads
        row_lo = self.layer_idx * H
        row_hi = row_lo + H
        k_cache[:, row_lo:row_hi, past_len:end_step, :] = k
        v_cache[:, row_lo:row_hi, past_len:end_step, :] = v
        # Read the active prefix back from THIS layer's slice.
        k_active = k_cache[:, row_lo:row_hi, :end_step, :]
        v_active = v_cache[:, row_lo:row_hi, :end_step, :]
        # GQA repeat for Q-side attention.
        n_rep = self.n_q_heads // self.n_kv_heads
        k_active = repeat_kv(k_active, n_rep)
        v_active = repeat_kv(v_active, n_rep)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k_active.transpose(-2, -1)) * scale
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1).to(v_active.dtype)
        out = attn @ v_active
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.n_q_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3StatefulBlock(nn.Module):
    def __init__(self, hidden: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                  intermediate: int, rms_eps: float, layer_idx: int):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden, eps=rms_eps)
        self.self_attn = Qwen3StatefulAttention(hidden, n_q_heads, n_kv_heads, head_dim,
                                                  rms_eps, layer_idx=layer_idx)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden, eps=rms_eps)
        self.mlp = Qwen3MLP(hidden, intermediate)
        # NO per-block buffers — caches live on the parent Qwen3StatefulModel.

    def forward(self, x, positions, cos, sin, attn_mask,
                 k_cache, v_cache, past_len, end_step):
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, positions, cos, sin, attn_mask,
                                k_cache, v_cache, past_len, end_step)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class Qwen3SingleBlockAttention(nn.Module):
    """Single-block attention with PRIVATE k/v cache (not consolidated).

    Designed for M8 layer-chunked conversion: each block ships as its
    own mlpackage with its own pair of MLState slots. Per the M6 bisect
    in docs/learn/ane-research/m6-findings.md, ANE runs exactly one
    block's stateful forward cleanly; chunking sidesteps the multi-layer
    crash by giving each block its own package.

    Cache layout (per block):
        k_cache, v_cache: [1, n_kv_heads, max_seq, head_dim]
    """
    def __init__(self, hidden: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                  rms_eps: float):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        q_out = n_q_heads * head_dim
        kv_out = n_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden, q_out, bias=False)
        self.k_proj = nn.Linear(hidden, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_eps)

    def forward(self, x, positions, cos, sin, attn_mask,
                 k_cache, v_cache, past_len, end_step):
        B, T_new, _ = x.shape
        q = self.q_proj(x).view(B, T_new, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T_new, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T_new, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, positions)
        # Private cache: no layer offset slicing.
        k_cache[:, :, past_len:end_step, :] = k
        v_cache[:, :, past_len:end_step, :] = v
        k_active = k_cache[:, :, :end_step, :]
        v_active = v_cache[:, :, :end_step, :]
        n_rep = self.n_q_heads // self.n_kv_heads
        k_active = repeat_kv(k_active, n_rep)
        v_active = repeat_kv(v_active, n_rep)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k_active.transpose(-2, -1)) * scale
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1).to(v_active.dtype)
        out = attn @ v_active
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.n_q_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3SingleBlockModel(nn.Module):
    """One Qwen3 transformer block packaged as a standalone stateful module.

    forward(hidden_state, causal_mask, position_offset)
       hidden_state:    [1, T_new, hidden]    fp16
       causal_mask:     [1, 1, T_new, end]    fp16 additive
       position_offset: [1] int32             absolute position of token[0]

    Returns: hidden_out: [1, T_new, hidden]   fp16

    Owns its own (k_cache, v_cache) MLState pair. Convert one
    instance per block index to produce 28 separate mlpackages.
    """
    def __init__(self, hf_config: dict, max_seq_len: int):
        super().__init__()
        self.config = hf_config
        hidden = hf_config["hidden_size"]
        n_q = hf_config["num_attention_heads"]
        n_kv = hf_config["num_key_value_heads"]
        head_dim = hf_config.get("head_dim", hidden // n_q)
        rms_eps = hf_config.get("rms_norm_eps", 1e-6)
        intermediate = hf_config["intermediate_size"]
        rope_theta = float(hf_config.get("rope_theta", 10000.0))
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv
        self.head_dim = head_dim

        self.input_layernorm = Qwen3RMSNorm(hidden, eps=rms_eps)
        self.self_attn = Qwen3SingleBlockAttention(hidden, n_q, n_kv, head_dim, rms_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden, eps=rms_eps)
        self.mlp = Qwen3MLP(hidden, intermediate)
        cos, sin = precompute_rope_cache(head_dim, max_seq_len, rope_theta, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # Private K and V caches owned by this block only.
        self.register_buffer("k_cache", torch.zeros(1, n_kv, max_seq_len, head_dim))
        self.register_buffer("v_cache", torch.zeros(1, n_kv, max_seq_len, head_dim))

    def forward(self, hidden_state, causal_mask, position_offset):
        B, T_new, _ = hidden_state.shape
        end_step = causal_mask.shape[-1]
        past_len = end_step - T_new
        offset = position_offset[0]
        positions = offset + torch.arange(T_new, dtype=torch.long,
                                           device=hidden_state.device)
        h = self.input_layernorm(hidden_state)
        x = hidden_state + self.self_attn(h, positions, self.rope_cos, self.rope_sin,
                                           causal_mask, self.k_cache, self.v_cache,
                                           past_len, end_step)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class Qwen3StatefulModel(nn.Module):
    """Stateful Qwen3 variant for ANE decode + prefill with CONSOLIDATED
    KV caches.

    forward(input_ids, causal_mask, position_offset)
       input_ids:      [1, T_new]            int32
       causal_mask:    [1, 1, T_new, end]    fp16 additive mask
       position_offset: [1] int32            position of token[0] in sequence

    KV cache layout:
       k_cache, v_cache: [1, n_layers * n_kv_heads, max_seq_len, head_dim]
       Layer i indexes into rows [i*n_kv_heads .. (i+1)*n_kv_heads).

    The consolidated layout (2 state slots vs the natural 56) is what
    makes the resulting mlpackage compile + run on ANE on macOS 26 +
    coremltools 9. See `_consolidated_state_spike.py` for the validation
    that drove this choice; the per-layer variant is rejected by
    ANECCompile (error -14).
    """
    def __init__(self, hf_config: dict, max_seq_len: int):
        super().__init__()
        self.config = hf_config
        hidden = hf_config["hidden_size"]
        n_q = hf_config["num_attention_heads"]
        n_kv = hf_config["num_key_value_heads"]
        head_dim = hf_config.get("head_dim", hidden // n_q)
        rms_eps = hf_config.get("rms_norm_eps", 1e-6)
        intermediate = hf_config["intermediate_size"]
        n_layers = hf_config["num_hidden_layers"]
        vocab = hf_config["vocab_size"]
        rope_theta = float(hf_config.get("rope_theta", 10000.0))
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_kv_heads = n_kv
        self.head_dim = head_dim

        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            Qwen3StatefulBlock(hidden, n_q, n_kv, head_dim, intermediate, rms_eps,
                                layer_idx=i)
            for i in range(n_layers)
        ])
        self.norm = Qwen3RMSNorm(hidden, eps=rms_eps)
        self.tie_word_embeddings = hf_config.get("tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
        cos, sin = precompute_rope_cache(head_dim, max_seq_len, rope_theta, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # Consolidated K and V caches. Shape: [1, n_layers*n_kv_heads, max_seq, head_dim].
        # For Qwen3-0.6B that's [1, 28*8=224, max_seq, 128] = ~7.3MB at max_seq=256.
        kv_rows = n_layers * n_kv
        self.register_buffer("k_cache", torch.zeros(1, kv_rows, max_seq_len, head_dim))
        self.register_buffer("v_cache", torch.zeros(1, kv_rows, max_seq_len, head_dim))

    def forward(self, input_ids, causal_mask, position_offset):
        B, T_new = input_ids.shape
        end_step = causal_mask.shape[-1]
        past_len = end_step - T_new
        offset = position_offset[0]
        positions = offset + torch.arange(T_new, dtype=torch.long,
                                            device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for blk in self.layers:
            x = blk(x, positions, self.rope_cos, self.rope_sin, causal_mask,
                    self.k_cache, self.v_cache, past_len, end_step)
        x = self.norm(x)
        if self.tie_word_embeddings:
            logits = F.linear(x, self.embed_tokens.weight)
        else:
            logits = self.lm_head(x)
        return logits


class Qwen3Model(nn.Module):
    """Forward-only Qwen3 transformer matching `Qwen3ForCausalLM`."""

    def __init__(self, hf_config: dict):
        super().__init__()
        self.config = hf_config
        hidden = hf_config["hidden_size"]
        n_q = hf_config["num_attention_heads"]
        n_kv = hf_config["num_key_value_heads"]
        head_dim = hf_config.get("head_dim", hidden // n_q)
        rms_eps = hf_config.get("rms_norm_eps", 1e-6)
        intermediate = hf_config["intermediate_size"]
        n_layers = hf_config["num_hidden_layers"]
        vocab = hf_config["vocab_size"]
        max_pos = hf_config["max_position_embeddings"]
        rope_theta = float(hf_config.get("rope_theta", 10000.0))

        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            Qwen3Block(hidden, n_q, n_kv, head_dim, intermediate, rms_eps)
            for _ in range(n_layers)
        ])
        self.norm = Qwen3RMSNorm(hidden, eps=rms_eps)
        # tie_word_embeddings — Qwen3-0.6B ties; no separate lm_head param.
        self.tie_word_embeddings = hf_config.get("tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        # RoPE cache. We register cos/sin as buffers so they ship with the
        # traced model. Buffer is positioned at construction time; for the
        # M3 stateful path we'll regenerate on a longer ctx without
        # retracing the model code.
        cos, sin = precompute_rope_cache(head_dim, max_pos, rope_theta, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Stateless forward — used for M2 validation + prompt prefill.

        input_ids: [B, T] int64 token IDs
        returns:   [B, T, vocab] logits (caller takes last-position slice)
        """
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        # Positions = 0..T-1. For chunked prefill in M3 we pass `position_offset`.
        positions = torch.arange(T, dtype=torch.long, device=x.device)
        # Causal additive mask. Build in fp32, large-negative for off-diag
        # upper triangle. We use -1e4 instead of -inf so fp16 conversion
        # doesn't overflow.
        mask = torch.full((T, T), -1.0e4, dtype=x.dtype, device=x.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        for blk in self.layers:
            x = blk(x, positions, self.rope_cos, self.rope_sin, mask)
        x = self.norm(x)
        if self.tie_word_embeddings:
            logits = F.linear(x, self.embed_tokens.weight)
        else:
            logits = self.lm_head(x)
        return logits


# =============================================================================
# Safetensors loader — maps HF names to our PyTorch module tree
# =============================================================================


def hf_state_to_qwen3(state: dict, n_layers: int, tie_word_embeddings: bool) -> dict:
    """Map the HF safetensors keys to our Qwen3Model state-dict names.

    HF Qwen3 names look like:
        model.embed_tokens.weight
        model.layers.N.input_layernorm.weight
        model.layers.N.self_attn.q_proj.weight
        model.layers.N.self_attn.q_norm.weight   # the QK-Norm pieces
        ...
        model.norm.weight
        lm_head.weight                            # absent if tied

    Our nn.Module tree drops the leading `model.` prefix and otherwise
    matches HF naming.
    """
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        out[nk] = v
    # Sanity: ensure every required key exists.
    expected = ["embed_tokens.weight", "norm.weight"]
    for i in range(n_layers):
        for sub in ["input_layernorm.weight", "post_attention_layernorm.weight",
                     "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                     "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                     "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                     "mlp.gate_proj.weight", "mlp.up_proj.weight",
                     "mlp.down_proj.weight"]:
            expected.append(f"layers.{i}.{sub}")
    if not tie_word_embeddings:
        expected.append("lm_head.weight")
    missing = [k for k in expected if k not in out]
    if missing:
        raise RuntimeError(f"missing tensors after rename: {missing[:6]}...")
    return out


def load_qwen3_from_hf(hf_dir: Path, stateful_max_seq: int = 0) -> tuple[nn.Module, dict]:
    """Returns either Qwen3Model (stateless) or Qwen3StatefulModel
    depending on `stateful_max_seq`. Pass 0 for stateless.
    """
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json at {cfg_path}")
    config = json.loads(cfg_path.read_text())
    assert config.get("model_type") == "qwen3", f"expected qwen3, got {config.get('model_type')}"

    # Find safetensors files. We support both single-file
    # (model.safetensors) and sharded layouts.
    shards = sorted(hf_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {hf_dir}")
    state: dict[str, torch.Tensor] = {}
    for s in shards:
        state.update(load_file(str(s)))

    if stateful_max_seq > 0:
        model = Qwen3StatefulModel(config, max_seq_len=stateful_max_seq)
    else:
        model = Qwen3Model(config)
    tie = config.get("tie_word_embeddings", True)
    sd = hf_state_to_qwen3(state, config["num_hidden_layers"], tie)
    # Cast to fp32 for trace (lossless w.r.t. bf16 source).
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    # Qwen3-0.6B's safetensors ships `lm_head.weight` even though
    # `tie_word_embeddings=true` — they're identical to embed_tokens.weight
    # in that case. Our `Qwen3Model` doesn't allocate an lm_head when ties
    # are on (`F.linear` reuses the embedding directly), so drop the
    # untyped sibling to avoid an `unexpected key` complaint.
    if tie and "lm_head.weight" in sd:
        sd = {k: v for k, v in sd.items() if k != "lm_head.weight"}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Suppress missing buffers — they're set up at __init__ time and don't
    # need to come from the HF checkpoint:
    #   - rope_cos / rope_sin   (RoPE tables, regenerated per init)
    #   - k_cache / v_cache     (consolidated stateful KV cache buffers,
    #                            registered on Qwen3StatefulModel root)
    def _is_init_buffer(name: str) -> bool:
        return (name.startswith("rope_")
                or name == "k_cache"
                or name == "v_cache")
    missing = [m for m in missing if not _is_init_buffer(m)]
    if missing:
        raise RuntimeError(f"could not load weights — missing: {missing}")
    if unexpected:
        raise RuntimeError(f"could not load weights — unexpected: {unexpected[:6]}")
    return model, config


# =============================================================================
# Parity check — sample logits to verify PyTorch arch matches HF reference
# =============================================================================


def parity_check(model: Qwen3Model, hf_dir: Path, tokens: list[int]) -> dict:
    """Run a quick forward and emit a diagnostic.

    We can't (cleanly) compare against MLX-Swift from Python in one process,
    so we emit (top-5 token IDs at last position, max logit value) and
    expect the caller to cross-check via `tinygpt sample` or `tinygpt
    hf-load --sample` on the same prompt.
    """
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([tokens], dtype=torch.long)
        logits = model(ids)
        last = logits[0, -1]
        topv, topi = torch.topk(last, k=5)
        return {
            "input_len": len(tokens),
            "last_logit_max": float(last.max().item()),
            "last_logit_min": float(last.min().item()),
            "top5_ids": [int(x) for x in topi.tolist()],
            "top5_logits": [float(x) for x in topv.tolist()],
        }


# =============================================================================
# CoreML conversion
# =============================================================================


def convert_stateful(model: 'Qwen3StatefulModel', max_seq_len: int,
                      out_path: Path, precision: str, compute_units: str) -> None:
    """Convert a Qwen3StatefulModel to a CoreML mlpackage with StateType
    KV-cache slots. The output is a SINGLE mlpackage that handles both
    prefill (input_ids: [1, T>1]) and decode (input_ids: [1, 1]) via
    RangeDim on the query and end_step dims. Per the canonical
    coremltools toy-attention test this is supported on macOS15+ /
    coremltools 9+, and our `_stateful_spike.py` confirmed it works on
    macOS 26 + ct 9.

    Trace input shapes:
      input_ids:       [1, 1]                 int32 (decode-shape sample)
      causal_mask:     [1, 1, 1, 1]           fp16
      position_offset: scalar long

    Convert input shapes (RangeDim, bounds [1, max_seq_len]):
      input_ids:       [1, q]
      causal_mask:     [1, 1, q, e]
    The state slots are introspected from `register_buffer` names of
    the form `layers.N.k_cache` / `layers.N.v_cache`.
    """
    import coremltools as ct

    model.eval()
    # Trace with the DECODE shape (T_new=1) — that's the hottest path. The
    # RangeDim at convert time lets the same model handle prompt prefill.
    ex_ids = torch.zeros(1, 1, dtype=torch.long)
    ex_mask = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    ex_pos = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(model, (ex_ids, ex_mask, ex_pos))

    compute_precision = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    compute_unit_map = {
        "ane":  ct.ComputeUnit.CPU_AND_NE,
        "gpu":  ct.ComputeUnit.CPU_AND_GPU,
        "all":  ct.ComputeUnit.ALL,
        "cpu":  ct.ComputeUnit.CPU_ONLY,
    }
    cu = compute_unit_map.get(compute_units.lower(), ct.ComputeUnit.CPU_AND_NE)

    # Consolidated KV caches — see Qwen3StatefulModel docstring.
    n_layers = model.config["num_hidden_layers"]
    n_kv = model.config["num_key_value_heads"]
    head_dim = model.config.get("head_dim", model.config["hidden_size"] // model.config["num_attention_heads"])
    kv_rows = n_layers * n_kv
    cache_shape = (1, kv_rows, max_seq_len, head_dim)
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
            name="k_cache"),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
            name="v_cache"),
    ]

    query_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
    end_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
    mlpkg = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, query_dim), dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=(1, 1, query_dim, end_dim), dtype=np.float16),
            ct.TensorType(name="position_offset", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=states,
        compute_precision=compute_precision,
        compute_units=cu,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlpkg.save(str(out_path))


def convert_stateless(model: Qwen3Model, max_prompt_length: int,
                       out_path: Path, precision: str, compute_units: str) -> None:
    import coremltools as ct

    model.eval()
    # Trace a fixed-length prompt. ANE prefers fixed shapes — using
    # RangeDim works but ANE often falls back to GPU on dynamic dims.
    # The serve path can ALWAYS prefill at exactly `max_prompt_length`
    # (pad or chunk) so this is the simpler interface.
    example = torch.zeros(1, max_prompt_length, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    compute_precision = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    compute_unit_map = {
        "ane":  ct.ComputeUnit.CPU_AND_NE,
        "gpu":  ct.ComputeUnit.CPU_AND_GPU,
        "all":  ct.ComputeUnit.ALL,
        "cpu":  ct.ComputeUnit.CPU_ONLY,
    }
    cu = compute_unit_map.get(compute_units.lower(), ct.ComputeUnit.CPU_AND_NE)

    mlpkg = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids",
                               shape=(1, max_prompt_length),
                               dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=compute_precision,
        compute_units=cu,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlpkg.save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True, help="HF dir with config.json + safetensors")
    parser.add_argument("--out", required=True, help="output .mlpackage path")
    parser.add_argument("--mode", choices=["stateless", "parity-only", "stateful"], default="stateless",
                         help="stateless = M2 prompt prefill; parity-only = skip CoreML convert; "
                              "stateful = M3 (NOT IMPLEMENTED YET)")
    parser.add_argument("--max-prompt-length", type=int, default=512,
                         help="fixed prompt length for tracing (ANE prefers fixed shapes)")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                         help="CoreML compute precision (fp16 = ANE-friendly)")
    parser.add_argument("--compute-units", choices=["ane", "gpu", "all", "cpu"], default="ane",
                         help="CoreML compute_units hint")
    parser.add_argument("--parity-prompt", default="The capital of France is",
                         help="prompt for the parity diagnostic")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    out_path = Path(args.out)

    print(f"[1/3] loading Qwen3 from {hf_dir} …")
    model, config = load_qwen3_from_hf(hf_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      ✓ loaded — {n_params/1e6:.1f}M params, {config['num_hidden_layers']} layers, "
          f"vocab={config['vocab_size']}")

    # We can run the parity probe without a real tokenizer by emitting
    # token IDs in a known prompt (caller can echo them through both
    # paths and compare). Pull the HF tokenizer if available.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(hf_dir))
        ids = tok.encode(args.parity_prompt)
        print(f"[2/3] parity prompt: {args.parity_prompt!r} → {len(ids)} tokens")
    except Exception as e:
        print(f"[2/3] no transformers tokenizer ({e}); using fixed dummy ids")
        ids = [464, 3139, 286, 4881, 374]
    diag = parity_check(model, hf_dir, ids)
    diag["parity_prompt"] = args.parity_prompt
    diag["parity_token_ids"] = ids
    print(json.dumps(diag, indent=2))

    if args.mode == "parity-only":
        print("[3/3] mode=parity-only — skipping CoreML conversion")
        return

    if args.mode == "stateful":
        # Re-load the model in stateful flavor — it has KV-cache buffers
        # registered per block that the convert path needs.
        del model
        max_seq = args.max_prompt_length
        print(f"[3/3] re-loading model in stateful flavor (max_seq_len={max_seq}) …")
        model, _ = load_qwen3_from_hf(hf_dir, stateful_max_seq=max_seq)
        print(f"      tracing decode-shape (T_new=1) + converting "
              f"(precision={args.precision}, compute_units={args.compute_units}) …")
        print("      RangeDim wires both prefill and decode into one mlpackage")
        convert_stateful(model, max_seq_len=max_seq, out_path=out_path,
                          precision=args.precision, compute_units=args.compute_units)
        print(f"      ✓ wrote {out_path} (stateful)")
        print()
        print(f"validate via:")
        print(f"  tinygpt ane-validate --coreml {out_path} --hf-dir {hf_dir} \\")
        print(f"      --prompt {args.parity_prompt!r}  --stateful")
        return

    # Pad the prompt to max_prompt_length so we can use it for an
    # end-to-end smoke check of the converted model later.
    print(f"[3/3] tracing + converting (max_prompt_length={args.max_prompt_length}, "
          f"precision={args.precision}, compute_units={args.compute_units}) …")
    print("      this step takes several minutes on a 0.6B model — be patient")
    convert_stateless(model,
                       max_prompt_length=args.max_prompt_length,
                       out_path=out_path,
                       precision=args.precision,
                       compute_units=args.compute_units)
    print(f"      ✓ wrote {out_path}")
    print()
    print(f"validate via:")
    print(f"  open {out_path}     # Xcode → Instruments → Core ML profile")
    print(f"  tinygpt ane-validate --coreml {out_path} --hf-dir {hf_dir} \\")
    print(f"      --prompt {args.parity_prompt!r}")


if __name__ == "__main__":
    main()
