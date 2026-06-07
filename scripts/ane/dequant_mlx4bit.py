#!/usr/bin/env python3
"""Dequantize MLX-4bit safetensors → fp16 HF safetensors.

MLX 4-bit format (per group of `group_size` values):
  weight:  uint32 array, packed 8 lanes per uint32 = 8 × 4 bits
  scales:  bf16 array, one per group
  biases:  bf16 array, one per group

Dequant formula:  value = nibble * scale + bias   (nibble ∈ [0, 15])

Used to convert LM Studio's Qwen3-14B-MLX-4bit (already downloaded)
into a drop-in HF dir for the M8 chunked ANE pipeline.

Output: single-file model.safetensors fp16 + copied config/tokenizer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def bf16_bytes_to_fp32(bf16_arr: np.ndarray) -> np.ndarray:
    """bf16 is top 16 bits of fp32. Pad with zeros to recover fp32."""
    u16 = bf16_arr.astype(np.uint16) if bf16_arr.dtype != np.uint16 else bf16_arr
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def dequant_4bit(packed_u32: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                   group_size: int) -> np.ndarray:
    """packed: [..., cols_packed] uint32
       scales/biases: [..., n_groups] fp32
       returns: [..., n_groups * group_size] fp32
    """
    *lead, cp = packed_u32.shape
    # Unpack 8 lanes per uint32 → [..., cp, 8]
    shifts = np.arange(8, dtype=np.uint32) * 4
    nibbles = (packed_u32[..., :, None] >> shifts) & np.uint32(0xF)
    nibbles = nibbles.astype(np.float32).reshape(*lead, cp * 8)  # [..., cols]
    n_groups = scales.shape[-1]
    assert nibbles.shape[-1] == n_groups * group_size, (
        f"unpacked cols {nibbles.shape[-1]} != n_groups {n_groups} * group_size {group_size}")
    nibbles = nibbles.reshape(*lead, n_groups, group_size)
    s = scales[..., None].astype(np.float32)
    b = biases[..., None].astype(np.float32)
    out = nibbles * s + b
    return out.reshape(*lead, n_groups * group_size)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="MLX-4bit HF-style dir")
    p.add_argument("--dst", required=True, help="output fp16 HF dir")
    args = p.parse_args()
    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    qcfg = cfg.get("quantization", cfg.get("quantization_config", {}))
    group_size = qcfg["group_size"]
    bits = qcfg["bits"]
    assert bits == 4, f"only 4-bit supported, got {bits}"
    print(f"dequant: group_size={group_size}, bits={bits}")

    # Pool all tensor names by base.
    shards = sorted(src.glob("*.safetensors"))
    print(f"loading {len(shards)} shards from {src.name}...")

    # Index every tensor to its shard.
    shard_for_key: dict[str, Path] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                shard_for_key[k] = shard
    print(f"  {len(shard_for_key)} tensors total")

    shard_readers = {s: safe_open(s, framework="pt").__enter__() for s in shards}

    # Find all .weight tensors. For each, check if sibling .scales/.biases exist.
    weight_keys = [k for k in shard_for_key if k.endswith(".weight")]
    print(f"  {len(weight_keys)} .weight tensors")

    out: dict[str, torch.Tensor] = {}
    t0 = time.time()
    n_dequanted = 0
    n_passthrough = 0
    for wkey in sorted(weight_keys):
        prefix = wkey[: -len(".weight")]  # e.g. "model.layers.0.mlp.gate_proj"
        scales_key = f"{prefix}.scales"
        biases_key = f"{prefix}.biases"
        if scales_key in shard_for_key and biases_key in shard_for_key:
            # Quantized: dequant.
            w_packed_t = shard_readers[shard_for_key[wkey]].get_tensor(wkey)
            s_t = shard_readers[shard_for_key[scales_key]].get_tensor(scales_key)
            b_t = shard_readers[shard_for_key[biases_key]].get_tensor(biases_key)
            # Pre-quant weights stored as uint32 (the dtype came through as int32
            # in our earlier probe). Use view to reinterpret without copy.
            if w_packed_t.dtype == torch.int32:
                w_np = w_packed_t.numpy().view(np.uint32)
            elif w_packed_t.dtype == torch.uint32:
                w_np = w_packed_t.numpy()
            else:
                raise TypeError(f"expected int32/uint32 for {wkey}, got {w_packed_t.dtype}")
            # bf16 → fp32 via uint16 view.
            s_np = bf16_bytes_to_fp32(s_t.view(torch.int16).numpy().view(np.uint16))
            b_np = bf16_bytes_to_fp32(b_t.view(torch.int16).numpy().view(np.uint16))
            dq = dequant_4bit(w_np, s_np, b_np, group_size)
            out[wkey] = torch.from_numpy(dq).to(torch.float16)
            n_dequanted += 1
            if n_dequanted % 50 == 0:
                print(f"  dequanted {n_dequanted}/{len(weight_keys)}, "
                        f"elapsed {time.time()-t0:.1f}s", flush=True)
        else:
            # Non-quantized (layernorms, etc.) → fp16 passthrough.
            t = shard_readers[shard_for_key[wkey]].get_tensor(wkey)
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float16)
            out[wkey] = t
            n_passthrough += 1

    elapsed = time.time() - t0
    print(f"\ndone: {n_dequanted} dequanted, {n_passthrough} passthrough, "
            f"{elapsed:.1f}s")

    # Write single shard
    out_path = dst / "model.safetensors"
    print(f"writing {out_path}...")
    save_file(out, str(out_path))

    # Copy non-safetensors files (config etc.); strip quantization keys from config.
    cfg_out = dict(cfg)
    cfg_out.pop("quantization", None)
    cfg_out.pop("quantization_config", None)
    (dst / "config.json").write_text(json.dumps(cfg_out, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "generation_config.json"):
        src_f = src / name
        if src_f.exists():
            shutil.copy(src_f, dst / name)
    pkg_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e6
    print(f"\nwrote dequant'd dir: {dst}  ({pkg_mb:.0f} MB)")


if __name__ == "__main__":
    main()
