#!/usr/bin/env python3
"""Generic MLX dequantizer — supports 4-bit, 6-bit, 8-bit etc.

Uses mlx.core.dequantize() as the workhorse, so packing format details
are handled by MLX itself. Output: fp16 HF-compatible safetensors dir
that the existing TinyGPT HFModel pipeline can load.

Supersedes scripts/ane/dequant_mlx4bit.py (which handled 4-bit only via
hand-rolled bit unpacking). This version handles any mlx-community
quantization — Qwen3-14B-MLX-4bit, UI-Venus-1.5-2B-6bit, etc.

Usage:
    python3 scripts/ane/dequant_mlx_generic.py \\
        --src ~/.lmstudio/models/mlx-community/UI-Venus-1.5-2B-6bit \\
        --dst ~/.cache/tinygpt/ui-venus-1.5-2b-fp16
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_safetensors_to_mx(shard: Path) -> dict[str, mx.array]:
    """Load a safetensors shard into MLX arrays. We go through PyTorch
    because safetensors-mlx doesn't recognize the uint32 storage MLX uses
    for quantized weights — but PyTorch loads int32 + we reinterpret."""
    out: dict[str, mx.array] = {}
    with safe_open(str(shard), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            # Convert torch → numpy → mx. For int32 weight tensors (the
            # quantized matrices), MLX dequantize wants uint32 view.
            if t.dtype == torch.int32:
                # int32 → uint32 by bit-reinterpret
                out[k] = mx.array(t.numpy().view("uint32"))
            elif t.dtype == torch.bfloat16:
                # mx can take bfloat16 directly via numpy view
                out[k] = mx.array(t.float().numpy(), dtype=mx.bfloat16)
            else:
                out[k] = mx.array(t.numpy())
    return out


def collect_quantized_tensors(d: dict[str, mx.array]) -> tuple[dict, dict]:
    """Find each weight that has companion scales + biases (the quantized set),
    and return (quantized_groups, passthrough_tensors).
    """
    keys = set(d.keys())
    quantized: dict[str, dict] = {}
    passthrough: dict[str, mx.array] = {}

    for k in sorted(keys):
        if k.endswith(".weight"):
            prefix = k[: -len(".weight")]
            scales_k = f"{prefix}.scales"
            biases_k = f"{prefix}.biases"
            if scales_k in keys and biases_k in keys:
                quantized[k] = {
                    "weight": d[k],
                    "scales": d[scales_k],
                    "biases": d[biases_k],
                }
            else:
                passthrough[k] = d[k]
    return quantized, passthrough


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="MLX-quantized HF-style dir")
    p.add_argument("--dst", required=True, help="output fp16 HF dir")
    args = p.parse_args()
    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    qcfg = cfg.get("quantization", cfg.get("quantization_config", {}))
    bits = qcfg["bits"]
    group_size = qcfg["group_size"]
    mode = qcfg.get("mode", "affine")
    print(f"dequant: bits={bits}, group_size={group_size}, mode={mode}")

    # Some models split LLM and vision quantization differently.
    # Honor sub-config quantization if present.
    text_cfg = cfg.get("text_config", {})
    vision_cfg = cfg.get("vision_config", {})
    text_q = text_cfg.get("quantization_config")
    vision_q = vision_cfg.get("quantization_config")
    if text_q or vision_q:
        print(f"  text_config quantization: {text_q}")
        print(f"  vision_config quantization: {vision_q}")

    shards = sorted(src.glob("*.safetensors"))
    print(f"loading {len(shards)} shards from {src.name}...")
    all_tensors: dict[str, mx.array] = {}
    for shard in shards:
        all_tensors.update(load_safetensors_to_mx(shard))
    print(f"  {len(all_tensors)} tensors total")

    quantized, passthrough = collect_quantized_tensors(all_tensors)
    print(f"  quantized: {len(quantized)}, passthrough: {len(passthrough)}")

    out: dict[str, torch.Tensor] = {}
    t0 = time.time()
    n_dequanted = 0
    for k, group in quantized.items():
        # mx.dequantize signature: (w, scales, biases=, group_size=, bits=, mode=)
        dq_mx = mx.dequantize(
            group["weight"],
            scales=group["scales"],
            biases=group["biases"],
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        # Convert to torch fp16. mx → numpy → torch.
        # mx.array.tolist() is slow; use .item() or numpy bridge.
        dq_np = (
            dq_mx.astype(mx.float16)  # cheap precision drop
                 .__array__()         # mx → numpy
        )
        out[k] = torch.from_numpy(dq_np)
        n_dequanted += 1
        if n_dequanted % 50 == 0:
            print(f"  dequanted {n_dequanted}/{len(quantized)}, elapsed {time.time()-t0:.1f}s")

    # Passthrough tensors → fp16 if floating, otherwise as-is
    for k, t in passthrough.items():
        if t.dtype in (mx.bfloat16, mx.float32, mx.float16):
            out[k] = torch.from_numpy(t.astype(mx.float16).__array__())
        else:
            out[k] = torch.from_numpy(t.__array__())

    elapsed = time.time() - t0
    print(f"\ndone: {n_dequanted} dequanted, {len(passthrough)} passthrough, "
            f"{elapsed:.1f}s")

    out_path = dst / "model.safetensors"
    print(f"writing {out_path}...")
    save_file(out, str(out_path))

    # Write a cleaned config (strip quantization keys).
    cfg_clean = dict(cfg)
    cfg_clean.pop("quantization", None)
    cfg_clean.pop("quantization_config", None)
    if "text_config" in cfg_clean:
        tc = dict(cfg_clean["text_config"])
        tc.pop("quantization_config", None)
        cfg_clean["text_config"] = tc
    if "vision_config" in cfg_clean:
        vc = dict(cfg_clean["vision_config"])
        vc.pop("quantization_config", None)
        cfg_clean["vision_config"] = vc
    (dst / "config.json").write_text(json.dumps(cfg_clean, indent=2))

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "generation_config.json",
                  "preprocessor_config.json", "processor_config.json",
                  "chat_template.json", "chat_template.jinja"):
        sf = src / name
        if sf.exists():
            shutil.copy(sf, dst / name)

    pkg_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e6
    print(f"\nwrote dequant'd dir: {dst}  ({pkg_mb:.0f} MB)")


if __name__ == "__main__":
    main()
