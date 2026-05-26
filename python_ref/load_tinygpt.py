"""
load_tinygpt.py — read a .tinygpt model file (the format exported by the
browser playground) into a PyTorch state_dict.

The .tinygpt file is self-describing — its JSON header carries a `manifest`
that names every tensor in PyTorch-compatible form (`token_embedding.weight`,
`blocks.0.attn.q_proj.weight`, …) plus its shape. So loading into the
python_ref/model.py reference model is a one-step mapping.

The browser exports include the AdamW optimiser state (m, v) interleaved with
each weight. By default we drop those and return only the weights — that's
what you want for further training or inference. Pass `with_optimizer=True` to
get the m/v moments too as separate entries (named `<weight>.m` and `<weight>.v`).

    python python_ref/load_tinygpt.py --in model.tinygpt --out model.pt
    python python_ref/load_tinygpt.py --in model.tinygpt --inspect

Or programmatically:

    from python_ref.load_tinygpt import load_tinygpt
    config, state_dict = load_tinygpt("model.tinygpt")
    model.load_state_dict(state_dict, strict=False)

The browser stores weights row-major in the same shape PyTorch expects, so no
transposes are needed.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MAGIC = b"TGPT"
SUPPORTED_VERSIONS = {1, 2}


def load_tinygpt(
    path: str | Path,
    with_optimizer: bool = False,
) -> tuple[dict[str, Any], dict[str, "np.ndarray | torch.Tensor"]]:
    """Read a .tinygpt file and return (config, state_dict).

    `state_dict` maps tensor name -> ndarray (or torch.Tensor if torch is
    installed). Keys match python_ref/model.py's PyTorch state_dict by default.
    """
    path = Path(path)
    with path.open("rb") as f:
        prefix = f.read(12)
        if len(prefix) != 12:
            raise ValueError(f"{path} too small to be a .tinygpt file")
        magic, version, header_len = struct.unpack("<4sII", prefix)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r} (expected {MAGIC!r})")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"{path}: unsupported version {version}")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path}: header truncated")
        header = json.loads(header_bytes)
        # v1 didn't include the manifest. v2 does.
        manifest = header.get("manifest")
        if manifest is None:
            raise ValueError(
                f"{path}: no tensor manifest in header (file is v1, pre-safetensors). "
                "Re-export from the browser to get a v2 file."
            )
        # State buffer layout: int32 step + per-param triplets of [w, m, v].
        f.read(4)  # skip step counter
        state_dict: dict[str, Any] = {}
        for entry in manifest:
            name = entry["name"]
            shape = tuple(entry["shape"])
            n = int(np.prod(shape))
            n_bytes = n * 4
            w = np.frombuffer(f.read(n_bytes), dtype=np.float32).reshape(shape).copy()
            m_buf = f.read(n_bytes)  # AdamW first moment
            v_buf = f.read(n_bytes)  # AdamW second moment
            state_dict[name] = w
            if with_optimizer:
                state_dict[f"{name}.m"] = (
                    np.frombuffer(m_buf, dtype=np.float32).reshape(shape).copy()
                )
                state_dict[f"{name}.v"] = (
                    np.frombuffer(v_buf, dtype=np.float32).reshape(shape).copy()
                )
        # Drop config out — surface it next to the state dict for convenience.
        config = header.get("config", {})
        # Include rich metadata too if present (v2 files include loss history + sample).
        config["_tinygpt_meta"] = {
            "savedAt": header.get("savedAt"),
            "finalLoss": header.get("finalLoss"),
            "bestVal": header.get("bestVal"),
            "sample": header.get("sample"),
            "lossHistory": header.get("lossHistory"),
        }
        if HAS_TORCH:
            state_dict = {k: torch.from_numpy(v) for k, v in state_dict.items()}
        return config, state_dict


def inspect(path: str | Path) -> None:
    """Print a human-readable summary of the file's tensors and metadata."""
    config, state_dict = load_tinygpt(path)
    meta = config.get("_tinygpt_meta", {})
    print(f"\nFile: {path}")
    print("-" * 64)
    print("Config:")
    for key in ("layers", "dModel", "ctx", "heads", "dMlp", "batchSize", "backend"):
        if key in config:
            print(f"  {key:14s} {config[key]}")
    if meta.get("finalLoss"):
        fl = meta["finalLoss"]
        train = fl.get("train")
        val = fl.get("val")
        step = fl.get("step")
        line = f"  final loss     train {train:.3f}"
        if val is not None:
            line += f", val {val:.3f}"
        line += f" @ step {step}"
        print(line)
    if meta.get("sample"):
        print(f"\n  sample:        {meta['sample'][:80]!r}")
    print(f"\nTensors ({len(state_dict)}):")
    total_params = 0
    for name, t in state_dict.items():
        shape = tuple(t.shape) if HAS_TORCH else t.shape
        n = int(np.prod(shape))
        total_params += n
        print(f"  {name:40s} {str(shape):20s} {n:,}")
    print("-" * 64)
    print(f"  total parameters: {total_params:,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in", dest="inp", required=True, help=".tinygpt file to read")
    p.add_argument("--out", help="output .pt file (PyTorch state_dict)")
    p.add_argument(
        "--with-optimizer",
        action="store_true",
        help="include AdamW m/v moments in the output",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="print the file's tensors + metadata without writing anything",
    )
    args = p.parse_args()

    if args.inspect:
        inspect(args.inp)
        return

    if not args.out:
        raise SystemExit("provide --out PATH (or --inspect)")
    if not HAS_TORCH:
        raise SystemExit("PyTorch is required for --out; install with `pip install torch`")

    config, state_dict = load_tinygpt(args.inp, with_optimizer=args.with_optimizer)
    out_path = Path(args.out)
    torch.save({"config": config, "state_dict": state_dict}, out_path)
    print(f"wrote {out_path}  ({len(state_dict)} tensors)")


if __name__ == "__main__":
    main()
