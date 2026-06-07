#!/usr/bin/env python3
"""M8 prototype — export one Qwen3 block as a standalone stateful mlpackage,
verify it runs on ANE, and check parity vs the torch reference.

Per M6 findings (docs/learn/ane-research/m6-findings.md), ANE runs a 1-layer
stateful Qwen3 cleanly and crashes on N≥2. M8 sidesteps that by packaging
each of the 28 blocks as its OWN 1-layer mlpackage with its own private
state. Swift orchestrates the 28 sequential predict calls per token.

This script validates the smallest unit: convert ONE block, ANE-load it,
predict, and compare to the same block run in PyTorch fp32.

Usage:
    python3 scripts/ane/m8_block_export.py \
        --hf-dir <Qwen3-0.6B HF snapshot> \
        --block-index 0 \
        --max-seq 128 \
        --out ~/.cache/tinygpt/ane/m8-block-0.mlpackage
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from qwen3_to_coreml import Qwen3SingleBlockModel, hf_state_to_qwen3
from safetensors.torch import load_file


def load_block(hf_dir: Path, block_index: int, max_seq: int):
    config = json.loads((hf_dir / "config.json").read_text())
    n_layers = config["num_hidden_layers"]
    if not (0 <= block_index < n_layers):
        raise ValueError(f"block_index {block_index} out of range [0, {n_layers})")

    shards = sorted(hf_dir.glob("*.safetensors"))
    state = {}
    for s in shards:
        state.update(load_file(str(s)))
    tie = config.get("tie_word_embeddings", True)
    sd = hf_state_to_qwen3(state, n_layers, tie)

    # Keep only this block's weights, renaming layers.<i>.<sub> → <sub>
    prefix = f"layers.{block_index}."
    block_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            block_sd[k[len(prefix):]] = v.float()

    model = Qwen3SingleBlockModel(config, max_seq_len=max_seq)
    missing, unexpected = model.load_state_dict(block_sd, strict=False)

    def _is_init_buffer(name: str) -> bool:
        return name.startswith("rope_") or name in ("k_cache", "v_cache")

    missing = [m for m in missing if not _is_init_buffer(m)]
    if missing:
        raise RuntimeError(f"missing tensors for block {block_index}: {missing[:6]}")
    if unexpected:
        raise RuntimeError(f"unexpected tensors for block {block_index}: {unexpected[:6]}")
    return model, config


def convert_one_block(model: Qwen3SingleBlockModel, max_seq: int, out_path: Path):
    import coremltools as ct

    model.eval()
    ex_hidden = torch.zeros(1, 1, model.config["hidden_size"], dtype=torch.float32)
    ex_mask = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    ex_pos = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(model, (ex_hidden, ex_mask, ex_pos))

    n_kv = model.n_kv_heads
    head_dim = model.head_dim
    cache_shape = (1, n_kv, max_seq, head_dim)
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape,
                                                   dtype=np.float16),
                       name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape,
                                                   dtype=np.float16),
                       name="v_cache"),
    ]
    query_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq, default=1)
    end_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq, default=1)
    hidden_size = model.config["hidden_size"]
    # FLOAT32 compute + FLOAT16 state — drops per-block fp16 error ~100×
    # (cos_sim 0.999995 → 1.0000000) at ~1.6× predict-time cost. Necessary
    # to keep 28-block chain drift under control. See m6-findings.md.
    mlpkg = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_state",
                            shape=(1, query_dim, hidden_size),
                            dtype=np.float32),
            ct.TensorType(name="causal_mask",
                            shape=(1, 1, query_dim, end_dim),
                            dtype=np.float32),
            ct.TensorType(name="position_offset",
                            shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="hidden_out", dtype=np.float32)],
        states=states,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlpkg.save(str(out_path))


def test_ane_run(out_path: Path, hidden_size: int) -> dict:
    import coremltools as ct
    result = {}
    print(f"loading {out_path.name} on CPU_AND_NE...", flush=True)
    t0 = time.time()
    ml = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    result["ane_load_sec"] = time.time() - t0
    print(f"  ANE load OK in {result['ane_load_sec']:.2f}s", flush=True)

    state = ml.make_state()
    # Fixed-seed hidden state for parity comparison.
    rng = np.random.default_rng(42)
    hidden = rng.standard_normal((1, 1, hidden_size), dtype=np.float32)
    mask = np.zeros((1, 1, 1, 1), dtype=np.float32)
    pos = np.zeros((1,), dtype=np.int32)

    print("  predict (first call may include ANE JIT)...", flush=True)
    t1 = time.time()
    out = ml.predict(
        {"hidden_state": hidden, "causal_mask": mask, "position_offset": pos},
        state=state,
    )
    result["ane_first_predict_ms"] = (time.time() - t1) * 1000
    print(f"  ANE predict OK in {result['ane_first_predict_ms']:.1f}ms — "
            f"hidden_out shape: {out['hidden_out'].shape}", flush=True)
    result["ane_hidden_out"] = out["hidden_out"].copy()

    # Steady-state: 5 more predict calls, take median.
    timings = []
    for i in range(5):
        t = time.time()
        _ = ml.predict(
            {"hidden_state": hidden, "causal_mask": mask, "position_offset": pos},
            state=state,
        )
        timings.append((time.time() - t) * 1000)
    result["ane_steady_predict_ms"] = float(np.median(timings))
    print(f"  ANE steady-state predict (median of 5): "
            f"{result['ane_steady_predict_ms']:.2f}ms", flush=True)
    result["ane_input_hidden"] = hidden
    result["ane_input_mask"] = mask
    return result


def torch_reference(model: Qwen3SingleBlockModel, hidden: np.ndarray,
                     mask: np.ndarray):
    """Same forward in fp32 PyTorch for parity check."""
    model.eval()
    # Reset cache
    model.k_cache.zero_()
    model.v_cache.zero_()
    with torch.no_grad():
        h = torch.from_numpy(hidden).float()
        m = torch.from_numpy(mask).float()
        p = torch.zeros(1, dtype=torch.long)
        out = model(h, m, p)
    return out.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--out", default=None,
                          help="output .mlpackage path (default: ~/.cache/tinygpt/ane/m8-block-N.mlpackage)")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir).expanduser()
    out_path = (Path(args.out).expanduser() if args.out
                 else Path.home() / ".cache/tinygpt/ane" /
                       f"m8-block-{args.block_index}.mlpackage")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        import shutil; shutil.rmtree(out_path)

    print(f"[1/4] loading block {args.block_index} from {hf_dir.name}...", flush=True)
    model, config = load_block(hf_dir, args.block_index, args.max_seq)

    print(f"[2/4] converting block {args.block_index} → mlpackage...", flush=True)
    t0 = time.time()
    convert_one_block(model, args.max_seq, out_path)
    convert_sec = time.time() - t0
    pkg_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6
    print(f"      convert OK in {convert_sec:.1f}s, .mlpackage size = {pkg_mb:.0f} MB",
            flush=True)

    print("[3/4] testing ANE load + predict...", flush=True)
    ane_result = test_ane_run(out_path, config["hidden_size"])

    print("[4/4] torch fp32 reference + parity vs ANE fp16...", flush=True)
    ref = torch_reference(model, ane_result["ane_input_hidden"],
                            ane_result["ane_input_mask"])
    ane = ane_result["ane_hidden_out"]
    diff = np.abs(ref.astype(np.float32) - ane.astype(np.float32))
    # Cosine similarity
    a = ref.astype(np.float32).reshape(-1)
    b = ane.astype(np.float32).reshape(-1)
    cos_sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"      torch ref shape: {ref.shape}", flush=True)
    print(f"      diff stats: mean={diff.mean():.4f}  median={np.median(diff):.4f}  "
            f"p95={np.quantile(diff, 0.95):.4f}  max={diff.max():.4f}", flush=True)
    print(f"      cosine sim:  {cos_sim:.6f}", flush=True)

    summary = {
        "block_index": args.block_index,
        "max_seq": args.max_seq,
        "pkg_size_mb": pkg_mb,
        "convert_sec": convert_sec,
        "ane_load_sec": ane_result["ane_load_sec"],
        "ane_first_predict_ms": ane_result["ane_first_predict_ms"],
        "ane_steady_predict_ms": ane_result["ane_steady_predict_ms"],
        "parity_mean_diff": float(diff.mean()),
        "parity_max_diff": float(diff.max()),
        "parity_cos_sim": cos_sim,
        "out_path": str(out_path),
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    summary_path = out_path.parent / f"m8-block-{args.block_index}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
