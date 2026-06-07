#!/usr/bin/env python3
"""M6 diagnostic — layer-count bisect of ANECCompile error -14.

Hypothesis: ANECCompile fails on the full 28-layer Qwen3-0.6B stateful
graph because the graph is too deep / too many ops, not because of any
specific op. Bisect by truncating to N layers, converting, and trying
to load on CPU_AND_NE — that's the step that fires ANECCompile.

Outputs `~/.cache/tinygpt/ane/m6-bisect-results.json` with one entry per
N tested, plus a printed summary table.

Usage:
    python3 scripts/ane/m6_layer_bisect.py \
        --hf-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha> \
        [--layers 1,4,8,14,20,28] [--max-seq 128]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Make sibling module importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from qwen3_to_coreml import (
    Qwen3StatefulModel,
    convert_stateful,
    hf_state_to_qwen3,
)
from safetensors.torch import load_file


def truncate_state_dict(sd: dict, n_keep: int) -> dict:
    """Drop weights for layer indices >= n_keep."""
    out = {}
    for k, v in sd.items():
        if k.startswith("layers."):
            rest = k[len("layers."):]
            idx_str, _, _ = rest.partition(".")
            try:
                idx = int(idx_str)
            except ValueError:
                out[k] = v
                continue
            if idx >= n_keep:
                continue
        out[k] = v
    return out


def build_truncated_model(hf_dir: Path, n_layers: int, max_seq_len: int):
    config = json.loads((hf_dir / "config.json").read_text())
    config = copy.deepcopy(config)
    config["num_hidden_layers"] = n_layers

    shards = sorted(hf_dir.glob("*.safetensors"))
    state = {}
    for s in shards:
        state.update(load_file(str(s)))
    tie = config.get("tie_word_embeddings", True)
    sd = hf_state_to_qwen3(state, n_layers, tie)
    sd = truncate_state_dict(sd, n_layers)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    if tie and "lm_head.weight" in sd:
        sd = {k: v for k, v in sd.items() if k != "lm_head.weight"}

    model = Qwen3StatefulModel(config, max_seq_len=max_seq_len)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    def _is_init_buffer(name: str) -> bool:
        return name.startswith("rope_") or name in ("k_cache", "v_cache")

    missing = [m for m in missing if not _is_init_buffer(m)]
    if missing:
        raise RuntimeError(f"N={n_layers}: missing tensors after truncate: {missing[:6]}")
    if unexpected:
        raise RuntimeError(f"N={n_layers}: unexpected tensors: {unexpected[:6]}")
    return model, config


def attempt_ane_load(mlpkg_path: Path, n_layers: int, n_kv: int, head_dim: int,
                       max_seq: int) -> dict:
    """Load .mlpackage with CPU_AND_NE and try a single decode step.
    The load+predict path is where ANECCompile actually runs.
    """
    import coremltools as ct
    result = {"ane_load_ok": False, "ane_decode_ok": False,
               "load_seconds": None, "decode_seconds": None,
               "error": None}
    t0 = time.time()
    try:
        ml = ct.models.MLModel(str(mlpkg_path),
                                compute_units=ct.ComputeUnit.CPU_AND_NE)
        result["load_seconds"] = time.time() - t0
        result["ane_load_ok"] = True
    except Exception as e:
        result["error"] = f"ANE_LOAD: {type(e).__name__}: {str(e)[:200]}"
        return result

    # Single decode step at T_new=1.
    try:
        state = ml.make_state()
        ids = np.zeros((1, 1), dtype=np.int32)
        mask = np.zeros((1, 1, 1, 1), dtype=np.float16)
        pos = np.zeros((1,), dtype=np.int32)
        t1 = time.time()
        _ = ml.predict(
            {"input_ids": ids, "causal_mask": mask, "position_offset": pos},
            state=state,
        )
        result["decode_seconds"] = time.time() - t1
        result["ane_decode_ok"] = True
    except Exception as e:
        result["error"] = f"ANE_DECODE: {type(e).__name__}: {str(e)[:200]}"
    return result


def attempt_gpu_load(mlpkg_path: Path) -> dict:
    """Sanity check: model loads on CPU+GPU. If yes, conversion is fine
    and only the ANE path is the problem.
    """
    import coremltools as ct
    result = {"gpu_load_ok": False, "error": None}
    try:
        _ = ct.models.MLModel(str(mlpkg_path),
                                compute_units=ct.ComputeUnit.CPU_AND_GPU)
        result["gpu_load_ok"] = True
    except Exception as e:
        result["error"] = f"GPU_LOAD: {type(e).__name__}: {str(e)[:200]}"
    return result


def run_one(hf_dir: Path, n_layers: int, max_seq: int, work_dir: Path) -> dict:
    print(f"\n[N={n_layers}] building truncated model + converting...")
    t0 = time.time()
    model, config = build_truncated_model(hf_dir, n_layers, max_seq)
    out_path = work_dir / f"bisect-n{n_layers}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)
    try:
        convert_stateful(model, max_seq_len=max_seq, out_path=out_path,
                           precision="fp16", compute_units="ane")
        convert_seconds = time.time() - t0
        convert_ok = True
        convert_error = None
    except Exception as e:
        return {
            "n_layers": n_layers, "convert_ok": False,
            "convert_seconds": time.time() - t0,
            "convert_error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    n_kv = config["num_key_value_heads"]
    head_dim = config.get("head_dim",
                            config["hidden_size"] // config["num_attention_heads"])
    pkg_size_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6

    print(f"[N={n_layers}]   convert OK in {convert_seconds:.1f}s, "
            f".mlpackage size = {pkg_size_mb:.0f} MB")
    print(f"[N={n_layers}]   attempting ANE load...")
    ane = attempt_ane_load(out_path, n_layers, n_kv, head_dim, max_seq)
    print(f"[N={n_layers}]   attempting GPU load (sanity check)...")
    gpu = attempt_gpu_load(out_path)

    # Cleanup: keep package only if the smallest two compiled (for reuse).
    if n_layers > 4:
        shutil.rmtree(out_path, ignore_errors=True)

    return {
        "n_layers": n_layers,
        "convert_ok": convert_ok, "convert_seconds": convert_seconds,
        "convert_error": convert_error,
        "pkg_size_mb": pkg_size_mb,
        **ane,
        **gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--layers", default="1,4,8,14,20,28",
                          help="comma-separated layer counts to test")
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--work-dir", default="~/.cache/tinygpt/ane")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir).expanduser()
    work_dir = Path(args.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    layer_counts = [int(x) for x in args.layers.split(",")]

    results = []
    for n in layer_counts:
        try:
            r = run_one(hf_dir, n, args.max_seq, work_dir)
        except Exception as e:
            traceback.print_exc()
            r = {"n_layers": n, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        results.append(r)

    out_json = work_dir / "m6-bisect-results.json"
    out_json.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 72)
    print(f"{'N':>4} | {'convert':>8} | {'ane-load':>9} | "
            f"{'ane-decode':>11} | {'gpu-load':>9} | size MB")
    print("-" * 72)
    for r in results:
        n = r.get("n_layers", "?")
        c = "OK" if r.get("convert_ok") else "FAIL"
        al = "OK" if r.get("ane_load_ok") else "FAIL"
        ad = "OK" if r.get("ane_decode_ok") else "FAIL"
        gl = "OK" if r.get("gpu_load_ok") else "FAIL"
        sz = f"{r.get('pkg_size_mb', 0):.0f}"
        print(f"{n:>4} | {c:>8} | {al:>9} | {ad:>11} | {gl:>9} | {sz:>7}")
    print(f"\nFull results: {out_json}")


if __name__ == "__main__":
    main()
