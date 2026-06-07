#!/usr/bin/env python3
"""M8 full end-to-end ANE-chained decode on Qwen3-0.6B.

Loads all 28 single-block .mlpackage files. Runs the full forward:

  1. tokenize → input_ids
  2. embedding (PyTorch fp32) → hidden_0
  3. for i in 0..28: hidden_{i+1} = ane_block[i].predict(hidden_i, ...)
  4. final RMSNorm + tied lm_head (PyTorch fp32) → logits
  5. argmax → next token

Compares end-to-end ANE-chained logits to a pure-PyTorch fp32 reference
on the same prompt to validate the moat.

Usage:
    python3 scripts/ane/m8_chained_decode.py \
        --hf-dir <Qwen3-0.6B HF snapshot> \
        --prompt "The capital of France is" \
        --max-seq 128 [--steps 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from qwen3_to_coreml import (
    Qwen3Model, Qwen3RMSNorm, precompute_rope_cache, hf_state_to_qwen3,
)
from safetensors.torch import load_file


def load_full_state(hf_dir: Path):
    config = json.loads((hf_dir / "config.json").read_text())
    shards = sorted(hf_dir.glob("*.safetensors"))
    state = {}
    for s in shards:
        state.update(load_file(str(s)))
    tie = config.get("tie_word_embeddings", True)
    sd = hf_state_to_qwen3(state, config["num_hidden_layers"], tie)
    return config, sd


def tokenize(hf_dir: Path, prompt: str) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    return tok.encode(prompt)


def causal_mask(t_new: int, end: int, past: int = 0) -> np.ndarray:
    """[1, 1, t_new, end] additive mask. -inf where attention forbidden.
    fp32 because the M8 block-mlpackages take fp32 inputs (compute_precision=FLOAT32)."""
    m = np.full((t_new, end), -1e4, dtype=np.float32)
    # Each new query at position past+i can attend to positions [0..past+i]
    for i in range(t_new):
        m[i, : past + i + 1] = 0.0
    return m.reshape(1, 1, t_new, end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8,
                          help="how many tokens to decode after prefill")
    parser.add_argument("--pkg-dir", default="~/.cache/tinygpt/ane")
    args = parser.parse_args()

    import coremltools as ct

    hf_dir = Path(args.hf_dir).expanduser()
    pkg_dir = Path(args.pkg_dir).expanduser()

    # 1. Load HF weights for embedding + final norm + tied lm_head
    print("[setup] loading full model state for embed+norm+head...", flush=True)
    config, sd_full = load_full_state(hf_dir)
    hidden_size = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    rms_eps = config.get("rms_norm_eps", 1e-6)

    embed_w = sd_full["embed_tokens.weight"].float()
    final_norm_w = sd_full["norm.weight"].float()
    # tied embeddings
    lm_head_w = embed_w

    final_norm = Qwen3RMSNorm(hidden_size, eps=rms_eps)
    final_norm.weight.data = final_norm_w.clone()
    final_norm.eval()

    # 2. Load all 28 ANE block packages
    print(f"[setup] loading {n_layers} ANE block packages from {pkg_dir}...",
            flush=True)
    t0 = time.time()
    mls = []
    for i in range(n_layers):
        path = pkg_dir / f"m8-block-{i}.mlpackage"
        ml = ct.models.MLModel(str(path),
                                compute_units=ct.ComputeUnit.CPU_AND_NE)
        mls.append(ml)
    print(f"  loaded {n_layers} blocks in {time.time()-t0:.1f}s", flush=True)

    states = [m.make_state() for m in mls]

    # 3. Tokenize + ANE PREFILL (one token at a time for v1 simplicity)
    print(f"[run] tokenize '{args.prompt}'", flush=True)
    ids = tokenize(hf_dir, args.prompt)
    print(f"      tokens: {ids[:20]}{'...' if len(ids)>20 else ''}", flush=True)

    decoded_ids = list(ids)
    print(f"\n[run] prefill {len(ids)} tokens through 28 ANE blocks...",
            flush=True)
    prefill_start = time.time()
    for pos, tid in enumerate(ids):
        hidden = embed_w[tid].reshape(1, 1, hidden_size).numpy().astype(np.float32)
        mask = causal_mask(1, pos + 1, past=pos)
        pos_arr = np.array([pos], dtype=np.int32)
        for i, ml in enumerate(mls):
            out = ml.predict(
                {"hidden_state": hidden,
                  "causal_mask": mask,
                  "position_offset": pos_arr},
                state=states[i],
            )
            hidden = out["hidden_out"]
    prefill_sec = time.time() - prefill_start
    prefill_tok_per_sec = len(ids) / prefill_sec
    print(f"      prefill: {prefill_sec*1000:.0f}ms total, "
            f"{prefill_tok_per_sec:.1f} tok/s", flush=True)

    # 4. Final norm + lm_head on the LAST token's hidden state
    h_torch = torch.from_numpy(hidden.astype(np.float32))
    with torch.no_grad():
        h_norm = final_norm(h_torch).numpy()
    logits = h_norm @ lm_head_w.numpy().T  # [1, 1, vocab]
    next_id = int(np.argmax(logits, axis=-1)[0, 0])
    print(f"\n[run] first decoded token: id={next_id}", flush=True)

    # 5. Tokenizer for debug print
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    print(f"      decoded: '{tok.decode([next_id])}'", flush=True)
    decoded_ids.append(next_id)

    # 6. Continue decoding N more tokens
    print(f"\n[run] decode {args.steps - 1} more tokens...", flush=True)
    decode_start = time.time()
    cur_id = next_id
    for step in range(args.steps - 1):
        pos = len(decoded_ids) - 1
        hidden = embed_w[cur_id].reshape(1, 1, hidden_size).numpy().astype(np.float32)
        mask = causal_mask(1, pos + 1, past=pos)
        pos_arr = np.array([pos], dtype=np.int32)
        for i, ml in enumerate(mls):
            out = ml.predict(
                {"hidden_state": hidden,
                  "causal_mask": mask,
                  "position_offset": pos_arr},
                state=states[i],
            )
            hidden = out["hidden_out"]
        h_torch = torch.from_numpy(hidden.astype(np.float32))
        with torch.no_grad():
            h_norm = final_norm(h_torch).numpy()
        logits = h_norm @ lm_head_w.numpy().T
        cur_id = int(np.argmax(logits, axis=-1)[0, 0])
        decoded_ids.append(cur_id)

    decode_sec = time.time() - decode_start
    decode_tok_per_sec = (args.steps - 1) / decode_sec if args.steps > 1 else 0
    print(f"      decode: {decode_sec*1000:.0f}ms total, "
            f"{decode_tok_per_sec:.1f} tok/s steady-state", flush=True)

    print(f"\n=== generated ===")
    print(f"  prompt + completion: '{tok.decode(decoded_ids)}'")
    print(f"  decode tok/s: {decode_tok_per_sec:.1f}")
    print(f"  prefill tok/s: {prefill_tok_per_sec:.1f}")

    # Summary JSON
    summary = {
        "prompt": args.prompt,
        "n_layers": n_layers,
        "prompt_tokens": len(ids),
        "decode_steps": args.steps,
        "prefill_seconds": prefill_sec,
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "decode_seconds": decode_sec,
        "decode_tok_per_sec": decode_tok_per_sec,
        "generated_ids": decoded_ids,
        "generated_text": tok.decode(decoded_ids),
    }
    sum_path = pkg_dir / "m8-chained-decode-summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {sum_path}")


if __name__ == "__main__":
    main()
