#!/usr/bin/env python3
"""m8_numerics_gate.py — automated numerics gate for M8 ANE chain changes.

Project rule: every perf path needs an automated numerics gate or it
doesn't ship. This is that gate for the int8/fp16-IO handoff work (#306).

Runs N fixed prompts × K greedy decode steps through a chunked ANE chain
and either saves the per-step logits as the baseline, or compares a
changed chain against a saved baseline.

PASS criteria (vs baseline):
  - 100% top-1 token match at every step of every prompt
  - per-step logit cosine similarity >= 0.999
  - ' Paris' canary: prompt "The capital of France is" decodes ' Paris' first

Usage:
  # 1. capture baseline from the current (fp32-IO) chain:
  python3 scripts/ane/m8_numerics_gate.py \
      --pkg-dir ~/.cache/tinygpt/ane \
      --hf-dir <Qwen3-0.6B HF snapshot> \
      --save-baseline ~/.cache/tinygpt/ane/m8-gate-baseline.npz

  # 2. after re-export / runtime change, compare:
  python3 scripts/ane/m8_numerics_gate.py \
      --pkg-dir ~/.cache/tinygpt/ane-fp16io \
      --hf-dir <Qwen3-0.6B HF snapshot> \
      --compare ~/.cache/tinygpt/ane/m8-gate-baseline.npz

Exit code 0 = gate PASS, 1 = gate FAIL, 2 = setup error.
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
from m8_chained_decode import load_full_state, tokenize, causal_mask
from qwen3_to_coreml import Qwen3RMSNorm

GATE_PROMPTS = [
    "The capital of France is",
    "2 plus 2 equals",
    "The opposite of hot is",
    "Once upon a time there was a",
    "The chemical symbol for gold is",
]
CANARY_PROMPT = "The capital of France is"
CANARY_TOKEN = " Paris"
STEPS = 8
COS_THRESHOLD = 0.999


def load_chain(pkg_dir: Path, n_layers: int):
    import coremltools as ct
    mls = []
    t0 = time.time()
    for i in range(n_layers):
        path = pkg_dir / f"m8-block-{i}.mlpackage"
        if not path.exists():
            print(f"FATAL: missing {path}", file=sys.stderr)
            sys.exit(2)
        mls.append(ct.models.MLModel(str(path),
                                       compute_units=ct.ComputeUnit.CPU_AND_NE))
    print(f"  loaded {n_layers} blocks in {time.time()-t0:.1f}s", flush=True)
    # Detect hidden_state input dtype from the first block's spec so fp16-IO
    # chains get fp16 inputs without silent fp32→fp16 casts inside predict.
    spec = mls[0].get_spec()
    hidden_dtype = np.float32
    for inp in spec.description.input:
        if inp.name == "hidden_state":
            # ArrayFeatureType.ArrayDataType: FLOAT16 = 65552, FLOAT32 = 65568
            if inp.type.multiArrayType.dataType == 65552:
                hidden_dtype = np.float16
    print(f"  hidden_state IO dtype: {np.dtype(hidden_dtype).name}", flush=True)
    return mls, hidden_dtype


def run_chain(pkg_dir: Path, hf_dir: Path, prompts: list[str], steps: int):
    """Greedy-decode each prompt; return per-prompt per-step logits + ids."""
    config, sd = load_full_state(hf_dir)
    n_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    embed_w = sd["embed_tokens.weight"].float()
    lm_head_w = sd.get("lm_head.weight", sd["embed_tokens.weight"]).float()
    final_norm = Qwen3RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6))
    final_norm.weight.data = sd["norm.weight"].float()
    final_norm.eval()

    mls, hidden_dtype = load_chain(pkg_dir, n_layers)

    all_logits = []   # [prompt][step] -> np.ndarray [vocab]
    all_ids = []      # [prompt] -> list[int] (decoded continuation only)

    for prompt in prompts:
        states = [m.make_state() for m in mls]
        ids = tokenize(hf_dir, prompt)

        def block_pass(token_id: int, pos: int) -> np.ndarray:
            hidden = embed_w[token_id].reshape(1, 1, hidden_size).numpy().astype(hidden_dtype)
            mask = causal_mask(1, pos + 1, past=pos)
            pos_arr = np.array([pos], dtype=np.int32)
            for i, ml in enumerate(mls):
                out = ml.predict(
                    {"hidden_state": hidden,
                     "causal_mask": mask,
                     "position_offset": pos_arr},
                    state=states[i])
                hidden = out["hidden_out"].astype(hidden_dtype)
            h = torch.from_numpy(hidden.astype(np.float32))
            with torch.no_grad():
                h_norm = final_norm(h).numpy()
            return (h_norm @ lm_head_w.numpy().T)[0, 0]  # [vocab]

        # prefill
        logits = None
        for pos, tid in enumerate(ids):
            logits = block_pass(tid, pos)

        # greedy decode `steps` tokens, capturing logits at each step
        step_logits = [logits.astype(np.float32)]
        decoded = [int(np.argmax(logits))]
        for s in range(steps - 1):
            pos = len(ids) + s
            logits = block_pass(decoded[-1], pos)
            step_logits.append(logits.astype(np.float32))
            decoded.append(int(np.argmax(logits)))

        all_logits.append(np.stack(step_logits))   # [steps, vocab]
        all_ids.append(decoded)
        print(f"  '{prompt[:40]}' → ids {decoded}", flush=True)

    return np.stack(all_logits), all_ids   # [P, steps, vocab]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkg-dir", type=Path, required=True)
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--save-baseline", type=Path)
    p.add_argument("--compare", type=Path)
    p.add_argument("--steps", type=int, default=STEPS)
    args = p.parse_args()
    if bool(args.save_baseline) == bool(args.compare):
        print("exactly one of --save-baseline / --compare required", file=sys.stderr)
        sys.exit(2)

    pkg_dir = args.pkg_dir.expanduser()
    hf_dir = args.hf_dir.expanduser()

    print(f"=== m8 numerics gate: running {len(GATE_PROMPTS)} prompts × "
          f"{args.steps} steps on {pkg_dir.name} ===", flush=True)
    logits, ids = run_chain(pkg_dir, hf_dir, GATE_PROMPTS, args.steps)

    if args.save_baseline:
        out = args.save_baseline.expanduser()
        np.savez_compressed(
            out, logits=logits,
            ids=np.array(ids, dtype=np.int64),
            prompts=np.array(GATE_PROMPTS),
            steps=args.steps)
        print(f"\nbaseline saved → {out}")
        # still verify the canary on the baseline itself
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(hf_dir))
        canary_idx = GATE_PROMPTS.index(CANARY_PROMPT)
        first = tok.decode([ids[canary_idx][0]])
        print(f"canary: first token = '{first}' "
              f"({'OK' if first == CANARY_TOKEN else 'MISMATCH — investigate baseline!'})")
        sys.exit(0 if first == CANARY_TOKEN else 1)

    # --- compare mode ---
    base = np.load(args.compare.expanduser(), allow_pickle=True)
    if int(base["steps"]) != args.steps or list(base["prompts"]) != GATE_PROMPTS:
        print("FATAL: baseline prompts/steps mismatch — re-capture baseline",
              file=sys.stderr)
        sys.exit(2)
    base_logits = base["logits"]          # [P, steps, vocab]
    base_ids = base["ids"].tolist()       # [P][steps]

    fails: list[str] = []
    min_cos = 1.0
    for pi, prompt in enumerate(GATE_PROMPTS):
        for s in range(args.steps):
            a, b = base_logits[pi, s], logits[pi, s]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            min_cos = min(min_cos, cos)
            if cos < COS_THRESHOLD:
                fails.append(f"cos {cos:.6f} < {COS_THRESHOLD} at prompt {pi} step {s}")
            if ids[pi][s] != base_ids[pi][s]:
                fails.append(f"top-1 mismatch at prompt {pi} step {s}: "
                             f"{ids[pi][s]} != {base_ids[pi][s]}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    canary_idx = GATE_PROMPTS.index(CANARY_PROMPT)
    first = tok.decode([ids[canary_idx][0]])
    if first != CANARY_TOKEN:
        fails.append(f"canary: first token '{first}' != '{CANARY_TOKEN}'")

    print(f"\nmin logit cosine across all steps: {min_cos:.7f}")
    if fails:
        print(f"\nGATE FAIL ({len(fails)} violations):")
        for f in fails[:20]:
            print(f"  {f}")
        sys.exit(1)
    print("GATE PASS — top-1 100% match, cosine >= 0.999, canary OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
