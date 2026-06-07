#!/usr/bin/env python3
"""M9 — speculative decoding combining:
  - Draft: Qwen3-0.6B running on ANE via the M8 chunked bundle
  - Verify: Qwen3-14B-MLX-4bit running on GPU via mlx_lm

Goal: hit decode speeds higher than either model alone, because:
  (a) the draft is essentially free thermally (runs on ANE in parallel)
  (b) the verify processes K candidates in ONE forward pass

Baselines on this Mac (M5 Pro, macOS 26):
  Qwen3-0.6B pure ANE chunked (Swift):  25.3 tok/s
  Qwen3-14B pure MLX 4-bit:              30.2 tok/s
  Qwen3-14B via LM Studio:               27.9 tok/s

Target: ~40-60 effective tok/s on the 14B output via spec dec.

Usage:
  python3 scripts/ane/m9_spec_decode.py \
      --draft-dir ~/.cache/tinygpt/ane \
      --draft-hf-dir <Qwen3-0.6B-HF> \
      --verify-dir ~/.lmstudio/models/lmstudio-community/Qwen3-14B-MLX-4bit \
      --prompt "The capital of France is" \
      --max-tokens 80 --k 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import coremltools as ct
import mlx.core as mx
from mlx_lm import load as mlx_load
from qwen3_to_coreml import Qwen3RMSNorm, hf_state_to_qwen3
from safetensors.torch import load_file
from transformers import AutoTokenizer


# ----------------- ANE draft side -----------------

class ANEDraft:
    """Wraps the 28 Qwen3 single-block .mlpackage files into a per-token
    decode interface. Caller drives positions; this class manages state."""

    def __init__(self, chunked_dir: Path, hf_dir: Path, max_seq: int):
        self.cfg = json.loads((hf_dir / "config.json").read_text())
        self.n_layers = self.cfg["num_hidden_layers"]
        self.hidden = self.cfg["hidden_size"]
        self.vocab = self.cfg["vocab_size"]
        self.eps = float(self.cfg.get("rms_norm_eps", 1e-6))
        self.max_seq = max_seq

        # Pool tensors from safetensors.
        state = {}
        for s in sorted(hf_dir.glob("*.safetensors")):
            state.update(load_file(str(s)))
        sd = hf_state_to_qwen3(state, self.n_layers, self.cfg.get("tie_word_embeddings", True))
        self.embed_w = sd["embed_tokens.weight"].float()  # [V, H]
        norm_w = sd["norm.weight"].float()
        self.final_norm = Qwen3RMSNorm(self.hidden, eps=self.eps)
        self.final_norm.weight.data = norm_w
        self.final_norm.eval()

        # Load 28 block mlpackages on CPU+ANE.
        self.blocks = []
        for i in range(self.n_layers):
            pkg = chunked_dir / f"m8-block-{i}.mlpackage"
            ml = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.blocks.append(ml)
        self.states = [ml.make_state() for ml in self.blocks]

    def reset_states(self):
        self.states = [ml.make_state() for ml in self.blocks]

    @staticmethod
    def _causal_mask(t_new: int, end_step: int) -> np.ndarray:
        m = np.full((t_new, end_step), -1e4, dtype=np.float32)
        past_len = end_step - t_new
        for j in range(t_new):
            abs_row = past_len + j
            for k in range(end_step):
                if k <= abs_row:
                    m[j, k] = 0.0
        return m.reshape(1, 1, t_new, end_step)

    def step(self, token_id: int, position: int) -> tuple[int, np.ndarray]:
        """One ANE forward at the given absolute position. Updates state.
        Returns (next_token_argmax, logits_fp32_shape_V).
        """
        end_step = position + 1
        hidden = self.embed_w[token_id].numpy().reshape(1, 1, self.hidden).astype(np.float32)
        mask = self._causal_mask(1, end_step)
        pos_arr = np.array([position], dtype=np.int32)
        for i in range(self.n_layers):
            out = self.blocks[i].predict(
                {"hidden_state": hidden, "causal_mask": mask, "position_offset": pos_arr},
                state=self.states[i],
            )
            hidden = out["hidden_out"]
        # Final norm + tied lm_head.
        h = torch.from_numpy(hidden.astype(np.float32))
        with torch.no_grad():
            h_norm = self.final_norm(h).numpy()
        logits = h_norm.reshape(self.hidden) @ self.embed_w.numpy().T
        return int(np.argmax(logits)), logits


# ----------------- MLX verify side -----------------

class MLXVerify:
    """Qwen3-14B (or any mlx_lm-loadable model) used as the spec-dec verifier.
    Returns top-1 predictions at every position via a single forward pass."""

    def __init__(self, model_path: str):
        # mlx_lm.load returns (model, tokenizer). Tokenizer mirrors HF; we use
        # ours directly via AutoTokenizer for consistency with the draft side.
        self.model, _ = mlx_load(model_path)
        self.model.eval()

    def top1_preds(self, token_ids: list[int]) -> list[int]:
        """Top-1 next-token predictions at each of the input positions.
        Returns a list of length len(token_ids); index k = pred for k+1."""
        tokens = mx.array(token_ids, dtype=mx.int32).reshape(1, -1)
        logits = self.model(tokens)  # [1, T, V]
        # Top-1 over vocab at each position.
        preds = mx.argmax(logits[0], axis=-1)
        mx.eval(preds)
        return [int(p) for p in preds.tolist()]


# ----------------- Spec dec loop -----------------

def spec_decode(
    draft: ANEDraft,
    verify: MLXVerify,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int,
    K: int,
) -> dict:
    prompt_ids = tokenizer.encode(prompt)
    output = list(prompt_ids)
    draft.reset_states()

    # Prefill: ANE state for the prompt tokens.
    t0 = time.time()
    for pos in range(len(prompt_ids)):
        draft.step(prompt_ids[pos], pos)
    prefill_sec = time.time() - t0
    print(f"  prefill (ANE, {len(prompt_ids)} tokens): {prefill_sec*1000:.0f}ms")

    # Decode loop.
    n_drafted = 0
    n_accepted = 0
    n_rounds = 0
    decode_start = time.time()
    accept_history = []

    while len(output) - len(prompt_ids) < max_tokens:
        # Draft K tokens with ANE.
        round_pos_start = len(output)
        draft_tokens = []
        for k in range(K):
            last_tok = output[-1] if not draft_tokens else draft_tokens[-1]
            pred, _ = draft.step(last_tok, round_pos_start + k - 1)
            draft_tokens.append(pred)

        # Verify with MLX 14B in one forward. Pass the FULL candidate (length T
        # = len(output)+K) — verify_preds[i] = top-1 prediction for token at
        # position i+1 given tokens[0..i]. verify_preds[T-1] is the "all
        # accepted, what comes next" prediction.
        candidate = output + draft_tokens
        verify_preds = verify.top1_preds(candidate)

        # Accept tokens up to first mismatch.
        accepted = 0
        correction = None
        for i in range(K):
            pos_in_seq = round_pos_start + i
            v_pred = verify_preds[pos_in_seq - 1]
            if v_pred == draft_tokens[i]:
                accepted += 1
            else:
                correction = v_pred
                break

        # Commit accepted + 1 correction (or +1 from verify if all accepted).
        if correction is not None:
            output.extend(draft_tokens[:accepted] + [correction])
            # Repair ANE state: its k_cache at position (round_pos_start + accepted)
            # currently holds draft_tokens[accepted]'s K/V, but we substituted
            # correction. Re-run that position with the correct token to overwrite.
            # The "stale" entries at positions [round_pos_start+accepted+1 ..
            # round_pos_start+K) won't be read on next round because end_step
            # will be (round_pos_start+accepted+2) — under the threshold.
            draft.step(correction, round_pos_start + accepted)
        else:
            # All K accepted. Take verify's NEXT prediction too (free token).
            v_next = verify_preds[round_pos_start + K - 1]
            output.extend(draft_tokens + [v_next])
            # ANE state needs the v_next position written. Re-run ANE there.
            draft.step(v_next, round_pos_start + K)

        n_rounds += 1
        n_drafted += K
        n_accepted += accepted
        accept_history.append(accepted)

    decode_sec = time.time() - decode_start
    n_decoded = len(output) - len(prompt_ids)
    tok_per_sec = n_decoded / decode_sec
    accept_rate = n_accepted / n_drafted if n_drafted else 0.0
    return {
        "n_decoded": n_decoded,
        "decode_sec": decode_sec,
        "tok_per_sec": tok_per_sec,
        "prefill_sec": prefill_sec,
        "n_rounds": n_rounds,
        "n_drafted": n_drafted,
        "n_accepted": n_accepted,
        "accept_rate": accept_rate,
        "accept_history": accept_history,
        "K": K,
        "output_text": tokenizer.decode(output[len(prompt_ids):]),
    }


# ----------------- CLI -----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft-dir", required=True, help="chunked ANE bundle dir")
    p.add_argument("--draft-hf-dir", required=True, help="HF dir matching the draft (for embed/norm/tokenizer)")
    p.add_argument("--verify-dir", required=True, help="mlx_lm-loadable model dir (e.g. Qwen3-14B-MLX-4bit)")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--k", type=int, default=4, help="draft length per round")
    p.add_argument("--max-seq", type=int, default=512)
    args = p.parse_args()

    print(f"[setup] tokenizer from {args.draft_hf_dir}")
    tok = AutoTokenizer.from_pretrained(args.draft_hf_dir)
    print(f"[setup] loading ANE draft from {args.draft_dir}")
    t0 = time.time()
    draft = ANEDraft(
        chunked_dir=Path(args.draft_dir).expanduser(),
        hf_dir=Path(args.draft_hf_dir).expanduser(),
        max_seq=args.max_seq,
    )
    print(f"        loaded {draft.n_layers} blocks in {time.time()-t0:.1f}s")
    print(f"[setup] loading MLX verify from {args.verify_dir}")
    t0 = time.time()
    verify = MLXVerify(args.verify_dir)
    print(f"        loaded in {time.time()-t0:.1f}s")

    print(f"\n[run] spec decode: prompt='{args.prompt}', K={args.k}, max={args.max_tokens}")
    result = spec_decode(draft, verify, tok, args.prompt, args.max_tokens, args.k)

    print(f"\n=== result ===")
    print(f"  generated {result['n_decoded']} tokens in {result['decode_sec']:.2f}s")
    print(f"  effective decode tok/s: {result['tok_per_sec']:.1f}")
    print(f"  acceptance rate: {result['accept_rate']*100:.1f}% "
            f"({result['n_accepted']}/{result['n_drafted']} draft tokens accepted)")
    print(f"  rounds: {result['n_rounds']}, accepts per round (median): "
            f"{int(np.median(result['accept_history']))}/{result['K']}")
    print(f"  output: {result['output_text']!r}")


if __name__ == "__main__":
    main()
