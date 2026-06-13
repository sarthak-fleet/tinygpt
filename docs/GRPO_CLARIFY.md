# GRPO-Clarify v1 — RL on the ambig dimension

Status: PRD (pre-experiment). Owner: Sarthak. Date: 2026-06-12.

## Goal

Break the 22% ambig ceiling. The 12-config drilldown
([docs/DRILLDOWN.md](DRILLDOWN.md)) showed every local model — including
the new champion Gemma-3-12B — tops out at ~22% on clarify fixtures while
oos (82%) and destructive (77%) are workable. Clarify is the unsolved
frontier and it's a *behavior* gap, not a knowledge gap: models guess
instead of asking. That's exactly the shape RL-against-a-verifier fixes.

Success = a Qwen3-4B LoRA that beats 22% ambig on held-out h2-ext with
**zero regression** on oos/destructive. Anything else gets discarded.

## Base model

**Qwen3-4B-Instruct** (the locked QLoRA base from the planner-candidates
drill — passes 3/5 dims zero-shot, fits training + rollouts on M5 Pro
48GB). LoRA adapters only; no full fine-tune.

## Reward = strict scorer

`scripts/eval_pace_unhappy.py --strict` is the reward function. The
lenient scorer is exploitable within minutes of GRPO (topic-word
stuffing, prompt echo, target stuffed in payload, degenerate
spokenText) — strict mode closes those holes and has an adversarial
self-test (`--self-test`, pure python, no HTTP) that must pass before
every training run.

### Shaped reward (proposal)

| outcome | reward |
|---|---|
| full strict PASS | +1.0 |
| correct `intent`, fails any strict check | +0.3 |
| wrong intent / unparseable JSON | 0.0 |
| `clarify` emitted on a **non-ambig** fixture | −0.5 |

Notes:
- The −0.5 anti-over-asking penalty is mandatory. A previous two-stage
  shim died from exactly this failure mode (clarify-on-everything); GRPO
  will rediscover it instantly if every training prompt is ambig. Mix
  ~30-40% non-ambig prompts (action/answer/oos) into every batch.
- Partial credit (+0.3) exists for gradient signal early on; consider
  annealing toward strict-only in late training so the policy can't farm
  intent-only reward (see risk register).
- Caveat: ultra-short clarify answers like `"which app?"` (2 words) fail
  strict's degenerate-spokenText check and earn only +0.3. That's
  acceptable shaping pressure toward fuller questions, but watch for it
  pinning reward.

## Training framework (researched 2026-06-12)

**Recommendation: [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
(v2.1.0, Apr 2026).** Native MLX GRPO on Apple Silicon, and — the
decisive feature — custom reward functions plug in as a python file:
`--reward-functions-file ./my_rewards.py --reward-functions "clarify_strict"`.
That means the strict scorer drops in directly. Supports Qwen3
(Qwen3-4B-Instruct appears in its examples), LoRA + 4/6/8-bit quantized
training, GRPO variants (Dr. GRPO, GSPO, DAPO), and since
[v0.9.9](https://x.com/ActuallyIsaak/status/1996657714810818741) its
GRPO backend uses native mlx-lm batch generation for rollouts.
Also on [PyPI](https://pypi.org/project/mlx-lm-lora/).

Alternatives considered:
- **[trl GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) on
  MPS** — transformers
  [auto-detects MPS](https://huggingface.co/docs/transformers/en/perf_train_special)
  and recent TRL releases fixed MPS-specific bugs, but the GRPO rollout
  path is built around vLLM/CUDA; on Mac it's the slow, less-tested
  road. Fallback only.
- **[Doriandarko/MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO)** —
  pure-MLX GRPO demo pipeline (GSM8K-oriented). Good reference code,
  less maintained than mlx-lm-lora.
- **[mlx-examples PR #1233](https://github.com/ml-explore/mlx-examples/pull/1233)** —
  the upstream GRPO PR (same author as mlx-lm-lora); the standalone
  package is ahead of it.

## Rollout token-budget math (M5 Pro 48GB)

Assumptions: 4B model ≈ 8 GB bf16 (or ~3 GB 4-bit), ~50 tok/s decode at
inference; budget training-loop decode at ~0.5-0.7× that (LoRA-attached
policy, batched rollouts, no serve-grade KV tricks). Response length
~120 tok (v11 JSON), prompt ~1.5k tok (system prompt + elements).

Per GRPO step (1 prompt, group size G):
- G=4: ~480 decode tok ≈ 10-19 s
- G=8: ~960 decode tok ≈ 19-38 s

One epoch over 200 training prompts at G=8: ~192k decode tokens ≈
65 min pure decode, call it **~2 h wall-clock** with prefill + backward
+ optimizer. G=4 halves that at the cost of noisier advantages — start
G=4 for shakeout, G=8 for the real run.

Eval cost (lenient + strict both run): n=40 ambig prompts × ~120 tok ≈
5k tok ≈ 2-3 min; full 3-dim gate (~130 fixtures) ≈ 10 min. Cheap enough
to gate every checkpoint.

Memory: model + LoRA grads + optimizer + G concurrent KV fits in 48 GB
with room. **Operational rule: LM Studio's 8 GB model must be unloaded
first — one large model at a time.** Run under `caffeinate`.

## Train/eval contamination policy

- The h2 and h2-ext fixture suites (`evals/fm-fixtures-{ambig,oos,destructive}-h2{,-ext}`)
  are **eval-only. NEVER training prompts.** No paraphrases of them
  either — paraphrase leakage is still leakage.
- Training prompts are generated fresh: new ambiguous scenarios
  (missing recipient/time/content/quantity, multi-candidate elements)
  plus non-ambig distractors, from templates + teacher paraphrase.
- Seed material: **149 DPO pairs** at
  `~/.cache/tinygpt/datasets/clarify-dpo-v1.jsonl` — use the *prompts*
  as scenario seeds and the chosen/rejected pairs to sanity-check the
  reward (chosen should score ≥ rejected). Audit first: some chosen
  completions (`"which app?"`) only earn partial credit under strict.
- h2 = checkpoint-selection eval; h2-ext = held-out ship gate, looked at
  once, at the end.

## Ship gate

1. Self-test green: `python3 scripts/eval_pace_unhappy.py --self-test`.
2. Ambig on held-out h2-ext (lenient scorer, for comparability with the
   drilldown numbers) **> 22%** (Gemma-3-12B champion).
3. oos and destructive on h2-ext: **zero regression** vs the same
   Qwen3-4B base + system prompt before RL.
4. Strict-mode ambig reported alongside (the honest number).
5. Any gate fails → discard the adapter. No partial ships.

## Reward-hacking risk register

| # | risk | mitigation |
|---|---|---|
| 1 | Topic-template collapse: policy memorizes the six canonical topics and emits generic "which X?" that passes substring checks without grounding | rotate topic phrasings in fresh training prompts; human-read 20 high-reward rollouts per run |
| 2 | Jaccard-threshold gaming: paraphrased echo just under 0.6 | spot-check near-threshold (0.45-0.6) samples; tighten threshold if seen |
| 3 | Partial-credit farming: always-correct intent + junk fields for a safe +0.3 | anneal to strict-only reward late in training; monitor strict-pass fraction, not mean reward |
| 4 | Over/under-asking collapse from a miscalibrated −0.5 penalty | track clarify-rate on the mixed batch every eval; healthy band ≈ the true ambig fraction |
| 5 | Min-length gaming: shortest question that clears MIN_SPOKEN_WORDS, repeated verbatim | monitor distinct-question ratio per eval; add diversity penalty only if it degenerates |
| 6 | KL drift to grammar-shaped gibberish that satisfies regex checks | keep GRPO KL penalty on; read raw samples at every checkpoint |
| 7 | Scorer bug = silent reward bug | `--self-test` is a hard precondition of every training launch; extend it whenever a new exploit is observed |
| 8 | Eval overfitting via checkpoint selection on h2 | h2-ext stays sealed until the single final gate run |

## Out of scope (v1)

- Full fine-tune, >4B bases, multi-dim reward beyond the shaped table,
  distillation from the 30B teacher (separate track), any cloud compute.
