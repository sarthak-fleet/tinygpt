---
name: E7 Local LLM-as-judge shim
status: not-started
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier E (E7)
task_tracker: #236
---

# PRD — `tinygpt judge`

## Goal

Ship `tinygpt judge` — a Swift subcommand that takes paired model
outputs (or single outputs) and asks a local Qwen / SmolLM3 to rate
them via the LLM-as-judge pattern. Unlocks AlpacaEval / MT-Bench /
RewardBench-style preference evals without an OpenAI API key.

## Why now

- Many of the "is this response good?" evals (instruction-following,
  helpfulness, coherence) are graded by a stronger LM acting as judge.
  The standard tools call OpenAI's GPT-4. **TinyGPT is local-first** —
  shipping a local-judge shim keeps the entire eval pipeline offline.
- A1 specialist (tool-caller) + later instruct-tuned variants need
  preference-style scoring. lm-eval (E3) covers loglikelihood + greedy
  generation; judge fills the gap for free-form responses.
- The HF Browser (today's app sprint) means a user can pull a Qwen3
  judge model in 2 minutes. Infrastructure already in place.

## Scope — in

- New file `Sources/TinyGPT/JudgeShim.swift`
- Two modes:
  1. **Single-rating** (--mode rate): for each `{prompt, response}`
     pair, ask the judge to rate the response 1-10 + give a short
     justification. Emit `EvalCompare.Row` with `metric="judge_rating"`,
     score in [0, 1] (rating / 10).
  2. **Pairwise** (--mode pairwise): for each `{prompt, response_a,
     response_b}`, ask the judge "which is better, a or b?" + justification.
     Emit row with `metric="judge_win_rate"`, score = win fraction
     (for paired runs).
- Subprocess-style: spawn `tinygpt-cli serve <judge.tinygpt>` OR call
  an external HF judge model via a separate serve (default: a Qwen3 or
  SmolLM3 path the user has downloaded).
- Read inputs from JSONL: one row per `{prompt, response_a, response_b?}`.
- Standard prompt templates baked in (mirroring AlpacaEval's exact
  judge prompt — license-permissive paraphrase).

## Scope — out (v2)

- Judge ensemble (multiple judges, average ratings).
- Calibration to a reference set (debias the judge).
- Streaming live ratings to a UI.
- Cross-validation against GPT-4 / Claude ground-truth — needs API
  keys, not the local-first target.

## Inputs the agent has

| Resource | Location |
|---|---|
| Pattern to copy | `Sources/TinyGPT/RunLmEval.swift` → `runViaServe` (subprocess + serve dance) |
| E0 schema | `Sources/TinyGPT/EvalCompare.swift` |
| Default judge model | `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/` (already pulled) — small judge for smoke; production uses larger HF judges via the App's HF Browser |
| AlpacaEval prompt template | https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/alpaca_eval.txt — paraphrase, don't copy verbatim (different license) |
| MT-Bench prompts | https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl — same license-care note |

## Acceptance criteria

1. `tinygpt judge --help` clean usage
2. Smoke against SmolLM2 as judge + two synthetic responses:
   ```
   echo '{"prompt":"What is 2+2?","response_a":"It is four.","response_b":"It is potato."}' > /tmp/judge-smoke.jsonl
   tinygpt judge /tmp/judge-smoke.jsonl --mode pairwise \
     --judge-model <SmolLM2 path or HF model id> \
     --serve-port 8102 --out /tmp/judge-out.jsonl
   ```
3. Output JSONL has a row with `metric="judge_win_rate"`, score in [0,1]
4. `tinygpt eval-compare /tmp/judge-out.jsonl --by task` renders it
5. Build passes

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/JudgeShim.swift` |
| **don't touch** | dispatch + plan files (PR diff for `TinyGPT.swift`) |

## Estimated effort

**~1 day.** Most of it is prompt-template wrangling — the subprocess
mechanics are the same pattern as RunLmEval.

## Coordination + risks

- Judge model size affects throughput. SmolLM2-135M is too weak to be
  a real judge (won't grade well) — document this. Production usage
  needs a Qwen3-3B or larger; smoke-test with whatever's on disk.
- Local judges are systematically biased toward responses that look
  like their own training distribution. Note this clearly in the
  output — `EvalCompare.Row` should include `metadata.judge_model`
  field (extending the schema slightly — agent should propose).
- Some judges (Claude / GPT-4) require API keys. Out of scope.

## Source links

- AlpacaEval: https://github.com/tatsu-lab/alpaca_eval
- MT-Bench: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
- RewardBench: https://huggingface.co/spaces/allenai/reward-bench
