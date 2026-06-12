# Drilldown — the experiments left untried, run to bedrock

Started 2026-06-11 after the "miner stopping 30 min before the diamond"
check. Closed 2026-06-12 with the full table below.

## What had been mined to bedrock as of session start

1. 0.6B specialist training (v1–v11). Capacity ceiling — proven.
2. Best-of-N test-time compute (BoN-8). Sampling more = more wrong w/ diversity.
3. Prompt engineering for clarify (4 variants). Binary regime — no smooth interior.
4. Small-corpus SFT on a strong base (clarify-v1, 38 rows on 4B). 47pp regression.

## Final results — h2 + h2-ext combined, n=130

All scored with `scripts/eval_pace_unhappy.py` against
`grammars/pace-system-prompt-v11.txt`, temperature 0, max_tokens 300.

| Model                            |   ambig   |    oos    | destructive | Notes |
|---|---|---|---|---|
| Qwen3-4B-Instruct (ship floor)   |  0/40 ( 0%) | 47/60 (78%) | 20/30 (67%) | reference |
| Qwen3-14B                        |  8/40 (20%) | 43/60 (72%) |  7/30 (23%) | gives back destructive |
| Apple FM (guided)                |  1/40 ( 2%) | 57/60 (95%) | 11/30 (37%) | refusal champ, can't ask |
| clarify-v1 LoRA (4B + 38 rows)   |  1/40 ( 2%) | 24/60 (40%) | 15/30 (50%) | 47pp interference |
| Pace v9-LoRA (0.6B)              |  0/40 ( 0%) | 13/60 (22%) |  1/30 ( 3%) | capacity wall |
| Pace v11-LoRA (0.6B)             |  0/40 ( 0%) |  9/60 (15%) |  5/30 (17%) | capacity wall |
| two-stage shim **v1**            |  0/40 ( 0%) | 34/60 (57%) | 21/30 (70%) | rules detect but scorer rejects |
| two-stage shim **v2** (topic fix)|  8/40 (20%) | 33/60 (55%) | 20/30 (67%) | topic-aware questions, over-triggers a bit |
| Qwen3-4B-Thinking                |  0/40 ( 0%) |  6/60 (10%) |  0/30 ( 0%) | think trace consumes 300-tok cap |
| DeepSeek-R1-Distill-Qwen-7B      |  0/40 ( 0%) |  2/60 ( 3%) |  0/30 ( 0%) | same — thinking-model token wall |
| Llama-3.1-8B-Instruct            |  1/40 ( 2%) | 17/60 (28%) | 20/30 (67%) | over-compliant on OOS |
| **Gemma-3-12B-it (qat-4bit)**    | **9/40 (22%)** | **49/60 (82%)** | **23/30 (77%)** | **wins all three dims** |

## Verdicts per diamond

1. **Reasoning-tuned models** (Qwen3-4B-Thinking, DeepSeek-R1-7B). Drilled.
   Empty output at max_tokens=300 — the entire budget goes to the
   `<think>` trace. NOT a model-quality verdict; would need
   max_tokens≥1024 and a stop-on-`</think>` strategy to give them a fair
   shot. **Parked** — re-evaluate only if Gemma fails in production.

2. **Larger non-thinking bases**. Drilled.
   - Llama-3.1-8B: worse than 4B on every dim.
   - Qwen3-14B: trades destructive for ambig.
   - Gemma-3-12B-it: **wins** on every dim. Drilled to bedrock.

3. **Rule-based ambiguity detector wrapper** (two-stage v2). Drilled.
   Got real movement on ambig (0→20%) at small cost to oos (78→55).
   Rules over-intercept "click" / pronoun cases when the planner could
   have handled them. With Gemma's 22% ambig zero-shot, the shim
   provides no additional lift on ambig and hurts the other dims — so
   **don't ship the shim on top of Gemma**. It's an option only if
   we end up shipping 4B-Instruct for footprint reasons.

4. **DPO on contrastive clarify pairs**. NOT drilled.
   149 pairs built (`~/.cache/tinygpt/datasets/clarify-dpo-v1.jsonl`),
   trainer not written. Given Gemma already clears the floor on every
   dim and DPO would target the 0.6B / 4B paths we now don't ship, the
   ROI is gone. **Parked** unless Gemma proves unworkable in production
   (UX, latency, memory).

## The recommendation

**Pace ships on Gemma-3-12B-it (mlx-community/gemma-3-12b-it-qat-4bit, ~8 GB).**

Reasons:
- Only model in this drill that beats the 4B baseline on ALL three
  unhappy-path dimensions while staying ≤14B.
- 82% OOS clears the "doesn't make stuff up" bar.
- 77% destructive — best in the entire matrix.
- 22% ambig is still poor in absolute terms (the unsolved frontier),
  but it's tied for best among everything tested.

Update `LocalPlannerModelIdentifier` in Pace's `Info.plist` from
`qwen3-4b-instruct-2507` to `google/gemma-3-12b` (LM Studio identifier).

## Rerunning this on the next model that drops

One command (this is the productized form of this whole document):

```bash
scripts/eval_planner.sh <lm-studio-model-id>
```

JIT-loads the model, runs all three suites (n=130), prints the table vs
the stored champion (`evals/planner-champion.json`) with a swap/no-swap
verdict. ~30 min for a 12B on M5 Pro.

## What stays open

- **Ambig is the unsolved dimension.** Best score across 12 configs is
  22%. Either: (a) larger Gemma / Qwen, (b) DPO on clarify pairs against
  Gemma, (c) re-run the reasoning models with adequate token budget.
  Owner decision: ship Gemma first, then revisit if user-data confirms
  ambig is the dominant failure mode.

- **Reasoning model retest.** Bump `max_tokens` to 1024 and add
  `stop=["</think>"]` to give Qwen3-4B-Thinking and DeepSeek-R1 a real
  scoreboard. Cheap experiment, scheduled as a TODO not a blocker.

- **tinygpt's role.** With training closed, tinygpt's keepers:
  benchmark + eval harness (this drilldown is the canonical example),
  serve runtime + grammar + int8 ANE, mech-interp tooling.
