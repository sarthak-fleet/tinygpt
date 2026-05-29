# LLM Quality Benchmark Landscape, May 2026 — Survey for tinygpt

*Research compiled by an Explore subagent on 2026-05-29 to cover the gap
between my Jan 2026 knowledge cutoff and current state. Includes URLs.*

## 1. Open LLM Leaderboard

HuggingFace's **Open LLM Leaderboard v2** (MMLU-Pro, IFEval, BBH, GPQA,
MuSR, MATH-Lvl5) has effectively been deprecated. Reports indicate the
official Space has been intermittently broken in May 2026, and a
v3-style overhaul ("Open LLM Leaderboard 2026") rolled out earlier in
2026 with anti-contamination controls causing the largest reshuffle in
its history. MuSR and MATH-Lvl5 saturated; current composition leans on
**MMLU-Pro, GPQA-Diamond, IFEval, LiveCodeBench, AIME 2025/2026,
MATH-500, SWE-bench Verified**. Plain MMLU is now considered
"outdated/saturated" and excluded from non-saturated leaderboards.

- [HF Space](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [OpenEvals 2026](https://huggingface.co/spaces/OpenEvals/every-leaderboards)
- [v3 writeup](https://agentmarketcap.ai/blog/2026/04/10/huggingface-open-llm-leaderboard-v3-2026)

## 2. Contamination-Resistant

- **LiveBench** still the gold standard; 21 tasks across 7 categories,
  monthly additions + 6-month full refresh, drawn from post-cutoff
  arXiv/news/IMDb/contests.
- **ZebraLogic** (1000 logic-grid puzzles, programmatically generated,
  controllable complexity via Z3 conflict count) — actively maintained,
  last updated May 2026.
- New entrants: **BeyondBench** (Sep 2025) and **ThinkBench** (dynamic
  OOD reasoning) explicitly target contamination resistance via
  synthesis.

## 3. Frontier Reasoning

- **ARC-AGI-2**: GPT-5.5 leads at 85%, GPT-5.4 Pro 83.3%, Gemini 3.1
  Pro 77.1% (May 2026). Still separates frontier models.
- **ARC-AGI-3** launched March 25, 2026 — interactive games, no
  instructions. All frontier models <1%, humans 100%.
- **HLE (Humanity's Last Exam)**: still low single-to-low-double digits
  for frontier models; agi.safe.ai hosts the live board.
- **FrontierMath**: T1–T3 nearing >95% per 2026 forecasts; T4 still
  discriminating.

## 4. Code/Agent

- **SWE-Bench Verified**: Claude Opus 4.7 leads at **87.6%** (Apr 2026).
- **LiveCodeBench**: contamination-free, continuously scraped from
  LeetCode/AtCoder/Codeforces — preferred over HumanEval/MBPP which are
  saturated.
- **GAIA**: Claude Sonnet 4.5 leads at 74.6% (Princeton HAL).
- **τ-bench**, **AgentBench**, **WebArena**, **OSWorld**: Berkeley RDI
  (2026) found 8 of these gameable to near-perfect without solving
  tasks — treat with caution.

## 5. Long-Context

- **RULER** (NVIDIA, 13 tasks × 4 categories) — still the credible
  synthetic bar. Frontier models effectively use 50–65% of advertised
  context.
- **LongBench v2**: Claude Opus 4.5 leads at 64.4% — nearing saturation.
- **MRCR v2** and **NoLiMa** are the newer harder needle variants.
  Advertised vs effective context diverges by 30–60 points past 200K
  tokens.

## 6. Runnable on a Single Mac

**Fully local, no judge model:**
- **LiveBench** (math/coding/IF programmatic graders; "language" subset
  uses LLM judge — skip or use a local Qwen3 judge)
- **ZebraLogic** (Z3-verified, deterministic)
- **RULER** (synthetic, exact-match)
- **LongBench v2** (multiple choice — exact match)
- **GSM8K, MATH-500, AIME, GPQA-Diamond, MMLU-Pro, IFEval, BBH,
  HumanEval/LiveCodeBench** — all via `lm-evaluation-harness` (supports
  MLX through `local-chat-completions`; watch the stop-sequence bug in
  0.4.11)
- **TinyStories PPL and bits-per-byte** stay valuable for byte-level
  models. The **NeurIPS 2025 E2LM** competition explicitly addresses
  early-training (<200B tokens) evaluation for sub-7B models —
  directly relevant to tinygpt.

**Requires paid API judge (cost gate):**
- MT-Bench, AlpacaEval, Arena-Hard, GAIA-text-judge, HLE auto-grader,
  GPQA open-ended — all need GPT-4/Claude grader
- SWE-Bench Verified runs Dockerized test suites locally but is heavy
  (50+GB images, hours per model)
- τ-bench requires a user-simulator LLM

## 7. New since Jan 2026

- **ARC-AGI-3** (Mar 2026) — interactive
- **AIME 2026** — fresh, contamination-free math
- **ReasonBench** (Dec 2025) — variance-aware reasoning evaluation
  across 11 methods × 4 models × 7 tasks
- **BeyondBench** (Sep 2025) — synthesis-based contamination resistance
- **LongBench Pro** — bilingual realistic long-context extension
- Berkeley RDI **agent benchmark exploit paper** (2026) — caveat on
  trusting τ/SWE/WebArena scores

## Recommendation for tinygpt

Wire in (all local, no judge cost):
1. **bits-per-byte on held-out TinyStories/Shakespeare** (already have
   it) — extend with TinyStories-eval split
2. **HellaSwag-easy + ARC-Easy** via lm-eval-harness for byte-level
   signal early
3. **GSM8K + IFEval + GPQA-Diamond** for HF-loaded Llama models
4. **RULER short-context (4K/8K)** — gives a credible
   context-effectiveness number
5. **LiveBench programmatic subset** — math + coding + reasoning, no
   judge

Skip until external budget: MT-Bench, Arena-Hard, GAIA, HLE, full
SWE-Bench.

## Sources

- [HF Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [OpenEvals 2026 leaderboards](https://huggingface.co/spaces/OpenEvals/every-leaderboards)
- [HF Open LLM Leaderboard v3 writeup](https://agentmarketcap.ai/blog/2026/04/10/huggingface-open-llm-leaderboard-v3-2026)
- [LiveBench paper](https://livebench.ai/livebench.pdf) / [GitHub](https://github.com/LiveBench/LiveBench)
- [ZebraLogic paper](https://arxiv.org/html/2502.01100v1) / [leaderboard](https://llm-stats.com/benchmarks/zebralogic)
- [BeyondBench](https://arxiv.org/pdf/2509.24210)
- [ThinkBench](https://arxiv.org/pdf/2502.16268)
- [ARC-AGI-2 paper](https://arxiv.org/pdf/2505.11831) / [leaderboard](https://llm-stats.com/benchmarks/arc-agi-v2)
- [ARC-AGI-3 launch](https://www.mindstudio.ai/blog/arc-agi-3-results-frontier-models-score-zero)
- [Humanity's Last Exam](https://agi.safe.ai/)
- [SWE-Bench Verified](https://www.swebench.com/verified.html)
- [Agent benchmark leaderboard 2026](https://benchmarkingagents.com/swe-bench/)
- [RULER paper](https://arxiv.org/pdf/2404.06654)
- [LongBench v2 paper](https://arxiv.org/pdf/2412.15204)
- [Long-context benchmarks survey](https://ofox.ai/blog/long-context-llm-benchmarks-200k-tokens-2026/)
- [NeurIPS 2025 E2LM (small-model eval)](https://arxiv.org/pdf/2506.07731)
- [ReasonBench](https://arxiv.org/pdf/2512.07795)
