# Small-model tool-calling: the SOTA playbook (what others do)

**What:** a survey of how the field builds SOTA small (1-8B) tool-callers — data,
SFT tricks, RL, eval, on-device serving — distilled to what's *stealable on a Mac*.
**Why:** we kept finding cheap wins one at a time (parser fixes, function masking)
because we worked bottom-up. This is the top-down map so we stop re-deriving the
known playbook. Surveyed 2026-06-14.

Companion: [tool-calling-frontier-parity.md](./tool-calling-frontier-parity.md) (our
own results) · [external-references.md](./external-references.md).

## 1. Data synthesis (how SOTA training sets are built)
- **APIGen** (xLAM-60k): 3,673 executable APIs, **three-stage verification — format →
  actual execution → LLM-semantic**. The *execution gate* is the key idea. [paper](https://arxiv.org/pdf/2406.18518)
- **APIGen-MT**: multi-turn via "blueprint (committee of LLM reviewers + ground-truth
  actions) → simulated human-agent trajectory." Beats GPT-4o/Claude-3.5 on τ-bench. [paper](https://arxiv.org/pdf/2504.03601)
- **ToolACE** (the set we used): self-evolving synthesis → 26k diverse APIs, multi-agent
  user/assistant/verifier, complexity-tiered, **dual-layer (rule + model) verification**. [paper](https://arxiv.org/abs/2409.00920)
- **ToolBench/ToolLLM**: 16k APIs, **DFSDT** search for valid multi-step trajectories. [paper](https://arxiv.org/pdf/2307.16789)

## 2. SFT tricks
- **Function masking** (Hammer): randomize ~33-50% of function/param *names* during
  training → model relies on *descriptions*, not memorized names. Robust to unseen
  catalogs. [paper](https://arxiv.org/abs/2410.04587)
- **Irrelevance/abstention augmentation** (Hammer): ~10% of data = correct tool removed,
  label = empty list → teaches "don't call when nothing fits." Cuts over-emission.
- **Decontamination** + **curriculum** (single→multi→long-horizon). LoRA suffices ≥8B;
  full-FT common at 1.7-4B. SOTA small bases: Qwen2.5 / Llama-3.1-8B.

## 3. RL / post-training
- **Graded reward >> binary** (ToolRL): reward in [-3,3] = tool-name (Jaccard) +
  param-name (Jaccard) + param-value (exact) + a separate format reward. Fine-grained
  partial credit beats binary at *all* sizes; **length rewards hurt small models**.
  ToolRL ≈ +10 over SFT. [paper](https://arxiv.org/html/2504.13958v1)
- **Single-turn RL barely moves** (EGPO got +2.15 on single-turn BFCL — *matches our +2*).
  Gains live in reward granularity + multi-call credit + cold-start. [EGPO](https://arxiv.org/html/2508.05118)
- **Stability/efficiency** (DAPO/Dr.GRPO): clip-higher (ε_hi≈0.28), drop std-normalization,
  token-level loss, and **dynamic sampling — discard prompts where all K rollouts pass or
  all fail** (we saw many such no-variance skips). Small batch is the #1 mistake. [GRPO tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- **RFT** (rejection-sampling best-of-N SFT, filtered by the AST matcher) is the cheap
  middle rung between SFT and full GRPO. **Cold-start GRPO can beat SFT-init GRPO** (SFT
  overfits). DPO/KTO cheaper but generalize worse on multi-step. [post-training 2026](https://llm-stats.com/blog/research/post-training-techniques-2026)
- **Credit assignment for multi-call** (CARL): segment-level advantages so a good and bad
  call in one trajectory get opposite credit — but the full version is cluster-scale. [CARL](https://arxiv.org/html/2605.27788)

## 4. Eval (and its traps)
- **Benchmarks**: BFCL v3/v4 (AST + executable + irrelevance/relevance; v4 adds agentic +
  **format-sensitivity**), **τ-bench/τ²-bench** (multi-turn, stateful, **Pass^k** reliability),
  HammerBench, ACEBench, MCP-Bench (real MCP schemas). [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) · [τ²](https://github.com/sierra-research/tau2-bench)
- **Traps**: gold under-determination + exact-match brittleness (*our hermes lesson*);
  **contamination** (FC specialists train on BFCL-like data → high scores partly by-design);
  **prompt-format sensitivity drops AST 13-19 pts** under paraphrase.
- **Our frontier-calibration gate ("~100% or broken") is industry-aligned.**
- **Small models cliff on multi-turn**: Command-R7B 69%→**5%** single→multi-turn; Llama-3.1-8B
  61%→9.6%. Purpose-tuned (xLAM-2-8b-fc) survive ~69%. **Our single-turn 78 likely overstates
  real agentic ability — we haven't tested multi-turn at all.**

## 5. On-device / Apple-Silicon serving
- **Quantization hurts FC disproportionately** — tool *selection* + arg fidelity degrade faster
  than general tasks at 4-bit. Prefer **8-bit / AWQ / QAT** for the tool-caller, BFCL-validate. [study](https://arxiv.org/pdf/2504.04823)
- **Constrained decoding**: guarantees valid JSON; *rescues* weak small models (3B beat 70B
  on BFCL under XGrammar), but a **"constraint tax"** — hard schema-locking can suppress
  reasoning (Qwen2.5-1.5B 91.5%→48%). **Adopt with a free-text reasoning prefix, not a hard
  global lock.** [XGrammar](https://arxiv.org/pdf/2601.04426) · [constraint tax](https://arxiv.org/abs/2605.26128)
- **Prefix/KV-cache the repeated tool-schema system prompt** — biggest *speed* win locally.
- **Rapid-MLX** (17 tool-parsers w/ auto-recovery, batching, ~2× Ollama) / oMLX / LM Studio. [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)

## STEAL / ADOPT priority list (mapped to our gaps)
1. **ToolRL graded reward** (name+param-name+param-value+format) replacing our per-call
   binary — the single highest-ROI fix to our flat +2 RL. *Reward-fn edit only.*
2. **Function masking + irrelevance augmentation** in SFT data — targets our `live_multiple`
   WRONG_FUNC (32%) + over-emission directly. *Cheap data edit.*
3. **Dynamic sampling** (drop zero-advantage groups) + clip-higher — fixes the no-variance
   skips we observed; sample-efficiency on Mac compute. *Few-line GRPO edit.*
4. **8-bit (not 4-bit) for the tool-caller**, BFCL-validated — 4-bit may be silently costing FC accuracy.
5. **Eval upgrades**: add irrelevance + format-sensitivity (paraphrase ×3) probes now;
   a multi-turn stateful slice later (where small models actually cliff).
6. **RFT** (best-of-N filtered by the AST matcher) as a cheap rung before full GRPO.
7. **Serving**: prefix-KV-cache the tool schema; constrained JSON *with* a reasoning prefix.
