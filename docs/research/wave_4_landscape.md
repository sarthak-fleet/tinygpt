# Wave 4 landscape — TML / Apple FM / code agents / Indic

**Date**: 2026-05-31
**Question**: What's the competitive + capability landscape tinygpt
should know about before training specialists?
**Outcome**: TML's "interaction model" framing maps onto Wave 2.6;
Apple owns the same architecture but walled; code agents are
cloud-dominated with a narrow local-first opening; Indic plan needs
the `desi-max` reference replaced.

## 1. Thinking Machines Lab (TML)

- **Founded Feb 2025** by Mira Murati + ex-OpenAI cohort (Schulman,
  Zoph, Weng, Tulloch, Metz). Raised $2B at $12B valuation in 5 months,
  reportedly chasing $50B by late 2025
  ([Built In](https://builtin.com/articles/what-is-thinking-machines-lab),
  [Wikipedia](https://en.wikipedia.org/wiki/Thinking_Machines_Lab)).
- **Tinker (Oct 2025)** — cloud distributed-training API for fine-tuning
  open-weight models (Qwen 4B-397B, Llama 1B-70B, DeepSeek V3.1, Kimi K2,
  Nemotron, GPT-OSS) via 4 primitives: `forward_backward`, `optim_step`,
  `sample`, `save_state`. LoRA-first; **not on-device**
  ([Tinker page](https://thinkingmachines.ai/tinker/),
  [DeepLearning.AI](https://www.deeplearning.ai/the-batch/thinking-machines-new-tinker-api-makes-it-easier-to-fine-tune-models-on-many-gpus)).
- **Research blog Connectionism**: "Defeating Nondeterminism in LLM
  Inference" (Horace He), "LoRA Without Regret" (Schulman, 2025-09-29),
  "On-Policy Distillation" (2025-10-27), "Modular Manifolds" (Bernstein)
  ([blog index](https://thinkingmachines.ai/blog/)).
- **"Interaction Models" thesis (May 2026)** — argues interactivity must
  be *native to the model*, with a **hybrid split**: a foreground
  "Interaction Model" doing 200ms micro-turns + a background model for
  async reasoning/tools
  ([interaction-models](https://thinkingmachines.ai/blog/interaction-models/)).
- Posture: **cloud-only, infrastructure-focused, research-lab** —
  selling APIs to researchers, not shipping a consumer on-device product.

**Implications for tinygpt**: TML's interaction-model framing maps
directly onto tinygpt's Wave 2.6, but their implementation is cloud-bound.
tinygpt's differentiator is doing the foreground-interaction model
*on-device* on Apple Silicon — that's a real differentiator, not
duplication. Steal the architectural vocabulary ("foreground interaction
model + background async") for the docs. Read "LoRA Without Regret"
before more LoRA work. Do **not** try to compete with Tinker; if cloud
fine-tune is ever needed, use it.

## 2. Apple Foundation Models + Private Cloud Compute

- **~3B on-device model**: 5:3 block-depth split with KV-cache sharing
  (37.5% memory reduction) + 2-bit QAT; supports 65K context; **15
  languages** via 150K-token vocab (up from 100K, only 25% more tokens)
  ([2025 updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates),
  [tech report arXiv 2507.13575](https://arxiv.org/abs/2507.13575)).
- **Server side**: Parallel-Track MoE (PT-MoE) cutting sync overhead
  87.5%, running on Apple-silicon servers under Private Cloud Compute.
- **Foundation Models framework (WWDC25)**: Swift API; `Tool` protocol
  handles guided generation + parallel/serial tool-call call graphs
  automatically; LoRA adapter fine-tuning supported via a Python toolkit
  ([WWDC25 #286](https://developer.apple.com/videos/play/wwdc2025/286/),
  [adapter training](https://developer.apple.com/apple-intelligence/foundation-models-adapter/)).
- **PCC routing logic is NOT publicly documented** — the whitepaper
  covers attestation, secure enclave, hardened OS, but the "when to
  escalate" decision is opaque
  ([PCC blog](https://security.apple.com/blog/private-cloud-compute/)).
- **Third-party model API access: none.** You get Apple's model via the
  Swift framework; you can ship LoRA adapters; you cannot substitute your
  own model into the framework or route your traffic through PCC.
  Adapters require re-training on every OS update.
- Outperforms Qwen-2.5-3B across all 15 languages on internal evals;
  competitive with 4B models on English.

**Implications for tinygpt**: Apple owns the exact architecture tinygpt
is chasing, but the wall is real — closed model, closed router,
adapter-only customization, OS-dependency on retraining. Position
tinygpt as the **open, hackable, multi-specialist** counterpart:
- Can swap base models (Llama/Qwen/Sarvam)
- Can do full SFT/DPO not just adapters
- Can route to *any* cloud (not just PCC)
- Targets devs/researchers Apple won't serve

**Steal**: KV-cache sharing, the `Tool` protocol shape (already similar
to the tool-call extractor plan), 2-bit QAT as a future quant target.
**Don't** try to plug into App Intents — there's no public hook for
third-party LLMs into Apple Intelligence. Stay parallel.

## 3. Code-agent architectures (Cursor / Continue / Cline / Aider)

- **Cline** uses a **ReAct loop with structured-output enforcement** —
  rejects plain text, forces a tool call every turn (Plan-mode dialogue
  goes through a `plan_mode_respond` tool). Plan/Act split is the
  differentiator
  ([deepwiki](https://deepwiki.com/cline/cline/3.4-plan-and-act-modes),
  [GitHub](https://github.com/cline/cline)). SWE-bench Verified
  high-70s with Sonnet 4.5.
- **Cursor Background Agent** + Cline both >59% on SWE-bench Verified
  with Claude Sonnet 4.6 — model dominates, the wrapper is a 3-5pt delta
  ([benchmark](https://awesomeagents.ai/leaderboards/swe-bench-coding-agent-leaderboard/)).
- **Continue.dev** is the most local-friendly: first-class Ollama
  provider on `localhost:11434`, `provider: ollama` config, supports
  VS Code / JetBrains / Neovim
  ([docs](https://docs.continue.dev/customize/model-providers/top-level/ollama)).
  Lightweight middleware, no agent loop — closer to a "copilot" than an
  "agent."
- **Aider** is terminal/git-first; "architect mode" splits planning
  model from editor model; uses **edit formats** (diff, diff-fenced,
  whole, editor-diff) as the structured output contract
  ([edit-formats](https://aider.chat/docs/more/edit-formats.html)).
  Architect mode 31.4% SWE-bench Verified — lower because of human-in-
  the-loop framing.
- **None of them run a serious specialist model locally by default.**
  Local-LLM support exists but it's a degraded mode — they all assume
  cloud Sonnet/GPT for the real work.
- Codeium/Windsurf excluded — fully cloud, closed.

**Implications for tinygpt**: The competitive gap is real but narrow.
If tinygpt ships a Mac dev tool *today* with on-device specialists, you
lose on raw SWE-bench (Sonnet 4.6 > anything we fit in 3B). You win on:
- **Latency**: sub-50ms TTFT vs 200-500ms cloud round-trip
- **Cost**: zero per-token
- **Privacy**: code never leaves device
- **Multi-specialist routing**: none of them have it

**Three concrete steals**:
- Cline's structured-output-enforcement-via-tool (tinygpt already does
  JSON mode — push harder)
- Aider's edit-format contracts (cleaner than raw diff text for small
  models)
- Continue's Ollama provider compatibility (ship a tinygpt provider for
  Continue and you're instantly in dev workflows)

Don't try to beat Cursor on SWE-bench; build the "local-first specialist
with cloud escalation" framing they can't ship.

## 4. Multilingual / India focus

- **desi-max is NOT a language model** — it's a 78-image LoRA on
  Qwen-Image-2512 for vintage South Asian *visual design* aesthetics
  ([HF card](https://huggingface.co/yenupam/desi-max)). Previously
  treated as a Hindi LLM base in `north_star_refined.md` — that was
  wrong; corrected 2026-05-31.
- **Sarvam 30B and 105B** (open-sourced 2026, Apache 2.0): MoE with GQA
  (30B) / MLA (105B), 128 sparse experts each, 16T/12T pretrain tokens,
  custom tokenizer covering 22 scheduled Indian languages / 12 scripts.
  Government-selected for India's sovereign LLM via IndiaAI Mission
  ([Sarvam blog](https://www.sarvam.ai/blogs/sarvam-30b-105b),
  [sovereign LLM](https://www.sarvam.ai/blogs/indias-sovereign-llm)).
  Plan includes **Sarvam-Edge** for on-device — direct overlap with
  tinygpt scope.
- **AI4Bharat Airavata**: Hindi instruction-tuned LLM fine-tuned from
  OpenHathi using machine-translated English instruction sets via
  IndicTrans2; instruction datasets released publicly
  ([arXiv 2401.15006](https://arxiv.org/pdf/2401.15006),
  [IndicInstruct repo](https://github.com/AI4Bharat/IndicInstruct)).
- **Krutrim** (Ola/Bhavish Aggarwal): covers 22 Indian languages but
  open-weight story is weaker than Sarvam
  ([Rest of World](https://restofworld.org/2026/india-frugal-ai-sarvam-krutrim-sovereign/)).
- **MILU** (NAACL 2025, AI4Bharat): 8 domains, 41 subjects, 11 Indic
  languages, India-centric (regional exams, festivals, local history).
  42 LLMs evaluated; GPT-4o leads at 74%
  ([MILU repo](https://github.com/AI4Bharat/MILU),
  [arXiv 2411.02538](https://arxiv.org/pdf/2411.02538)).
- **IndicGenBench**: generative tasks across 29 Indic languages, extends
  Cross-Sum/XQuAD/XorQA/FLORES
  ([arXiv 2404.16816](https://arxiv.org/pdf/2404.16816)).
- **Tokenizer reality check**: smollm2 and Qwen3 are NOT optimized for
  Devanagari. Sarvam's tokenizer is the right choice for serious Indic
  work; falling back to Qwen3 gives 2-4× token bloat on Hindi text.

**Implications for tinygpt**: Replace `desi-max` with **Sarvam-Edge (when
released) or Airavata** as the Indic specialist starting point. The
Indic specialist isn't a one-week dropin — tokenizer choice matters
(likely retokenize with Sarvam's vocab or accept Qwen3 token-bloat
penalty), and the eval harness needs MILU + IndicGenBench before
claiming Hindi support. Sarvam is the most credible upstream — Apache-2.0
release + planned Edge variant means standing on their shoulders rather
than training from scratch. Steal Airavata's translate-English-instructions-via-IndicTrans2 trick to bootstrap Hindi instruction data
for specialists cheaply.

## Top 3 actions (next 3 months)

1. **Continue.dev provider adapter for tinygpt** — Ollama-compatible
   endpoint that drops into Continue/Cline/Aider configs. Lowest-effort
   path to real users, validates the local-first thesis against the
   actual code-agent ecosystem, and provides a benchmark surface
   (SWE-bench mini) to track against Cursor/Cline. Pairs with the SSE
   streaming work already shipped.
2. **Adopt Apple's `Tool` protocol shape + Cline's structured-output-
   enforcement contract for the tool-call extractor** (Wave 2.6 mini-
   router). Don't reinvent — Apple's call-graph handling for
   parallel/serial tools is the right abstraction; Cline's "reject plain
   text, require tool call" is the right enforcement. Makes the
   eventual screen-reader specialist drop into a known-good shape.
3. **Fix the Indic plan**: replace desi-max with Sarvam-Edge / Airavata
   + wire MILU + IndicGenBench into the eval harness. Before any Hindi
   specialist training, run base Qwen3 / smollm2 through MILU to get a
   real baseline; that number tells you whether tokenizer-swap to
   Sarvam's vocab is worth the engineering cost.

## Key correction

`desi-max` in `docs/roadmap/north_star_refined.md` was previously
referenced as a Hindi LLM base. It is in fact a *text-to-image* LoRA on
Qwen-Image-2512 for vintage Indian visual design. Replace with
Sarvam-Edge (forthcoming) or Airavata. Corrected this revision.

## Sources

- [TML Tinker](https://thinkingmachines.ai/tinker/),
  [TML Interaction Models](https://thinkingmachines.ai/blog/interaction-models/),
  [Connectionism blog](https://thinkingmachines.ai/blog/)
- [Apple FM 2025 updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates),
  [Apple FM tech report](https://arxiv.org/abs/2507.13575),
  [PCC](https://security.apple.com/blog/private-cloud-compute/),
  [WWDC25 #286](https://developer.apple.com/videos/play/wwdc2025/286/),
  [Apple adapter training](https://developer.apple.com/apple-intelligence/foundation-models-adapter/)
- [Cline GitHub](https://github.com/cline/cline),
  [Aider edit formats](https://aider.chat/docs/more/edit-formats.html),
  [Continue+Ollama](https://docs.continue.dev/customize/model-providers/top-level/ollama),
  [SWE-bench leaderboard](https://awesomeagents.ai/leaderboards/swe-bench-coding-agent-leaderboard/)
- [Sarvam 30B/105B](https://www.sarvam.ai/blogs/sarvam-30b-105b),
  [Airavata paper](https://arxiv.org/pdf/2401.15006),
  [MILU repo](https://github.com/AI4Bharat/MILU),
  [IndicGenBench paper](https://arxiv.org/pdf/2404.16816),
  [desi-max HF card](https://huggingface.co/yenupam/desi-max)
