---
title: Competitive landscape — fine-tune, eval, interpretability
description: A 2026 map of the products around "make a model good at your task" and "is my agent any good," with the Mac-first whitespace called out. Evidence behind docs/sessions/2026-06-13-market-landscape-mac-first.md.
---

# Competitive landscape (2026)

Factual map of the players a Mac-first SLM toolkit competes with or
positions against. The *strategy* derived from this lives in
[`docs/sessions/2026-06-13-market-landscape-mac-first.md`](../sessions/2026-06-13-market-landscape-mac-first.md);
this page is the citable evidence. Researched 2026-06-13; numbers move,
re-verify before quoting externally.

## Fine-tuning platforms

| Player | Hosting | Wedge | Mac/MLX? |
|---|---|---|---|
| OpenAI fine-tuning | cloud | best base models, zero infra; SFT/DPO + RFT | no |
| [Together AI](https://www.together.ai/fine-tuning) | cloud | broad open-model menu, cheap LoRA/full-FT | no |
| [Fireworks AI](https://fireworks.ai/blog/reinforcement-fine-tuning) | cloud | managed RFT, PyTorch pedigree | no |
| Predibase → [Rubrik](https://www.rubrik.com/blog/company/25/rubrik-predibase-bipul-sinha) | cloud/VPC | first to productize RFT (acquired Jun 2025) | no |
| [Lamini](https://www.lamini.ai/pricing) | cloud + on-prem/air-gap | Memory Tuning; enterprise privacy | no |
| Modal / Replicate | cloud | serverless GPU infra (where FT jobs run) | no |
| [Castform](https://castform.com/) | cloud | RL on agent-trace/RAG envs; export weights | no |
| [Tinker](https://thinkingmachines.ai/news/announcing-tinker/) (Thinking Machines) | cloud | low-level distributed-LoRA API | no |
| OpenPipe / ART → [CoreWeave](https://www.coreweave.com/news/coreweave-to-acquire-openpipe-leader-in-reinforcement-learning) | OSS + cloud | agent RL trainer (acquired Sep 2025) | no |
| [Unsloth](https://unsloth.ai/) | OSS | fastest QLoRA; MPS 3–5× slower, native MLX "coming" | partial |
| Axolotl / TorchTune / LLaMA-Factory | OSS | config-driven FT | CUDA-centric (no 4/8-bit on Mac) |
| [Kiln AI](https://github.com/kiln-ai/kiln) | OSS, local UX | local-first workbench — but delegates training out | orchestrates, doesn't train on-device |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) (Apple) | OSS, local | **the** native Apple-Silicon LoRA/QLoRA/DoRA path | yes (library/CLI) |

**Read:** the commercial market is one shape — *rent our GPUs, send us
your data, pay per token or per GPU-hour.* Mac-native training as a
**product** is unowned; only Apple's MLX-LM library + thin wrappers serve
it, and Kiln's local UX delegates the actual training elsewhere.

## Agent eval / observability

| Player | Hosting | Wedge | Local story |
|---|---|---|---|
| [Braintrust](https://www.braintrust.dev/) | cloud (+ enterprise self-host) | integrated eval + experiment + monitor | minimal |
| [LangSmith](https://www.langchain.com/langsmith/evaluation) | hybrid | LangChain-native tracing + eval | enterprise tier only |
| [Langfuse](https://langfuse.com/) → ClickHouse | OSS + cloud | most-adopted OSS observability (acquired Jan 2026) | strong (self-host) |
| [Arize Phoenix](https://arize.com/phoenix/) | OSS + SaaS | OTel-based, self-hostable | strong |
| [Galileo](https://galileo.ai/) | cloud | guardrail models + "Insights" root-cause | no |
| [Patronus AI](https://www.patronus.ai/) | cloud | proprietary eval models (Lynx/GLIDER/Percival) | no |
| Humanloop → [Anthropic](https://techcrunch.com/2025/08/13/anthropic-nabs-humanloop-team-as-competition-for-enterprise-ai-talent-heats-up) | — | dead as standalone (acqui-hire Aug 2025) | — |
| [Promptfoo](https://www.promptfoo.dev/) → OpenAI | OSS CLI | eval + red-team, local by default (acquired Mar 2026) | strong |
| [Comet Opik](https://github.com/comet-ml/opik) | OSS + cloud | Apache-2.0 tracing/eval | solid (self-host) |
| [W&B Weave](https://wandb.ai/site/evaluations/) → CoreWeave | cloud | one-line tracing + eval dashboards | limited |
| [DeepEval](https://deepeval.com/) (Confident AI) | OSS + cloud | 50+ metrics, local-first | strong (VPC self-host) |
| [Ragas](https://www.ragas.io/) | OSS lib | reference-free RAG metrics | runs anywhere |

**Read:** "self-host" here means *your K8s/VPC*, not *your Mac*. True
on-device eval is a gap, but local-eval alone is commoditizing (Promptfoo,
DeepEval, Langfuse, Phoenix all do it). The bigger gap is **mechanistic**
"why did it do that" — every "root cause" feature is just an LLM
summarizing traces, not model internals.

## Interpretability tooling

| Player | State |
|---|---|
| [Goodfire](https://www.goodfire.ai/) (Ember) | public API → partnership-only (Feb 2026); not self-serve |
| [Neuronpedia](https://www.neuronpedia.org/) | OSS, research/safety-funded; closest to "productized" interp |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) / SAELens | actively-maintained research libraries; no product |
| Anthropic interpretability | stays research; not a sold feature |

**Read:** nobody sells activation patching / logit lens / SAEs as a paid
agent-debugging feature. The eval community and the interp community barely
overlap.

## The whitespace (one line each)

1. **Mac-first training as a product** — owned by a library (MLX-LM), not
   a product. → TinyGPT's B6 + B31.
2. **Eval + interp + local, fused** — category-of-one; nobody combines all
   three. → TinyGPT already ships the fusion.
3. **Academic agent benchmarks as a local CI gate** — BFCL/τ-bench are
   leaderboards, not products. → TinyGPT wrapped both (E1/E2); reframe as a
   workflow primitive (B32).

## Consolidation (the market is being rolled up)

Predibase→Rubrik · W&B Weave→CoreWeave · OpenPipe→CoreWeave ·
Humanloop→Anthropic · Langfuse→ClickHouse · Promptfoo→OpenAI ·
Goodfire→partnership-only. The buyers are GPU-infra companies and frontier
labs. A `$0-marginal-cost + data-stays-on-device + OSS-inspectable` tool is
structurally outside that roll-up — no GPU meter to acquire, no ingestion
revenue to absorb.
