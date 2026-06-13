---
title: Advanced architecture, RAG/agents, eval & ML system design — interview-grade map
description: Senior/staff topics spanning modern transformer architecture, the linear-algebra↔transformers whiteboard, RAG, agents, evaluation (LLM-as-judge, perplexity, contamination), ML system-design rounds, and classic-ML depth — each with the best external source and repo anchor.
---

# Advanced architecture, RAG/agents, eval & system design

The breadth half of staff interviews. Format: **what's probed**, **best
source**, **in the repo**. Transformer *basics* (attention mechanics,
gradient descent) live in the `session-0X` series + `llm-mechanics-fundamentals.md`;
this doc is the senior-depth layer on top.

## Modern architecture

**1. The 2025-era decoder block.** Defend each deviation from the 2017
paper: pre-norm vs post-norm (gradient stability at depth), RMSNorm
(drops mean-centering, cheaper, same quality), SwiGLU gated FFN.
"Walk me through a modern decoder layer and justify each change."
*Learn:* [Transformer Design Guide](https://rohitbandaru.github.io/blog/Transformer-Design-Guide-Pt2/) · *senior*
*In repo:* the Qwen3 block in `TinyGPTModel`; mechanics in `llm-mechanics-fundamentals.md`.

**2. RoPE.** Rotary embeddings encode *relative* position via complex-plane
rotation of Q/K, parameter-free; extension via NTK/YaRN frequency interp
(see [`advanced-llm-inference.md`](advanced-llm-inference.md) §14). "Why does
RoPE generalize to longer context than learned absolute positions?"
*Learn:* [RoFormer](https://arxiv.org/abs/2104.09864) · *senior/staff*

**3. Attention as matmuls — the whiteboard.** Live derivation: X∈[n,d] →
Q,K,V via [d,d_k] projections; scores QKᵀ∈[n,n]; row-softmax; out [n,d_v].
**Must** explain the 1/√d_k factor: dot products of unit-variance vectors
have variance d_k, so scaling renormalizes to keep softmax out of the
saturated/vanishing-gradient regime. Expect FLOPs + O(n²) memory questions.
This is the "relate linear algebra to transformers" round.
*Learn:* [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · *senior*
*In repo:* `session-06-tokenization-embeddings.md` + `llm-mechanics-fundamentals.md` build this ground-up.

**4. Why decoder-only won.** Causal masking enables the KV-cache and cheap
autoregressive serving; in-context learning + scaling. "When would you
still reach for an encoder (BERT/embeddings)?" *Learn:* [Decoder-Only Workhorse](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) · *senior*
*In repo:* `model-vs-agent.md`; `session-04-ml-paradigms.md` on encoder–decoder.

(MoE routing → [`advanced-llm-training.md`](advanced-llm-training.md) §7;
GQA/MLA + KV economics → [`advanced-llm-inference.md`](advanced-llm-inference.md) §12.)

## RAG & agents

**5. RAG retrieval design.** Chunking (fixed/semantic/recursive, overlap)
as a product choice; hybrid BM25+dense (sparse catches exact IDs/SKUs,
dense catches paraphrase). "Rare-SKU queries fail — fix retrieval."
*Learn:* [Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/) · *senior*
*In repo:* Pace `PaceLocalRetrieval` is BM25 lexical + best-effort embedding
re-ranker today (`docs/prds/local-rag-layer.md`).

**6. Rerankers & embeddings.** Bi-encoder (fast ANN first stage) vs
cross-encoder (accurate rerank of top-k); when to fine-tune embeddings;
MTEB for selection. "Recall fine, top-3 precision poor — what stage?"
*Learn:* [Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) · *senior*
*In repo:* `tinygpt rerank-train` / `rerank-eval` / `eval-mteb`; mxbai-embed.

**7. RAG evaluation.** Measure retrieval (recall@k, MRR/nDCG) *separately*
from generation faithfulness/groundedness; knowledge conflict
(parametric vs retrieved). "Answers wrong but retrieval looks right —
localize it." *Learn:* [RAGAS](https://arxiv.org/abs/2309.15217) · *senior/staff*

**8. When NOT to use RAG / query rewriting.** Adaptive transforms
(multi-query, HyDE, decomposition) on weak retrieval; skip RAG for small
stable corpora (fine-tune / long-context) or low-latency paths — the
RAG-vs-long-context staff debate. *Learn:* [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) · *staff*

**9. Agents: ReAct, planning, memory, tools.** Agent = LLM + planning +
memory + tools; reason↔act loop; short-term (context) vs long-term (vector)
memory; structured function-calling + error recovery + loop bounding.
"Design a travel-booking agent — where does it loop/fail, how do you bound it?"
*Learn:* [Lil'Log: LLM Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) · *senior*
*In repo:* `tinygpt agent`; Pace's plan-act-observe loop; `agent-context-hierarchy.md`.

**10. Agent evaluation (trace-based).** Score tool-call correctness,
planning quality, task completion via step-level traces — not just the final
answer. "Eval an agent with 20 tool calls when only the last output shows."
*Learn:* [Agent Eval Guide](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide) · *staff*
*In repo:* B23 agent-eval-protocol (pass@1 over repeated runs ± σ);
`tinygpt eval-bfcl` / `eval-tau-bench` — and **today's A1 run** is this end-to-end.

## Evaluation depth

**11. LLM-as-judge + its failure modes.** The real signal is naming biases:
position, length/verbosity, self-preference, concreteness, prompt-injection.
Mitigate: pairwise + order randomization, rubric + CoT, validate vs human
labels. "Your judge prefers longer answers — prove it and correct it."
*Learn:* [Hamel: Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) · *senior/staff*
*In repo:* `tinygpt judge` (E7, `JudgeShim.swift`); strict-scorer mode.

**12. Perplexity — definition, uses, limits.** PPL = exp(mean per-token NLL)
= exp(cross-entropy). The catch: comparable **only under the same tokenizer**,
measures next-token confidence not correctness, lower PPL ≠ better
generations. "When is perplexity meaningful vs misleading?"
*Learn:* [HF Perplexity](https://huggingface.co/docs/transformers/en/perplexity) · *senior*
*In repo:* training loss in `Train.swift` is cross-entropy (exp it for PPL);
the eval suite (`eval-matrix-2026-06-08.md`) is the "PPL isn't enough → use task evals" argument made concrete.

**13. Contamination, pass@k, calibration.** Test leakage inflates scores —
detect/guard via held-out/private sets, canary strings, decontamination;
pass@k for code (functional correctness over k samples); calibration (ECE,
reliability diagrams). *Learn:* [Contamination Survey](https://arxiv.org/abs/2406.04244) · *staff*
*In repo:* `mac-assistant-judgment` benchmark ships a real contamination
check (Jaccard ≥0.6 vs train); `tinygpt eval-humaneval` is pass@k.

## ML system design & classic-ML depth

**14. Recommender / ranking / feed design.** The retrieval → ranking →
re-ranking funnel; two-tower/ANN candidate gen vs heavy ranker; objectives
(engagement vs relevance vs freshness vs diversity); cold-start. The
most-asked staff system-design prompt. *Learn:* [Eugene Yan: System Design for Discovery](https://eugeneyan.com/writing/system-design-for-discovery/) · *staff*

**15. Training-serving skew & feature stores.** Skew from features computed
differently train vs serve; the fix is one feature store + point-in-time
joins (no label leakage). "Offline AUC great, online tanked — debug."
*Learn:* [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) · *senior/staff*

**16. A/B testing & guardrail metrics.** Goal vs guardrail metrics,
power/MDE/sample size, novelty effects, peeking/multiple-comparison
corrections, shadow→canary→rollback. "Goal up but a guardrail moved — ship?"
*Learn:* [Microsoft ExP: Metric Pitfalls](https://exp-platform.com/Documents/2017-08%20KDDMetricInterpretationPitfalls.pdf) · *senior/staff*

**17. Classic-ML senior depth.** Bias-variance as capacity lever; L1/L2
(sparsity vs shrinkage); **calibration** (Platt/isotonic — critical for
credit risk); **causal inference / uplift** (two-model / meta-learners,
propensity, DiD, doubly-robust) under class imbalance; imbalance handling
(class weights, PR-AUC over accuracy). Maps directly to a credit-risk
background. *Learn:* [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) · *senior/staff*
*In repo:* the SAE-vs-PCA contrast ([`speech-and-systems-topics.md`](speech-and-systems-topics.md) §5)
is the dimensionality-reduction version of this.

## Suggested order

3 and 11–13 are the highest-differentiation for this profile (the
linear-algebra whiteboard + judge-bias enumeration + calibration). 14–17
are pure system-design/DS study; ESL is the slow material — budget for it.
