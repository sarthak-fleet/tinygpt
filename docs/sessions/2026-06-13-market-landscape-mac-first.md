# Strategy session — the market landscape and the Mac-first wedge

**Date:** 2026-06-13
**Premise:** A wave of products now does "make a model good at your task"
(fine-tune SaaS) and "is my agent any good" (eval/observability). Where
does a Mac-first SLM toolkit fit, and what is genuinely unowned? Triggered
by the castform.com scan; researched the broader landscape.

**Method:** two parallel research sweeps (fine-tune platforms; agent
eval/interp platforms). Full citable map at
[`docs/learn/competitive-landscape.md`](../learn/competitive-landscape.md) —
this doc is the *strategy*, that page is the *evidence*.

## The one-paragraph finding

Every commercial player in both categories monetizes the exact cost a
Mac-first tool zeroes out: **cloud GPU rent** (per-token / per-GPU-hour)
for fine-tuners, **trace ingestion** (per-event SaaS) for eval vendors.
Their business model *is* the thing we don't charge for. That's not a
coincidence to route around — it's the wedge. And the market is
consolidating fast (six acquisitions in twelve months), which means the
independents are being absorbed by the infra + frontier-lab players a
local OSS-leaning tool is structurally outside of.

## Three unowned spaces

### 1. Mac-first *training* as a product (not a primitive)

The CUDA stacks treat Apple Silicon as a degraded afterthought —
bitsandbytes is CUDA-only, so Axolotl/TorchTune/LLaMA-Factory can't do
4/8-bit on Mac; Unsloth's MPS path is 3–5× slower with native MLX still
"coming soon." The *only* truly Mac-native training path is **Apple's own
MLX-LM** plus thin wrappers (mlx-tune, unsloth-mlx). Kiln AI has the
local-first UX but *delegates* training to external providers.

So "fine-tune on your laptop" is owned by a *library* (MLX-LM), not a
*product*. Nobody ships the packaged data → train → eval → deploy loop as
one Mac-native app. **That is the lane.** TinyGPT already has the full
pipeline (pretrain / SFT / DPO / distill / quantize / serve) on MLX-Swift;
the gap to fill is the product wrapper (B6 Factory tab) + the distribution
surface (B31 gallery + project pins).

### 2. Eval + interpretability + local, fused

The eval/observability field answers "is my agent good" with traces +
LLM-as-judge + RAG metrics. Nobody answers "*why* did it do that" at the
mechanistic level — the "root cause" features (Galileo Insights, LangSmith
trace analysis) are just another LLM summarizing traces, not model
internals. Interpretability tooling (Neuronpedia, TransformerLens, SAELens)
is research-funded OSS with **zero eval integration**; Goodfire's Ember
went partnership-only in Feb 2026. The two communities barely overlap.

TinyGPT already ships the fusion nobody else has: eval harnesses (BFCL /
τ-bench / lm-eval wrappers) **plus** an interpretability lab (SAE, activation
patching, logit lens, ROME/MEMIT, causal trace) **plus** a local agentic
leaderboard. Fused + local = category-of-one. This is the strongest
differentiation, stronger than local-training alone — because local
*training* is a cost story (compelling but copyable) while local
*eval+interp* is a capability story (nobody has the combination).

### 3. Academic agent benchmarks as a local CI gate

BFCL (Berkeley) and τ-bench are leaderboards, not products — no commercial
harness wraps them into "gate my SLM in CI." TinyGPT already wrapped both
(E1 / E2 shipped). The unfilled step is *framing* them as a developer
workflow primitive: `tinygpt eval` as a pre-commit / CI gate that fails the
build when a specialist regresses. That reframes shipped infra as a
product surface for ~zero new code (filed below).

## The consolidation signal (why now)

| Independent | Absorbed by | When |
|---|---|---|
| Predibase (RFT) | Rubrik | Jun 2025 |
| W&B Weave | CoreWeave | Mar 2025 |
| OpenPipe / ART | CoreWeave | Sep 2025 |
| Humanloop | Anthropic (acqui-hire) | Aug 2025 |
| Langfuse | ClickHouse | Jan 2026 |
| Promptfoo | OpenAI | Mar 2026 |
| Goodfire Ember | → partnership-only | Feb 2026 |

The pattern: GPU-infra companies (CoreWeave, Rubrik) and frontier labs
(OpenAI, Anthropic) are buying the tooling layer. A tool whose value is
**$0 marginal cost + data never leaves the device + OSS-inspectable** has
nothing for that consolidation to roll up — no GPU meter, no ingestion
revenue. It's a different shape of business (sell the tool, not the
compute), which is exactly why it survives the roll-up.

## Positioning moves (concrete)

1. **Lead with the capability story, support with the cost story.**
   Headline: *"Train, evaluate, and understand a specialist entirely on
   your Mac — and see why it works."* The interp+eval+local fusion is the
   moat; the $0-cloud-cost is the conversion lever.

2. **Price against the meter, not within it.** Every competitor's revenue
   is per-token/per-GPU-hour/per-trace. Sell the tool (OSS core + flat
   paid tier / one-time license), structurally un-matchable by anyone
   whose P&L is GPU rent.

3. **Own "fine-tune on your laptop" as a brand, not a feature.** MLX-LM
   owns the primitive; nobody owns the product. B6 Factory tab + B31
   gallery/pins are the surfaces that convert the primitive into a named
   product.

4. **Ship `tinygpt eval` as a CI gate.** Reframe shipped E1/E2 as a
   developer-workflow primitive (pre-commit / GitHub Action). Near-zero
   code; turns a benchmark wrapper into a product surface. Filed as **B32**.

5. **Privacy/compliance as the enterprise long-tail.** Air-gapped local
   training is a *requirement* in healthcare/legal/finance; only Lamini
   serves it, expensively, at enterprise tier. The indie/solo/regulated-
   small-team segment is unserved.

## What this does NOT change

- The north-star (the 3-axis Pareto win — quality ≥90%, speed ≥10×,
  memory ≤1/100 — from
  [`2026-06-06-mac-specialist-platform.md`](2026-06-06-mac-specialist-platform.md))
  is unchanged. This doc is *positioning*, not *strategy pivot*.
- A1 specialist is still the unlock. The market analysis sharpens *how we
  talk about it*, not *what we build next*. Without one shipped specialist
  the whole pitch is theoretical.

## Filed from this session

- **B32. `tinygpt eval` as a CI/pre-commit gate** — `docs/prds/B32-eval-ci-gate.md`
- **B33. One-command laptop-finetune onboarding** — `docs/prds/B33-laptop-finetune-onboarding.md`
- Competitive-landscape evidence page — `docs/learn/competitive-landscape.md`
