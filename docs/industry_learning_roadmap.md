# Industry learning roadmap

This is the external learning track for TinyGPT. Use it after the repo-local
[`learning_roadmap.md`](learning_roadmap.md): CS336 is the spine, and company
docs/blogs are the applied case studies.

The goal is not to copy frontier-scale infrastructure. The goal is to extract
small, testable ideas that fit TinyGPT: better data, cleaner evals, stronger
specialist training, and Mac-first runtime discipline.

## How to read

1. Read the source.
2. Write the one sentence lesson.
3. Map it to a TinyGPT artifact: code, doc, eval, or explicit skip.
4. Only implement if it improves a current Tier A/B item in `PLAN.md`.

## Module 0 - Course spine: Stanford CS336

Source: [Stanford CS336 - Language Modeling from Scratch](https://cs336.stanford.edu/)

Why it matters: CS336 is almost exactly TinyGPT's educational contract. It
walks through tokenizer/model/optimizer basics, systems profiling,
FlashAttention, distributed memory efficiency, scaling laws, data filtering and
deduplication, and SFT/RL-style post-training.

TinyGPT mapping:

| CS336 piece | TinyGPT anchor |
|---|---|
| Assignment 1: basics | `python_ref/`, `tests/test_phase1.py` |
| Assignment 2: systems | `wasm/`, `webgpu/`, FA2 notes |
| Assignment 3: scaling | `configs/`, `bench/`, `docs/benchmark_harness_design.md` |
| Assignment 4: data | `tinygpt download-dataset`, `dedupe`, dataset registry |
| Assignment 5: alignment/reasoning RL | `sft`, `dpo`, future RLVR/Tier 5 reasoning |

Action: add CS336 as the default external course for anyone learning the repo.
Do not import assignments wholesale; use it as a reading and audit checklist.

## Module 1 - Small model data recipes

Sources:

- [Hugging Face SmolLM blog](https://huggingface.co/blog/smollm)
- [FineWeb / FineWeb-Edu paper](https://arxiv.org/abs/2406.17557)

Lesson: small models do not win by architecture alone. They need unusually good
data: educational-quality text, code subsets, deduplication, and scale-aware
evaluation.

TinyGPT actions:

- Keep B10: quality classifier on pretrain data.
- Keep dedupe and MinHash dedupe in the default data path.
- Add every future training run to a manifest with corpus hash, filter settings,
  and eval set.

## Module 2 - Open post-training recipes

Sources:

- [Ai2 Tulu 3](https://allenai.org/tulu)
- [Tulu 3 technical blog](https://allenai.org/blog/tulu-3-technical)
- [Tulu 3 report](https://arxiv.org/abs/2411.15124)

Lesson: post-training is a recipe, not a single dataset. The useful shape is
SFT -> preference tuning -> verifiable-reward RL, with explicit data mixtures
and evaluation.

TinyGPT actions:

- Keep `docs/training/` as the canonical pretrain/SFT/DPO pipeline.
- Add held-out task evals before claiming any specialist win.
- Treat RLVR as a Tier 5 learning experiment until SFT/DPO specialists are real.

## Module 3 - Reasoning and RLVR

Source: [DeepSeek-R1 official repo/report](https://github.com/deepseek-ai/DeepSeek-R1)

Lesson: reasoning gains come from verifiable rewards and long rollouts, but
this is only meaningful after the base model and SFT path are stable.

TinyGPT actions:

- Use math/code tasks with exact checkers before any LLM judge.
- Start with GRPO/DAPO as mental models, not immediate production features.
- Log full trajectories, rewards, and token ids if any RL-style run happens.

## Module 4 - Agent design

Sources:

- [Anthropic - Building effective agents](https://www.anthropic.com/research/building-effective-agents/)
- [Mistral Agents docs](https://docs.mistral.ai/capabilities/agents/)
- [Mistral handoffs docs](https://docs.mistral.ai/agents/handoffs/)

Lesson: most useful agent systems are simple workflows with good tools. Multi-
agent handoffs help only when the boundary is crisp.

TinyGPT actions:

- Prefer tool quality and eval harnesses over more agent layers.
- Keep the router/specialist boundary explicit: one specialist per task family.
- Add handoff only after one specialist can prove it should delegate.

## Module 5 - Agentic coding and eval discipline

Source: [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive)

Lesson: the stealable pieces are not 30T tokens or 6,144 GPUs. They are data
mixing discipline, repeated agent evals, token-preserving trajectories, and
careful sandbox budgets.

TinyGPT actions:

- Keep B21: micro-AutoMixer for specialist data mixes.
- Keep B22: token-preserving agent trajectory recorder.
- Keep B23: repeated pass@1 agent eval protocol.
- Keep B24: Muon only after a large/proxy-scale re-benchmark.

## Module 6 - Evals as product infrastructure

Sources:

- [OpenAI Evals cookbook](https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals)
- [OpenAI structured-output eval example](https://cookbook.openai.com/examples/evaluation/use-cases/structured-outputs-evaluation)
- [OpenAI Model Spec evals note](https://openai.com/index/our-approach-to-the-model-spec/)

Lesson: evals should be scenario-shaped and rubric-shaped, not just aggregate
leaderboard numbers. Structured-output tasks need schema validity plus semantic
grading.

TinyGPT actions:

- Add custom evals for each specialist before training the specialist.
- For JSON/tool/storyboard outputs, score both schema validity and task success.
- Keep public benchmark scores separate from repo-local product evals.

## Module 7 - General foundation model reports

Sources:

- [Meta Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [Qwen docs: key concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)

Lesson: foundation-model reports are useful for phase structure, eval breadth,
tokenizer and multilingual choices, and safety/post-training taxonomy. They are
not directly actionable at TinyGPT scale.

TinyGPT actions:

- Read these for vocabulary and comparison tables.
- Do not chase 100B-scale architecture changes unless a tiny proxy can falsify
  the idea first.

## Module 8 - Mac/local runtime

Sources:

- [Apple MLX open-source project](https://opensource.apple.com/projects/mlx/)
- [Apple MLX M5 LLM research note](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

Lesson: TinyGPT's differentiator is not beating CUDA. It is making local,
inspectable model training and inference work well on Apple Silicon.

TinyGPT actions:

- Keep Mac-native MLX paths first-class.
- Measure TTFT, throughput, memory, and energy before adding runtime tricks.
- Treat CoreML/ANE/GPU changes as measured runtime work, not roadmap glamour.

## Module 9 - Specialized visual/video systems

Sources:

- [Lamina Labs](https://laminalabs.ai/)
- [Qwen Image technical direction](https://arxiv.org/abs/2605.10730) for a
  larger-scale multimodal comparison point.

Lesson: for TinyGPT, the feasible first step is a structured explainer compiler,
not pixel-native video generation.

TinyGPT actions:

- Keep Tier 5.7 scoped to prompt/doc -> script -> storyboard DSL -> deterministic
  render.
- Train a visual-planner specialist only after storyboard data and evals exist.

## Running source queue

Read in this order when updating the roadmap:

1. CS336
2. SmolLM / FineWeb-Edu
3. Tulu 3
4. Anthropic agents
5. OpenAI evals
6. Poolside Laguna
7. Apple MLX
8. Llama/Qwen reports
9. DeepSeek-R1
10. Lamina/video references

Each time a new source is added, update this file with one of:

- **Adopt now**: exact Tier A/B task.
- **Adopt later**: exact trigger.
- **Study only**: why it is context, not roadmap.
- **Skip**: why it does not fit TinyGPT.
