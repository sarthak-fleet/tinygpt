# Strategy session — what TinyGPT actually is, what's possible on a Mac

**Date:** 2026-06-06
**Premise:** TinyGPT shouldn't be framed as "small models beat large ones on
benchmarks." The real thesis is: **a Mac app where individuals build a
specialist for their specific task — bring data, pick a teacher, ship a
fast/cheap/comparable-quality model.** Distillation, fine-tuning, and
quantization are the toolkit that makes this actually work.

This doc captures the strategic frame, the memory math, and the product
roadmap implications.

## What TinyGPT is, restated

> **"A Mac app where you bring data, pick a teacher, and ship a
> specialist that runs 15-100× faster than the teacher with comparable
> quality on your specific task."**

Not framework. Not benchmark contender. Not research project. A **platform
for individual task-specialization on consumer hardware.**

The benchmark-beating angle is a *demo* of what the platform can do — not
the product itself.

## Success criterion (refined)

The user explicitly clarified the bar: **comparable quality + 20-150× speed
+ 150× less memory** is a complete win. Then optimize performance further
from there.

That maps to a 3-axis Pareto win:

| Axis | Required against teacher |
|---|---|
| Quality | ≥ 90% of teacher on the specialized task |
| Speed | ≥ 10× tokens/sec on the same Mac |
| Memory | ≤ 1/100 the teacher's footprint |

If a specialist hits this triangle, ship it. Headline-grade "beats teacher
on absolute quality" is a bonus, not a requirement.

## Mac memory math — what's possible at each scale

The 3B "limit" everyone repeats applies *only to pretraining from scratch.*
Different operations have radically different memory footprints, and the
practical ceiling jumps significantly as you go down this table:

| Operation | Cost per param | 48 GB Mac ceiling |
|---|---|---|
| Pretrain from scratch (Adam bf16) | ~6-8 bytes | ~3-4B |
| Full fine-tune (Adam bf16) | ~6-8 bytes | ~3-4B |
| **LoRA fine-tune** | ~2 bytes base + tiny delta | **~13-15B** |
| **QLoRA** (4-bit base + LoRA) | ~0.5 byte base + tiny delta | **~30-40B** |
| Distillation (teacher → student) | Teacher inference + student training (sequential or split) | Teacher up to 30B+; student up to 13B (LoRA) |
| Inference bf16 | 2 bytes | ~20-25B |
| Inference Q8 | 1 byte | ~40B |
| **Inference Q4** | 0.5 byte | **~70-80B** |

**Implications for what the platform should offer:**

- "Train from scratch on your Mac" — only works for ≤3B. Limited niche.
- "Fine-tune a pretrained model on your Mac" — works up to ~13B via LoRA
  (free, fast, no quality loss), up to ~30B via QLoRA (small quality
  trade for memory).
- "Distill from a larger teacher" — teacher can be a downloaded ≤30B model
  OR an API call (Claude, GPT-4). Student is whatever the user wants.
- "Quantize and ship" — final output can target up to 70B at Q4 for
  consumer inference, but most specialists will be ≤7B for real speed.

The real product story is **not** "train tiny models from scratch."
It's **"fine-tune or distill any model up to ~30B on your Mac for your
specific task."**

## Why 4B / 5B / 6B models barely exist (the gap)

Open-weights ecosystem clusters at 1B / 3B / 7B / 13B / 30B / 70B. The
4-6B niche is empty. Reasons:

1. **Consumer GPU memory tier alignment.** 4 GB → 1B; 8 GB → 3B or
   7B-Q8; 16 GB → 13B-Q8. A 5B model awkward — wastes 8 GB headroom,
   doesn't reach 13B-tier quality. Labs pick sizes that match buyers'
   actual hardware.
2. **Training-budget round numbers.** Cluster commit goes 3B → 7B (2×
   compute); 5B is "would rather just do 7B." Economics, not physics.
3. **Mistral 7B anchored the "quality sweet spot."** Every lab now
   benchmarks against 7B to be in the conversation.
4. **Differentiation thin.** 5B isn't meaningfully better than 3B for
   most tasks, worse than 7B. No clean marketing story.
5. **Recipe inheritance.** Most open labs use a Llama-style scaling
   recipe that only varies depth/width to hit Llama-class sizes.

**This is an opportunity, not a constraint.** Distillation lets you
target any size including 4-6B. A TinyGPT-Mega-tuned variant at 5-6B
would slot into "better than 3B, faster than 7B, fits 8 GB GPUs at Q8."
Niche worth investigating once the base recipe works.

## The realistic Mac-specialist sizes

| Student size | Memory (Q4 inference) | Train path on Mac | Likely speed vs 7B teacher |
|---|---|---|---|
| 22M (TinyGPT-Huge — what we're training) | ~12 MB | Pretrain from scratch ✓ | ~50-100× |
| 76M (Mega) | ~38 MB | Pretrain from scratch ✓ | ~30-50× |
| 1B | ~500 MB | Pretrain from scratch or fine-tune ✓ | ~10-20× |
| 3B | ~1.5 GB | Pretrain from scratch (tight) or fine-tune ✓ | ~5-10× |
| 7B | ~3.5 GB | LoRA fine-tune only on Mac | ~2-4× |
| 13B | ~6.5 GB | LoRA fine-tune (tight) | ~1-2× |

The sweet spot for "user's specialized task on their Mac" depends entirely
on the task. Routing/intent → 22M is plenty. Function calling → 76M-1B is
the band. Code completion → 3-7B. The platform's job is to **pick the
right size automatically** based on user data + task type.

## What's already built (status as of 2026-06-06)

| Component | Status |
|---|---|
| Mac app shell (Sample/Train/Fine-tune/Interp/Server tabs) | ✅ shipped |
| CLI: train, sample, sae, eval-*, judge, serve, to-coreml, to-safetensors, gguf-extract | ✅ shipped |
| Multiple presets (Tiny/Small/Huge/Mega) | ✅ shipped |
| HF model downloader (for teacher selection) | ✅ shipped |
| OpenAI-compatible serve (with batched-prompt fix shipped today) | ✅ shipped |
| GGUF + CoreML + safetensors exporters | ✅ shipped |
| 3-view eval comparison (by step/model/task) | ✅ shipped |
| Browser eval-leaderboard + sae-timeline viewers | ✅ shipped |
| Distillation recipe doc | ✅ shipped (today) |
| Quantization (Q5/Q6/Q8) | ✅ shipped |
| LoRA fine-tuning | ✅ shipped (in app's Fine-tune tab) |
| Eval pipeline E0/E1/E2/E3/E5/E7/E8 | ✅ shipped (E0/E3 mine, rest from elf) |

## Working principles (clarified 2026-06-06)

| Principle | Operational meaning |
|---|---|
| **Capture everything, execute by ROI** | Strategy doc holds the full surface; Tier 1-4 backlog ranks by ROI; ideas don't get lost when not yet acted on |
| **One canonical best per slot** | Not 20 distillation variants; one. Not PPO+DPO+GRPO competing; the consensus best per axis. |
| **AI coding flattens effort estimates** | "5 min vs 1 hour" — choosing the right thing matters far more than estimating effort precisely. Don't waste decision time on small effort differences. |
| **Outsider perspective is a feature** | The platform's product framing is being shaped by non-ML-insider thinking. Defers to expert advice on implementation; trusts gut on framing/strategy. Use advisor when committing. |
| **Adopt OSS only when it's the best; reinvent when we materially improve (10%+ quality/perf/integration)** | Default isn't "always adopt." Default is "adopt unless reinvention gives a real, measurable boost." If we're reinventing for nothing, clean it up immediately. |
| **Verify existence before claiming gap** | Before adding any item to a "gap list," grep the dispatch table + source tree. Don't plan to build what's already there. (See 2026-06-06 audit correction — I claimed ~7 gaps that were already shipped.) |
| **Results first; architecturally interesting later** | (Clarified 2026-06-06 EOD.) The user explicitly said: "we will get architecturally interesting. but we need results." The platform's first job is shipping working specialists that beat baselines. BLT-style adaptive tokenization, MoE, Mamba hybrids, music gen, Apple Intelligence integration are all interesting — but they DON'T ship until A1 + Distillation + Eval-tab + Templates + QLoRA all land first. Architecturally-interesting items move to Tier 4 unless directly required by a "results" item. |

## OSS audit — what to adopt vs build

Honest read: **~70% of the Tier 1 backlog has strong existing OSS to adopt.** Work shrinks from "implement from scratch" to "adapt + integrate." Updated effort estimates assume adoption.

### Tier 1 gaps × OSS sources

| Capability | Adopt from | Build effort |
|---|---|---|
| Distillation feature | TRL distillation patterns | 2 days (was 3-4) |
| QLoRA | mlx-lm 4-bit + LoRA primitives (compose, don't rewrite) | 2 days (was 3-5) |
| DPO | TRL `DPOTrainer` loss + ref-model trick (~20 lines) | 1-2 days (was 2-3) |
| Constrained decoding | llama.cpp GBNF format (text grammars, no engine to write) | 2 days (was 2-3) |
| KV cache in serve | Our own `sample` command already has it — port internally | 1-2 days |
| Safetensors loader in serve | mlx-lm architecture mappers (Llama/Qwen/Mistral/Phi/Gemma ready) | 2-3 days (was 3-5) |
| `tinygpt prep` data CLI | `datatrove` (HF) patterns + our existing MinHash + quality filter | 2 days (was 2-3) |
| `tinygpt eval-custom` | lm-eval-harness YAML custom-task spec | 1 day (was 2-3) |
| TIES model merging | `mergekit` core algorithm (~100 lines) | 1-2 days (was 2-3) |
| Q4_K_M quantization | llama.cpp dequant code | 1-2 days |
| Tokenizer trainer | HF `tokenizers` (Rust crate) wrapper | 2 days (was 2-3) |
| Domain-adapt SFT | Config preset on existing train loop | 1 day (was 1-2) |

### Tier 2 modality OSS sources

| Modality | Adopt from |
|---|---|
| VLM | MLX-VLM (mlx-community) — Mac-native LLaVA port |
| Whisper | Apple's `ml-whisper` (CoreML, ANE-accelerated) |
| Image gen | Apple's `ml-stable-diffusion` (SDXL Turbo / SD 1.5, ANE-accelerated) |
| TTS | Kokoro-82M (tiny, open, recent) OR F5-TTS |
| Embeddings | sentence-transformers framework + BGE-M3 weights |
| Reranker | BGE rerankers (BAAI) |
| Reward model | TRL training pattern |
| Music gen | MusicGen-small (Meta, open) |

### Multi-model architectures OSS

| Pattern | Adopt from |
|---|---|
| LoRA hot-swap | `peft` adapter registry pattern |
| RAG | LlamaIndex patterns (don't take whole framework) |
| Reranking | BGE rerankers + MTEB eval patterns |
| Phone-a-friend | No specific OSS — build (it's 50 lines) |

### What's actually ours to write from scratch

The integration glue and product UX:
- Local-teacher labeling pipeline (no one has this at this scope)
- Unified `tinygpt prep` CLI wrapper
- "Create Specialist" app wizard
- Templates for common patterns (router, classifier, JSON-extractor, etc.)
- Quality predictability heuristics
- Mac-specific Apple Intelligence / Shortcuts integration

The rest is **adapt and integrate**, not invent. With this discipline, the Tier 1 estimate revises from 4-6 weeks → **~3 weeks of focused work.**

### Audit as a recurring practice

For each new gap that surfaces in future sessions:
1. **First action**: grep GitHub / awesome-ml-X lists for prior art
2. **Second action**: read the closest 2-3 OSS implementations
3. **Third action**: decide adopt / fork / wrap / build, in that order of preference
4. **Only build from scratch** if nothing usable exists OR the license precludes adoption

The session-doc gap-table format above should be **filled in** for every new capability proposal, not skipped.

## What's missing — the disciplined backlog

### Tier 1 — text-only platform completion (CORRECTED 2026-06-06 after dispatch-table audit)

**Earlier draft of this list claimed ~7 gaps that turned out to be already shipped.** Corrected list below; verify any claimed gap by `grep 'case "X"' native-mac/Sources/TinyGPT/TinyGPT.swift` before adding.

**Already shipped (DO NOT re-plan):** `distill`, `dpo`, `gptq`, `hqq`, `prune-unstructured`, `prune-structured`, `bon`, `agent`, `cloud`, `escalate` (phone-a-friend), `dedupe`, `rome`, `memit`, `patch`, `causal-trace`, plus `ConstrainedGen.swift` (804 lines for constrained decoding) and `KVCache.swift` (885 lines).

**Actual remaining gaps (verified by absence in dispatch / source tree):**

| Gap | One-best impl | Effort (with OSS adoption) |
|---|---|---|
| **QLoRA** | 4-bit base + LoRA training recipe (primitives exist: `gptq`/`hqq` + LoRA — just wire them together) | 2 days |
| **TIES model merging** | Port from mergekit (~100 lines algorithm) | 1-2 days |
| **Safetensors loader for serve** (foreign architectures) | Adopt mlx-lm's architecture mappers | 2-3 days |
| **KV cache wired into `serve`** | `KVCache.swift` exists; needs serve integration | 1-2 days |
| **`tinygpt prep`** wrapper | Bundle existing `dedupe` + `quality-filter` + extract + format normalize | 1 day |
| **`tinygpt eval-custom`** | Custom-task YAML on top of existing `run-lm-eval` | 1-2 days |
| **Q4_K_M quantization level** | Port from llama.cpp | 1-2 days |
| **Tokenizer trainer** | Wrap HF `tokenizers` Rust crate | 2 days |
| **Domain-adapt SFT mode** | May already be a `train` flag — verify before building | 0-1 day |
| **App-native "Create Specialist" wizard** | Composes existing CLI subcommands behind GUI | ~1 week |
| **Templates for common patterns** | Router, classifier, JSON-extractor, tool-caller, summarizer | ~1 week |
| **Quality predictability** | Heuristics from prior runs | hard, iterative |

**Revised Tier 1 effort: ~2 weeks of focused work.** Much closer to done than original 4-6 week estimate suggested.

## Reinvention cleanup candidates (audit before removing)

The flip side of "adopt OSS only when best": **audit existing code for places we reinvented without payoff.** Candidates (need verification before action):

| File / area | Possible OSS alternative | Verdict |
|---|---|---|
| `ExtractorData.swift` (525 lines) + `GitHubCorpus.swift` (525 lines) | `datatrove` (HF) | **Investigate** — if our impl predates datatrove and offers no Mac-specific advantage, migrate |
| `Score.swift` (782 lines) | overlaps with `eval-compare` / `run-lm-eval`? | **Investigate** — may be a predecessor that should consolidate into the newer eval stack |
| `Serve.swift` HTTP layer (1258 lines) | swift-nio / Hummingbird / Vapor | **Keep** — requires MLX integration + Ollama + OpenAI + serial inference queue. No good Swift HTTP framework fits. |
| `GGUFReader.swift` (558 lines) | `gguf` crate, llama.cpp's reader | **Keep** — loads directly into MLX tensors; no Swift+MLX alternative |
| 3 Rust crates (parquet, hf-downloader, sandbox) | arrow-rs (used), hf-transfer crate, no Mac sandbox alt | **Keep all three** — parquet uses arrow-rs (correct); hf-downloader replaces Python (justified for speed); sandbox is Mac-specific |

**Highest priority cleanups to investigate this week:** `Score.swift` consolidation into the newer eval stack; `ExtractorData.swift` vs datatrove migration analysis.

### Tier 2 — multimodal (user-confirmed priorities: vision + voice; image gen; video defer)

Each modality is real engineering — substantial scope, not a knob. Ordered by ROI.

| Modality | The one best path | Effort | Notes |
|---|---|---|---|
| **Vision input (VLM)** | **LLaVA-style**: CLIP-ViT encoder → projection → LLM body. LoRA-train the projection + LLM. | ~2-3 weeks | Highest-ROI modality after text. Unlocks specialists for "describe screenshots / extract from PDFs / classify designs / chart QA." |
| **Speech-in (Whisper)** | Bundle CoreML-compiled Whisper-small as audio frontend → text token stream → existing LLM specialist | ~1 week | Easiest win. Apple's own apps already use this. |
| **Image generation** | SDXL or SD 1.5 + LoRA training via Apple's `ml-stable-diffusion` (CoreML) | ~3-4 weeks | Well-trodden Mac path. LoRA fine-tuning for personal style is the killer feature. |
| **TTS / voice cloning** | F5-TTS or XTTS — open weights, voice-cloning support | ~2 weeks | Real accessibility / agent feature. |
| **Video understanding** | Frame sampling (1 fps) → run VLM on each → aggregate | ~1 week after VLM | Trivial once VLM lands. Loses temporal info but useful. |
| **Embeddings** | sentence-transformers-style — small encoder, contrastive training | ~3-5 days | Useful for RAG / retrieval specialists. |
| **Video generation** | Stable Video Diffusion / AnimateDiff / CogVideoX-2B | ~3-4 weeks | **Punt to v3+.** Local quality far below Sora/Veo/Kling; SOTA video needs datacenter GPUs. Not credible to ship in v1-v2. |

### Tier 3 — advanced (defer until Tier 1+2 land)

| Capability | When to revisit |
|---|---|
| Reasoning RL (GRPO / DeepSeek-R1 style) | Only if DPO + distillation + constrained decoding can't close gap on reasoning tasks |
| Mixture of Experts (MoE) | If a 4-6B MoE proves competitive with dense 7B at lower latency |
| Mamba / SSM hybrid | If long-context tasks need >8K tokens efficiently |
| AWQ smarter quantization | If GGUF Q4 quality insufficient for shipping |
| Pruning | Only if specific use case it solves better than quantization |
| **Music generation** (MusicGen / Stable Audio) | Tier 4 modality — real demand but not blocking the v1 platform |

### Tier 4 — explicitly out of scope (or skip / research-only)

| Capability | Why skipped |
|---|---|
| Video generation | Mac quality far below Sora/Veo/Kling; SOTA needs datacenter GPUs |
| 3D generation (DreamGaussian, Instant-NGP) | Niche; only add if target audience asks |
| World models (Dreamer-V3, GENIE) | Research-only; not consumer-facing |
| Diffusion language models | Research-only; not yet useful |
| Time series, graph neural | Different domain, not language-model adjacent |

## Judgment-family models (related but distinct from rerankers)

Beyond generating text, models can *evaluate* text. This unlocks self-refinement, RL data generation, automated eval, and quality gating. Should be a first-class capability slot:

| Model type | What it does | Output shape | Use case |
|---|---|---|---|
| **Judge** (LLM-as-judge) | Evaluate quality of a single output | Score + reasoning | Eval (✅ E7 shipped — pairwise + rate modes), preference data, feedback |
| **Reranker** | Score N candidates, return ranking | Numeric score per candidate | Best-of-N selection, search/retrieval |
| **Verifier** | Check if output is correct (hard signal) | Binary + reasoning | Math, code (compile+run), JSON schema validity |
| **Reward model** | Scalar reward for RL training | Single number | RLHF/DPO data generation. Lets DPO work without preference annotations — judge labels them. |
| **Critic** | Detailed critique for revision | Free-form critique | Self-refinement loops (output → critic → revised output) |
| **Constitutional / principle-based** | Judge against a set of rules | Pass/fail per rule | Safety, style, persona consistency |

**Status for our platform:** judge shipped (E7); the others are gaps. The one-canonical-best per slot:

| Slot | Recommended impl | Effort |
|---|---|---|
| **Verifier** | Task-specific hardcoded checkers (Python sandbox for code, JSON-schema validator, math eval) + a small learned classifier where rules aren't enough | ~3-5 days |
| **Reward model** | Distill a judge into a smaller faster scoring head (reuse the E7 judge as the data labeler) | ~3-5 days |
| **Critic** | LoRA on a small base model with critique-formatted training data, OR prompt a strong judge | ~3 days for prompt-based; longer for trained |
| **Reranker** | Small encoder-only model (BERT-style) trained on (query, candidate, score) triples | ~1 week |

These compose with the multi-model architectures below — e.g., generator → reranker → verifier → output is a high-quality production pattern.

## Multi-model architectures (how multiple models compose at inference)

A specialist doesn't have to be a single model. Several patterns compose multiple models for better quality, lower cost, or new capabilities:

| Pattern | What it does | Status |
|---|---|---|
| **Speculative decoding** | Small draft proposes N tokens; large verifies in one pass. Lossless ~2-3× speedup. | ✅ shipped (B14) |
| **Phone-a-friend / escalation routing** | Small tries; if low-confidence, escalate to large. Saves cost when easy queries dominate. | gap — serve-level addition (~3-5 days) |
| **Cascade / waterfall** | Multi-tier version (cheap → mid → expensive). Each tier handles what the previous couldn't. | gap (extension of phone-a-friend) |
| **Teacher-student distillation** | Big model labels offline; small serves inference. (Bulk of "build a specialist" recipe.) | recipe doc; Tier 1 build |
| **Self-consistency** | Same model N times with different seeds, vote. Free quality boost for math/reasoning. | trivial to add at serve |
| **Ensemble** | N different models in parallel, vote on output. N× cost for quality bump. | trivial |
| **Multi-agent / debate** | Models discuss until consensus. Slow but high quality. | research-y; defer |
| **Tool-augmented agent loop** | Model decides actions, tools execute, results feed back. Browser-use pattern. | A1 specialist enables this |
| **RAG** (retrieval-augmented) | Small model + retrieval over big corpus. "Phone a database." | needs embeddings (Tier 2) + retriever |
| **Hierarchical reasoning** | Planner breaks task → workers solve → coordinator aggregates | application-level |
| **Reranking** | Generator produces N candidates; reranker scores; best wins. | needs reranker model (small) |
| **Verification loop** | Generator → verifier → loop until valid. Math/code use this. | application-level |
| **Sketch + refine** | Small drafts; large polishes. Trades roundtrips for quality. | application-level |
| **LoRA hot-swap** | One base + many LoRA adapters, swap per task. One model serves many specialists. | gap — high-value addition |
| **TIES merging** | Weight-averaging multiple fine-tunes. Often beats any individual. | planned (Tier 1) |
| **Critic-actor** | Actor proposes, critic evaluates. RL inheritance. | research |

**Most valuable for our platform to add:** phone-a-friend / cascade (serve-level), LoRA hot-swap (one base, many specialists), reranking (small encoder for best-of-N selection). All ~3-5 days each.

## Structured output — beyond JSON

JSON is solved because its grammar is well-defined. **The same mechanism (grammar-constrained decoding) handles any format with a definable grammar.** Our Tier-1 "constrained decoding" slot should be **grammar-engine general**, not JSON-specific:

| Format | Grammar | Use case |
|---|---|---|
| JSON | built-in | tool calls, API responses |
| YAML / TOML | Lark grammars | config files |
| XML | grammar | legacy systems |
| **SQL** (per table schema) | schema-derived | text-to-SQL specialists |
| **Protobuf** (per .proto) | schema-derived | binary-format generation, RPC payloads |
| **GraphQL** (per schema) | schema-derived | query generation |
| Cron | small grammar | scheduling DSLs |
| LaTeX (math subset) | partial grammar | math notation generation |
| Markdown with required structure | grammar | structured doc generation |
| CSV / TSV (per schema) | row grammar | tabular output |
| HTML / JSX (constrained) | tree grammar | UI generation |
| Regex pattern enforcement | FSM-direct | phone numbers, dates, IDs |
| Bash / shell | iffy grammar | risky but possible |
| **Custom DSL** | user-provided grammar | domain-specific applications |

**Engine choice (one canonical):** GBNF (llama.cpp grammar format). Reasons: well-documented, expressible enough for any format above, ecosystem familiarity (llama.cpp users know it), text-based grammar files easy to ship. Alternative: xgrammar (NVIDIA, faster) or Outlines (Python, most flexible). Pick GBNF for ecosystem fit.

**For protobuf specifically:** the .proto file IS the grammar. Generator translates .proto → GBNF → constrained decoder that produces byte-perfect serializable protobuf. Feature, not research.

## Diffusion models — speed reality check

| Diffusion type | Status | Speed on Mac |
|---|---|---|
| **Image diffusion (fast variants — SDXL Turbo / Flux Schnell / LCM)** | production-ready | **1-2 sec per image** (1-4 denoising steps). Real-time-feeling. |
| Image diffusion (traditional, 20-50 steps) | available | 3-8 sec per image |
| Audio diffusion | available | ~1-2 sec for short clips |
| **Text diffusion** (Mercury Coder, LLaDA, DiffuLLaMA, SEDD) | early-commercial | **Mercury / Mercury Coder (Inception Labs, 2025)** is the famous example — claims 1000+ tok/sec on H100 by generating tokens in parallel via iterative denoising. Real demo. Open-weights stack still emerging; ecosystem (training tooling, evals) years behind autoregressive. **Status: keep watching, don't build yet.** Revisit when the first strong open-weights diffusion coder ships. |
| Video diffusion | (Tier 4 — skipped) | Slow; quality far below SOTA on Mac |

**For our image-generation slot (Tier 2):** SDXL Turbo or Flux Schnell + LoRA fine-tuning via Apple's `ml-stable-diffusion` (CoreML). Runs on Neural Engine. That's the "fast diffusion on Mac" pitch — real and shippable.

## Inference performance — what's possible on Mac

For context on the "fast and cheap" half of the pitch:

**What Groq / Cerebras / SambaNova do:**

| Vendor | Hardware | Why it's fast |
|---|---|---|
| Groq LPU | Custom inference chip, SRAM-only (230 MB on-die), deterministic execution | Skips DRAM bottleneck. Llama-3-70B at 500+ tok/s. |
| Cerebras WSE-3 | Entire wafer as one chip (900K cores, 44 GB on-die SRAM) | No off-chip memory at all. Llama-3-70B at 1800+ tok/s. |
| SambaNova RDU | Reconfigurable dataflow | Custom dataflow per model |

The win is ~80% hardware (SRAM-only memory hierarchy bypasses GPU's HBM bottleneck), ~20% software co-design.

**What we can stack on Mac:**

Not Groq-fast, but "feels real-time" is achievable:

| Lever | Status | Estimated speedup |
|---|---|---|
| Q4 quantization | ✅ (GGUF) | 4× memory pressure reduction |
| KV cache | ✅ in sample, gap in serve | 5-10× on long sessions |
| Speculative decoding (B14) | ✅ shipped | 2-3× lossless |
| Apple Neural Engine inference (M5) | gap (queued #193) | 3-5× for INT8 workloads |
| Continuous batching | gap | 2-3× under load |
| Compiled kernels (MLX compile) | ✅ available in sample | 1.5-2× |

**The memory-bandwidth ceiling (hard physics):**

Autoregressive inference reads the entire model per token. So:

```
max_tokens_per_second = memory_bandwidth / model_size_in_bytes
```

| Hardware | Bandwidth | 7B Q4 ceiling | 70B Q4 ceiling |
|---|---|---|---|
| M5 Pro Mac (~600 GB/s) | 600 GB/s | ~170 tok/s | ~17 tok/s |
| H100 (3 TB/s HBM3) | 3000 GB/s | ~850 tok/s | ~85 tok/s |
| Groq LPU (~80 TB/s SRAM) | 80,000 GB/s | ~22,000 tok/s | ~2,200 tok/s |
| Cerebras WSE-3 (~125 TB/s on-die) | 125,000 GB/s | ~35,000 tok/s | ~3,500 tok/s |

The Groq/Cerebras advantage is ~130× memory bandwidth (on-chip SRAM vs DRAM). That's where their tok/s wins come from — not magic, not algorithmic.

**Realistic stacked ceiling on M5 Pro (48 GB):**

| Model | Theoretical ceiling | Practical with all levers stacked |
|---|---|---|
| 22M (Huge) at Q4 | 100,000+ tok/s | **~500 tok/s** (overhead dominates at this size) |
| 3B specialist at Q4 | ~170 tok/s | **~80-120 tok/s** |
| 13B (QLoRA fine-tune) at Q4 | ~40 tok/s | **~25-30 tok/s** |
| 70B (inference only) at Q4 | ~17 tok/s | **~10-15 tok/s** (barely fits) |

**What can't be 100×-ed on Mac:** the bandwidth ceiling. You'd need 100× more SRAM bandwidth, which means Groq/Cerebras-class hardware ($10K-100K). Not happening on a laptop.

**What can:** the **size of the specialist itself.** 22M vs 3.8B is already ~150× the param count. Combined with the inference speedups, the "20-150× faster than teacher" claim in the success criterion is achievable.

The platform pitch isn't "as fast as Groq" — it's "fast enough for your specific task, on your own machine, at zero cost." A 300 tok/s specialist locally beats a 1000 tok/s cloud model when you factor in 50-200ms network round-trip + cost + privacy.

## Market landscape (what's out there as of mid-2026)

For situating where TinyGPT fits in the broader inference / specialization market:

| Category | Players | What they sell |
|---|---|---|
| **Fast inference on standard models** | Groq, Cerebras, SambaNova, Together, Fireworks, DeepInfra | Hosted Llama/Mistral/Qwen at 500-1800 tok/s |
| **Embeddings / retrieval** | Voyage AI, Jina, Cohere Embed | Small specialized encoders |
| **Reranking** | Cohere Rerank, Jina Reranker | Cross-encoder models, ~150M, very fast |
| **Speech (ASR)** | Whisper API, Speechmatics, AssemblyAI, Deepgram | Real-time transcription <100ms |
| **TTS / voice cloning** | ElevenLabs, Cartesia, Resemble, PlayHT | Specialized TTS, ultra-low latency for agents |
| **Summarization (specialized)** | Cohere Summarize, AssemblyAI LeMUR, Cresta, NICE | Often not LLMs — sometimes encoder-decoder T5-class |
| **Code-specific** | Cursor, Continue.dev, Codeium, Tabnine, Mercury Coder | Distilled code specialists or routed agents |
| **Diffusion text (emerging)** | Mercury (Inception Labs), LLaDA, SEDD | Parallel-decoding via diffusion, 1000+ tok/s claims |
| **Mobile / on-device** | Apple Intelligence, Gemini Nano, Anthropic-on-iOS | 3B-class distilled for phone CPUs |
| **Specialized agents** | Cognition Devin, browser-use, Stagehand, Cline, Aider | Agent loops over standard models with task-specific scaffolding |
| **Local-Mac inference** | Ollama, LM Studio, MLX-LM, llama.cpp, **TinyGPT** | Run open weights on consumer Mac |

**TinyGPT's positioning:** the only player in "local-Mac inference" that ALSO covers the full build/upgrade lifecycle (data → train → eval → deploy). Ollama / LM Studio run pretrained models; TinyGPT builds and ships specialists end-to-end on the same machine.

### Competitive landscape — deeper research (added 2026-06-06)

#### Direct Mac/local-inference competitors

| Player | What it is | Differentiator |
|---|---|---|
| **LM Studio** | GUI for GGUF/MLX models; no fine-tuning | "LM Studio runs models. TinyGPT *makes* them." |
| **Ollama** | Local inference daemon; loads adapters | Complement — export target. "Bring your fine-tune to Ollama; build it in TinyGPT." |
| **Jan** | OSS ChatGPT desktop, inference-only | Jan = chat client; TinyGPT = factory. |
| **GPT4All** | Inference + LocalDocs RAG | Different lane — RAG vs distilled specialist. |
| **MLX-LM** | Apple library: train + inference on Apple Silicon | **Closest direct competitor.** TinyGPT = product/workflow/eval loop atop MLX-LM as library. |
| **Unsloth + Ollama** | Linux/CUDA fine-tune + export | "Unsloth needs a 4090; TinyGPT runs on your laptop." |

#### Cloud fine-tuning (we don't compete)

| Player | Verdict |
|---|---|
| Together AI ($0.48-$12/M tokens, ≤405B) | Different scale + buyer. Concede. |
| Fireworks (≤16B, $0.50/M tokens) | Same. Potential export target. |
| Anyscale (enterprise HIPAA/SOC2) | Wrong buyer. |
| **OpenAI fine-tune API** | **DEPRECATED May 2026.** Major tailwind. |
| Anthropic fine-tune | Not publicly available. |

#### Eval platforms

| Player | Verdict |
|---|---|
| **lm-eval-harness** (EleutherAI) | Use it; don't rebuild. Already wrapped via E3. |
| **Inspect AI** (UK AISI) | Use for agent/tool evals. |
| **Arize Phoenix** | Complement — they watch prod; we score pre-deploy. |
| LangSmith / Comet / HumanLoop | Wrong buyer; ignore. |

#### Synthetic data / distillation

| Player | Verdict |
|---|---|
| **Distilabel** (HF) | Use as engine; wrap with opinionated recipes. |
| Gretel / Mostly AI | Tabular; ignore. |
| Lilac (Databricks) | Complement; different pipeline stage. |

#### Apple-specific frontier

| Player | Verdict |
|---|---|
| **Apple Foundation Models** (iOS 26) | Both partner and competitor. Same on-device thesis; locked to Apple's base + App Store. |
| **MLX / mlx-community** | Build on it; contribute back. |

### Top 5 positioning statements

1. **"Run-only tools chat with models. TinyGPT *builds* them."** — vs LM Studio / Ollama / Jan / GPT4All.
2. **"Fine-tune on the Mac you already own — no 4090, no cloud bill, no data leaving the device."** — vs Together / Fireworks / Unsloth.
3. **"OpenAI killed their fine-tune API in May 2026. Specialist models shouldn't be a hosted API in the first place."** — market-signal positioning.
4. **"One opinionated path — pick a teacher, bring data, get a fast specialist — wrapped around the best OSS pieces (MLX-LM, lm-eval-harness, Distilabel)."** — honest about adoption; integrated workflow is the differentiator.
5. **"Apple's Foundation Models give you one 3B model locked to iOS. TinyGPT gives you any open base from 22M to 13B, deployable anywhere."** — vs Apple's on-device push; no ecosystem lock-in.

### Where TinyGPT honestly does NOT compete

- 70B+ production fine-tuning — Together/Fireworks win; concede
- LangChain-tied production observability — LangSmith owns it
- RAG-over-my-files chat — GPT4All's LocalDocs is mature
- Tabular synthetic data — Gretel/Mostly own enterprise tabular
- iOS App Store distribution for AI — Apple's Foundation Models wins; export to it

### Wave 2 landscape — AI coding / agents / inference / reasoning / RAG (researched 2026-06-06)

#### AI coding tools

| Player | Verdict | Notes |
|---|---|---|
| Cursor, Cody, Codeium, Augment, Windsurf | **Ignore** (cloud-IDE plays, different surface) | Wrong layer for us |
| **Continue.dev, Aider** | **Complement** — they consume our `serve` endpoint | "Point Continue at a TinyGPT specialist trained on your repo" |
| **Tabnine** | **Direct competitor on "per-repo fine-tune" claim** | They charge enterprise $$, cloud-clone of proprietary. **Tagline: "Tabnine's per-repo fine-tune, but on your Mac, on open weights, free."** |

#### Agent frameworks

| Player | Verdict | Notes |
|---|---|---|
| **LangGraph, smolagents (HF), Pydantic AI** | **Complement, high priority** | Ship cookbook pages — these are zero-effort distribution channels |
| AutoGen, CrewAI | Complement, lower priority | Smaller / less-aligned audiences |
| Magentic | Ignore | Tiny audience |

**Sharp insight:** None of these *produce* specialists. They orchestrate. **TinyGPT is the missing model-producer layer underneath.** Cookbook pages for smolagents + Pydantic AI reach high-intent audiences cheaply.

#### Inference framework wars

| Player | Verdict | Notes |
|---|---|---|
| vLLM, SGLang, TensorRT-LLM | Ignore on Mac (CUDA) | Reference architectures |
| **HF TGI** | **DEAD** (maintenance mode since March 2026) | Strike from reference lists |
| llama.cpp | Compete on Mac | They need GGUF conversion; we serve `.tinygpt` native |
| MLX-LM serve | Direct Mac competitor | We bundle train+eval+serve; they ship serve-only |
| **vllm-mlx** (community fork) | **THREAT** — claims 4.3× throughput over llama.cpp on Apple Silicon w/ continuous batching | See strategic decision below |

#### Reasoning models

| Player | Verdict | Notes |
|---|---|---|
| **DeepSeek-R1-Distill-Qwen-1.5B** | **Default base for `tinygpt grpo`** (Tier 3) | Only realistic single-Mac GRPO target |
| **QwQ-32B (Q4)** | **Mac flagship reasoning teacher** | 32B fits 48GB at Q4; Apache 2.0 |
| DeepSeek-R1 (671B) | Use as teacher via distillation traces | Not runnable on Mac |
| o1 / o3 / Claude thinking / Gemini thinking | **Forbid as teacher** (our positioning is "no API spend") | — |

**Sharp insight:** Half of HF reasoning model cards mention GRPO; it's table stakes. If we ship `tinygpt grpo`, DeepSeek-R1-Distill-Qwen-1.5B + GSM8K reward is the recipe.

#### RAG infrastructure

| Player | Verdict |
|---|---|
| **LanceDB** | **Default for TinyGPT RAG** — embedded, no server, matches our "single binary" ethos |
| ChromaDB | Optional default for tutorials |
| Qdrant | Optional (richer filters) |
| Pinecone, Turbopuffer | Ignore (SaaS) |
| Weaviate | Ignore for default (server process) |

**Sharp insight:** Every RAG stack assumes OpenAI embeddings. **None treat "I trained my own 22M embedding specialist" as a first-class workflow.** That's a wedge.

### Three strategic decisions surfaced by wave 2

These warrant explicit decisions, not deferral:

#### Decision 1: How to handle vllm-mlx serve threat

vllm-mlx is claiming 4.3× throughput at 16 concurrent requests on Apple Silicon. Our hand-rolled `serve` has no continuous batching. Three options:

| Option | Effort | Differentiation |
|---|---|---|
| (a) Match vllm-mlx | Months of paged-KV + scheduling work | Strong but eats `results-first` priority |
| (b) Wrap vllm-mlx as backend | ~1-2 weeks | None on serving; differentiate on train/eval loop instead |
| (c) Status quo | 0 | Lose throughput war within 6 months |

**Recommended: (b) — wrap.** Aligns with the `results-first` principle. Our value is train+distill+eval+ship loop, not serve-throughput. Worth a dedicated PRD to evaluate vllm-mlx maturity + plan the integration.

#### Decision 2: Add "specialist embedder" as Tier 2 modality

**New opportunity surfaced:** every RAG stack assumes OpenAI embeddings. A `tinygpt embed-train` recipe + LanceDB integration is weekend-buildable, reuses our existing specialist-training stack, opens the RAG category without us shipping a vector DB.

**Recommended: yes, add to Tier 2.** Slot between "Speech-in (Whisper)" and "Image gen" by priority. PRD-sized.

#### Decision 3: Distribution strategy — agent frameworks > editor integrations

Cookbook pages for **smolagents** and **Pydantic AI** reach high-intent audiences (people already running local models, already caring about tool-call + structured-output reliability) more cheaply than chasing the Continue.dev / Aider crowd (who already have working models).

**Recommended: write 2 cookbook pages first, before any new editor-integration work.** Each ~1 day. Use the same recipe as the "Tabnine alternative" positioning.

### New positioning statements (from wave 2)

Add to the top 5:
- **"Tabnine's per-repo fine-tune, but on your Mac, on open weights, for free."** (cuts cleanly against the only incumbent with "specialist code model" as a productized story)
- **"Every RAG stack assumes OpenAI embeddings. Train your own."** (wedge into RAG without building a vector DB)

## Flagship example applications

These are **apps built on top of the platform** — not new modalities. They demonstrate what the platform can do:

| Example app | Built on | Notes |
|---|---|---|
| **Browser agent (browser-use / Stagehand style)** | VLM (Tier 2) + tool-call specialist (A1) + Playwright integration (~1 week) | Specialist outputs structured DOM actions (`click[id=42]`, `type[id=87]=text`). VLM disambiguates visual elements. Eval against WebArena / OSWorld / Mind2Web. |
| **Per-language code specialist** | Distillation (Tier 1) + LoRA + custom eval | Distill DeepSeek-Coder-33B / Qwen2.5-Coder-32B into a 3B specialist for ONE language (Rust, TS, Swift, Python, etc.). Match teacher within ~5-10pp on that language, run 5-10× faster. **Killer demo of the platform.** |
| **Voice command → action** | Whisper (Tier 2) + tool-call specialist (A1) | Speak a request → transcribed → routed to specialist → action executed |
| **Personal-image-style generator** | Image gen (Tier 2) + LoRA training | Dreambooth-style: train a LoRA on user's photos, generate in their style |
| **Document QA specialist** | VLM (Tier 2) + custom eval | Read PDF / screenshot of doc → answer questions about it |
| **Local research agent** | Tool-call specialist (A1) + web search tool | Smaller alternative to Perplexity / GPT-4 web browsing |
| **Text-to-SQL specialist** | Tool-call + constrained decoding (per-schema grammar) | Generate valid SQL for the user's actual table schema. Killer feature for analytics-adjacent users. |
| **Protobuf / RPC generator** | Constrained decoding (per-.proto grammar) | Generate valid RPC payloads for the user's services. Specialist for backend devs. |
| **Phone-a-friend smart router** | Local small specialist + cloud / larger-local escalation | App-level pattern: 90% of queries handled locally, 10% escalate. Pareto-optimal cost/quality. |

The platform's value is in being the *substrate* these are built on. Once Tier 1 + 2 land, each of these is ~1-2 weeks of integration work, not a research project.

## Language-specific code distillation — the killer recipe

Concrete recipe (e.g., for a Rust specialist):

| Step | Detail |
|---|---|
| Teacher | DeepSeek-Coder-V2-33B at Q4 (~16 GB) OR Qwen2.5-Coder-32B (similar) — strongest open code models |
| Inputs | 100K-500K target-language prompts: completion contexts, refactor tasks, lint-fix, doc generation. Sources: the-stack-smol (cached), GitHub scrape of target language, synthetic from teacher |
| Labeling | Teacher labels prompts locally — ~2-3 days on Mac at 5 tok/s for 100K examples |
| Student | 3B (fine-tune Qwen2.5-Coder-3B base) OR LoRA on a 7B base. Use Tier-1 distillation feature + QLoRA. |
| Training | 1-2 days on Mac |
| Eval | HumanEval-{lang} + MBPP-{lang} + custom user examples for that language's tricky patterns |
| Outcome | Match teacher within 5-10pp on target language; run 5-10× faster on same Mac |

**Why this works particularly well:**
- Code has hard correctness signal (compiles or doesn't)
- Long-tail language-specific patterns (Rust borrow checker, Swift @MainActor, Python f-strings) are exactly where small specialists shine
- General LLMs (GPT-4, Claude) trained on all languages → mediocre at each specific one
- Per-language specialists are a clean differentiator vs general assistants

## Tokenization — why tokens (not characters / bytes)

Tokens are just contiguous character groups the tokenizer decided to chunk together. Why we use them instead of raw bytes/characters comes down to one number: **attention cost grows as O(N²) in sequence length.** Shorter sequences = exponentially less compute.

| Approach | Vocab | "hello world" | Tokens per KB of English |
|---|---|---|---|
| Byte-level | 256 | 11 | ~5,000 |
| Character-level | ~150 | 11 | ~5,000 |
| **Subword BPE (SmolLM2)** | **49K** | **2** | **~250** |
| WordPiece (BERT) | ~30K | 2 | ~300 |
| Word-level (legacy) | ~50K-1M | 2 (but OOV explodes) | ~200 |
| Patch-based (Megabyte) | byte-level + grouping | 2-3 patches | ~700 |

Subword BPE produces ~20× fewer tokens than byte-level for English. That translates to:
- ~400× less attention compute (O(N²))
- 20× shorter KV cache
- 20× more text fits in the same context window

That's why every production LLM uses subword tokenization. The cost is a larger embedding matrix (49K × d_model extra params) — amortized to nothing per token.

### Tradeoffs

| Approach | Pros | Cons |
|---|---|---|
| **Byte-level (✅ supported)** | 256 vocab; no OOV; multilingual / emoji / typo-robust; no tokenizer to train | 20× longer sequences; 20× slower per doc |
| **BPE / SentencePiece (✅ supported via SmolLM2)** | 20× shorter sequences; mature tooling | Language-biased vocab; rare languages fragment; modality-bound |
| **Patch-based (Megabyte, 2023; BLT, 2024)** | Byte-level robustness + ~80% of BPE speed | New, less mature ecosystem |
| **Hybrid (char encoder → BPE decoder)** | Best of both | Complex, few open implementations |
| **Pixel-based** (render text as image, VLM) | Solves multilingual + visual layout together | Research-only |

### Platform recommendation — when to use which

| Tokenizer | When |
|---|---|
| **Byte-level (built-in)** | Tiny specialists (<50M), narrow tasks, multilingual, robustness-critical |
| **SmolLM2 BPE (49K)** | Default for English / code; compatibility with existing models |
| **Custom BPE (Tier 1 — tokenizer-trainer)** | Domain-specific corpus (medical, legal, code-only) |

### Frontier ideas worth tracking

- **Megabyte** (Meta 2023) — patch transformer over bytes
- **BLT (Byte Latent Transformer)** (Meta 2024) — dynamic byte grouping based on entropy; high-entropy spans get more compute
- **CANINE / ByT5** — pure character/byte transformers, accept the cost
- **Pixel-based** — text-as-image + VLM

The disciplined call: keep BPE as default. Watch BLT — if it becomes mature open weights, byte-with-patches becomes the right answer for multilingual + code + typo-robust specialists.

### Why this is a research frontier (and why TinyGPT is positioned for it)

Tokenization is one of the few hardcoded preprocessing steps left in modern LLMs — everything else (embeddings, attention patterns, position encoding, expert routing) has been "learned" over the past decade. Letting the model learn its own input units is **active 2024-2025 research**.

Known problems with fixed BPE tokenization that motivate this:
- Multilingual fragmentation (non-English languages get split into many tokens)
- Typo brittleness ("the" vs "teh" become unrelated)
- Math breaks ("1+2=3" tokenizes differently than "1 + 2 = 3")
- Code identifier inconsistency (`getUserById` = 1 token or 4?)
- Number reasoning collapses (large numbers get inconsistent splits)
- No morphological awareness inside tokens

**The deeper theme: adaptive compute.** Tokenization is one example of a broader idea — uniform per-token compute is wasteful, concentrate compute where signal is. Related techniques:

| Technique | What's adaptive | Status |
|---|---|---|
| BLT | Patch size (input granularity) | Recent (Meta 2024) |
| Mixture of Experts (MoE) | Which experts process each token | Production |
| Mixture of Recursions (MoR) | Layers per token | Research (2024) |
| Speculative decoding | Speculate easy, verify in bulk | Production (we have) |
| Test-time training | Update weights per input | Research |
| Latent reasoning (Quiet-STaR) | Internal "thinking tokens" | Research |
| Adaptive computation time (ACT) | Compute steps per token | Recently revived |

**TinyGPT's positioning is unusual:** most platforms (HF transformers, llama.cpp, etc.) are BPE-locked. We **already** support byte-level + SmolLM2 BPE as first-class options. Adding BLT-style adaptive patches when open weights mature is **incremental**, not a rewrite. That's a strategic asset worth preserving.

Three honest tiers of involvement:
1. **Conservative (now):** BPE default, byte fallback. Ship the product.
2. **Adopt (12-24 months):** integrate BLT when first strong open-weights release lands. Mac-native multilingual + typo-robust + code/text unified specialists.
3. **Research (12+ months, optional):** implement adaptive tokenization ourselves. Real research contribution if it works on tiny models — small models benefit *disproportionately* from adaptive compute because they have less compute to waste on easy tokens.

## What "upgrading a model" covers

"Build a model" is half the story. "Upgrade an existing model" is the other half. Most have primitives shipped; gap is packaging as canonical upgrade operations:

| Upgrade type | Status | Gap |
|---|---|---|
| Continued pretrain | `--resume` ✓ | — |
| Domain adaptation | gap | `--domain-adapt` mode (Tier 1) |
| Knowledge injection (edit specific facts) | MEMIT ✓ | — |
| Capability injection (add skill via LoRA) | LoRA ✓ | merge-LoRA-into-base for shipping |
| Preference correction | gap | DPO (Tier 1) |
| Quantize for shipping | GGUF Q5/Q6/Q8 ✓ | Q4_K_M (Tier 1) |
| Distill into smaller | recipe doc | feature (Tier 1) |
| Merge with another model | gap | TIES merging (Tier 1) |
| Revert / repair (rollback to checkpoint) | save-history ✓ | `tinygpt revert --step N` flow |

## Local-teacher architecture (confirmed 2026-06-06)

The product runs entirely offline. No API dependencies.

| Teacher | Q4 size | Mac tokens/sec | 10K-label run | Use when |
|---|---|---|---|---|
| Phi-3-mini-4k | 1.5 GB | ~30-50 | ~30 min | default; strong general teacher |
| Qwen3-7B | 3.5 GB | ~15-25 | ~1 hr | stronger general; instruct-tuned |
| Llama-3.1-8B-Instruct | 4 GB | ~10-15 | ~2 hrs | strong general alternative |
| Qwen3-14B | 7 GB | ~7-10 | ~4-5 hrs | high-end teacher |
| Mixtral 8x7B (MoE) | 15 GB | ~3-5 | ~12 hrs | premium teacher; overnight runs |
| DeepSeek-V2-Lite (16B) | 8 GB | ~10-15 | ~3 hrs | strong + reasonable speed |
| Qwen2.5-32B / Llama-3-30B class | 15-20 GB | ~3-5 | ~12-24 hrs | overnight; near-frontier quality |

**The architectural call:** teacher labeling runs as an overnight or background process. Zero API spend. Users download the teacher once; reuse across all specialists. This is the differentiator.

## Today's PRD ship list (2026-06-06)

Net 14 PRDs in `docs/prds/`. Status snapshot as of end-of-session:

**Shipped by elf today (all marked `status: shipped-2026-06-06`):**
- E1 BFCL, E2 τ-bench, E5 HumanEval+sandbox, E7 judge, E8 train-time hook
- eval-leaderboard.astro, sae-timeline.astro
- Rust hf-downloader, Rust parquet-decoder
- dataset-decode-verify
- **adam-state-persistence** (B12 v2)
- **cleanup-extractordata-datatrove**
- **cleanup-score-consolidation** (`RunBench.swift` + `GenerationUtils.swift` exist)
- **app-train-controls-thermal** (TrainController.swift +174 lines, TrainView.swift +103 lines; some pieces like CLI `--throttle` flag may not be in this batch)

**Shipped by elf today (status: shipped-2026-06-06):**
- **app-eval-tab** — Eval tab in Mac app GUI (drafted + shipped same session)

**Not started — top of the queue:**
- **train-controls-gap-closure** — closes 3 audit-identified gaps (`--throttle` CLI flag + Adam verification + GitHubCorpus delete-after-parity), drafted 2026-06-06 EOD

**Verified via `swift build` + grep:** 839 insertions / 117 deletions across 24 files since the previous elf sync; `--no-save-opt-state` flag visible in `train --help` (confirms Adam state persistence shipped).

## Sequencing

1. **Now**: N02 base finishes → A1 tool-call distillation proves the
   recipe works end-to-end. Without this, building UI on an unproven
   recipe is premature.
2. **Then**: `tinygpt specialist` CLI subcommand wraps the proven flow
   (data prep → teacher label → SFT/LoRA → eval). Headless API.
3. **Then**: App wizard built on top of the CLI. App = GUI for the
   proven flow.
4. **Then**: Templates for the 5-10 most common patterns ship as
   one-click flows.
5. **Then**: QLoRA path opens 13-30B fine-tuning. App can now offer
   "fine-tune a 13B model for your task on your Mac."

## Honest probability of the full vision landing

Assuming sustained execution at the current pace:

- **A1 tool-call specialist lands within 2 weeks of N02 finishing:** ~70%
- **`tinygpt specialist` CLI wrapping the proven flow:** ~85% (mostly packaging)
- **App wizard built on the CLI:** ~75% (Mac UI work, real effort)
- **5 production-quality templates:** ~60% (need empirical tuning per template)
- **QLoRA path opens 13-30B fine-tuning:** ~70% (requires real ML engineering)
- **Full vision shipping as a coherent v1 product:** ~50%

The bottleneck is *not* technical capability. It's:
1. Sustained execution against scope creep
2. UX/templates (the unsexy work)
3. Quality predictability (telling users honestly when their task won't work)

## What this changes about today's plans

- Stop framing TinyGPT as "training tiny models from scratch on your Mac."
  That's *one* capability, not the product.
- Start framing it as "**fine-tune or distill any model up to ~30B on
  your Mac for your specific task.**" That's the actual platform.
- The 22M-from-scratch project becomes a *proof of recipe* — it shows the
  pipeline (data → train → eval → ship) works end-to-end at tiny scale.
  Same recipe scales to 76M, 1B, 3B, 7B+LoRA.
- The eval pipeline shipped today is the *measurement infrastructure* for
  any specialist — not just our 22M one. Reusable for every fine-tune.
- The distillation recipe (`docs/recipes/distillation-fc.md`) is the
  *training-side template* for the same product.

## Design principle: laptop-thermal-aware defaults

**Not a new direction — overdue translation.** TinyGPT has been explicitly Mac-first since day one. The audience target has always been the individual on a personal Mac. But the existing build's *defaults* (training loop, throttle behavior, fan-load assumptions) still implicitly inherit datacenter-era assumptions from upstream tools — nanoGPT's "for step in 0..<steps" loop, MLX's "use all the silicon you can" philosophy, ML papers' max-throughput benchmark culture. The platform was Mac-targeted at the API level but hadn't yet retrofitted the *defaults* to reflect "this is running on a laptop."

**This section codifies the overdue fix**: defaults should actively reflect the Mac-laptop audience that's been the thesis from day one.

| Surface | TinyGPT-correct default |
|---|---|
| **App Train tab — throttle slider** | Default position: **75%** or **"Auto"** (thermal-state-driven). NOT 100%. |
| **App Train tab — first-run banner** | "Training runs your Mac at full load. Slide down for cooler operation, or leave on Auto to throttle automatically when heat rises." |
| **CLI `tinygpt train`** | Default `--throttle 1.0` (CLI users opt in to what they want; no surprise behavior). |
| **App Train tab — auto-throttle toggle** | **On by default.** Maps `ProcessInfo.thermalState` → automatic throttle adjustment (`.nominal` 100% / `.fair` 75% / `.serious` 50% / `.critical` 25%). |
| **Pre-launch confirm for long runs** | Show estimated thermal impact: "This will run at ~50W sustained for ~8 hours. [✓] My Mac is on a hard surface with good airflow." |

This makes TinyGPT the **first** Mac-native ML platform with laptop-aware defaults out of the box. The discipline is: **assume the user's machine is a laptop, not a server.** Every default decision flows from there.

## Maintainer learning roadmap (added 2026-06-06)

Personalized for outsider-perspective + AI-augmented + shipping-while-learning context. Not an academic curriculum — just-in-time depth on what the platform touches.

### Highest-ROI sequence

| Resource | Time | Why |
|---|---|---|
| **Karpathy "Zero to Hero" YouTube series** (6 videos, ~10 hrs) | 1 week of evenings | Best practical ML pedagogy that exists. Builds transformer from scratch. After this, the matmul-stack mental model is locked in. |
| **3blue1brown "Neural Networks" series** (~3 hrs) | 1 evening | Visual intuition for matmul + gradient descent. Pairs with Karpathy. |
| **Read nanoGPT line by line** (~300 lines PyTorch) | 2-3 hrs | Karpathy's reference impl. Once this clicks, ~80% of TinyGPT's `Train.swift` is legible. |
| **The Annotated Transformer (Harvard NLP)** | 2-3 hrs | Original 2017 paper, line-by-line with running code. Canonical reference. |
| **Read your own codebase actively** | ongoing | `TransformerBlock.swift`, `Train.swift`, `Sample.swift`. Best teacher is the system you shipped. |
| **Papers, chronologically, just-in-time** | on-demand | Transformer (2017) → GPT-2 (2019) → Scaling Laws (2020) → Chinchilla (2022) → LLaMA (2023) → DPO (2023) → BLT (2024). Don't pre-read; read when relevant to current work. |

### What NOT to spend time on

- Linear algebra textbooks (don't need matrix calculus by hand)
- PhD-level deep learning courses (Stanford CS224N — too academic)
- ML theory papers (NTK, information bottleneck — interesting but not actionable)
- Following every new paper on Twitter (drowns signal)

### The pattern that works

- **Build, then learn what you needed to build it** — by the time you need to know X deeply, you've already used it pragmatically. Theory comes second.
- **Use advisor as a tutor** — cheap to ask, fast turnaround
- **Just-in-time before commits** — 2-3 hrs reading the relevant paper + strongest blog + one OSS implementation before adding a new capability
- **Read companion docs:** `docs/learn/llm-mechanics-fundamentals.md` is the matmul-first explainer this session produced

## Open questions for future sessions

1. **Should TinyGPT ship its own pretrained 3B base?** Or rely on
   downloaded Qwen3-3B / Phi-3.5-mini as starting points? Argument for:
   architectural consistency, one binary handles everything. Argument
   against: weeks of training, doesn't differentiate from existing
   strong bases.
2. **Where does QLoRA fit?** Implementing 4-bit base + LoRA adapters
   isn't trivial in MLX-Swift. Might require waiting for MLX upstream
   or doing the integration ourselves.
3. **Cloud teacher option.** Allow user to point at Claude / GPT-4 API
   for labeling instead of local teacher? Lower friction, costs $.
4. **App template marketplace?** Other people's specialist recipes
   shareable as JSON configs. Long-term play.
5. **"Teacher RBS" clarification.** User mentioned this in passing during
   2026-06-06 strategy session; acronym unrecognized. Best guesses:
   teacher-reranker-student variant, or a paper-specific term. Treated as
   variant of teacher-student in current architecture list; revisit if
   user confirms a specific pattern.

6. **Should the embedder specialist ship BEFORE the LLM specialist (A1)?**
   (Parked 2026-06-06 EOD — decide after N02 lands.) The embedder track
   has fundamentally tighter eval loops: forward pass is ~10× cheaper
   than autoregressive generation; eval signal is exact (was-the-doc-in-top-K)
   not fuzzy; standard benchmarks (MTEB, BEIR, MS-MARCO) ship labeled
   pairs; eval takes seconds-to-minutes vs minutes-to-hours for LLM evals;
   metrics are cross-comparable (MRR/NDCG/Recall@K are universal).

   **Concretely**: embedder PoC end-to-end in ~1 week vs LLM specialist
   ~2-4 weeks. If "results first" is the operating principle, embedder
   might genuinely beat A1 to the first credible "small beats big on Mac"
   demo.

   **Why parked**: real comparison needs the N02 base done so we can
   score both paths empirically rather than estimate. Decision deferred
   to post-N02 discussion. PRD `docs/prds/specialist-embedder.md` is
   currently `blocked-on-heavy-training-approval` reflecting this gate.

## Related docs (for full context)

- `docs/recipes/distillation-fc.md` — concrete distillation recipe
- `docs/recipes/b25-scaledown.md` — sibling specialist track
- `docs/sessions/2026-06-05-eval-first.md` — yesterday's eval pipeline session
- `docs/PLAN.md` — canonical roadmap
- `HANDOFF.md` — pickup state for next session

## 2026-06-07 morning addendum — post-mortem + carry-forward

### Lesson from the /tmp incident

Lost ~14 hours of N02 output (best.tinygpt val_loss 4.434, canonical at step 89,901, all step-N history + JSONL) when the Mac rebooted overnight and macOS wiped `/tmp`. Fix: `docs/prds/persistent-training-output.md` (filed 2026-06-07).

**Generalized lesson**: audit all "ephemeral by default" choices in the platform. `/tmp` for outputs, in-memory state without a sync hook, anything that depends on a process staying alive. **Make persistence the default; volatility is the explicit opt-in.** This is a platform-credibility issue — a tool that silently destroys 14 hours of work on reboot is not a tool people trust with serious training.

### What we actually validated yesterday (despite the loss)

Even though we lost the weights, the *recipe* and *learnings* are intact:

1. **Loss trajectory is real signal**: 11.34 → 4.4-4.7 plateau over 90K steps. Honest interpretation: the Huge preset (22M params) + FineWeb-Edu may be under-capacity. Decay phase (180K-200K) might drop loss further, but not dramatically. **Probably need Mega (76M) or distillation-from-bigger-teacher to get a base usable as A1 starting point.**
2. **PowerMonitor auto-pause works** — caught thermal critical, saved checkpoint cleanly. Safety net validated.
3. **App's external-run detection works** — surfaces live + exited training runs, renders loss curve from JSONL, ETA computed.
4. **75% throttle works but isn't enough margin** — sustained 75% still hits thermal critical. **For laptop training, default should be 50% or even 40%** to never hit `.critical`.
5. **JSONL log truncates on `--resume`** (write mode instead of append) — workaround was reconstructing history from text logs. **Real bug; add to gap-closure PRD.**

### Updated default recommendations

| Setting | Old | New (post-2026-06-07) |
|---|---|---|
| `--out` default | `/tmp/<name>.tinygpt` | `~/.cache/tinygpt/runs/<name>/<name>.tinygpt` |
| App throttle slider default | 75% | 50% |
| Pre-launch confirmation threshold | >30 min | >2 hrs (most users will skip 30-min runs through it without thinking) |
| `--log-jsonl` on `--resume` | write mode (truncates) | append mode (preserves history) |

### Re-prioritization

**Today (2026-06-07) priorities**:
1. **Fix `/tmp` default** (filed PRD) — credibility-critical
2. **App polish pass** — locale formatter ("2,00,000" → "200,000"), throttle row layout, click targets, seed placeholder
3. **N02 re-run tonight** — persistent path, 50% throttle from start
4. **SFT smoke** — even against SmolLM2 baseline if our base doesn't recover in time, just to validate recipe

**Defer**: embedder vs A1 sequencing, vllm-mlx investigation, cookbook content. All blocked on having a real base anyway.

## Related docs (for full context)

- `docs/recipes/distillation-fc.md` — concrete distillation recipe
- `docs/recipes/b25-scaledown.md` — sibling specialist track
- `docs/sessions/2026-06-05-eval-first.md` — eval pipeline session
- `docs/prds/persistent-training-output.md` — 2026-06-07 post-mortem PRD
- `docs/PLAN.md` — canonical roadmap
- `HANDOFF.md` — pickup state for next session

## TL;DR for myself in 3 months

You decided TinyGPT is **a Mac app for individual task-specialization**.
The product is "bring data → pick teacher → ship a fast specialist."
Distillation + LoRA + QLoRA + Q4 inference lift the ceiling from "3B
trained from scratch" to "30B fine-tuned locally + 70B-class quality via
distillation." The infrastructure is mostly built; the work is packaging,
templates, and honest quality predictability. The A1 specialist landing
is the proof-of-recipe that unblocks the platform pitch.

**One day in 2026-06 we lost 14 hrs of training output to a `/tmp` default.** It taught us the persistence-by-default principle. Don't repeat that lesson.
