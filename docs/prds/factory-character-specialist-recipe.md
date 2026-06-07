---
name: Character specialist recipe — fully-free end-to-end pipeline for NPC / persona / character models
status: shipped-docs-v1-2026-06-07-cli-gaps-deferred
owner: unassigned
created: 2026-06-07
priority: P2 — directly enables specialist demos for HN-launch story; Aliveville is the concrete pull
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md
---

# PRD — Fully-free character specialist pipeline

## 2026-06-07 ship note

Shipped docs v1:
- `docs/recipes/cookbook-character-specialist.md`
- linked from `docs/recipes/README.md`
- uses existing commands only: `download-model`, `download-dataset`,
  `synthesize` through a local OpenAI-compatible teacher, `filter`, `dedupe`,
  `sft`, and `serve`

Deferred CLI gaps remain as follow-up work:
- `synthesize --teacher-local`
- `sft-format-merge`
- `eval rubric`

## Goal

Ship a **zero-API-spend, end-to-end recipe** for training small (~500M)
character/persona specialist models in TinyGPT. The deliverable is a
cookbook page (`docs/recipes/cookbook-character-specialist.md`) plus the
few CLI surface adjustments needed to make the pipeline one command per
phase. No new training mechanisms — uses pieces that already shipped.

The concrete pull is Aliveville's NPC AI (separate game project at
`~/Desktop/fleet/ai-game/`), but the recipe generalizes to any
character-or-persona-specialist use case (customer-support bot, narrative
AI tool, brand-voice writer, etc.).

## Why ship

1. **Validates the product framing.** TinyGPT's positioning ("Mac
   platform for building specialists for your specific tasks") needs a
   concrete, repeatable, demonstrably-works recipe. Character specialists
   are the most user-graspable instance.
2. **All pieces exist; integration doesn't.** TinyGPT already has
   `distill`, `sft`, `lora`, `synthesize`, `train-quality-classifier`,
   `download-model`, plus full PEFT bundle. The PRD pulls them into a
   single documented flow.
3. **Free constraint matters for adoption.** Most existing tutorials
   assume Claude/GPT-4 as teacher (API spend). This recipe uses
   **local teachers** (Llama-3.1-70B-Instruct int4 or
   Qwen2.5-72B-Instruct int4) so the entire pipeline costs $0 in
   recurring fees. Required only: time + electricity.
4. **HN-launch artifact.** A "free character/NPC specialist in 4 hours
   on a Mac" demo is more shareable than "look at our loss curves."

## Scope — in

### The recipe (cookbook page)

End-to-end, four phases. All commands map to existing TinyGPT
subcommands (CLI gaps listed below if any).

#### Phase 1 — Pull the base model

```bash
tinygpt download-model Qwen/Qwen2.5-0.5B
# or: SmolLM2-360M, Llama-3.2-1B, Qwen2.5-1.5B (size/quality trade-off)
```

Default pick: **Qwen 2.5-0.5B** (Apache 2.0, strongest small base on HF
leaderboard).

#### Phase 2 — Pull free datasets

```bash
tinygpt download-dataset chimbiwide/RolePlay-NPC          # 18.5K rows, Apache 2.0
tinygpt download-dataset agentlans/multi-character-dialogue  # 10K+ multi-NPC scenes
tinygpt download-dataset NousResearch/CharacterCodex      # broad creative scenarios
```

Optional (commercial-caution): `PygmalionAI/PIPPA` (1M lines, scraped
from Character.AI without explicit permission — fine for research/proto,
risky for commercial release).

#### Phase 3 — Synthesize domain-specific data with a LOCAL teacher

```bash
# Pull a local teacher (one-time, ~40 GB on disk at int4)
tinygpt download-model Qwen/Qwen2.5-72B-Instruct --quantize int4
# alternative: meta-llama/Llama-3.1-70B-Instruct --quantize int4

# Synthesize character-specific data using the local teacher
tinygpt synthesize \
    --teacher-local Qwen2.5-72B-Instruct \
    --prompt-file character-prompts.jsonl \
    --num-samples 20000 \
    --temperature 0.8 \
    --out character-data.jsonl
```

Each `character-prompts.jsonl` entry is a Aliveville-style scenario
prompt (location, recent events, character mood, prior dialog). The
teacher generates plausible in-character responses. Output is
SFT-formatted.

#### Phase 4 — LoRA SFT on combined data

```bash
# Merge public + synthetic, dedupe, format
tinygpt sft-format-merge \
    public:chimbiwide/RolePlay-NPC \
    public:agentlans/multi-character-dialogue \
    local:character-data.jsonl \
    --out merged-sft.jsonl

# LoRA SFT
tinygpt sft \
    --base Qwen2.5-0.5B \
    --data merged-sft.jsonl \
    --lora-rank 16 \
    --lora-alpha 32 \
    --neftune-alpha 5 \
    --epochs 3 \
    --max-lr 1e-4 \
    --out character-specialist.lora
```

#### Phase 5 — Evaluate

```bash
tinygpt sample Qwen2.5-0.5B \
    --lora character-specialist.lora \
    --prompt-file eval-scenarios.jsonl \
    --temperature 0.8 \
    --tokens 200 \
    --out eval-results.jsonl

# Compare against the base (LoRA off)
tinygpt sample Qwen2.5-0.5B \
    --prompt-file eval-scenarios.jsonl \
    --temperature 0.8 \
    --tokens 200 \
    --out eval-baseline.jsonl
```

Side-by-side rubric (in-character consistency, plausibility, novelty)
documented in the cookbook page.

### CLI gaps to fill

- `tinygpt synthesize --teacher-local <model>` — currently `synthesize`
  expects an OpenAI-compatible URL. Add a local-teacher path that loads
  a model directly and runs synthesis in-process (no HTTP overhead).
- `tinygpt sft-format-merge` — combine multiple dataset sources into a
  single SFT-format JSONL with dedup and shuffling. Currently has to
  be scripted; deserves a first-class command.
- `tinygpt eval rubric` — semi-structured eval comparing two model
  variants on a prompt list, emitting comparable JSON (no judge model
  needed for the first pass).

### Cookbook page contents (`docs/recipes/cookbook-character-specialist.md`)

- Why this recipe exists (the small-specialist thesis)
- Hardware requirements (M5 Pro 48 GB; 70B teacher needs int4)
- Step-by-step copy-paste commands
- Time budget per phase (~1 hr setup + 1 hr synthesis + 2 hr SFT)
- Eval rubric and example outputs (base vs LoRA)
- Iteration tips (where to focus when first run is mediocre)
- License and ethics notes (PIPPA caveat, synthetic-data provenance)
- Cross-link to Aliveville integration notes (Phase 6 — outside this PRD)

## Scope — out

- **Aliveville-specific integration.** The cookbook gives generic NPC
  recipes; the Aliveville bridge (game-state prompts, in-game inference
  serving, multi-character dispatch) is its own PRD.
- **API-teacher path.** Already supported by existing `synthesize`. Not
  in this PRD's scope to document.
- **New training mechanisms.** This is recipe + integration; no new
  PEFT variants, no new optimizers, no new architecture work.
- **Player-footage imitation learning.** Session 7's behavior-learning
  story applies, but collecting and training on player footage is a
  Phase 7+ artifact (once Aliveville is shipping and producing data).

## Acceptance criteria

1. Cookbook page exists at `docs/recipes/cookbook-character-specialist.md`
   with all phases copy-paste runnable.
2. `tinygpt synthesize --teacher-local` works end-to-end on M5 Pro 48 GB
   with a 70B int4 teacher and a 100-sample prompt file.
3. `tinygpt sft-format-merge` produces a deduplicated combined JSONL
   from public + local sources.
4. Smoke test: pull Qwen 2.5-0.5B + 1000-sample synthesis from local
   Qwen2.5-72B + 100-step SFT completes in <4 hours wall on M5 Pro.
5. Eval rubric produces JSON comparing base vs LoRA for a 20-prompt
   scenario list.
6. PLAN.md updated to mark this recipe shipped under Cookbooks /
   Recipes section.

## Out-of-scope but worth flagging

- **`huge-v2` preset.** Current `huge` preset (12L, d=256, vocab=256
  byte-level by default) does NOT enable SwiGLU / RoPE / RMSNorm / YOCO
  by default, all of which TinyGPT supports. A modernized `huge-v2`
  preset with these on would be a free quality bump for any future
  small base trained from scratch. Separate PRD.
- **Free-teacher leaderboard.** Once 1-2 character specialists ship,
  publishing a "free 500M specialist beats X" leaderboard becomes the
  natural HN-launch story. Separate PRD.

## Risks

- **70B local teacher may be too slow.** Generation on M5 Pro at int4
  is ~3-8 tok/s. 20K examples × ~150 tokens each = 3M tokens of
  generation = ~10-30 hrs wall. Need to either accept the wall time,
  ship `--batch-synthesize` to parallelize, or document Llama 3.1-8B
  as a "fast but lower-quality" teacher.
- **Synthetic data quality without an API teacher.** Local 70B is good
  but not Claude/GPT-4 quality. Recipe should document expected gap
  and what failure modes look like.
- **Cookbook drift.** TinyGPT CLI moves quickly. Cookbook pages need a
  smoke-test CI step or they decay.
