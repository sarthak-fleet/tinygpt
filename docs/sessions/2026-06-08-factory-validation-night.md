# Session — 2026-06-08, the factory-validation night

**Date**: 2026-06-07 evening → 2026-06-08 early morning
**Premise**: N02 base finished. Test the factory by trying to ship a
real FC specialist. Find every gap.

## Tonight's actual wins

| # | Shipped | Significance |
|---|---|---|
| 1 | **GQA + head_dim fix** in attention path (PRD shipped) | Qwen3 family (1024 hidden, 16 heads, 128 head_dim) loads + trains. The reshape bug was off-by-2× because we assumed `head_dim = hidden_size/num_heads`. |
| 2 | **QK-Norm support** (Qwen3 RMSNorm on Q/K) shipped | Qwen3 base attention was mathematically wrong without this. Step-1 SFT loss dropped 12.8 → 0.68 (19× improvement) after wiring. |
| 3 | **DoRA serialization** fix (both writers) shipped | Default `tinygpt sft` uses DoRA. Both writers (`LoraIO` + `LoraHF`) only checked `as? LoraLinear` — every DoRA-trained adapter was silently empty (209 bytes). Now DoRA round-trips as LoRA (m vector deferred to v2). |
| 4 | **Theme-completer trained from scratch** in 38 seconds | 842K-param Tiny preset, 5050 hex palette pairs, loss 5+ → 0.114. Path-1 from-scratch validated on a non-text task. Echoes input colors (memorization on tiny corpus) but produces valid hex completions. |
| 5 | **Pace specialist arc end-to-end prep** | System prompt extracted, tool-tag GBNF grammar drafted, recipe at `docs/recipes/pace-planner.md`. Tomorrow's flagship arc is ready to fire. |

## Honest failures + what they taught

| Attempt | Outcome | Real lesson |
|---|---|---|
| N02 FC SFT (22M base, ctx=256, hermes-fc 1150-token avg) | Empty LoRA (loss=0 from truncation) | From-scratch bases need ctx ≥ 2048 for instruction-following SFT. N02 is dead for downstream SFT. Future runs must train at bigger context. |
| Qwen3-0.6B FC SFT (first attempt) | Reshape crash | Surfaced the head_dim override gap. Fixed. |
| Qwen3-0.6B FC SFT (second, post-GQA-fix) | Oscillating loss 9-16, no convergence | QK-Norm missing → mathematically wrong attention. Fixed. |
| Phi-3 FC SFT (multiple attempts, parallel) | Thermal pressure (fan 100%) | One training run at a time. **Hard rule.** |

## The four factory gaps surfaced and closed

1. ~~`head_dim` override in Qwen3 family~~ ✅ shipped (`factory-gqa-support.md`)
2. ~~QK-Norm RMSNorm on Q/K (Qwen3)~~ ✅ shipped (covered in same PRD)
3. ~~DoRA serialization in writers~~ ✅ v1 shipped (`factory-dora-serialization.md`)
4. **LoRA application for `TinyGPTModelHF`** ⬜ tomorrow (#250)

Without (4), we cannot inference-test the adapters we trained. Tomorrow's first priority.

## State of artifacts

| Path | Size | What |
|---|---|---|
| `~/.cache/tinygpt/runs/n02-20260606-1128/huge-base-v1.tinygpt` | 253 MB | N02 22M from-scratch, val 4.32. Useful for interp/B13. Not for SFT. |
| `~/.cache/tinygpt/runs/theme-v1/theme-v1.tinygpt` | ~3 MB | Tiny theme-completer. Validates path-1. |
| `~/.cache/tinygpt/runs/qwen3-fc-v1/qwen3-fc-v1.lora` | 8.8 MB | First real Qwen3 LoRA. Trained pre-QK-Norm fix → numerics suspect. Replace with v2. |
| `~/.cache/tinygpt/runs/qwen3-fc-v2/qwen3-fc-v2.lora` | in flight | Real Qwen3 FC SFT with QK-Norm + `--no-dora`. Expected size ~8-15 MB. |

## Tomorrow's plan

| Priority | Task | Effort |
|---|---|---|
| 1 | LoRA support for TinyGPTModelHF (#250) — `hf-load --lora` + `serve --lora` | 2-3 hrs |
| 2 | Verify qwen3-fc-v2 generates FC-shaped output for hermes-fc test prompts | 30 min |
| 3 | Pace specialist arc fire: pace prompts → `tinygpt synthesize` (LM Studio teacher) → `tinygpt sft` Qwen3 → eval against Pace fixtures | half day |
| 4 | DoRA format v2 (round-trip with `m` vector) | 2-4 hrs |
| 5 | Theme-completer v2: scrape ColorHunt for 10K palettes, retrain at Small preset | half day |

## Hard rules learned tonight

1. **One training run at a time.** Thermal pressure is real and bypasses `pmset` warnings — fan-by-ear is the early signal.
2. **Use `--no-dora` until DoRA format v2 ships.** Default SFT writes header-only files otherwise.
3. **From-scratch bases need ctx ≥ 2048 if downstream SFT is intended.** N02 at ctx=256 was a recipe-validation experiment, not a deployable base.
4. **GPU contention from parallel training is harder to spot than expected.** A "killed" SFT that completed silently kept running; the duplicate launch went to 2× memory pressure.

## What changes about the strategy

Nothing fundamental — but factory completeness now means more than just shipping primitives. It means shipping primitives THAT ACTUALLY WORK end-to-end on a real modern HF base. Tonight closed three architectural gaps that would have blocked every specialist arc on Qwen3/Llama-3/Mistral.

The "factory is mostly done" claim from last session was true at the primitive level but had silent gaps at the integration level. Tonight surfaced and closed them.
