# GEPA prompt evolution — automated system-prompt iteration

**Date**: 2026-06-09
**Status**: PARKED — activate after v11 ships
**Trigger**: when a v12 planner is needed AND v11 ship-gate eval suite is stable

---

## What

Apply **GEPA** (Genetic-Pareto Prompt Evolution, [ICLR 2026 Oral](https://github.com/gepa-ai/gepa)) to automate the system-prompt + action-description iteration loop. Stop hand-tuning prompts; let evolutionary search find the Pareto-front.

Reference application: [NousResearch/hermes-agent-self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution) — a Hermes-specific wrapper. The wrapper is illustrative; we'd use GEPA directly, not the wrapper.

## Why now (motivation)

10 versions of Pace planner system prompts (v3 → v10) were all hand-tuned. Each iteration cost a person-day of analysis + writing. GEPA's claim: **$2-10 per optimization run** with trace-driven mutations against a fixed eval suite. That replaces 4-6 of our cycles with one API-driven loop.

Prerequisite — already done as of 2026-06-09:
- ✅ Locked eval suite (v11 ship gate)
- ✅ 6 dimensions, fixed thresholds, immutable
- ✅ Multi-objective fitness available (accuracy_v2, BFCL, abstention, disambiguation, schema, safety, plus prompt-length as cost axis)

Without those, GEPA = noise. With them, GEPA = the right tool.

## How

```
read current pace-system-prompt-v10-actions.txt
       │
       ▼
GEPA: propose N=8 mutations targeting observed failure patterns
       │  (uses execution traces from v9/v10 runs against the eval suite)
       ▼
serve each candidate prompt against fm-fixtures-v2, BFCL-12, OOS, AMBIG, DESTRUCT
       │
       ▼
compute Pareto front: (accuracy, prompt_length, TTFW)
       │
       ▼
human PR review on top 2 candidates
       │
       ▼
ship if all 6 ship-gate thresholds clear AND prompt ≤ existing length × 1.2
```

GEPA reads *why* each candidate fails (not just *that* it fails) — failures inform the next mutation cycle. This is reflective prompt evolution, not blind random search.

## What it optimizes for Pace

| Target | Why GEPA helps |
|---|---|
| `pace-system-prompt-v10-actions.txt` | Direct hand-tuning replacement |
| `grammars/v10-actions/registry.json` action descriptions | These shape tool selection — GEPA can tighten descriptions for ambiguity reduction |
| Future v11/v12 system prompts | First-time prompts get a Pareto-front exploration instead of "engineer's first guess" |

## License notes

| Component | License | Use |
|---|---|---|
| GEPA (`gepa-ai/gepa`) | MIT | ✅ Direct dependency |
| DSPy (`stanfordnlp/dspy`) | MIT | ✅ Optional, GEPA-adjacent |
| Hermes-Agent-Self-Evolution wrapper | MIT | ⚠️ Reference only, don't import |
| Darwinian Evolver (`imbue-ai/darwinian_evolver`) | **AGPL v3** | ❌ DO NOT integrate; license-incompatible with commercial Pace |

## Cost / runtime

GEPA's API-driven loop calls an LLM (we'd use local Qwen3-14B + thinking, no $$$). Self-hosted: ~3-5 hours wall per optimization run on Qwen3-14B local teacher with no API spend. The published "$2-10/run" assumes paid API.

## ROI

| | Estimate |
|---|---|
| Integration effort | ~12-16h |
| P(meaningful accuracy gain) | ~60% |
| Quality of gain | 5-10pp on prompt-sensitive dimensions, with NO new training |
| ROI | ~0.2-0.3 (below most queued tasks) |
| **Multiplier** | Compounds with v11 training. After v11 trains, run GEPA on v11's prompt for a cheap second derivative gain. |

## Activation gate

Two conditions must hold:
1. v11 has shipped OR v11 has failed and we're planning v12
2. The v11 ship-gate eval suite remains the canonical fitness function (do not invent a new one for GEPA)

If both → activate. If either fails → wait.

## What NOT to do with GEPA

- Don't use it to chase a single metric (the Pareto formulation prevents this naturally; don't break it)
- Don't apply it to training data — that's not what GEPA is for
- Don't pull the Hermes wrapper as a dependency — read it for ideas, then write our own ~150-line integration
- Don't run it during a training run — GPU contention

## Related

- [pace-planner-v11-ship-gate.md](pace-planner-v11-ship-gate.md) — the eval suite GEPA optimizes against
- [pace-planner-v11-training-data.md](pace-planner-v11-training-data.md) — data work is orthogonal; GEPA evolves prompts not weights
- Memory: [[feedback-research-first-doctrine]] — we verified this is ICLR 2026 Oral, not pre-print noise
