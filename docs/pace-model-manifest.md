# Pace model manifest — every model role, decided and on disk

Canonical inventory of all models Pace needs (goal set 2026-06-10: all
roles decided with runnable artifacts by end of day). One row per role;
this doc is the single source of truth — update it when any decision
changes. Embedded/ANE ports are perf infrastructure, not blockers: interim
runtime is LM Studio / tinygpt serve, which Pace already uses.

| Role | Model | Artifact (on this machine) | Size | Status 2026-06-10 |
|---|---|---|---|---|
| Planner (runtime default) | qwen/qwen3-30b-a3b | LM Studio | 18.6 GB | ON DISK, wired (Pace default) |
| Planner (specialist) | v11 DoRA on Qwen3-0.6B, served int8 g64 | `~/.cache/tinygpt/runs/pace-planner-v11/baked-hf` | ~0.5 GB eff. | TRAINING — ship-gate verdict tonight; ships only if it beats the gate, else 30B stays |
| Planner (ANE perf variant) | M8 28-block chain, int8 per-block | `~/.cache/tinygpt/ane-w8b/` | 476 MB | ON DISK, numerics gate PASS, 17–22 tok/s |
| Screen VLM | A/B tonight: UI-Venus-1.5-2B-6bit vs Qwen3-VL-2B-Instruct-6bit (UI-Venus-8B as third column) | all three in LM Studio | 2.2–5.8 GB | DECIDING tonight (`scripts/eval_pace_vlm_ab.py`, 12 fixtures, FakePace baseline 5/12); M4 Swift port (#266) follows the winner |
| ASR (voice) | WhisperKit openai_whisper-large-v3-turbo | `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3-v20240930_turbo` | 1.5 GB | ON DISK, qualified 2026-06-08 (9× realtime on ANE, perfect "Pace" accuracy); app integration = #292 (Swift work, not a model gap) |
| TTS | Apple AVSpeechSynthesizer (system) | none — OS framework (`LocalTTSClient.swift`) | 0 | DONE — no model needed, zero-cloud by construction |
| Embeddings (RAG) | Qwen3-Embedding-0.6B (decided 2026-06-09, replaced mxbai) | `~/.cache/tinygpt/models/qwen3-embedding-0.6b/` | 1.2 GB | ON DISK; runtime retrieval is currently Spotlight-first (`PaceSpotlightRetrievalConnector`), embedding RAG is a stub contract (`PaceLocalRetrieval.swift`) — model ready when wiring lands |
| Dictation post-proc | Stage A rule-based (no model) | in pace repo, 26 tests | 0 | SHIPPED; Stage B model is conditional on a v11 BFCL trigger, deliberately not built |

## Disk budget

On-disk total for the shipping set (30B planner + VLM winner + ASR +
embeddings): **~22–24 GB**, dominated by the 30B. If v11 ships and replaces
the 30B for planning, the set drops to **~5–6 GB**. The specialist question
is therefore also a 4× bundle-size question.

## What "done today" means per role

- **Planner**: v11 verdict lands tonight → either v11 ships (specialist) or
  the 30B is confirmed as the frozen decision. No third option; planner
  freezes either way.
- **VLM**: A/B runs right after the pipeline frees LM Studio → winner locked
  for #266.
- **ASR / TTS / embeddings / dictation**: already decided + on disk (audited
  today). Remaining work on these is app wiring, not model selection.

## Stale items this manifest supersedes

- #299 (mxbai CoreML conversion) — obsolete since the Qwen3-Embedding swap.
- "Pace runtime planner = our specialist" assumptions anywhere — the default
  is the 30B until v11 *proves* otherwise on the gate.

## Related

- `docs/prds/pace-planner-v11-ship-gate.md` — tonight's planner verdict
- `docs/prds/vlm-ab-uivenus-vs-qwen3vl.md` — tonight's VLM decision
- `docs/prds/pace-task-loop-v1.md` — first consumer of the full set
- Memory: `whisperkit-qualified-2026-06-08`, `embedding-swap-2026-06-09`,
  `serve-quantize-2026-06-10`, `m8-int8-shipped-2026-06-10`
