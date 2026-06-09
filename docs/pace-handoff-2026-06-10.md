# Pace handoff — tinygpt deliverables status (2026-06-10)

tinygpt (the factory) refocuses on local-LLM research. This doc freezes the
state of everything tinygpt owes or has delivered to Pace (the consumer),
so either repo can pick up any thread without re-deriving context.

## Delivered — done, shipped, in use

| Deliverable | Where | State |
|---|---|---|
| v9 planner LoRA (shipping specialist) | `~/.cache/tinygpt/runs/pace-planner-v9/` | 33.3% v2 / 70% compose; **not** Pace's runtime default (see below) |
| ANE M8 28-block chain | tinygpt ANE scripts | 17 tok/s, ' Paris' top-1 correct |
| v10 action registry + schema | `grammars/v10-actions/` → copied into `pace/leanring-buddy/Resources/` | Pace bundles its own copy; registry validates at app startup |
| Serve: grammar constraint, EOS stops, prompt cache, TTFW 119ms warm, partial-JSON streaming | TinyGPTServe | Pace consumes via localhost HTTP |
| WhisperKit qualification | memory + benchmarks | large-v3-turbo: 1.5GB, 9× realtime — Pace's provider scaffold awaits streaming wiring |
| Qwen3-Embedding-0.6B qualification | memory | qualified for RAG; Pace's v1 retrieval shipped lexical instead — embedding upgrade is a future option |
| Stage A dictation post-processor | `pace/leanring-buddy/PaceDictationPostProcessor.swift` + 26 tests | committed 2026-06-10 |
| v11 eval suites (60 unhappy fixtures + 96 BFCL-12) | `pace/evals/fm-fixtures-{oos,ambig,destructive}` + `~/.cache/tinygpt/datasets/bfcl/BFCL_v3_pace12.json` | held-out; baselines locked in `docs/v11-baselines-2026-06-09.md` |

## Ready to fire — one command, needs GPU free

**v11 planner train-and-gate**: `bash scripts/v11_pipeline.sh [--amplify]`

- Corpus: 492 rows merged now; `--amplify` adds ~150-180 judge-filtered rows (~90 min, needs LM Studio + Qwen3-14B)
- Trains DoRA r32/a64, bakes, serves with v11 grammar (7 intents), runs all 6 ship-gate dimensions + non-regression + formula score
- Verdict against `docs/prds/pace-planner-v11-ship-gate.md` — ship iff ALL six clear
- **Important reframe**: Pace's runtime planner default is currently `qwen/qwen3-30b-a3b` via LM Studio (set in Pace's Info.plist), NOT the tinygpt specialist. v11's real bar is beating the 30B on (speed × accuracy) / cost, not just beating v9. Add a 30B comparison row to the eval before deciding ship.

## Owed — open tinygpt work Pace is waiting on

| # | Work | Why Pace needs it | Size |
|---|---|---|---|
| #266 | VLM in-process runtime (Qwen3-VL or UI-Venus M4 port) | Pace's `ScreenAnalysisProvider: inProcess/coreML/mlx` options are stubs falling back to LM Studio HTTP | days — the next big factory project |
| #308 | VLM A/B (UI-Venus vs Qwen3-VL-2B) | picks the #266 port target | 1-2 days; needs screenshots + GPU |
| #292-adj | WhisperKit streaming runtime wiring | Pace's `TranscriptionProvider: whisperKit` scaffold falls back to Apple Speech | mostly Pace-side; tinygpt advises |
| #306 | macOS 26 int8 ANE handoff into M8 | ~1.8× ANE speedup primitive | perf research track |
| #303/#305 | Quantization pass + Swift QuantizedLinear | smaller shipping models | perf research track |

## Decisions frozen (do not re-litigate without new data)

- Planner iteration FREEZES after one v11 run — ship or fail, no v11.1/v12 this month ([ship gate](prds/pace-planner-v11-ship-gate.md))
- Nemotron is a full ASR replacement, not a dictation post-processor — Stage A is regex; Stage B (trained) only if telemetry shows formatting errors matter
- APIGen-MT-5k is CC-BY-NC — commercial training data must come from Hermes-FC-v1 (Apache) or own synthesis
- GEPA prompt evolution parked until after v11 ([PRD](prds/gepa-prompt-evolution.md))

## Coordination notes

- Another agent works in pace repo; tinygpt-side work avoids contention
- pace branch: `eval/fm-fixtures-v2` (6 commits ahead as of this doc, pushed)
- GPU/LM Studio is a shared resource — check `lms ps` + `ps aux | grep eval-planners` before loading models
