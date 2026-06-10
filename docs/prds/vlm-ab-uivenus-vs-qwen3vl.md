# VLM A/B — UI-Venus-1.5-2B vs Qwen3-VL-2B for Pace's vision pillar

Status: PRD (2026-06-09). The 2026-06-09 research sweep flagged Qwen3-VL-2B as a competitor to our UI-Venus-1.5-2B port target. This A/B decides which one we actually port to MLX-Swift / CoreML / ANE for #266.

## Why this matters

#266 (UI-Venus M4 port) is 6-9 days of focused MLX-Swift work. We should not port the wrong model. If Qwen3-VL-2B wins the A/B on Pace's actual use cases:
- One Qwen3 family across planner + embedding + VLM (architectural simplification, one tokenizer, one quant pipeline)
- Already validated for "native computer/mobile control" per research (UI-TARS-2 paper benchmark)

If UI-Venus wins:
- Microsoft tuned it specifically on UI screenshots (Windows/web/mobile)
- Smaller specialization variance
- Stay with our current port plan

## Models on disk (downloaded 2026-06-09)

- `~/.lmstudio/models/mlx-community/UI-Venus-1.5-2B-6bit` — already there for the port
- `~/.cache/tinygpt/models/qwen3-vl-2b-instruct/` — 2.8 GB, just downloaded

Both available for A/B without further setup.

## Test plan (1-2 days, no porting required)

Run both via mlx_lm in Python (no Swift port needed for A/B). Compare on a Mac-specific fixture set.

### Fixture set construction

**`pace/evals/fm-vlm-fixtures-mac-v1/`** — 30 fixtures, captured via `tinygpt ax-capture`:

Categories:
1. **Identity** (8): "what app is this" on Mac apps Microsoft probably didn't train on — Xcode, Final Cut, Logic, OmniFocus, Bear, Things, Tot, Drafts
2. **Read-on-screen** (8): "read the error message" / "what's the title of this email" / "what's the current playback time"
3. **Click target** (8): "where do I click to send this" / "find the New Folder button" — model emits (x,y) or label, scored against ground-truth AX bounds
4. **Activity context** (6): "what am I doing right now" — multi-modal answer (app + visible content + task)

Capture process: 30 manual `ax-capture` sessions across daily Mac use (existing tool, ~30 min).

### Eval methodology

For each model:
1. Load via mlx_lm chat
2. For each fixture: pass the screenshot + AX-tree-as-text + the question
3. Score via:
   - **Identity**: string match on app name
   - **Read**: substring containment of expected text in response
   - **Click**: response label matches AX-tree label OR (x,y) lies inside ground-truth bounds
   - **Activity**: LLM-as-judge (Qwen3-14B teacher) scores 0-2 on coverage

Eval script: `scripts/eval_pace_vlm_ab.py` (write — extends eval_pace_v2.py pattern).

### Metrics captured

| Metric | UI-Venus-1.5-2B | Qwen3-VL-2B |
|---|---|---|
| Identity accuracy (/8) | ? | ? |
| Read accuracy (/8) | ? | ? |
| Click accuracy (/8) | ? | ? |
| Activity accuracy (/6) | ? | ? |
| **Total (/30)** | ? | ? |
| Per-call latency (MPS) | ? ms | ? ms |
| Per-call latency (ANE via CoreML if convertible) | ? ms | ? ms |
| Disk (post-quantization to 4-bit MLX) | ? MB | ? MB |
| Formula score | ? | ? |

### Tiebreakers (when scores are close)

1. **Same family wins** (Qwen3) — one tokenizer + one MLX pipeline for planner + embedding + VLM. Worth ~5pp.
2. **Smaller bundle wins** — Pace's total bundle budget matters for v1 ship.
3. **Better ANE conversion path wins** — whichever model converts to ANE-chunked CoreML with less drama.

## Decision tree

| Result | Action |
|---|---|
| Qwen3-VL ≥ UI-Venus by ≥ 5pp on Mac fixtures | **Switch port target to Qwen3-VL-2B.** Same family alignment compounds the win. |
| UI-Venus ≥ Qwen3-VL by ≥ 5pp | **Stay on UI-Venus.** Continue M4 port. |
| Within 5pp tie | **Pick Qwen3-VL** — family alignment + ecosystem momentum (Qwen3-VL native computer/mobile control per research). |

## What "Mac-tune later" looks like (out of scope here)

Either winner can be fine-tuned later on Pace's ax-capture corpus once Pace is deployed and we have real screen+AX pairs. The A/B picks the BASE; tuning is a separate factory pass.

## Done when

- 30 Mac fixtures captured + ground-truth labeled
- Both models scored on the same fixtures
- Formula scores computed
- Decision recorded in memory
- Port target locked for #266

## Related

- #266 UI-Venus M4 port (consumer of this decision)
- `project-embedding-swap-2026-06-09` (parallel: same-family-wins logic applied to embedding)
- `feedback-tinygpt-north-star` (formula bar applied here)
- `feedback-research-first-doctrine` (this A/B IS the research-first response to the Qwen3-VL surfacing)
