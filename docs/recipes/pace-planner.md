# Recipe — Pace planner specialist

The factory's first product-validating arc. Distill Pace's
qwen3-30b-a3b planner into a small student. See
`docs/prds/specialist-pace-planner.md` for full spec.

This recipe is the executable form — the actual commands to run.

## Prerequisites (verified 2026-06-07)

- ✅ Qwen3-0.6B base on disk + SFT-compatible (GQA + head_dim fix landed)
- ✅ `tinygpt synthesize` ships (OpenAI-compat teacher endpoint)
- ✅ `tinygpt sft` ships
- ✅ Pace eval fixtures at `clickyLocal/evals/fixtures/`
- ✅ Pace system prompt extracted (see "system prompt" section below)
- ✅ Pace tool-tag GBNF grammar drafted at `grammars/pace-tool-tags.gbnf`
- ⬜ LM Studio + qwen3-30b-a3b running (needed at step 2)
- ⬜ Pace integration conformer (`TinyGPTPlannerClient.swift` in clickyLocal)

## Pace's system prompt (canonical)

```
you're pace, a voice companion in the user's menu bar. the user just spoke
to you via push-to-talk and you can see their screen. your reply is read
aloud, so write the way you'd actually talk.

rules:
- default to one or two sentences. be direct.
- all lowercase, casual, warm. no emojis.
- write for the ear. no lists, no bullets, no markdown.
- spell out small numbers, no "e.g." or "i.e.".
- if the question relates to what's on their screen, reference what you see.
  otherwise just answer the question.
- never say "simply" or "just".

pointing:
you have a cursor that can fly to and point at things on screen. point
whenever it would help. when you point, append [POINT:x,y:label] at the
very end. if pointing wouldn't help, append [POINT:none].
```

## Tool-tag taxonomy (from fixtures)

| Tag | Use | Example |
|---|---|---|
| `[POINT:x,y:label]` | Point cursor at element | `[POINT:412,40:save button]` |
| `[POINT:none]` | No pointing needed | `[POINT:none]` |
| `[CLICK:x,y:label]` | Click element (action mode) | `[CLICK:412,40:save button]` |
| `[DOUBLE_CLICK:x,y:label]` | Double-click | |
| `[TYPE:text]` | Type text | `[TYPE:hello world]` |
| `[SCROLL:dir:amount]` | Scroll | `[SCROLL:down:200]` |
| `[KEY:keys]` | Press keys | `[KEY:cmd+s]` |
| `[OPEN_APP:name]` | Launch app | `[OPEN_APP:Notes]` |
| `[VOLUME:dir]` | Adjust volume | |
| `[BRIGHTNESS:dir]` | Adjust brightness | |

## Fixture categories (the eval signal)

| Fixture | Tests | Expected output shape |
|---|---|---|
| `qa-no-screen.json` | Pure-knowledge Q&A | `[POINT:none]` + natural sentence |
| `screen-referential.json` | "save it for me" with screen state | `[POINT:x,y:label]` with valid coords |
| `multi-turn-continuation.json` | Follow-up question | Conversational, no rehash |
| `action-mode-off.json` | Action mode OFF, user asks for action | Refuse action tags, only `[POINT:]` |

## Pipeline

### 1. Generate input pool (no teacher needed)

Extract or synthesize prompts in Pace's shape.

Option A — use clickyLocal's existing tool:
```bash
cd /Users/sarthak/Desktop/fleet/clickyLocal/
python scripts/generate-intent-corpus.py --count 1000 --out ~/.cache/tinygpt/datasets/pace-prompts.jsonl
```

Option B — start with the 4 fixtures and synth variants:
```bash
# Parse fixtures, extract user messages, mutate variants
python scripts/pace-prompts-from-fixtures.py \
    --fixtures /Users/sarthak/Desktop/fleet/clickyLocal/evals/fixtures/ \
    --variants-per 100 \
    --out ~/.cache/tinygpt/datasets/pace-prompts.jsonl
# (TODO: write this; ~50 lines)
```

### 2. Label with teacher (qwen3-30b-a3b via LM Studio)

```bash
tinygpt synthesize \
    --teacher http://localhost:1234/v1 \
    --teacher-model qwen/qwen3-30b-a3b \
    --inputs ~/.cache/tinygpt/datasets/pace-prompts.jsonl \
    --input-field prompt \
    --system-file docs/recipes/pace-system-prompt.txt \
    --grammar grammars/pace-tool-tags.gbnf \
    --temperature 0.0 \
    --parallel 4 \
    --rate-limit 30 \
    --out ~/.cache/tinygpt/datasets/pace-labeled.jsonl
```

Wall time: ~30-60 min for 1K-10K samples, depending on LM Studio's tok/s.

### 3. Distill into Qwen3-0.6B

```bash
QWEN_DIR=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<HASH>
mkdir -p ~/.cache/tinygpt/runs/pace-planner-v1

tinygpt sft "$QWEN_DIR" \
    --data ~/.cache/tinygpt/datasets/pace-labeled.jsonl \
    --template chatml \
    --rank 16 --alpha 32 \
    --steps 2000 \
    --lr 1e-4 \
    --max-seq 2048 \
    --out ~/.cache/tinygpt/runs/pace-planner-v1/pace-planner-v1.lora
```

Wall time: ~30 min on M5 Pro.

### 4. Eval against Pace's fixtures

```bash
# Spin up student as OpenAI-compat endpoint
tinygpt serve ~/.cache/tinygpt/runs/pace-planner-v1/pace-planner-v1.lora \
    --base "$QWEN_DIR" \
    --port 8765 \
    --grammar grammars/pace-tool-tags.gbnf &

# Run Pace's eval against our endpoint
cd /Users/sarthak/Desktop/fleet/clickyLocal/
LOCAL_PLANNER_URL=http://127.0.0.1:8765/v1 \
LOCAL_PLANNER_MODEL=pace-planner-v1 \
python scripts/eval-planners.py
```

Compare to qwen3-30b-a3b's 15/15 baseline + 925ms latency.

### 5. Ship into Pace

Add a `TinyGPTPlannerClient.swift` conformer in clickyLocal that
implements `BuddyPlannerClient` against the serve port. Toggle via:

```
PlannerProvider = tinygpt-local
LocalPlannerModelIdentifier = pace-planner-v1
LocalPlannerURL = http://127.0.0.1:8765/v1
```

## Acceptance criteria

| Criterion | Target |
|---|---|
| Fixture pass rate | ≥ 14/15 (one miss tolerable) |
| Latency p50 | < 200ms |
| RAM footprint | < 1.5 GB |
| Tag format compliance | 100% (constrained decoding) |
| Daily-driver in Pace | 1 week without rollback |

## Status — 2026-06-07 EOD

- Pipeline ready end-to-end
- All factory primitives validated
- Awaiting: clean Mac thermals + LM Studio with qwen3-30b loaded
- Fire morning of 2026-06-08
