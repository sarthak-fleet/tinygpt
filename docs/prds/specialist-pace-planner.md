---
name: Pace planner specialist — first factory customer
status: active-2026-06-08-v6_1-fixture-gate-unblocked-specialist-still-pending
owner: maintainer (manual recipe, drives factory verification)
created: 2026-06-07
parent_plan: docs/sessions/2026-06-06-mac-specialist-platform.md (first validating customer)
upstream_product: /Users/sarthak/Desktop/fleet/clickyLocal (Pace)
---

# PRD — Pace planner specialist

## 2026-06-07 status note

Owner approval for heavy work was granted. LM Studio is reachable at
`http://127.0.0.1:1234/v1/models` and includes `qwen/qwen3-30b-a3b`.

Cached teacher-label files exist, but are not clean enough to use as the
main specialist corpus:

- `pace-labeled-v2.jsonl`: 389 rows, only 116 non-empty outputs
- `pace-labeled-v3.jsonl`: 51 rows, only 36 non-empty outputs

The narrower v6.1 SFT attempt in `factory-pace-planner-v6_1.md` did not
meet fixture acceptance, even with schema-shaped correction data. This
broader specialist remains active and quality-blocked; the next viable
step is a cleaner synthesis pipeline that strips Qwen3 `<think>` behavior,
validates every row before SFT, and trains against the exact serve prompt
and schema path used by Pace.

## 2026-06-08 focused investigation note

The v6.1 and specialist failures were not independent. A shared
train/eval alignment issue was found: v6.1 SFT rows omitted the Pace system
prompt while serve/eval included it. The v6.1 builders now train through the
same `system:` + `user:` ChatML path used by serve. The eval harness also
had a brace-counting JSON extractor that misread valid strings containing
`}`; it now uses Python's JSON decoder.

This moved the narrow v6.1 fm-fixture result from the earlier broken
2/19 fixture-gold attempt to 17/19 with the 300-step prompt-aligned
fixture adapter:

`~/.cache/tinygpt/runs/pace-planner-v6_1-fixture-system-300/pace-planner-v6_1-fixture-system-300.lora`

The final 19/19 gate required one more serving fix: TinyGPT's JSON Schema
parser/FSM previously ignored `minLength` and `maxLength`, so the constrained
decoder allowed degenerate one-character `spokenText` values such as `}`.
After implementing string length enforcement and raising Pace `spokenText`
`minLength` to 2, the same 300-step adapter passes **19/19 fm-fixtures**.

The broader specialist should not proceed with teacher distillation until
the same constraints are locked:

- every synthesized row must include the exact Pace system prompt path used
  by serve;
- every teacher output must validate with the same JSON/schema parser as
  acceptance;
- degenerate but valid `spokenText` values such as `}` should be rejected
  during data validation;
- v7-style dynamic schemas should enum-constrain visible target labels and
  `perform.action` values instead of leaving all target fields as free
  strings.

## Goal

Distill Pace's qwen3-30b-a3b planner (current production model, scoring
15/15 on internal fixtures, 925ms mean latency) into a small student
(target: 600M-1.5B) that:

1. **Matches 15/15** on the existing fixture suite at
   `clickyLocal/evals/fixtures/`
2. **Latency < 200ms p50** (vs 925ms today — ~5× faster)
3. **Footprint ≤ 1.5 GB Q4** (vs 18.6 GB today — ~12× less RAM)
4. Drops in as `PlannerProvider=tinygpt` via the existing
   `BuddyPlannerClient` Swift protocol

If this lands, Pace's planner becomes free + 5× faster + fits comfortably
beside the VLM. **This is the factory's first product validation.**

## Why first

Pace is shovel-ready in a way no other specialist is:

| Asset | Status |
|---|---|
| Production model in use | qwen3-30b-a3b via LM Studio at :1234 — already running |
| Eval fixtures | `clickyLocal/evals/fixtures/*.json` — 3 categories shipped |
| Eval runner | `clickyLocal/scripts/eval-planners.py` (34 KB) — exists |
| Tool-call schema | embedded in system prompt (`[POINT:x,y:label]`, `[CLICK ...]`, etc.) |
| Pluggable conformer | `BuddyPlannerClient.swift` protocol — designed for swap |
| Production traffic | Sarthak's daily usage |

This is the closest thing to "if you build it they will come" — there's
already a "they."

## Recipe — uses the factory primitives that just shipped

### Step 1: Synthesize labels from the teacher (via `tinygpt synthesize`)

Inputs: a pool of Pace-shaped prompts. Source options ranked by quality:
- **Best**: real Pace usage logs (we don't have, but `generate-intent-corpus.py`
  exists in clickyLocal/scripts) → produce N=10,000
- **Good**: synthesize input prompts via Qwen3-30b (also as Pace usage)
  then synthesize tool-call outputs

For v1, hybrid: 500 hand-curated from fixtures + 9,500 synthesized.

```bash
# Label inputs against the running LM Studio teacher
tinygpt synthesize \
    --teacher http://localhost:1234/v1 \
    --teacher-model qwen/qwen3-30b-a3b \
    --inputs pace-prompts.jsonl \
    --input-field prompt \
    --system-file pace-system-prompt.txt \  # Pace's actual system prompt
    --temperature 0.0 \
    --parallel 4 \
    --rate-limit 30 \
    --out pace-labeled.jsonl
```

### Step 2: Distill into a small student (via `tinygpt distill`)

```bash
# Soft distillation (just shipped today) for best transfer
tinygpt distill \
    --teacher qwen3-30b-a3b.tinygpt \  # if local; else use hard-only
    --student qwen3-0.6b.tinygpt \
    --corpus pace-labeled.jsonl \
    --mode soft \                       # default
    --temperature 4.0 \
    --alpha 0.7 \                       # 0.7 KL + 0.3 NLL
    --steps 5000 \
    --tokenizer qwen3-shared/ \
    --out pace-planner.tinygpt
```

If teacher doesn't load (30B too big to also fit), fall back to hard
mode (the synthesize output IS the supervision signal — just NLL on
teacher's text).

### Step 3: Constrained decoding for tool-tag compliance

Pace's tool tags (`[POINT:x,y:label]`, `[CLICK ...]`, etc.) are
deterministic patterns. A GBNF grammar enforces correct format,
eliminating the "model improvised a malformed tag" failure mode.

Pace's `BuddyPlannerClient` already parses these tags; constrained
decoding makes the model never emit malformed ones.

### Step 4: Eval against Pace's existing fixtures

```bash
# Spin up the student as an OpenAI-compat endpoint
tinygpt serve pace-planner.tinygpt --port 8765 &

# Run Pace's existing eval runner against our endpoint
cd /Users/sarthak/Desktop/fleet/clickyLocal/
LOCAL_PLANNER_URL=http://127.0.0.1:8765/v1 \
LOCAL_PLANNER_MODEL=pace-planner \
python scripts/eval-planners.py
```

Compare student's score to qwen3-30b-a3b's 15/15 baseline. Compare
latency vs the 925ms baseline.

### Step 5: Drop into Pace

In Pace's Info.plist:
```
PlannerProvider = tinygpt-local
LocalPlannerModelIdentifier = pace-planner
```

(or whatever the conformer name will be; `BuddyPlannerClient.swift`
already has a protocol — add a third conformer in clickyLocal that
points at our serve port.)

## Acceptance criteria

| Criterion | Target | Measurement |
|---|---|---|
| **Fixture pass rate** | ≥ 15/15 on existing suite | `eval-planners.py` |
| **Latency p50** | < 200ms (vs 925ms) | logged TTFSW from same runner |
| **Latency p95** | < 400ms | same |
| **Memory** | ≤ 1.5 GB peak RSS | `ps` during inference |
| **Format compliance** | 100% (tool tags well-formed) | constrained decoding guarantees |
| **Pace integration** | Pace daily-driver replaces `qwen3-30b-a3b` with student | Sarthak running for 1 week without rollback |

## Risk + mitigations

| Risk | Mitigation |
|---|---|
| Student too small (0.6B can't match 30B) | Fall back to Qwen2.5-1.5B-Instruct base; even larger if needed (still ≤ 1/10 the teacher) |
| Soft distill OOMs (teacher + student both in RAM) | Hard mode + synthesized labels — same recipe, less RAM |
| Synthesis data doesn't cover edge cases in Pace fixtures | Hand-augment with fixtures themselves; teacher then teacher-fills 9 more variants per fixture |
| Real Pace usage looks different from synthesis | Run for 1 week + log; if regressions, re-train with real logs |
| Format compliance regresses without grammar | Always run with `--grammar pace.gbnf` (already shipped feature) |

## Scope — out (v2)

- Vision specialist (VLM) — Pace also runs Qwen3-VL-8B; that's a
  separate, bigger distillation project
- Multi-modal training (text + screen state) — v1 = text-only
  conditioning, screen state is just text encoded in user prompt
- Online learning from Pace usage — v1 = static distillation

## Estimated effort

**~1 week of focused work** assuming the factory primitives work:
- Day 1: extract Pace's system prompt + generate prompt pool (use
  `generate-intent-corpus.py`)
- Day 2: `tinygpt synthesize` against LM Studio overnight (~1-2 hrs
  wall, depends on parallel)
- Day 3: distillation run on Qwen3-0.6B base — overnight
- Day 4: eval against fixtures, iterate
- Day 5: write `pace.gbnf` grammar, re-eval with constrained decoding
- Day 6: integrate into Pace (write the `TinyGPTPlannerClient` conformer)
- Day 7: daily-drive, polish, write up

## Source files in clickyLocal

| File | Role |
|---|---|
| `leanring-buddy/BuddyPlannerClient.swift` | Protocol — the swap point |
| `leanring-buddy/LocalPlannerClient.swift` | Reference OpenAI-compat conformer (LM Studio) |
| `leanring-buddy/AppleFoundationModelsPlannerClient.swift` | Reference Apple FM conformer |
| `scripts/eval-planners.py` | Eval runner |
| `evals/fixtures/*.json` | Fixture cases (qa-no-screen, multi-turn-continuation, screen-referential) |
| `scripts/generate-intent-corpus.py` | Synthetic prompt generator |
| `AGENTS.md` | Full architecture doc |

## Why this validates the factory

If Pace planner ships:
- Confirms `tinygpt synthesize` works end-to-end with a real teacher
- Confirms soft distillation transfers reasoning capability
- Confirms constrained decoding solves format compliance
- Confirms our serve OpenAI-compat surface is consumer-ready
- Proves the "comparable quality at much smaller and faster" thesis
- Gives Pace's users a 5× speedup + 12× RAM reduction
- Becomes the platform's first defensible "we used the factory to ship this" story

## Open questions for the maintainer

1. Do you want to start with 0.6B student or jump to 1.5B? (Lower risk
   with 1.5B; bigger speedup demo with 0.6B.)
2. Synthesize how many examples? 10K vs 50K — diminishing returns past 10K
   for narrow tasks, but cheap to generate more.
3. After Pace, KB embedder? Or another Pace specialist (VLM, action-
   safety classifier)?
