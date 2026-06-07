---
name: E1 BFCL eval wrapper
status: shipped-2026-06-05
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier E (E1)
task_tracker: #231
---

# PRD — `tinygpt eval-bfcl` wrapper

## Goal

Ship `tinygpt eval-bfcl <model.tinygpt>` — a Swift subcommand that runs the
Berkeley Function Calling Leaderboard harness against a TinyGPT model and
emits results in the shared E0 `EvalCompare.Row` schema.

## Why now

- A1 specialist target is a **tool-caller**. Without BFCL scoring it, the
  next 2 days of training produce a model with no measurable function-calling
  number — sample outputs only.
- The harness source is already on disk at
  `~/.cache/tinygpt/datasets/_external/gorilla-bfcl/berkeley-function-call-leaderboard/`
  — pulled in a previous session but never wired. Pulling source ≠ usable
  evaluator; this PRD is the wiring.
- The serve + lm-eval pattern (`Sources/TinyGPT/RunLmEval.swift`'s
  `runViaServe`) is the proven template. This task is *adapt the template
  for a different Python harness* — not greenfield engineering.

## Scope — in

- New file `Sources/TinyGPT/EvalBFCL.swift` mirroring `RunLmEval.swift`'s
  subprocess-via-serve pattern.
- Add a single switch case to `Sources/TinyGPT/TinyGPT.swift`:
  ```
  case "eval-bfcl":
      EvalBFCL.run(args: Array(args.dropFirst()))
  ```
  **DO NOT push this directly.** That file is serialized; include the exact
  diff in the PR description and the maintainer will merge.
- Parse BFCL's per-category accuracy JSON, emit one `EvalCompare.Row` per
  (subtask, metric) pair to the target JSONL.
- BFCL subtasks to preserve as `subtask` field (BFCL emits these as JSON
  keys; pass through verbatim):
  - `simple`
  - `multiple`
  - `parallel`
  - `parallel_multiple`
  - `relevance`
  - `irrelevance`
  - `live_simple`
  - `live_multiple`
  - `live_parallel`
  - `live_parallel_multiple`
- Per-subtask metric is `accuracy` (BFCL's headline number).

## Scope — out (deferred to v2)

- Sandbox isolation for BFCL's `exec` category — E5 HumanEval will solve
  sandboxing generically; v1 should either skip `exec` (`--exclude exec`)
  or trust the model's output (BFCL's exec category runs sample-generated
  code).
- Streaming partial results — emit JSONL at end-of-run only.
- Multi-model parallel scoring — sequential is fine for v1; the user can
  invoke the wrapper N times against N models if they want a leaderboard.
- Bundling BFCL's Python deps into a venv. Assume the user has run BFCL's
  `pip install -e .` once (document in the help text).
- `tinygpt eval-bfcl --hf-model <dir>` — for v1, only `--tinygpt-model` is
  required (matches the user's actual training target). HF-model support
  is a one-line addition once v1 works.

## Inputs the agent has

| Resource | Location |
|---|---|
| BFCL harness source | `~/.cache/tinygpt/datasets/_external/gorilla-bfcl/berkeley-function-call-leaderboard/` (read its `README.md` first — entry point details there) |
| Pattern to copy | `native-mac/Sources/TinyGPT/RunLmEval.swift`, specifically `runViaServe(...)` — well-commented ~250 lines |
| E0 schema to write | `EvalCompare.Row` in `native-mac/Sources/TinyGPT/EvalCompare.swift` |
| Serve binary | Resolved via `CommandLine.arguments.first` (self-invocation) — see how `RunLmEval.runViaServe` does it |
| Smoke target model | `/tmp/huge-smoke-30min.tinygpt` (exists by the time agent runs; ~6% expected, pipeline correctness is what matters) |
| Tokenizer for the model | `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/<sha>/` |

## Acceptance criteria

1. `tinygpt eval-bfcl --help` prints a clean usage block (model + tokenizer
   + out path + tasks subset + serve-port flags).
2. The end-to-end command runs without crashing:
   ```
   tinygpt eval-bfcl /tmp/huge-smoke-30min.tinygpt \
     --tokenizer ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/<sha>/ \
     --out /tmp/bfcl-smoke.jsonl \
     --limit 5 \
     --serve-port 8097
   ```
3. The output JSONL has rows conforming to `EvalCompare.Row` — verify with
   `python3 -c "import json; [print(json.loads(l)['task'], json.loads(l)['subtask'], json.loads(l)['score']) for l in open('/tmp/bfcl-smoke.jsonl')]"`.
4. `tinygpt eval-compare /tmp/bfcl-smoke.jsonl --by task` renders the
   per-subtask scores. (The `task` field is `"bfcl"`; `subtask` carries
   the BFCL subcategory name.)
5. Build passes: `cd native-mac && swift build -c release`.

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/EvalBFCL.swift` |
| **read for pattern** | `native-mac/Sources/TinyGPT/RunLmEval.swift`, `native-mac/Sources/TinyGPT/EvalCompare.swift` |
| **don't touch** | `Sources/TinyGPT/TinyGPT.swift` (dispatch line — maintainer merges), `docs/PLAN.md`, `HANDOFF.md`, anything in `Sources/TinyGPTServe/`, `Sources/TinyGPTModel/`, `Sources/TinyGPTApp/` |

## Sketch of `EvalBFCL.swift`

Copy `RunLmEval.runViaServe` then replace the inner subprocess block. The
shape should look like:

```swift
import Foundation

enum EvalBFCL {
    static func run(args: [String]) {
        // Parse: --tinygpt-model, --tokenizer, --tasks (CSV subset of subcats),
        //        --serve-port, --out, --limit, --model-name, --model-step, --baseline
        // ...

        // 1. Resolve tinygpt-cli (self-invocation via CommandLine.arguments.first)
        // 2. Spawn `tinygpt-cli serve <model> --port N` in background
        // 3. Poll http://127.0.0.1:N/v1/models for readiness (timeout 60s)
        // 4. Spawn the BFCL harness CLI (Python). Their entrypoint is
        //    `python -m bfcl_eval.main` per their README — verify.
        //    Pass --model OpenAI compatible (point at our serve), --base-url,
        //    --test-category <CSV>, --output-dir <work-dir>
        // 5. Wait for the harness; parse the resulting score_*.json
        //    (BFCL writes one JSON per (model, category) combo into a
        //    `score/` subdir — see their docs).
        // 6. For each (subcategory, score) row, append an EvalCompare.Row
        //    with task="bfcl", subtask=<subcategory>, metric="accuracy".
        // 7. Terminate the serve process.
        // 8. Print summary + the eval-compare command.
    }
}
```

## Estimated effort

**~1 day focused.** Breakdown:

- 1-2 hrs: read BFCL README + figure out the right CLI invocation
- 2-3 hrs: write `EvalBFCL.swift` (mostly mechanical copy from `RunLmEval`)
- 1-2 hrs: smoke-test + parse the actual BFCL score JSON shape
- 1 hr: round-trip via `eval-compare`
- 1-2 hrs: handle the inevitable edge case (BFCL's harness expects HF
  tokenizer in a way that may need extra args; their model API might want
  a specific JSON shape)

## Coordination

Work in a branch or worktree. PR description must include:

1. The exact diff line for `Sources/TinyGPT/TinyGPT.swift` (the new switch
   case) — maintainer applies this.
2. The BFCL CLI invocation you settled on (paste the full subprocess
   arguments).
3. A copy of `/tmp/bfcl-smoke.jsonl` from your smoke run (or a snippet of
   the first 3 rows).
4. Build log showing `ok (build complete)`.

Maintainer will:
- Merge the dispatch line
- Mark task #231 complete
- Update `HANDOFF.md` / `docs/PLAN.md`

## Known risks

- **BFCL's CLI API may have shifted** since the snapshot at
  `_external/gorilla-bfcl/`. If their entrypoint moved or their flags
  changed, fall back to invoking their `openfunctions_evaluation.py` or
  similar score-runner script directly. Worst case: read their pipeline
  source and reimplement just enough to get a score JSON out.
- **BFCL expects an OpenAI-shaped client**. Our `tinygpt serve` is
  OpenAI-compatible; BFCL may try to use the official `openai` Python
  package, which works against any `base_url`. Set the
  `OPENAI_API_KEY=anything` env var so it doesn't refuse to start.
- **Long prompts** — BFCL's function-spec system prompts can hit ~2K
  tokens. Our serve's default `maxContext` may truncate. Pass
  `--max-context 4096` to serve when needed.

## Source links

- BFCL leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- BFCL repo: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- Local snapshot: `~/.cache/tinygpt/datasets/_external/gorilla-bfcl/`
- Existing pattern: `native-mac/Sources/TinyGPT/RunLmEval.swift` (read first)
- E0 schema: `native-mac/Sources/TinyGPT/EvalCompare.swift`
