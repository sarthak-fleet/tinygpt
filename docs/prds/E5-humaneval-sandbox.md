---
name: E5 HumanEval sandboxed code-execution scorer
status: not-started
owner: unassigned (parallel-agent task)
created: 2026-06-05
parent_plan: docs/PLAN.md §3 Tier E (E5)
task_tracker: #234
---

# PRD — `tinygpt eval-humaneval` + Rust sandbox

## Goal

Ship `tinygpt eval-humaneval <model.tinygpt>` — a Swift subcommand that
generates Python code from TinyGPT, executes it inside a Rust-isolated
sandbox, and scores pass@1 against HumanEval + MBPP test suites.

## Why now

- Code specialist evaluation is the long-pole. lm-eval-harness has a
  HumanEval task but its execution path uses the host `python` —
  unsafe for arbitrary model output.
- Sandboxing is the actual hard part; the eval scoring is rote pattern
  (gen code → exec → check assertions → tally). Doing the sandbox
  right enables every future code-evaluation work.
- HumanEval (164 problems) + MBPP sanitized (257 problems) test JSONLs
  are already on disk at `~/.cache/tinygpt/datasets/{humaneval,mbpp}-test.jsonl`.

## Scope — in

- **Rust sandbox crate** at `scripts/humaneval-sandbox/`:
  - `Cargo.toml` with `nix` (for `setrlimit` + `seccomp` on Linux,
    `sandbox-exec` shim on macOS) + `serde_json` + `clap`
  - `src/main.rs` — CLI:
    ```
    humaneval-sandbox --code <file.py> --test <file.py> \
                      --timeout 10 --memory-mb 256 \
                      [--allow-network=false]
    ```
    Returns JSON `{"passed": bool, "stdout": str, "stderr": str,
    "exit_code": int, "wall_seconds": float}` on stdout
- **Sandbox approach (macOS)**: `sandbox-exec` with a profile that:
  - denies all network (`(deny network*)`)
  - denies write outside `/private/tmp/sandbox-<uuid>/` (`(deny file-write*)`
    with allow exception for the tmp dir)
  - allows fork/exec only for Python itself
- **Resource limits**: `setrlimit` for CPU time + memory + open files
  via Rust's `nix` crate, applied before `exec`-ing python
- **Swift wrapper** at `Sources/TinyGPT/EvalHumanEval.swift`:
  - Read HumanEval/MBPP test JSONL
  - For each problem: build the prompt → `tinygpt serve` → get model
    completion → write to a tmp file → invoke `humaneval-sandbox` →
    parse the pass/fail
  - Emit one `EvalCompare.Row` per (problem, metric=`pass@1`); aggregate
    to a single `subtask="humaneval"` and `subtask="mbpp"` row with
    pass@1 fraction

## Scope — out (v2)

- pass@k for k > 1 (needs multi-sample generation; v1 = greedy pass@1)
- Linux sandboxing (seccomp filters etc.) — v1 ships macOS-only since
  TinyGPT's primary platform is Apple Silicon
- Docker-based sandboxing (heavier, more deps; defer)
- Live execution during evaluation (streaming UX) — v1 batch

## Inputs the agent has

| Resource | Location |
|---|---|
| Test sets | `~/.cache/tinygpt/datasets/humaneval-test.jsonl` (164 problems), `~/.cache/tinygpt/datasets/mbpp-test.jsonl` (257) |
| Pattern to copy (Swift side) | `Sources/TinyGPT/RunLmEval.swift` → `runViaServe` |
| E0 schema | `Sources/TinyGPT/EvalCompare.swift` |
| HumanEval reference exec | https://github.com/openai/human-eval/blob/master/human_eval/execution.py |
| `sandbox-exec` docs | `man sandbox-exec`, profile examples in `/System/Library/Sandbox/Profiles/` |

## Acceptance criteria

1. `cargo build --release` produces `target/release/humaneval-sandbox`
2. Sandbox refuses network access (verify with a test program that
   tries `socket.connect`) and OOM-kills programs over `--memory-mb`
3. `tinygpt eval-humaneval --help` clean usage
4. End-to-end smoke:
   ```
   tinygpt eval-humaneval /tmp/huge-smoke-30min.tinygpt \
     --tokenizer <SmolLM2 dir> \
     --limit 3 --serve-port 8101 \
     --suites humaneval,mbpp \
     --out /tmp/humaneval-smoke.jsonl
   ```
5. Output JSONL has `task="humaneval"` / `task="mbpp"`, `metric="pass@1"`,
   `score=<0..1>` rows
6. `tinygpt eval-compare /tmp/humaneval-smoke.jsonl --by task` renders

## File paths

| Action | Path |
|---|---|
| **create** | `scripts/humaneval-sandbox/Cargo.toml` |
| **create** | `scripts/humaneval-sandbox/src/main.rs` |
| **create** | `scripts/humaneval-sandbox/macos-sandbox.sb` (profile file) |
| **create** | `native-mac/Sources/TinyGPT/EvalHumanEval.swift` |
| **don't touch** | `TinyGPT.swift` (dispatch — PR diff), `PLAN.md`, `HANDOFF.md` |

## Estimated effort

**~1-2 days.** Sandbox is the unknown:

- 1-2 hrs: scaffold the Rust crate + basic Python subprocess wrapper
- 2-4 hrs: macOS `sandbox-exec` profile + verify deny rules work
- 2-3 hrs: `setrlimit` integration for time/memory caps
- 2-3 hrs: Swift wrapper + JSONL emit
- 2 hrs: smoke against the 5 simplest HumanEval problems

## Coordination

PR description must include:
1. `cargo build --release` output + binary `otool -L` showing minimal deps
2. Sandbox negative test: a Python script attempting network access
   that gets killed (paste the output)
3. Sandbox positive test: a passing HumanEval problem solution (e.g.,
   `def add(a, b): return a + b` for `task_id=HumanEval/0`)
4. Smoke JSONL with rows for `humaneval` and `mbpp`

Maintainer merges dispatch line, marks E5 done.

## Known risks

- **sandbox-exec is deprecated on macOS** but still functional. Apple
  hasn't shipped a replacement that's CLI-usable. If it breaks in a
  future macOS, fall back to plain `setrlimit` + denied-by-default
  PATH (don't even let Python find `curl` or `wget`).
- **Python startup overhead**: 0.3s per problem × 164 = 50s of pure
  Python startup. Use a persistent Python interpreter via a long-lived
  subprocess + `print(json.dumps(...))` IPC, instead of `python new.py`
  per problem.
- **HumanEval test assertions can be cleverly evil** (intentionally —
  some "tests" attempt to detect that they're being scored). Trust
  the standard test suite verbatim; don't try to make it "safer."
- **MBPP's test format differs from HumanEval** — MBPP has `test_list`
  (multiple `assert` strings); HumanEval has a single `test` Python
  function. Document the difference; v1 handles both.

## Source links

- HumanEval paper: https://arxiv.org/abs/2107.03374
- HumanEval repo: https://github.com/openai/human-eval
- MBPP paper: https://arxiv.org/abs/2108.07732
- Local test sets: `~/.cache/tinygpt/datasets/{humaneval,mbpp}-test.jsonl`
- macOS sandbox-exec: `man 5 sandbox`
