---
name: B22 token-preserving agent trajectory recorder
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B22)
related_prds: B21-micro-automixer.md, B23-agent-eval-protocol.md (Poolside-discipline sibling PRDs)
---

# PRD — Store input_ids/output_ids/tool-calls/rewards through agent rollouts

## Goal

Every agent rollout (`tinygpt agent`, the AgentLoop in serve, BFCL +
τ-bench harness runs) writes a structured trajectory record carrying
the *raw token IDs* — input_ids, sampled output_ids, tool_call args,
tool_result text — alongside the rewards and checkpoint fingerprint.
This is the substrate for off-policy SFT, DPO, and (later) full RLVR
without re-tokenization drift mid-pipeline.

The discipline is Poolside's ([Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive)).
Their finding: keeping token IDs end-to-end avoids the subtle BPE
re-encoding mismatches that silently corrupt off-policy gradients.

## Why now

- Today, our agent rollouts log human-readable strings; SFT recipes
  re-tokenize them. For most cases that's fine. For a tool-call whose
  args contain `\n` or non-ASCII, the re-tokenized sequence can differ
  from the originally-sampled one in a way that biases gradient
  signal toward wrong behavior.
- A1 specialist + B5 cloud-escalate training data both want
  agent-trace inputs. Both are blocked on a clean substrate; this PRD
  unblocks both.
- Cheap to add — the agent loop already has the token IDs in scope
  during sampling; we just don't currently persist them.

## Scope — in

- `Sources/TinyGPTModel/AgentTrajectory.swift` — record format:
  ```swift
  struct AgentTrajectoryStep {
      let role: String          // "system" | "user" | "assistant" | "tool"
      let content: String       // for human inspection
      let input_ids: [Int]?     // present when role generated this turn
      let output_ids: [Int]?    // present for assistant; sampled tokens
      let tool_call: ToolCall?  // structured tool args, if any
      let tool_result: String?  // raw tool output, if any
      let reward: Double?       // task-defined; None if not scored
  }
  struct AgentTrajectory {
      let id: String                       // UUID
      let model_fingerprint: String        // SHA-256(weights)
      let checkpoint_path: String          // origin model
      let steps: [AgentTrajectoryStep]
      let summary: [String: Any]           // task name, final reward, etc.
  }
  ```
- File format: one trajectory per `.atraj` file (JSON + per-step token
  ID arrays). Optional gzip via `.atraj.gz` extension for long
  trajectories.
- `tinygpt agent --trajectory-dir <dir>` writes every rollout.
- `tinygpt eval-bfcl --trajectory-dir <dir>` writes BFCL's per-sample
  trajectories. Same for `eval-tau-bench`.
- Reader API: `AgentTrajectory.load(path)` returns the struct. Used by
  future SFT/DPO recipes that consume trajectories without
  retokenization.
- Backwards-compat: when `--trajectory-dir` isn't set, nothing
  changes.

## Scope — out

- **Trajectory-based training itself** (off-policy DPO from `.atraj`).
  Separate PRD; B22 ships only the substrate.
- **Replay** of a trajectory through a different checkpoint. Useful
  but adds significant code; defer.
- **Compression beyond gzip** — protobuf, packed-bits etc. Premature.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/AgentTrajectory.swift` | new — record types + reader/writer |
| `Sources/TinyGPT/Agent.swift` | thread the recorder through the loop; emit a `.atraj` per rollout |
| `Sources/TinyGPT/AgentLoop.swift` | same — serve-side multi-turn loop |
| `Sources/TinyGPT/EvalBFCL.swift` | add `--trajectory-dir` flag |
| `Sources/TinyGPT/EvalTauBench.swift` | same |
| `Tests/TinyGPTModelTests/AgentTrajectoryRoundtripTests.swift` | new — write, reload, assert byte-equality of input_ids/output_ids |
| `docs/agent_runtime.md` | "Token-preserving trajectories" section |

## Acceptance criteria

- [ ] `tinygpt agent --trajectory-dir /tmp/traj/ "ping the API"` writes
  one `.atraj` file containing the full conversation with token IDs.
- [ ] Reload the file; `output_ids` for the assistant turn decodes to
  exactly the same string that was generated.
- [ ] BFCL run with `--trajectory-dir` produces one `.atraj` per
  sample; total file size is < 1 MB per typical sample.
- [ ] Roundtrip test passes.
- [ ] Documentation references the format and the consumer API.

## Reference patterns

- `Sources/TinyGPT/Agent.swift` — already has all the token IDs in
  hand during sampling; instrument the existing loop.
- `Sources/TinyGPTServe/Trace.swift` (inference tracer) — analogous
  artifact-per-request pattern. Same `<UUID>.json` shape, different
  schema.
- [Poolside Laguna deep dive](https://poolside.ai/blog/laguna-a-deeper-dive)
  — cite for the design rationale.

## Open questions

- Whether to ship trajectories from `serve` requests too (not just
  CLI-driven agent rollouts). **Recommendation:** opt-in via a
  `serve --trajectory-dir` flag in a follow-up; the OpenAI surface
  doesn't expose token IDs by default and we'd want a careful design.
- File-per-trajectory vs append to a single JSONL. **Recommendation:**
  file-per-trajectory — trivially shardable, easy to delete one,
  matches the inference-tracer pattern.
