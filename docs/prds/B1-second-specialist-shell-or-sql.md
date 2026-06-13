---
name: B1 second specialist (shell or SQL)
status: not-started (blocked-by A1)
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B1)
related_prds: A1-first-specialist-tool-caller.md (the template; B1 cookie-cuts off it)
---

# PRD — Second specialist after A1 ships

## Goal

Once A1 (tool-caller) lands, run the same recipe shape against a
*different domain* — shell command generation OR SQL — to validate
the platform isn't accidentally A1-specific. Same base (qwen3-4b),
same SFT + LoRA path, different data + different eval.

Two specialists ship the platform thesis: "you can do this for *any*
task" is much stronger than "we did this for one task."

## Why now

- Blocked on A1. A1 ships the template recipe + acceptance scaffolding
  this PRD reuses verbatim.
- Choice of shell vs SQL: pick whichever has cleaner public eval
  scaffolding by the time A1 lands. **Default recommendation: shell**
  via [InterCode-Bash](https://intercode-benchmark.github.io/) (multi-
  turn, sandboxed, verifiable). SQL via [Spider](https://yale-lily.github.io/spider)
  is the fallback if shell sandbox setup is friction.

## Scope — in

- Pick the domain (shell vs SQL) at A1-ship-time based on eval-
  scaffolding state. Document the call in `docs/decision_log.md`.
- Pull the dataset: `scripts/recipes/b1-<domain>.sh` mirrors
  A1's recipe template.
- Training data: domain-specific public corpora (e.g.
  `glaiveai/glaive-function-calling-v2` already pulled for tool
  context; need shell-specific or SQL-specific equivalents).
- New eval: `tinygpt eval-<domain>` — a thin wrapper around the
  domain's public harness (InterCode-Bash or Spider's official
  scorer).
- Ship gate: domain-eval avg ≥ base + 3pp under B23 K=3 protocol.
- Artifact: `adapters/b1-<domain>.lora`.

## Scope — out

- **Combining A1 + B1 into a single multi-task adapter.** Distinct
  adapters keep the experiment clean; merging is its own arc.
- **Cross-domain transfer ablation.** Useful research; defer.

## Files to touch

| File | Change |
|---|---|
| `scripts/recipes/b1-<domain>.sh` | new — recipe (cookie-cut from A1) |
| `Sources/TinyGPT/EvalShell.swift` OR `EvalSpider.swift` | new — harness wrapper |
| `docs/specialists/b1-<domain>.md` | new — brief |
| `docs/research/mac_slm_leaderboard_v0.md` | regenerate with the new row |
| `docs/PLAN.md` | B1 ⬜ → ✅ on ship |

## Acceptance criteria

- [ ] B1 specialist beats base by ≥ 3pp on the domain eval (K=3).
- [ ] A1's eval scores ≥ stay at parity (the new specialist
  doesn't regress; orthogonal adapter).
- [ ] Leaderboard shows ≥ 3 distinct rows: A1, B1, base.

## Reference patterns

- `docs/prds/A1-first-specialist-tool-caller.md` — the template.
  This PRD is the second instance; if the recipe doesn't generalize
  easily, that's the finding.
- [InterCode-Bash](https://intercode-benchmark.github.io/) — the
  shell-side eval scaffolding.
- [Spider](https://yale-lily.github.io/spider) — the SQL-side eval
  scaffolding.

## Open questions

- Which domain. **Recommendation:** defer the binary choice to
  A1-ship-time; decide based on InterCode-Bash vs Spider setup
  friction.
