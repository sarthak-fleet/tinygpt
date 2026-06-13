---
name: B33 one-command laptop-finetune onboarding
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B33)
parent_strategy: docs/sessions/2026-06-13-market-landscape-mac-first.md (move #3)
related_prds: A1-first-specialist-tool-caller.md (the recipe this wraps), B6-mac-app-demo.md (the GUI version), B31-gallery-and-project-pins.md (resolves the base model)
---

# PRD — `tinygpt quickstart`: data → trained specialist in one command

## Goal

`tinygpt quickstart <data.jsonl>` takes a user's task data and walks them
to a trained, evaluated, runnable specialist on their Mac with **one
command and zero prior knowledge** — auto-picks a sensible base from the
gallery, infers the recipe, trains, evals against a baseline, and drops
them into a chat with the result. The CLI sibling of B6's GUI Factory tab.

The landscape finding: "fine-tune on your laptop" is owned by a *library*
(MLX-LM), not a *product*. The conversion barrier is the gap between
"MLX-LM can technically do this" and "a person who isn't an ML engineer
actually does it." B33 is that bridge for the CLI; B6 is it for the GUI.

## Why now

- A1 ships the proven recipe; B33 wraps it so a user doesn't have to read
  the recipe. The recipe being *good* and the recipe being *reachable*
  are different problems — A1 solves the first, B33 the second.
- Every competitor's onboarding is "sign up, add a credit card, upload to
  our cloud." A `brew install` → `tinygpt quickstart mydata.jsonl` flow
  with no account and no cloud is a categorically different first-run.
- B31 gallery + project pins give `quickstart` the base-model resolution
  it needs for free.

## Scope — in

- `Sources/TinyGPT/Quickstart.swift` — the wizard:
  1. **Inspect the data** — detect format (chat JSONL / instruction /
     raw text / tool-call traces); report row count + a sample; bail
     with a clear message if it can't parse.
  2. **Pick a base** — heuristic from data shape + size: tool-call data →
     the A1 base; chat → a small instruct base; raw text → a from-scratch
     config sized by corpus. Resolved via the B31 gallery. User can
     override with `--base`.
  3. **Infer the recipe** — sequence-packing on, sensible LoRA rank,
     WSD schedule (B11) once shipped, NEFTune, `--eval-every`. Print the
     resolved recipe and ask for confirmation (unless `--yes`).
  4. **Train** — call the shipped trainer; show the C10 live view URL.
  5. **Eval** — run a relevant suite vs the base 0-shot; print the
     before/after delta (the "did this help?" answer).
  6. **Try it** — drop into an interactive `tinygpt chat` against the
     new specialist, with the base available for A/B.
  - Writes a `tinygpt.project.json` (B31) so the result is reproducible
    + shippable.
- `--dry-run` prints the full resolved plan without training.

## Scope — out

- **GUI** — that's B6 (Factory tab). B33 is the CLI; they share the
  recipe-resolution logic (factor it into a shared
  `RecipeResolver` used by both).
- **Multi-stage pipelines** (SFT→DPO→quantize chains). V1 = single SFT
  pass. The recipe resolver can grow stages later.
- **Cloud fallback** for users without enough RAM. The pitch is local;
  if the data + base don't fit, fail with a clear "this needs NGb;
  try a smaller base" message, not a silent cloud upload.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/Quickstart.swift` | new — the wizard |
| `Sources/TinyGPTModel/RecipeResolver.swift` | new — data-shape → (base, recipe); shared with B6 |
| `Sources/TinyGPT/TinyGPT.swift` | `case "quickstart"` |
| `Sources/TinyGPT/Chat.swift` | reuse or extend for the "try it" step (if a chat REPL exists; else thin new one) |
| `evals/quickstart-smoke.sh` | new — tiny fixture data → resolved plan (--dry-run) asserts a sane recipe |
| `docs/quickstart.md` | new — the "your first specialist in 10 minutes" page |
| `docs/PLAN.md` | B33 ⬜ → ✅ on ship |

## Don't touch

- The trainer / eval internals — quickstart orchestrates shipped
  subcommands.

## Acceptance criteria

- [ ] `tinygpt quickstart sample-toolcalls.jsonl --yes` trains an adapter,
  evals it, and lands in a chat — end to end on M5 Pro, no account, no
  network beyond an initial base-model pull.
- [ ] `--dry-run` prints the resolved (base, recipe) plan and exits
  without training.
- [ ] Data it can't parse fails with an actionable message naming the
  expected formats.
- [ ] The run writes a valid `tinygpt.project.json` (passes B31's
  `validate()`).
- [ ] `docs/quickstart.md` reproduces on a clean `brew install`.

## Reference patterns

- `docs/prds/A1-first-specialist-tool-caller.md` — the recipe quickstart
  defaults to.
- `docs/prds/B6-mac-app-demo.md` — the GUI sibling; share `RecipeResolver`.
- `docs/prds/B31-gallery-and-project-pins.md` — base-model resolution +
  the project file quickstart emits.

## Open questions

- How smart the base-picker should be in V1. **Recommendation:** three
  buckets (tool-call / chat / raw-text) by data-shape heuristic, with
  `--base` override. Don't build a classifier; the heuristic + override
  covers the first-run case.
- Whether "try it" requires an interactive REPL or just generates N
  sample completions side-by-side. **Recommendation:** sample N
  side-by-side first (non-interactive, CI-testable); REPL is a nice
  follow-up.
