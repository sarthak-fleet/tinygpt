---
name: B30 prompt reasoning-depth classifier
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B30)
parent_learn: docs/learn/castform-rl-finetune.md (Steal #3)
related_prds: B10-quality-classifier.md (sibling classifier; reasoning-depth is the orthogonal axis),
              B29-trace-to-training-data.md (downstream consumer of the labels)
---

# PRD — Classify training prompts by reasoning depth

## Goal

`tinygpt reasoning-classify <corpus.jsonl>` tags every prompt with
its reasoning depth: `single-hop`, `multi-hop`, `comparison`, or
`other`. The label feeds B29's trace-to-training-data pipeline (so
the training mix is balanced by depth) and the leaderboard (so
per-depth accuracy is reportable, not just averaged).

Lifted from Castform's reasoning-classification step
(`docs/learn/castform-rl-finetune.md` §3). They use the labels to
mix training data; we use them for both that AND for the
leaderboard's per-category breakdown.

## Why now

- Once B29 ships, the SFT corpus comes from trace dumps — a
  mixed-depth blob. Balancing the depth ratio is a known knob
  that needs a classifier first.
- Castform's published mix examples cite single-hop dominance
  hurting comparison-task generalization. Same effect would hit
  us; cheaper to control upfront.
- Half-day if we mimic B10's bag-of-ngram fastText shape — the
  classifier is small.

## Scope — in

- `Sources/TinyGPT/ReasoningClassify.swift` — new subcommand:
  - `--train <labeled.jsonl> --out reason.cls` — trains on a
    labeled set
  - `--score <corpus.jsonl> --model reason.cls --out
    scored.jsonl` — adds a `reasoning_depth` field to each row
  - `--filter <scored.jsonl> --target-mix
    "single=0.3,multi=0.5,comparison=0.2" --out balanced.jsonl`
    — downsamples to the target mix
- Implementation: bag-of-trigram features → softmax-4. Same shape
  as B10; defactor a `BagOfNgramClassifier` shared utility.
- V1 label source: small hand-labeled seed (~500 prompts) +
  bootstrap via LLM-judge labeling of a larger pool (~5K) using
  E7 `tinygpt judge`.

## Scope — out

- **Continuous-depth scoring** (e.g. "this is 1.7-hops"). V1 is
  4-way categorical.
- **Heuristic-only classification** (string-pattern matching for
  "and then / also / compare"). Heuristics are too brittle; the
  small ngram classifier is the right floor.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/ReasoningClassify.swift` | new — subcommand |
| `Sources/TinyGPTModel/BagOfNgramClassifier.swift` | new — factored from B10 |
| `Sources/TinyGPT/QualityClassify.swift` | refactor to use shared classifier (B10 small follow-up) |
| `Sources/TinyGPT/TinyGPT.swift` | `case "reasoning-classify"` |
| `evals/reasoning-classifier-smoke.sh` | new — 200-prompt fixture, assert macro-F1 ≥ 0.5 |
| `docs/recipes/balanced-training-mix.md` | new — recipe for using the classifier in front of B29 |
| `docs/PLAN.md` | B30 ⬜ → ✅ on ship |

## Acceptance criteria

- [ ] Trained classifier reaches macro-F1 ≥ 0.5 on a 4-way held-out
  set of 200 prompts.
- [ ] `--score` mode adds the field; throughput ≥ 5 MB/s on M5 Pro.
- [ ] `--filter` mode produces a corpus whose final mix matches
  `--target-mix` within ±5pp per category.
- [ ] B29 with `--reasoning-balance` flag invokes B30 transparently
  on the synthesized data.

## Reference patterns

- B10 quality-classifier — sibling; same architecture, different
  label.
- `Sources/TinyGPTModel/LinearProbe.swift` — softmax-N head
  pattern.
- [Castform site](https://castform.com/) — original copy.

## Open questions

- Whether to include a "destructive" depth label (overlap with
  Pace's destructive-intent classifier). **Recommendation:** no —
  reasoning depth and intent are orthogonal axes; keep them
  separate.
