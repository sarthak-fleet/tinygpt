---
name: B25 ScaleDown Challenge specialist
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B25)
related_prds: A1-first-specialist-tool-caller.md (sibling specialist; different domain),
              E6 `tinygpt eval-scaledown` (the harness — still ⬜, doc-only at docs/recipes/b25-scaledown.md)
---

# PRD — Extractive context-compression specialist for ScaleDown leaderboard

## Goal

Train a Mac-runnable specialist that takes `(query, long_context)` and
returns the subset of sentences relevant to the query — i.e. extractive
compression. Submit to the public
[ScaleDown Challenge leaderboard](https://main.d3hbeukddvrxcc.amplifyapp.com/leaderboard)
as "competitive task SLM trained from scratch on a Mac." A second
publicly-scored proof-point alongside A1's BFCL.

The architectural trick is a token-level relevance classifier head on
the residual stream → sentence-level aggregation → threshold-keep
(no autoregressive generation). Different from A1's tool-caller in
shape: not a generic chat model, a domain-shaped specialist.

## Why now

- A1 (tool-caller) is the first specialist. B25 is the second, and
  the only one that ships with a *public external scoreboard* — the
  exact "shipped specialist trained on a Mac" proof-point the
  platform needs for credibility beyond our docs.
- The harness (E6 `tinygpt eval-scaledown`) is the smaller PRD — it's
  one of the still-⬜ Tier-E items. Once it lands, B25 has its scoring.
- The training cost is small (~half-week wall-clock on M5 Pro per the
  recipe at `docs/recipes/b25-scaledown.md`); the leaderboard
  submission cycle is short.

## Scope — in

- **Base model:** `qwen3-4b-instruct-2507` (consistent with A1; same
  Mac runtime story).
- **New module:** `Sources/TinyGPTModel/RelevanceHead.swift` — a small
  Linear(d_model → 2) classification head trained on per-token labels
  (relevant / not). Attaches at the final residual stream (post-norm).
- **New subcommand:** `tinygpt compress <query> --doc <doc.txt>
  --model <base+adapter> --threshold 0.5 --out compressed.txt`.
  Token-level scores → sentence-level mean → threshold filter.
- **Training data:** MS-MARCO + Natural Questions (D3 pulls these
  for B25). Teacher-label per-sentence relevance using a stronger
  local model (Qwen3-9B or cloud-escalate Gemma-3-12B); seeds saved
  to `docs/research/data/scaledown-seed.jsonl`.
- **Loss:** token-level BCE on the relevance head; mask-out non-
  context tokens (query and instruction) so we score document tokens
  only.
- **Recipe:** `scripts/recipes/b25-scaledown.sh` — data prep, SFT
  with the new head (frozen base + trained head + LoRA on a few
  blocks), eval, ScaleBench submission.
- **Eval:** the E6 `tinygpt eval-scaledown` harness scores
  downstream F1/EM after compression vs full-context.
- **Submission:** the leaderboard accepts a model + a small wrapper
  script; we package both.

## Scope — out

- **Abstractive compression** (rewriting / summarizing). Extractive is
  the V1 path; abstractive is a separate specialist arc.
- **Multi-document compression** (long-context retrieval across N
  docs). V1 takes one doc.
- **Speed / latency optimization** for the compression itself — V1
  uses the same base's forward + the head, with no special inference
  path. Optimization is V2.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPTModel/RelevanceHead.swift` | new — classification head |
| `Sources/TinyGPTModel/PeftVariants.swift` | wire "relevance-head + LoRA on top blocks" composite path |
| `Sources/TinyGPT/Compress.swift` | new — subcommand |
| `Sources/TinyGPT/TinyGPT.swift` | `case "compress"` |
| `Sources/TinyGPT/SFT.swift` | optional `--loss relevance-bce` mode |
| `scripts/recipes/b25-scaledown.sh` | new — end-to-end recipe |
| `docs/specialists/b25-scaledown.md` | new — user-facing brief |
| `docs/recipes/b25-scaledown.md` | update — now points at the recipe instead of being the recipe |
| `docs/PLAN.md` | B25 ⬜ → ✅ on ship + leaderboard rank |

## Don't touch

- A1 specialist artifacts — orthogonal recipe.
- The OpenAI surface in `tinygpt serve` — V1 ships compress as a
  CLI, not a server endpoint.

## Acceptance criteria

- [ ] `tinygpt compress "what is RoPE?" --doc rope_paper.txt
  --model qwen3-4b-instruct-2507+b25.lora --threshold 0.5` returns a
  ≤ 30%-original-length excerpt that, on a held-out QA eval, scores
  within 5pp F1 of the full document.
- [ ] `tinygpt eval-scaledown <model+adapter>` (E6) produces a row
  in the shared E0 schema.
- [ ] Submission to the [ScaleDown leaderboard](https://main.d3hbeukddvrxcc.amplifyapp.com/leaderboard)
  succeeds; we land a public rank, regardless of where.
- [ ] `docs/specialists/b25-scaledown.md` includes our rank + the
  full recipe.
- [ ] `evals/b25-acceptance.sh` re-runs the gate from a clean checkout.

## Reference patterns

- `Sources/TinyGPTModel/LinearProbe.swift` — the small classification-
  head pattern. Relevance head is the same shape, different label.
- A1 specialist's recipe (`docs/specialists/a1-tool-caller.md`) — the
  user-facing brief template.
- [ScaleDown blog](https://tinyml.substack.com/p/how-we-train-small-language-models)
  + their public methodology — cite, don't redocument.

## Open questions

- Whether to share the relevance head across LoRA layers vs train
  one independently. **Recommendation:** one shared head + LoRA on
  the top 4 blocks. Simplest path that still adapts; more aggressive
  is V2.
- Whether to publish the teacher-labeled training data on HF.
  **Recommendation:** yes if licensing of the source corpora allows;
  open-data submission is more credible than "we trained on
  something undisclosed."
