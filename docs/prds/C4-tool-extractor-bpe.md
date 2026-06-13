---
name: C4 tool-call extractor — BPE tokenizer support
status: not-started
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier C (C4)
related_prds: factory-planner-v7-tools-in-prompt.md (the planner this extractor sits behind)
---

# PRD — Extend `tinygpt train-extractor` to BPE tokenizers

## Goal

`tinygpt train-extractor` (the mini-router-trainer that ships the
"intent + tool" classifier on top of the residual stream) currently
assumes a byte-level tokenizer. Bases the user actually wants to
specialize (Qwen3-4B, Gemma-3, anything HF) use BPE. The extractor
silently misaligns when fed BPE bases; ship the BPE-aware path.

This is the smallest piece blocking B2 (mini-router on real BFCL
data) — the router can't be evaluated against a 4B-base specialist
unless the extractor speaks its tokenizer.

## Why now

- A1 (tool-caller specialist) ships on Qwen3-4B. If the extractor
  is the cheap-routing companion to A1, it needs to share Qwen3's
  tokenizer.
- The byte-level path has been working internally for the from-scratch
  models; extending it doesn't break that. Same code, two tokenizer
  backends.
- Existing infrastructure: `Sources/TinyGPTModel/HFTokenizer.swift`
  already loads BPE via swift-transformers for the serve path.

## Scope — in

- `Sources/TinyGPT/TrainExtractor.swift` — detect the base's
  tokenizer flavor at load time; route through either the byte-level
  or HF-BPE path uniformly.
- The label-alignment step (mapping char-spans of the training labels
  to token positions) gains a BPE branch using `HFTokenizer`'s
  `encodeWithOffsets` (already shipped in `HFTokenizer.swift`).
- New flag: none. Detection from the base is automatic.
- Smoke: `evals/extractor-bpe-smoke.sh` trains a 100-step extractor
  on Qwen3-4B over a 20-sample fixture, asserts loss < random.

## Scope — out

- **Tokenizer-agnostic alignment** (some unified API that doesn't
  distinguish at all). Engineering for engineering's sake; the
  two-branch path stays readable.
- **Training the extractor on multiple bases at once.** Defer.

## Files to touch

| File | Change |
|---|---|
| `Sources/TinyGPT/TrainExtractor.swift` | tokenizer-flavor branch in alignment + data pipeline |
| `Sources/TinyGPTModel/HFTokenizer.swift` | expose `encodeWithOffsets` if not already public |
| `evals/extractor-bpe-smoke.sh` | new — see above |
| `docs/decision_log.md` | one-line entry on the two-branch decision |

## Acceptance criteria

- [ ] `tinygpt train-extractor --base qwen3-4b-instruct-2507 ...`
  runs without "tokenizer mismatch" errors and produces a `.tre`
  sidecar.
- [ ] On a fixed BFCL-extracted training set (20 samples, 100 steps),
  cross-entropy loss decreases monotonically (no NaN, no plateau at
  random).
- [ ] The byte-level path continues to work on the from-scratch
  shakespeare base (no regression).
- [ ] Smoke passes in CI.

## Reference patterns

- `Sources/TinyGPTServe/Serve.swift` — already routes between
  byte-level and HF tokenizers at boot; copy the detection pattern.
- `Sources/TinyGPTModel/HFTokenizer.swift` — the BPE backend.
- `factory-planner-v7-tools-in-prompt.md` — the surrounding planner
  PRD; the extractor's output schema is fixed there.

## Open questions

- Whether to also accept SentencePiece bases (Llama, Gemma).
  **Recommendation:** swift-transformers already covers SP via the
  same HFTokenizer surface — should be free; verify with the smoke.
