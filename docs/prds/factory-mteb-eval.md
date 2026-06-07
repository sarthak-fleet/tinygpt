---
name: MTEB eval — retrieval benchmark for embedder specialists
status: shipped-2026-06-07
owner: unassigned
created: 2026-06-07
priority: P1 — blocks KB embedder validation
---

# PRD — `tinygpt eval-mteb` for embedders

## 2026-06-07 ship note

`tinygpt eval-mteb` is implemented with `native-mac/Sources/TinyGPT/EvalMTEB.swift`
and `scripts/eval-mteb-adapter.py`. It wraps the Python `mteb` library for
HF baseline embedders and emits E0-compatible JSONL rows. TinyGPT-native
`.tinygpt-embed` models still wait on the embedder model format; the command
fails clearly for those instead of pretending an embed CLI exists.

## Goal

Add an `eval-mteb` subcommand that scores an embedder model against
MTEB (Massive Text Embedding Benchmark) subtasks. Required to validate
the KB embedder specialist (or any custom embedder we ship).

## Why ship

When we distill BGE-M3 → 22M-110M embedder, the natural quality bar is
"compare MTEB scores against the teacher and a generic baseline." No
embedder eval today.

KB specialist arc is blocked on this eval surface existing.

## Scope — in

### CLI surface

```
tinygpt eval-mteb \
    --model my-embedder.tinygpt-embed \
    --tasks BEIR/scifact,BEIR/nfcorpus,MTEB/StackOverflowDupQ \
    --limit 500 \                              # cap docs per task
    --out results.jsonl
```

Wraps the Python `mteb` library as a subprocess (similar pattern to
`run-lm-eval`'s wrap of lm-eval-harness). Output rows match E0 schema.

### Tasks (v1 priorities)

Pick 5-10 BEIR/MTEB tasks for v1:
- `BEIR/scifact` — small, fast smoke task
- `BEIR/nfcorpus` — medical retrieval
- `MTEB/StackOverflowDupQ` — programming retrieval
- `MTEB/AmazonCounterfactualClassification` — classification embedding use
- `MTEB/Banking77Classification` — banking domain

### Metrics

Emit E0-conformant rows with:
- `metric`: `ndcg@10`, `recall@10`, `mrr`, `accuracy` (depending on task)
- `task`: e.g., `mteb/scifact`
- `score`: float

### Inference shape

Embedder receives strings → emits fixed-size float vectors. The
subprocess wrapper:
1. Spawns the embedder as a service (or batch-mode CLI)
2. Asks mteb to use it via a thin Python adapter
3. Parses mteb's standard JSON output

## Scope — out

- Full MTEB (hundreds of tasks). v1 = 5-10 representative.
- Multilingual MTEB. v1 = English only.
- Image / multimodal MTEB. v1 = text only.

## Acceptance

1. Run against an off-the-shelf HF embedder (e.g., `BAAI/bge-small-en`)
   first — should match published scores within 1pp
2. Run against a `tinygpt`-trained embedder (when shipped) — produces
   E0 rows
3. `tinygpt eval-compare results.jsonl --by model` renders correctly

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/EvalMTEB.swift` |
| **create** | `scripts/eval-mteb-adapter.py` — thin Python that registers a TinyGPT embedder with the `mteb` lib |
| **modify** | `native-mac/Sources/TinyGPT/TinyGPT.swift` — dispatch |

## Estimated effort

**~2-3 days.** mteb library is mature; mostly wiring.

## Source

- MTEB: Muennighoff et al. 2023 (https://arxiv.org/abs/2210.07316)
- mteb lib: https://github.com/embeddings-benchmark/mteb
