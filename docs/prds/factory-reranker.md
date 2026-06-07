---
name: Reranker — cross-encoder training + eval
status: shipped-v1-lexical-2026-06-07
owner: unassigned
created: 2026-06-07
priority: P2 — needed by KB specialist v2
---

# PRD — `tinygpt rerank-train` + `tinygpt rerank-eval`

## 2026-06-07 ship note

`tinygpt rerank-train` and `tinygpt rerank-eval` ship a lightweight lexical
cross-encoder baseline:
- trains on `{query,pos_doc,neg_doc}` JSONL triples
- writes a `.tinygpt-rerank` JSON artifact
- evaluates pairwise ranking with `accuracy` and `mrr`
- emits E0 JSONL rows

This closes the factory surface. A transformer cross-encoder remains a quality
upgrade after the Mac model path is ready for it.

## Goal

Add training + eval surface for cross-encoder rerankers. Distill from
jina-v2-base / BGE-Reranker into a domain-specialized smaller reranker.
Reuses the embedder distillation pattern but with cross-encoder
architecture (scores `[query, doc]` pairs jointly).

## Why ship

KB stack uses jina-v2-base reranker today (25s cold load, CPU-bound).
A domain-distilled smaller reranker would:
- Cold-load in <2s
- Better domain accuracy
- Smaller binary

Plus rerankers compose with embedders: "embed for recall, rerank for
precision" is the canonical RAG pattern.

## Scope — in

### Training CLI

```
tinygpt rerank-train \
    --teacher BAAI/bge-reranker-base \
    --triples (query, pos_doc, neg_doc) pairs JSONL \
    --student-preset rerank-tiny|rerank-small \
    --steps 5000 \
    --out my-reranker.tinygpt-rerank
```

Pairwise hinge loss on (query, pos, neg) triples — standard
cross-encoder training.

### Eval CLI

```
tinygpt rerank-eval \
    --model my-reranker.tinygpt-rerank \
    --task BEIR/scifact \
    --base-retriever bge-small-en \              # first-stage retriever
    --top-k 100 \
    --out results.jsonl
```

Two-stage: base retriever produces top-K candidates, reranker rescores.
Report NDCG@10 + MRR.

### Architecture

Single-tower transformer with two-pooler-output head (binary
relevance). Smaller than full-encoder embedder because it's a
classification head on top of a transformer body.

## Scope — out

- ColBERT-style late-interaction rerankers (different paradigm, v2)
- Cross-lingual reranking
- Multi-task reranking heads

## Acceptance

1. Train smoke completes on Mac in <30 min with 1K triples
2. Eval on BEIR/scifact: reranker improves NDCG@10 over base retriever
   by ≥5pp
3. Result rows match E0 schema; render via `eval-compare`

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/RerankTrain.swift` |
| **create** | `native-mac/Sources/TinyGPT/RerankEval.swift` |
| **create** | `native-mac/Sources/TinyGPTModel/RerankerModel.swift` |
| **modify** | `TinyGPT.swift` dispatch |

## Estimated effort

**~3-5 days.** Cross-encoder is straightforward; main work is the
two-stage eval pipeline + integrating with the embedder file format.

## Sources

- BGE-Reranker docs: https://huggingface.co/BAAI/bge-reranker-base
- ColBERT v2 (for reference; out of scope here):
  https://arxiv.org/abs/2112.01488
