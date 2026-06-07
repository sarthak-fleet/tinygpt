# Recipe — B25 ScaleDown specialist (extractive context compression)

**Goal:** train a small TinyGPT model that takes `(question, long_document)`
and outputs an *extractive* compressed document — a subset of the original
sentences that preserves the answer-relevant span — and submit to the
[ScaleDown.ai](https://scaledown.ai) challenge.

**Status as of 2026-06-05:** unblocked on compute (N02 produces the base);
blocked on the canonical training data pull (D3 — MS-MARCO + NQ). A
synthesis fallback path is documented below so we can start on something
the moment N02 lands.

## The challenge in one paragraph

ScaleDown rewards models that compress LLM context windows without losing
answer fidelity. The compression must be *extractive* — selecting actual
sentences from the source — not abstractive paraphrasing. The downstream
metric is: how often does a frozen QA model still answer correctly when
fed the compressed context instead of the full one? A high score means
you preserved the load-bearing sentences and removed the noise.

## Data we need

Each training row: `{question, long_doc, label_mask}` where `label_mask` is
a 0/1 vector over sentences in `long_doc` marking the ones a reader needs
to answer the question.

### Canonical sources (D3 — to be pulled)

| Dataset | Why | Path it will land |
|---|---|---|
| MS-MARCO (passage config) | question + passage + answer span | `~/.cache/tinygpt/datasets/ms-marco.jsonl` |
| Natural Questions (long+short answer) | natural web passages with explicit long-answer spans | `~/.cache/tinygpt/datasets/natural-questions.jsonl` |
| HotpotQA (supporting_facts) | multi-hop — explicitly labelled supporting sentences | not yet planned; ~115 MB |

### Synthesis fallback (if D3 stays blocked)

If we can't get MS-MARCO/NQ pulled before training time, derive supervision
from data we already have:

1. **From `hermes-fc.jsonl` (50 MB, cached)** — each tool-call example
   contains a "reasoning" block + the final structured call. The call's
   field values are extractable spans from the reasoning. Train compression
   = "output the reasoning sentences that mention the field values."
2. **From `ultrafeedback.jsonl` (1.2 GB, cached)** — instruction + reference
   answer + bad answer. Compressed form = the sentences from the reference
   that justify the rating. Weaker signal but plenty of volume.
3. **From FineWeb-Edu (the N02 corpus)** — synthesize questions via a
   frozen LLM (Qwen 0.6B via `tinygpt sample`), label the source sentences
   that contain the answer span. Bootstrap.

The synthesis path is `scripts/scaledown-prep.py`:

```bash
# Permissive (3.5% yield, ~391 rows, avg keep-ratio 0.70 — weak signal):
python3 scripts/scaledown-prep.py

# Strong-compression-only (0.8% yield, ~85 rows, keep-ratio ≤ 0.60):
python3 scripts/scaledown-prep.py --max-keep-ratio 0.60 \
    --out ~/.cache/tinygpt/datasets/scaledown-train-strong.jsonl
```

The script extracts tools-pruning supervision from hermes-fc: each row's
"full instruction" is the long context, the "compressed form" keeps only
the tool definitions actually called by the response. This is enough to
**validate the SFT loop runs end-to-end** but not enough volume (~85
strong rows) for a real specialist. Use D3 / NQ / HotpotQA when they
land for real signal.

## Training recipe

Once N02 lands:

```bash
tinygpt train \
    --base ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt \
    --preset huge \
    --tokenizer <SmolLM2 dir> \
    --corpus <scaledown-train.jsonl> \
    --task scaledown \
    --steps 50000 \
    --batch 4 --accum 4 \
    --lr-schedule wsd --warmup 500 --decay-steps 5000 \
    --max-lr 1e-4 \
    --save-every 5000 --save-history \
    --val-split 0.01 --val-every 250 \
    --log-jsonl ~/.cache/tinygpt/runs/scaledown-sft/scaledown-sft.jsonl \
    --out ~/.cache/tinygpt/runs/scaledown-sft/scaledown-sft.tinygpt
```

Output format expected (per ScaleDown submission schema):

```json
{
    "question": "<original>",
    "selected_sentence_indices": [3, 7, 12, 14],
    "compressed": "<sentences[3]> <sentences[7]> <sentences[12]> <sentences[14]>"
}
```

## Eval recipe — E6 (per PRD)

```bash
tinygpt eval-scaledown /tmp/scaledown-sft.tinygpt \
    --benchmark <scaledown-bench.jsonl> \
    --judge HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --out docs/artifacts/scaledown-e6-<date>.jsonl
```

Score = downstream QA accuracy on the compressed context, normalized by
compression ratio. E0 row shape — same `eval-compare` consumes it.

## Submission flow

ScaleDown is a leaderboard challenge with a public submission API. Their
submission expects:

1. JSONL of `{question_id, selected_sentence_indices}` over the public
   benchmark
2. The model card (params, training recipe, eval methodology)
3. Optional: open-weights link

The submission script (`scripts/scaledown-submit.py`) is a thin POST
wrapper — write after E6 produces a benchmark-format JSONL.

## Differentiation story

- **Local + tiny**: most ScaleDown entries will be massive proprietary
  models with prompt-engineering tricks. A 22M-param SLM that runs on
  laptop CPU shipping a non-trivial score is the demo.
- **Mechanistic interp**: if we run B13 (`tinygpt sae --checkpoint-dir`)
  across the SFT checkpoints, we can show the feature that fires on
  "this sentence is load-bearing." First publicly-watchable feature
  for a context-compression task.
- **Open recipe**: the docs/recipes/ + docs/artifacts/ trail (this file
  + the score JSONLs) IS the submission's reproducibility appendix.

## Open questions

1. **Compression ratio target.** The challenge has multiple tracks
   (compress to 25%, 50%, 75% of input length). Pick one. Start with 50% —
   easier supervision, still useful.
2. **Sentence vs span granularity.** Sentence-level is easier to label
   and explain; span-level may score higher. v1 sentence; v2 try span.
3. **Tokenizer choice.** SmolLM2 BPE works for English; if ScaleDown
   benchmark has code or non-English, may need to swap.

## File layout

| File | Role |
|---|---|
| `docs/recipes/b25-scaledown.md` | this doc |
| `scripts/scaledown-prep.py` | data synthesis from existing cached JSONLs (TODO) |
| `scripts/scaledown-submit.py` | challenge-submission POST wrapper (TODO) |
| `~/.cache/tinygpt/datasets/scaledown-train.jsonl` | prepared training data |
| `~/.cache/tinygpt/datasets/scaledown-bench.jsonl` | held-out evaluation set |

## Links

- Challenge: https://scaledown.ai/
- ScaleBench (canonical eval framework — to be cloned under
  `_external/`): mentioned in PLAN.md §3 E6
- Related: HotpotQA supporting-facts paper (Yang et al. 2018)
- Related: MS-MARCO passage ranking (Bajaj et al. 2018)
