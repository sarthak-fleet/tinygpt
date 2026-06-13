---
title: Speech & systems topics — interview-grade map
description: Voice-pipeline latency, WER, speech-to-speech, fine-tuning debugging, feature selection, queues vs websockets, FSDP2 — each mapped to the best external source and to where this codebase (or Pace) actually does it.
---

# Speech & systems topics — an interview-grade map

Eight topics that came up in a real technical interview, mapped the house
way: **Learn it** (best external source — we don't re-teach), **why it
matters to THIS project**, and **in the repo** where applicable. Several
of these aren't theory here — tinygpt/Pace has shipped them.

Where we're starting: you've built the cascade (Pace is literally
speech → text → LLM → text → speech, all local) — this doc connects what
you've built to the named concepts interviewers ask about.

## 1. Voice pipeline tail latency (p50 vs p95)

**What:** p95/p99 latency is dominated by rare slow events (GC, cold
caches, model loads), not the average path; you fix tails differently
than you fix means.

**Why it matters here:** Pace's doctrine is 100ms-on-local. The TTFW
hunt (330→119ms) was a tail-latency exercise: the fix was caching
`constraint_init` vocab iteration — a one-time cost that poisoned every
first request. Same shape as any p95 story.

**Learn it:** [The Tail at Scale (Dean & Barroso)](https://research.google/pubs/the-tail-at-scale/) — the canonical paper, 8 pages.

**In the repo:** `native-mac/Sources/TinyGPTServe/Serve.swift` (token-mask
caching at boot); `scripts/profile-pace-turn.sh` in the Pace repo breaks a
voice turn into stage timings — that's a p95 budget in script form.

## 2. ASR accuracy — WER and text post-processing

**What:** Word Error Rate = (substitutions + insertions + deletions) /
reference words. Post-ASR NLP (normalization, punctuation, entity
fixing — what spaCy gets used for) reduces *effective* WER without
touching the acoustic model.

**Why it matters here:** WhisperKit large-v3-turbo was qualified at
"perfect accuracy on 'Pace' without phrase biasing" — that qualification
WAS a WER eval. The Stage-A dictation post-processor specialist is
exactly the "fix text after ASR" pattern.

**Learn it:** [Hugging Face Audio Course](https://huggingface.co/learn/audio-course) (unit on ASR evaluation); [jiwer](https://github.com/jitsi/jiwer) for computing WER in 3 lines; [spaCy docs](https://spacy.io/usage/linguistic-features) for the NLP layer.

**In the repo:** Pace repo `WhisperKitTranscriptionProvider.swift`
(streaming re-transcription strategy is documented in its header comment);
dictation post-processor under Pace's specialist work (task #295/#318).

## 3. Cascaded vs direct speech-to-speech

**What:** The cascade (ASR → LLM → TTS) is modular and debuggable but
loses prosody/emotion and stacks latency; direct S2S transformers
(speech tokens in, speech tokens out) collapse the stack at the cost of
training difficulty and control.

**Why it matters here:** Pace IS the cascade, deliberately — each stage
is independently swappable (we just swapped the middle for Gemma-3-12B
with zero changes to ASR/TTS). Knowing exactly what the cascade gives up
is knowing Pace's ceiling.

**Learn it:** [Moshi (Kyutai)](https://arxiv.org/abs/2410.00037) — the
reference full-duplex S2S paper; [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)
for the "thinker-talker" hybrid that keeps a text spine.

**In the repo:** the whole Pace pipeline; tinygpt's contribution is the
middle box (planner eval + serve). `docs/DRILLDOWN.md` shows why the
modular middle is a feature — you can't A/B 12 planners inside a fused S2S model.

## 4. Fine-tuning a decoder-only model when loss won't drop

**What:** The debugging ladder: overfit one batch first; check the loss
mask (are you training on prompt tokens?); check the chat template
matches the base model exactly; LR warmup; then schedule (WSD) and
stability tricks.

**Why it matters here:** v1–v11 of the Pace specialist plus clarify-v1
were eleven rounds of this ladder, including a real catastrophic-
interference case (47pp OOS regression from 38 training rows) and a
thinking-mode template mismatch that silently burned an hour.

**Learn it:** [Karpathy — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — still the canonical debugging ladder.

**In the repo:** `native-mac/Sources/TinyGPT/Train.swift` (WSD schedule,
loss-spike recovery + replay); `docs/learn/session-08-training-mechanics.md`
covers the mechanics ground-up; `docs/RETROSPECTIVE.md` for the v1–v11 arc.

## 5. Feature selection — PCA vs recursive elimination vs isolation forest

**What:** Three different jobs often conflated: PCA *transforms* features
into orthogonal components (loses interpretability, keeps variance); RFE
*selects* original features by iteratively dropping the weakest; Isolation
Forest is an *outlier detector* (short isolation paths = anomalies), used
in feature pipelines to clean data, not select features.

**Why it matters here:** the closest analog in this repo is a genuinely
good interview answer: SAEs vs PCA. PCA finds orthogonal directions of
max variance; a sparse autoencoder finds an *overcomplete, sparse* basis —
which is why interp uses SAEs (features stay monosemantic) where classic
DS uses PCA.

**Learn it:** scikit-learn docs are the fastest route:
[PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca),
[RFE](https://scikit-learn.org/stable/modules/feature_selection.html#rfe),
[IsolationForest](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest);
[Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) ch. 3 + 14 for the theory.

**In the repo:** the SAE work (tasks #195/#199/#224, `docs/learn/` SAE
notes) — bring the SAE-vs-PCA contrast when asked about dimensionality
reduction.

## 6. Credit risk modeling

**What:** Predicting default probability from tabular features; the
industry stack is logistic regression / gradient-boosted trees + careful
feature engineering, with interpretability as a regulatory requirement
(why deep nets lag here).

**Why it matters here:** it doesn't — this is the one topic with no repo
anchor. Learn it as the canonical *tabular + interpretability-constrained*
ML setting, the opposite pole from everything tinygpt does.

**Learn it:** [scikit-learn's gradient boosting guide](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) +
[ESL](https://hastie.su.domains/ElemStatLearn/) ch. 10; for the domain
framing, search "PD/LGD/EAD modeling" — probability of default, loss
given default, exposure at default are the three regulated quantities.

## 7. Queues vs WebSockets for voice bots

**What:** A WebSocket is a transport (one live duplex pipe); a queue is a
decoupling primitive (backpressure, retry, burst absorption, consumer
independence). Voice bots want both: sockets at the edge for low latency,
queues between internal stages so a slow TTS render doesn't stall ASR ingest.

**Why it matters here:** Pace solves this in-process — the TTS sidecar
renders sentence N+1 while N plays (a 1-deep pipeline queue), and audio
samples accumulate in a lock-guarded buffer while transcription runs on
its own cadence. Same pattern, no broker.

**Learn it:** [Designing Data-Intensive Applications](https://dataintensive.net/) ch. 11
(stream processing — the queues-vs-direct-connection tradeoff in full).

**In the repo:** Pace repo `WhisperKitTranscriptionProvider.swift`
(sample buffer + 1.2s partial cadence = producer/consumer with
backpressure); the Kokoro sidecar pipelining described in Pace `AGENTS.md`.

## 8. GPU parallelism — FSDP2 and friends

**What:** The parallelism menu: data parallel (replicate model, split
batch), FSDP/ZeRO (shard params+grads+optimizer state, gather per-layer
just-in-time), tensor parallel (split individual matmuls), pipeline
parallel (split layers across devices). FSDP2 is PyTorch's rewrite with
per-parameter DTensor sharding — composable with the others.

**Why it matters here:** tinygpt is deliberately the *opposite* regime —
one Mac, unified memory, zero inter-device communication. Knowing FSDP2
is knowing exactly what you're NOT paying for (all-gather bandwidth,
sharding bugs) and why single-device MLX training tops out where it does
(the Mega-bf16 OOM was solved by config, not sharding — there's no
second device to shard to).

**Learn it:** [HF Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — the
current canonical treatment of all parallelism forms, interactive;
[PyTorch FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for the FSDP2 API itself.

**In the repo:** `native-mac/Sources/TinyGPT/Train.swift` is the
single-device counterexample; `docs/learn/session-05-scaling.md` covers
why scale-up beat scale-out for this project.

## Suggested order

1–4 and 7 first (they're anchored in code you own — read the anchor, then
the source). 8 next (one playbook read). 5–6 are pure-external study;
ESL chapters are the slowest material here, budget accordingly.
