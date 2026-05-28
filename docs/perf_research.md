# Performance research — what's done, what's plausibly next, what's mythology

A working document for the "best in market" target. Tonight's session
landed KV-cached sampling (~2× sustained), 4-bit palettization (6×
smaller files), and the ANE conversion path. This doc surveys the rest
of the levers — what's worth pursuing, with honest expectations.

## Where we are tonight (M5 Pro, 48 GB, MLX-Swift 0.31.3)

| Workload                       | Tonight     | Baseline (browser WebGPU) | Lift |
|-------------------------------|------------:|---------------------------:|-----:|
| Huge training (9.6M)          | 47 ms/step  | 720 ms/step                | 15× |
| Mega training (76M)           | 212 ms/step | n/a (browser can't)        |  ∞  |
| Behemoth training (404M)      | 1.0 s/step  | n/a                        |  ∞  |
| Titan training (1.3B)         | 2.0 s/step  | n/a                        |  ∞  |
| Huge sampling, short prompt   | 164 tok/s   | ~50 tok/s                  | 3.3× |
| Huge sampling, 500 tokens     | 304 tok/s   | ~50 tok/s                  | 6.1× |
| Core ML ANE forward (fp32)    | 365 pass/s  | n/a                        |  —   |
| 4-bit palettized model size   | 4.9 MB      | 18 MB (fp16 gallery)       | 0.27×|

## Levers in priority order

### 1. Batched sampling (B>1 sequences in parallel) — **2-4× pending**

Right now sampling is B=1. The GPU is heavily under-utilised — at
B=1, each generated token is a tiny matmul (1 × d_model × vocab).
Running 4-8 prompts in parallel (B=8 sample) multiplies token
throughput by roughly the batch size, capped by GPU compute.

**Engineering:** 1-2 hours. Generalise `forwardCached` to handle
B > 1, add a batched-sample CLI that takes N prompts and streams
N parallel outputs. Each completion gets its own KV cache (or a
shared batch-cache if all prompts are the same length).

**Realistic gain:** 4× tokens/sec, 2× tokens/sec/sequence. Practical
mostly when you have multiple users / multiple prompts to evaluate.

### 2. Speculative decoding — **3-5× sampling speedup**

A small "draft" model (Tiny / Small preset) generates 4-8 tokens
sequentially; the large "target" model (Huge / Mega) verifies all
8 with one parallel forward. If the target accepts K tokens, we
saved K-1 large-model forwards.

**Engineering:** 3-5 days. Train a draft model on the same corpus.
Implement the verify-and-accept loop. Tune acceptance threshold.
Common in production LLM serving (vLLM, Llama.cpp).

**Realistic gain:** 3-5× sampling tok/s. The acceptance rate
typically lands at 50-70% for well-matched draft/target pairs.

### 3. Continued quantization — **6× size today, 5-10× speed tomorrow**

What worked tonight: 4-bit palettization via coremltools — 6×
smaller files, comparable speed. Speed gain didn't materialise
because Core ML's palettize_weights is storage-side; at inference
it dequantises back to fp16.

**Real int-compute on ANE** is gated on Apple shipping the Stateful
Models API in coremltools (rumored late 2026). Once available,
the same 4-bit weights will run as int4 GEMM on ANE — historically
5-10× over fp16 GPU on transformer matmuls.

**Engineering today:** Adopt 4-bit for the browser side. Update
`finalize_gallery.mjs` to also emit a `.mlpackage-pal4` for Mac
distribution; ship the 5 MB version.

**Engineering tomorrow (when coremltools ships):** Re-quantize the
existing .mlpackage with the new path, benchmark. No code changes
needed — the conversion script will Just Work.

### 4. Mixed precision training (bf16) — **1.5-2× training**

We didn't actually verify fp16 training speedups tonight — the
preliminary numbers were ambiguous because the `mx.fast` ops
already auto-cast for some kernels. bf16 has the same range as
fp32 with half the bytes; less risky than fp16 for accumulator
overflow.

**Engineering:** 1-2 days. Set `Device.setDefault(.gpu(precision: .bfloat16))`
once MLX-Swift exposes it; otherwise cast model parameters
explicitly. Add the numerics gate from the browser-side
perf_quest framework.

**Realistic gain:** 1.5-2× training throughput on Mega/Behemoth;
negligible on Huge (already memory-bound to MLX-Fast).

### 5. Flash Attention 3 — **diminishing returns**

FA2 is in MLX-Fast already. FA3 (mid-2024 paper from Tri Dao) adds
warp specialisation for Hopper. On Apple Silicon there's no
equivalent — the M-series doesn't have warp specialisation. The
"FA3 equivalent for Apple" would be a co-designed
attention-kernel-for-AMX, which neither MLX nor Apple has shipped.

Probably wait. Don't chase this lever.

### 6. Distillation — **smaller model, same task**

Train a Tiny / Small student to mimic the Huge teacher's logits on
the same corpus. End up with a 1M-param model that samples at
~1000 tok/s and produces nearly-as-good text for a narrow domain.

**Engineering:** 1 week. Knowledge distillation loss
(KL between student & teacher distributions), train student on
teacher-labeled batches. Standard procedure.

**Use case:** Real-time on-device sampling, where you don't need
the full Huge model's quality but want millisecond response.

### 7. Sliding-window attention — **enables ctx > 1024**

Current Behemoth tops out at ctx=1024 because attention is O(T²)
in memory. With sliding-window (only attend to last K positions),
ctx scales to 4096+ with the same memory footprint.

**Engineering:** 2-3 days. Modify `CausalSelfAttention` to apply
a windowed mask; update positional embedding for ctx > 1024.

**Use case:** Longer-context Mac app (write a whole short story
in one continuous generation). Not relevant for the 256-token
browser gallery models.

## Datasets — beyond Project Gutenberg

Tonight: 34 MB across 19 books (Shakespeare, Bible, Tolstoy, Dickens,
Hugo, etc.). For "best in market" we need genuinely diverse
multi-genre text:

| Source                     | Size  | Notes |
|---------------------------|------:|-------|
| WikiText-2 (raw)           | ~12MB | Wikipedia article fragments, clean prose |
| WikiText-103 (raw)         | ~525MB | Same, 50× more |
| OpenWebText (sample)       | ~38GB | Curated web text, GPT-2 training set |
| Common Crawl (filtered)    | TB    | Massive but very noisy |
| ArXiv abstracts            | ~5GB  | Scientific writing, structured |
| Stack Exchange (text only) | ~20GB | Q&A, code-adjacent prose |
| Pile-CC subset             | ~50GB | EleutherAI's curated mix |
| GitHub code-only           | 100GB | Code in many languages |

**Practical next:** WikiText-2 is the right unlock — 10× our current
corpus, clean enough for byte-level, downloadable in seconds. Easy
fetch:

    curl -L https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    unzip -p wikitext-2-raw-v1.zip wikitext-2-raw/wiki.train.tokens > wikitext-2.txt

**Stretch:** WikiText-103 + the existing Gutenberg corpus gives ~560 MB
of mixed fiction + factual prose. Plenty for a serious training run.

## What "best in market" actually requires

Honest scope: the in-browser side is **already at the frontier** —
no one else trains GPTs from scratch in a browser tab today. The Mac
app pushes further than nanoGPT-style references (which are
inference-only or research-only), but loses to dedicated production
stacks (vLLM, MLC-LLM) on raw sampling tok/s for the same model size.

To genuinely lead in the "small-model from-scratch trainer" niche we
need:
- ✅ Browser path with one-click gallery (done)
- ✅ Mac path with arbitrary checkpoint sizes (done)
- ✅ End-to-end train + save + load + sample + eval (done)
- ⬜ LoRA fine-tuning (the killer feature; ~3-5 days)
- ⬜ Speculative decoding (3-5× sampling; ~3-5 days)
- ⬜ WikiText-103-trained gallery models (~1 day setup, multi-hour training)
- ⬜ Real ANE int4 compute path (gated on Apple)

## Open questions for the next push

1. Should we ship a "default gallery" that uses 4-bit palettized
   models? File size drops to ~5MB per model — 4× faster cold load.
2. LoRA fine-tuning first, or speculative decoding first? Lora
   answers a real user question ("can I make it write like me"),
   speculative just makes sampling faster.
3. What's the dataset story for the public gallery v2? Same
   classics, more compute? Or branch out into curated domain
   models (legal text, song lyrics, code in 5 languages)?

The framing has shifted from "feasibility" (proven) to "polish +
positioning." Pick the next 2-3 levers based on what readers
will most easily understand and most readily share.
