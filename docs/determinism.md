# Determinism contract

`tinygpt train --seed <UInt64>` seeds **MLXRandom** before any model
construction. This makes the following reproducible across runs:

- Model parameter initialization (He / Xavier / etc. — all sample from MLXRandom)
- GPU-side dropout, embedding noise (NEFTune), and any other MLX-sourced
  randomness inside the forward/backward pass
- Any MLX random op invoked between `seed()` and the next call to it

## What is NOT yet covered (v1 limitation)

Batch sampling in `ByteCorpus.sampleBatchRaw` and
`TokenizedCorpus.sampleBatchRaw` uses Swift's stdlib
`Int.random(in:)`, which routes through `SystemRandomNumberGenerator` and
is **not seedable**. So two runs with the same `--seed` will:

- Initialize identical weights ✅
- See different training batches ❌

Empirically this produces step-1 losses within ~1% of each other (same
init weights, different randomly-sampled context windows) and final
losses that converge to similar but not bit-identical values.

## Verifying determinism

The simplest manual check is two runs side-by-side:

```bash
tinygpt train --preset tiny --steps 3 --seed 42 --no-spike-detect \
  --corpus data/examples/tiny-corpus.txt --out /tmp/det-A.tinygpt
tinygpt train --preset tiny --steps 3 --seed 42 --no-spike-detect \
  --corpus data/examples/tiny-corpus.txt --out /tmp/det-B.tinygpt
```

Step-1 losses should agree to within roughly the variance of one
sampled batch.

## Roadmap to full bit-exact replay (v2)

To make step-1 losses bit-identical across runs we need to seed the
**host** RNG used by `sampleBatchRaw` too. Plan:

1. Add a deterministic `RandomNumberGenerator` (e.g., a small
   xoshiro256** implementation) to `TinyGPTModel` as a `public struct`.
2. Thread an `inout generator` parameter through every `sampleBatchRaw`
   variant.
3. Seed it from `--seed` alongside MLXRandom in `Train.run`.
4. Decide on prefetcher behaviour: either disable prefetch when a seed
   is set (cheapest), or give the prefetcher its own deterministic
   stream advanced by `(step, lane)`.

Until v2 lands, `--seed` covers model init and on-device randomness;
batch ordering remains the source of run-to-run drift.

## Where this matters

- **Spike investigations.** A reproducible init means you can re-run
  the same configuration to see whether a loss spike is intrinsic
  (recurs every run) or sampling-driven (occurs in one). See
  `--no-spike-detect` and `--spike-window` / `--spike-factor` flags
  on `tinygpt train`.
- **A/B sweeps.** When comparing `--lr-schedule cosine` vs `wsd` or
  two `--depth` values, fixing `--seed` removes one source of variance.
- **Crash recovery.** `--resume <path.tinygpt>` already restores
  weights; resume + the same `--seed` gets you as close to "continue
  the exact same run" as v1 supports.

See `docs/PLAN.md` §3 C9 for status. v2 work is queued; not currently
in progress.
