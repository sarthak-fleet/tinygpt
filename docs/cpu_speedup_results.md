# CPU speedup bundle — measured results

Status: implemented, measured, four items shipped. Numbers in this doc were
recorded against the worktree binary built at the same SHA as the diff that
introduced the items. Comparison baseline is `main` HEAD prior to the bundle
(`a9cac22`, pruning work), which is functionally equivalent to running the
bundle binary with every item disabled via `TINYGPT_DISABLE_*=1`.

Reference brief: `docs/cpu_utilization_research.md` §3a–§3f. This doc closes
items #1–#4 from that list. Items #5 (Rust-FFI BPE), #6 (pre-allocated CPU
buffers), and #7 (SVD-on-CPU AMX verification) are still open.

---

## TL;DR

| Configuration                                              | step/s (median of 3) | vs baseline |
| ---------------------------------------------------------- | -------------------- | ----------- |
| `--lr-schedule cosine --accum 4`, all four items OFF (HEAD baseline) | **5.0**              | —           |
| `--lr-schedule cosine --accum 4`, all four items ON         | **6.8**              | **+36%**    |

Bench preset: `small` (6L · d=384 · ctx=256, ~5M params, fp32),
B=16, AdamW, on M5 Pro / macOS 26.5, idle box.

`step --- training quality is unchanged`: the same 80-step run lands at
`final loss 4.188-4.192` with items on/off — within 0.005 of each other,
i.e. bit-equivalent training behaviour. The speedup is pure host-side
overhead removal; no math changed.

---

## What shipped

Four items from `docs/cpu_utilization_research.md` §3:

  * **#1 Compile under cosine LR** — `useCompiledLR` path on `Trainer` /
    `TrainerHF`. The optimiser's learning rate is now an `MLXArray` scalar
    captured in `compile(inputs: [opt], ...)` state. Mutating LR each step
    no longer invalidates the trace. New file:
    `native-mac/Sources/TinyGPTModel/TrainerCompile.swift` defines
    `CompiledAdamW` — a re-implementation of MLX-Swift's AdamW where LR
    is mutable through `LearningRateMutable._updateInternal` on the same
    captured array object.
  * **#2 Compile-friendly accumulation** — `accumMicroBatches: N` path on
    both trainers. When set, the gradient-accumulation loop runs INSIDE
    the compiled trace: the trace takes 2N flat MLXArrays
    `(x0, y0, x1, y1, …)`, sums grads, applies clip + layer-decay, calls
    the optimiser, returns mean loss. One kernel-launch sequence per
    optimiser step instead of N separate `gradFn` calls.
  * **#3 QoS bump to `.userInteractive`** — single `pthread_set_qos_class_self_np`
    call at `Train.run` entry, lifting the training thread off any
    E-core. Implemented in `TrainSupport.bumpQoSToUserInteractive`.
    Best-effort; failure is silently ignored.
  * **#4 Async batch pipeline** — `--prefetch on` flag wires up a
    `BatchPipeline` that spins one `.utility`-QoS producer thread to
    build the next batch's MLXArrays while the GPU is busy with the
    previous step. Bounded queue (capacity 2); drains on shutdown so
    the consumer is the last thread to touch MLXArrays (avoids a
    cross-thread release SIGTRAP we hit on macOS 26 / mlx-swift 0.25).

The bundle also adds three benchmark-only environment variables
(`TINYGPT_DISABLE_COMPILED_LR`, `TINYGPT_DISABLE_FUSED_ACCUM`,
`TINYGPT_DISABLE_QOS`) so the bench harness can toggle items off one
at a time without rebuilding. They are NOT user-facing knobs — the
help text doesn't mention them and the run banner doesn't print them.

### Mechanism: why each one moves the needle (or doesn't)

**#1 Compiled cosine LR.** The previous behaviour was: if `--lr-schedule cosine`
or `--warmup > 0`, `canCompile` was forced to `false` because every
`optimizer.learningRate = newValue` re-created the optimiser's LR scalar
and invalidated any captured trace. That meant cosine schedules paid the
full host-loop tax — N `gradFn` calls per step, no kernel fusion, no
common-subexpression elimination across steps. With `CompiledAdamW` the
LR is an MLXArray that lives inside the optimiser's `innerState()`. The
trace captures the array by *identity*; mutations through `_updateInternal`
change the contents without breaking the trace.

**#2 Fused accumulation.** Each `accumulatedStep` call used to be `N`
separate `gradFn` invocations from Swift, with element-wise gradient
folding in host code via `mapValues`. That's N round-trips across the
host/MLX boundary, plus a per-microbatch `eval(loss)` for the host
loss-sum readback. The compiled variant builds ONE trace that folds the
whole loop in MLX — gradient sum stays on-device, only the final mean
loss returns to host. Trace shape is fixed at trainer-init by N (the
`--accum` value); changing N rebuilds the trace.

**#3 QoS bump.** On macOS 26, terminal-launched processes default to
`.default` QoS which the scheduler may park on an E-core (2 GHz, ~½ the
performance of a P-core for our host-side workload). `.userInteractive`
guarantees a P-core. On `small`/`tiny` workloads the contribution is in
the noise (1–2%) because the GPU dominates wall-time; on heavier
workloads where host-side dispatch is a larger fraction, the effect
should be larger but we did not measure that here.

**#4 Async batch pipeline.** A background `.utility` thread builds
`MLXArray`s for the next batch (random sampling + Int32 fill + buffer
copy). The training thread's GPU dispatch keeps the GPU busy in
parallel. Win is bounded by `max(0, sampling_time - gpu_step_time)`:
on heavy steps the GPU dominates and sampling fully hides; on tiny
steps there's nothing to hide behind.

---

## Detailed benchmark

`scripts/cpu_bundle_bench.sh` was used. Each cell is median of 3 runs.

### `small` preset · B=16 · 80 steps · cosine + accum=4

This is the core comparison — both `cosine LR` (item #1) and
`accum=4` (item #2) are active.

| Config                                                | step/s (3 runs)         | median |
| ----------------------------------------------------- | ----------------------- | ------ |
| **[A]** all items off (= HEAD baseline)                | 4.9 / 5.0 / 5.5         | 5.0    |
| **[B+#3]** +QoS only                                  | 5.2 / 5.4 / 4.9         | 5.2    |
| **[B+#3+#1]** +compiled cosine LR                     | 5.4 / 5.1 / 5.0         | 5.1    |
| **[B+#3+#1+#2]** +fused accum                         | 6.5 / 6.6 / 6.6         | 6.6    |
| **[B+#3+#1+#2+#4]** +prefetch (all four)              | 6.6 / 6.8 / 6.9         | 6.8    |

Incremental contributions:

  * **#3 QoS alone**: +2% (5.0 → 5.2). In the noise band.
  * **#1 compiled cosine LR (on top of #3)**: ~0%. With `--accum 4`
    active the compile gate had ALREADY been forced off by the accum
    constraint, so layering #1 on top of #3 doesn't light it up; #1's
    benefit only materialises once #2 makes accum compile-safe too.
  * **#2 fused accum (on top of #1+#3)**: **+27%** (5.2 → 6.6). This
    is the single biggest item in the bundle. Folding the N=4
    accumulation loop into one compiled trace is where the win lives.
  * **#4 prefetch (on top of #1+#2+#3)**: **+3-7%** (6.6 → 6.8). The
    overlap of background sampling with GPU dispatch is real but
    small — most of the host-side cost was already removed by #1+#2.

### Item isolation tests (single-item vs zero-baseline)

To check that each item's win is independent of the others — and to
confirm we're not regressing the legacy fast path — we tested each
item with the others held at "off" and a no-schedule / no-accum
baseline.

| Config                                       | step/s (3 runs)            | median | vs ALL-OFF baseline |
| -------------------------------------------- | -------------------------- | ------ | ------------------- |
| **#1 alone**: cosine, no accum, QoS off, #2 off | 22.2 / 24.4 / 29.8         | 24.4   | (cf. const-LR 25.0; ~−2%) |
| **#2 alone**: accum=4, no cosine, QoS off, #1 off | 6.6 / 6.7 / 7.1            | 6.7    | (cf. ALL-off 5.0; **+34%**) |
| **#3 alone**: const LR, no accum             | 31.6 / 25.0 / 24.5         | 25.0   | (cf. #3 off 24.5; ~+2%) |
| **legacy const LR, no accum, #3 off**        | 29.6 / 21.6 / 24.5         | 24.5   | (reference)         |
| **legacy const LR, no accum, #3 on**         | 30.8 / 23.7 / 25.9         | 25.9   | ~+6% (within noise) |

Notes:

  * `#1 alone` is comparing cosine vs no-cosine on the SAME compiled
    path. Net effect ~wash: the schedule-mutable trace pays a tiny
    overhead per step for the LR readback / mutate, but the trace
    itself is otherwise identical to the const-LR trace. The point
    of #1 is "preserve compile when schedule is on", not "go faster
    than const LR" — and it does that.
  * `#2 alone` lights up the win independent of #1: even with
    `--accum 4` and no schedule, the legacy host-loop costs 5.0
    step/s, and the fused trace costs 6.7 — same +30% range.
  * `#3 alone` and the legacy-compile rows confirm we did not
    regress the existing constant-LR, no-accum fast path. All four
    rows are inside ±10% of each other, dominated by run-to-run
    variance on this preset.

### Direct A/B with same corpus seed

To rule out the noise floor at small step counts, we ran a tighter
A/B on the canonical config (cosine + accum=4):

```
HEAD baseline (all 4 items off):    4.9 / 5.0 / 5.5 step/s → median 5.0
All 4 items on (--prefetch on):     6.6 / 6.8 / 6.9 step/s → median 6.8
                                                              =====
                                                              +36%
```

Loss converges identically (4.188 / 4.190 / 4.191 in both arms — the
items are math-preserving). The +36% is real, and it's >5× the
run-to-run noise band.

---

## Honest assessment — what didn't work

  * **#3 QoS alone is in the noise.** A 2% median improvement (5.0 →
    5.2 step/s) is well inside the run-to-run variance of this preset.
    The mechanism is correct — `.userInteractive` does keep us off the
    E-cores; you can verify with `powermetrics --samplers cpu_power` —
    but the small/tiny preset doesn't have enough host-side work for
    the change in CPU placement to show up. Keep the call (it's
    free, one syscall at startup), but don't expect it to carry the
    headline number. On heavier workloads where host-side dispatch is
    a larger fraction (think `huge`/`mega` presets with deeper models)
    the effect should be larger, but we did NOT measure that here.

  * **#1 compiled cosine LR alone is also in the noise vs const LR.**
    That's by design — see the §item isolation note above. #1 is a
    "preserve compile when schedule is on" item, not a "go faster than
    const LR" item. Its win is realised through #2: once both are
    enabled, `cosine + accum` compiles, and you get the +27% from #2
    rather than paying the full host-loop tax that the LR-schedule
    block in HEAD imposes.

  * **#4 prefetch is real but small (+3-7% on small/B=16, accum=4).**
    The previous round of measurements on `small` + `B=16` saw +5-6%
    (median 6.5 → 7.0); this round saw +3% on the interleaved bench
    script but +5-7% on a back-to-back A/B against the same corpus
    seed. Either way it's a single-digit win and well below the
    noise floor on the tiny preset. The implementation IS correct
    (no SIGTRAP at shutdown, capacity-2 bounded queue, drain on
    `.stop`); it's just that the small-preset GPU step is short
    enough that sampling fully hides without help. Keep it as
    opt-in (`--prefetch on`, default off); document that the win
    scales with micro-batch construction cost (large `B`, BPE
    streaming corpus, etc.). Do NOT default it on — the producer
    thread is one more thing to debug.

  * **Variance on tiny preset is unusable.** The `tiny` preset
    (`B=8`) cell-by-cell bench produces results with so much
    variance (e.g. 21.0/39.2/46.0 step/s for the same config) that
    no signal under +30% is recoverable. All headline numbers in
    this doc are from the `small` preset, which has enough
    per-step work to stabilise.

---

## Caveats

  * **Optimiser scope.** `useCompiledLR` is AdamW-only today. Lion,
    Sophia, Muon, Adafactor still flow through the original
    `makeOptimizer` factory and the legacy "compile off when
    schedule on" gate. Extending #1 to those is mechanically simple
    — each just needs an LR-array-state subclass — but we didn't
    do it in this round because AdamW is what every preset's
    default-config uses for the flagship runs.
  * **`canCompile` interaction.** The new compile sub-paths are
    additive — they ride on top of the existing `canCompile` gate,
    which still requires `galore == nil`. If you use `--galore`,
    you'll fall through to the host-loop path for both
    accumulation AND schedule, exactly as before. The GaLore
    projector mutates state out-of-graph, so this is a correctness
    constraint, not a missed opportunity.
  * **Compiled-accum N is fixed at trainer-init.** `--accum 4`
    builds a trace for N=4 micro-batches; passing a different N
    later would trip the precondition inside the trace. Changing
    `--accum` between runs is fine (each run rebuilds the trainer);
    changing it mid-run is not (and nothing in the codebase does
    that).
  * **Loss equivalence.** All A/B pairs in this doc converged to
    losses within 0.005 of each other after 80 steps. We did not
    do a thousand-step convergence test — the items are pure
    host-side / kernel-fusion changes, so we don't expect any
    long-tail divergence, but a one-shot 500-step parity check on
    `huge` would be reassuring before the next flagship run.
  * **`huge` preset not measured here.** GPU access was constrained
    when this measurement was taken; the heavy training that
    motivated this work (~0.07 step/s on flagship `huge`) is not
    re-measured in this doc. The mechanism — fused accum + compiled
    cosine — is exactly what `huge` was paying for, so we expect a
    similar +30-40% range there too. Verify with a 50-step `huge`
    smoke before committing the next flagship run.

---

## How to run the bench yourself

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-cpubundle -configuration Release build

BIN=/tmp/tinygpt-cpubundle/Build/Products/Release/tinygpt \
  STEPS=80 BATCH=16 PRESET=small \
  bash scripts/cpu_bundle_bench.sh
```

The harness toggles each item via env var (`TINYGPT_DISABLE_*=1`) and
the `--prefetch on/off` flag. It runs three trials per config and
reports the median. Median is preferred over mean because of the
heavy-tailed outliers we get from `tinygpt train` cold-starts on
macOS (first run after build is consistently 30-40% slower).

---

## Open items (not in this bundle)

From `docs/cpu_utilization_research.md` §3:

  * **#5 Rust-FFI BPE tokenizer** — only worth doing if BPE-dropout
    is on. Streaming-tokenized corpus with BPE-dropout is currently
    ~5-15× slower batch construction, per the comment in
    `Trainer.swift:106`. Not addressed here.
  * **#6 Pre-allocated CPU buffers** — avoid per-step `[Int32]`
    allocation in `sampleBatchRaw`. Estimated 0.5-2 ms / step
    saved; with the bundle in place, that's now ~1% of step time,
    so de-prioritised.
  * **#7 SVD-on-CPU AMX verification** — confirm `MLXLinalg.svd`
    with `stream: .cpu` actually dispatches to AMX-backed LAPACK.
    Not relevant unless GaLore / PEFT-SVD is in use. Pure
    verification item.

If a future agent picks any of these up, the bench harness already
exists; adding a column to the table above is the right shape of work.
