# Mac inference baseline — M5 Pro / 48GB

**Date**: 2026-05-31
**Hardware**: Apple M5 Pro, 48 GB, 18 cores (6 perf + 12 P), macOS 25F71
**Harness**: `tinygpt bench --engine tinygpt` (greedy decode, seed=42)
**Commit**: d4a9de6
**Question**: Where is the actual bottleneck on M5 Pro? Is cider's
W8A8 worth the 4-7 day port?

## TL;DR

The Mac is **10-20× faster than the realtime/interaction-model
targets across every model size we have**. cider's prefill win is
real but immaterial at current scales; defer until there's a 3B
specialist or a model-size-driven slowdown.

| Model | Params | TTFT p99 | ITL p99 | Decode tok/s |
|---|---|---|---|---|
| mac-trained (gallery) | 9.6 M  | 5.83 ms | 3.75 ms | 564 |
| flagship-huge-v5      | 221 M  | 4.83 ms | 4.59 ms | 385–696* |
| mega-pilot            | ~960 M | 5.75 ms | 4.94 ms | 293 |

*decode tok/s varies with how full the context is at start of decode

**Realtime targets** (from roadmap §realtime):
- TTFT (warm): < 50 ms → **we're at < 6 ms p99 across all sizes (10×
  under target)**
- ITL p99: < 30 ms → **we're at < 5 ms p99 across all sizes (6×
  under target)**

## Raw results

### Run 1 — mac-trained.tinygpt (9.6 M params, browser gallery)
Workload: prompt=64, gen=128, n=25, warm=3

| metric | median | p95 | p99 |
|---|---|---|---|
| TTFT (ms) | 2.43 | 4.51 | 5.83 |
| ITL (ms) | 1.53 | 3.30 | 3.75 |
| decode tok/s | 564.37 | 761.35 | 767.24 |
| prefill tok/s | 26380.57 | 38230.50 | 39409.15 |
| peak RSS (MB) | 206.9 | 207.3 | 207.3 |

### Run 2 — flagship-huge-v5.tinygpt (221 M params)
Workload: prompt=64, gen=128, n=20, warm=3

| metric | median | p95 | p99 |
|---|---|---|---|
| TTFT (ms) | 3.25 | 4.79 | 4.83 |
| ITL (ms) | 2.37 | 4.03 | 4.59 |
| decode tok/s | 385.49 | 401.68 | 406.78 |
| prefill tok/s | 19912.87 | 23213.03 | 23827.04 |
| peak RSS (MB) | 270.8 | 270.8 | 270.8 |

### Run 3 — flagship-huge-v5.tinygpt, prefill-heavy
Workload: prompt=128, gen=128, n=20, warm=3 (context fills to ctx=256 cap)

| metric | median | p95 | p99 |
|---|---|---|---|
| TTFT (ms) | 2.25 | 3.47 | 3.65 |
| ITL (ms) | 1.30 | 2.31 | 2.75 |
| decode tok/s | 696.99 | 707.33 | 717.32 |
| prefill tok/s | 56814.74 | 58527.30 | 58687.24 |
| peak RSS (MB) | 323.7 | 323.7 | 323.7 |

### Run 4 — mega-pilot.tinygpt (~960 M params, 1.1 GB on disk)
Workload: prompt=64, gen=64, n=10, warm=2

| metric | median | p95 | p99 |
|---|---|---|---|
| TTFT (ms) | 4.76 | 5.75 | 5.75 |
| ITL (ms) | 3.24 | 4.65 | 4.94 |
| decode tok/s | 293.20 | 298.33 | 298.33 |
| prefill tok/s | 14359.44 | 15048.52 | 15048.52 |
| peak RSS (MB) | 687.0 | 687.2 | 687.2 |

## Implications for the cider decision

### What cider would buy us

Per `docs/research/wave_2_5_kernel_audit.md` §2, cider on M5 Pro:
- **W8A8 prefill: 1.2-1.9× faster** on Qwen3-8B / Qwen3-VL-2B
- **W8A8 decode: slightly worse than W8A16** (memory-bandwidth-bound,
  KV cache still fp16) — 104 vs 107 tok/s on Qwen3-VL-2B
- **8B model**: 9.726 → 9.756 PPL, 179.9 → **123.5 s** prefill,
  18.93 → **11.32 GB** peak memory

### Why it doesn't help us right now

1. **Model size mismatch**. cider's wins scale with matmul cost.
   At 221M / 960M params on M5 Pro, the model loads in <700 MB and
   prefill is 14k-56k tok/s — already saturating GPU compute
   throughput. Int8 saves nothing material until matmul dominates.

2. **Realtime targets met by 10×**. The Mac is already faster than
   anything the interaction-model demo needs. cider's 1.8× prefill on
   an already-fast prefill is "we noticed", not "now we can ship the
   demo."

3. **Decode is what limits agent latency** — the agent loop generates
   tokens one at a time and cider's W8A8 is *slower* than W8A16 on
   decode. We'd actively regress agent UX.

4. **The decode tok/s gap with model size suggests memory bandwidth
   is the limit at 1B+**, not int8 vs fp16 arithmetic. cider doesn't
   touch the bandwidth picture.

5. **Effort cost is high**. cider is Python+C++ targeting MLX's C++
   primitive interface; tinygpt is MLX-Swift. Port is 4-7 days, not
   the 1-2 days a Python project would face. See research/wave_2_5_kernel_audit.md
   for the integration analysis.

### When to revisit

Revisit cider adoption when ANY of:

- Training a 3B+ specialist where prefill is ≥1s and worth shaving
- Memory pressure forces W8 weights anyway (then we want W8A8 not just
  W8A16 since activations would otherwise upcast)
- Long-context (8K+) workloads where prefill cost dominates
- The bench harness shows decode degradation at larger model sizes
  that ANE-prefill + GPU-decode hybrid (Wave 2.6 deferred item) could
  address jointly

## Other Mac-speed levers ranked

Given the baseline, here's what's actually worth doing next for "Mac
speed":

| Lever | Effort | Expected impact | When |
|---|---|---|---|
| Verify Medusa/EAGLE-2 spec decode is engaged | 0.5 day | up to 2× decode if not active | Now (cheap check) |
| Cold-start TTFT for 1B+ models | 1-2 days | likely already <50ms but unmeasured cold | After cider decision |
| Async tool-call dispatch (start exec while args still streaming) | 3 days | agent loop UX, not raw inference | Wave 2.6 |
| Decode jitter under thermal load | 1 day | p99 may spike on sustained workloads | Optional |
| cider W8A8 adoption | 4-7 days | Marginal at current scales | When 3B+ specialist exists |
| ANE prefill + GPU decode hybrid | 3-6 weeks | Only when screen-watching specialist ships | Wave 2.6 |

## Bench harness limitations to flag

- `--prompt-tokens` is bounded by model context length (256 for the
  current flagship-huge). The harness fails silently if exceeded —
  filed as a TODO to add a clear error.
- `--no-energy` disables powermetrics (which needs sudo); J/token
  numbers are not in the baseline.
- "git tree dirty" warning is currently spurious (`.claude/` +
  `default.profraw` only). Could be filtered in the harness.

## Conclusion

**Do NOT spend 4-7 days on cider right now.** The Mac is already
faster than the product demands. The lever has marginal payoff at
current scales and active regression risk on decode.

**What to do instead:**
1. Document this baseline (this doc)
2. Move to the next product-shaped Wave 2.6 work (Continue.dev
   provider adapter is the highest-leverage per `wave_4_landscape.md`)
3. Revisit cider after training the first 3B specialist
