# Roadmap — Tier 4 (skip — not worth it for us)

These are deliberately not built because better alternatives exist.
Distinct from [`blockers.md`](blockers.md), which lists things we
**want** but can't currently build.

- **fp16 mixed-precision training** — bf16 is strictly better
  (already shipped, see [`docs/memory_tradeoffs.md`](../memory_tradeoffs.md)).
- **ZeRO / FSDP / pipeline parallelism** — multi-device only.
- **State space models (Mamba, RWKV)** — different architecture
  entirely; ~2-3 week port; better as a side project.
- **PagedAttention / continuous batching** — multi-user inference.
- **Tree attention / lookahead decoding** — marginal over
  speculative decoding.
- **Adapter modules (Houlsby/Pfeiffer)** — LoRA's older cousin,
  superseded.
- **BitFit** — train biases only; quality is poor.
- **Hyena / long-conv** — different architecture.
- **fp8 training** — needs H100/Blackwell hardware.
