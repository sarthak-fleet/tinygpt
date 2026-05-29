# State of LLM Inference Benchmarks (May 2026)

*Research compiled by an Explore subagent on 2026-05-29 to cover the gap
between my Jan 2026 knowledge cutoff and current state. Includes URLs for
verification.*

## 1. Suite landscape

- **MLPerf Inference v6.0** (April 2026) is the gold standard for credibility.
  Five of eleven datacenter tests are new/updated: text-to-video, **GPT-OSS
  120B**, vision-language models, DLRMv3, YOLOv11. v5.1 (Sept 2025) added
  DeepSeek-R1 reasoning, Whisper-v3 speech, Llama-3.1-8B small-LLM tracks.
  Closed Division audits weights/precision; Open allows quantization tricks.
  Record 27 submitters in v5.1.
- **vLLM benchmark / NVIDIA GenAI-Perf / llmperf (Anyscale)** remain the
  de-facto load generators but are not normalized across vendors.
- **Bench360** (arXiv 2511.16682, Nov 2025) — newest credible academic suite
  for *local* inference. Modular: task engine + workload controller
  (single-stream, batch, server) + backend abstraction (TGI, vLLM, SGLang,
  LMDeploy) + metrics collector incl. energy. Most directly relevant prior
  art to what we're building.
- **TokenPowerBench** (arXiv 2512.03024, Dec 2025, AAAI) — first open
  framework specifically for energy/power at prefill vs decode, GPU/node/
  system levels. Reports energy/token scales ~7.3x from 1B→70B (vs 70x param
  growth).
- **Long-context**: **RULER v2** (OpenReview ZU9tRffRSA, 2025) extends
  original 13-task RULER with reasoning categories; **LongBench v2** (ACL
  2025) — 503 multi-choice items, 8k–2M words, best non-reasoning model =
  50.1%, o1-preview = 57.7%. **MRCR** is the third pillar of the Awesome
  Agents long-context leaderboard.
- **HELM Efficiency** continues but has lost mindshare to MLPerf+Bench360
  for systems work.

## 2. Apple Silicon specifically

There is **no MLPerf-equivalent for M-series** yet — this is a real gap.
The closest credible artifacts:

- **arXiv 2511.05502** "Production-Grade Local LLM Inference on Apple
  Silicon" (Nov 2025) — comparative study of MLX, MLC-LLM, Ollama,
  llama.cpp, vLLM-MLX. Probably the most-cited third-party Mac comparison
  right now.
- Community numbers converge on: **MLX leads llama.cpp by 20–87% under
  ~14B**, gap collapses at 27B+ where 546 GB/s bandwidth on M4 Max
  saturates both. M4 Max ~70 tok/s on 70B Q4.
- **Orion** project (referenced in 2511.05502) — first open ANE programming
  via private APIs; reported 170+ tok/s on GPT-2 124M on M4 Max. Neither
  MLX nor llama.cpp currently uses ANE; CoreML can but size-caps make it
  impractical >7B.
- MLX-LM and llama.cpp publish numbers in GitHub Discussions
  (ggml-org/llama.cpp #4167) but these are user-submitted and inconsistent
  on prompt length, batch size, thermal state.

## 3. Metrics that matter

Standard set to report: **TTFT, ITL/TPOT, decode tok/s at batch=1/4/16/64,
prefill tok/s, peak RSS + unified memory high-water mark, sustained tok/s
under thermal load** (Mac-specific!), **energy per output token (J/tok)**
via `powermetrics`, model coverage matrix (dense + MoE + quantization).
Long-context: report at 4k/32k/128k/1M with cache-hit and cache-miss
separately.

## 4. Reproducibility bar

For "we beat X by Y%" to survive review: pin exact engine commit hashes,
model SHA, quantization scheme, KV-cache dtype, seed, sampling params
(temp, top-p), prompt corpus (ShareGPT-v3 or LMSYS-Chat-1M are
conventional), batch/concurrency profile, hardware SKU + RAM tier +
macOS build + thermal state (cold vs steady-state, ambient temp), ≥3
runs report median + p95/p99. MLPerf-style **submitter README + log
replay** is the gold bar. Bench360 ships YAML configs you can copy.

## 5. Publishable gaps

Real opportunities — none of these are well-measured publicly as of
May 2026:

1. **ANE utilization during serving** — no public benchmark reports ANE
   residency %, ANE↔GPU handoff latency, or ANE energy/token. Orion is
   a proof-of-concept, not a benchmark.
2. **Energy/token on Apple Silicon** — TokenPowerBench is NVIDIA-only;
   Bench360's energy module doesn't cover unified-memory `powermetrics`
   semantics.
3. **Prompt-cache hit-rate-dependent TTFT** on-device — production
   servers report ~90% KV-cache hit rates (llm-d data); no Mac suite
   varies hit-rate as an axis.
4. **Sustained vs burst** — fan-curve / thermal-throttle behavior over
   10-minute serving windows.
5. **MoE on unified memory** — expert-cache thrash on M-series is
   unmeasured (arXiv 2604.18788 on MoE+NPU is the closest).

## Sources

- [MLCommons MLPerf Inference v6.0](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/)
- [MLPerf v5.1 small-LLM (Llama-3.1-8B)](https://mlcommons.org/2025/09/small-llm-inference-5-1/)
- [HPCwire MLPerf v5.1 recap](https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/)
- [Bench360 paper (arXiv 2511.16682)](https://arxiv.org/abs/2511.16682) / [GitHub slinusc/bench360](https://github.com/slinusc/bench360)
- [TokenPowerBench (arXiv 2512.03024)](https://arxiv.org/html/2512.03024v1)
- [Production-Grade Local LLM Inference on Apple Silicon (arXiv 2511.05502)](https://arxiv.org/pdf/2511.05502)
- [LongBench v2 (ACL 2025)](https://aclanthology.org/2025.acl-long.183/) / [arXiv 2412.15204](https://arxiv.org/abs/2412.15204)
- [RULER v2 (OpenReview)](https://openreview.net/pdf?id=ZU9tRffRSA)
- [Long-Context Leaderboard (MRCR/RULER/LongBench v2)](https://awesomeagents.ai/leaderboards/long-context-benchmarks-leaderboard/)
- [llama.cpp Apple Silicon performance thread #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [2026 Mac inference framework comparison (MACGPU)](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)
- [llm-d intelligent inference scheduling (Apr 2026)](https://medium.com/@yakovbeder/llm-d-the-inference-scheduler-that-fixes-what-more-gpus-cant-03644ac55504)
- [Don't Break the Cache — prompt caching for agentic tasks (arXiv 2601.06007)](https://arxiv.org/html/2601.06007v1)
- [NVIDIA LLM benchmarking fundamentals](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [MoE LLM inference with Apple Silicon NPUs (arXiv 2604.18788)](https://arxiv.org/html/2604.18788v1)
