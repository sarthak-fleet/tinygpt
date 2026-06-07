# vllm-mlx Investigation

Date: 2026-06-06

Recommendation: partial-proceed. Do not wrap `vllm-mlx` into `tinygpt serve`
yet. Keep native serve as the default and as the eval/logprob backend. Treat
`vllm-mlx` as a production-chat candidate after an approved local install and
benchmark pass.

## Sources Checked

- Repo: https://github.com/waybarrios/vllm-mlx
- Server guide: https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/server.md
- CLI reference: https://github.com/waybarrios/vllm-mlx/blob/main/docs/reference/cli.md
- Continuous batching guide: https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/continuous-batching.md
- LLM benchmarks: https://github.com/waybarrios/vllm-mlx/blob/main/docs/benchmarks/llm.md
- Local tinygpt serve: `native-mac/Sources/TinyGPTServe/Serve.swift`
- Local safetensors exporter: `native-mac/Sources/TinyGPT/ToSafetensors.swift`

## 1. Maturity

The project is real and active enough to watch:

- Public Apache-2.0 repo, Python package, Apple Silicon target.
- GitHub API on 2026-06-06 showed 1,302 stars, 182 forks, 67 open
  issues/PRs, and latest `main` HEAD `015e080ff2c2ab95fb4368c736a12f58bcde63a9`.
- Latest pushed commit was 2026-05-31.
- Repo docs advertise OpenAI and Anthropic APIs, continuous batching,
  paged KV cache, prefix cache, model acquisition/conversion, and
  `bench-serve`.

But it is not mature enough to become the default tinygpt serving backend:

- `pyproject.toml` marks it as `Development Status :: 3 - Alpha`.
- Recent open issues include continuous-batching crashes and MLX stream
  failures. Examples checked from the open issue feed:
  - `#592` reports `NoneType` iteration crashes under continuous batching.
  - `#593` reports `/v1/chat/completions` returning 500 because generation
    runs on a thread without the expected MLX stream context.
- It is moving quickly, which is good for upstream velocity but bad for a
  stable wrapper contract today.

Decision: do not default to it. If used, keep it opt-in and isolated from
eval infrastructure.

## 2. Model Formats

vllm-mlx expects an MLX/HuggingFace-style model identifier or local model
directory:

- `vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit`
- `vllm-mlx serve /path/to/local/model`
- `vllm-mlx model convert <hf-model> --output <dir> ...`

It does not load `.tinygpt` checkpoints directly.

TinyGPT already has `tinygpt to-safetensors`, but that currently writes a
single `model.safetensors` with HF-Llama-ish names. That is not yet a complete
MLX/HF model directory contract for vllm-mlx because serving needs compatible
`config.json`, tokenizer files, and architecture expectations. TinyGPT's
from-scratch byte-level models also do not map cleanly to an upstream HF model
family without a wrapper config.

Decision: a direct `.tinygpt -> vllm-mlx` adapter is not a small flag. The
minimum honest bridge is:

1. Export weights.
2. Emit a compatible model directory (`config.json`, tokenizer, generation
   config, `model.safetensors`).
3. Verify `vllm-mlx serve <dir>` loads it.

Until that exists, vllm-mlx can serve external MLX/HF models, not arbitrary
TinyGPT checkpoints.

## 3. OpenAI Surface Parity

vllm-mlx documents:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- streaming chat/completion responses
- Anthropic `/v1/messages`
- embeddings, health, metrics, status

That is good enough for production chat clients.

The critical eval question is `logprobs + echo` for
lm-eval-harness/local-completions. This remains unproven. The server docs and
CLI reference checked today document chat/completions and streaming, but do not
document the exact completions `echo: true` plus `logprobs` behavior that
TinyGPT's native `Serve.scoreLogprobs` supports for eval scoring.

Decision: keep native serve as the eval backend. Even if vllm-mlx is adopted
for production chat, eval code should continue to route through native serve
until `logprobs + echo` is curl-verified.

## 4. Performance Reality Check

No local benchmark was run in this pass. Installing vllm-mlx and benchmarking
it against a checkpoint would require Python package installation, model
download/conversion, and sustained inference loops. The repo's macOS safety
rules require explicit approval before that class of workload.

Upstream benchmark data is still useful for triage:

- M4 Max results claim 402.3 tok/s for Qwen3-0.6B-8bit and 463.6 tok/s for
  Llama-3.2-1B-Instruct-4bit.
- Continuous batching docs claim 1.5x to 3.39x throughput improvement for
  five concurrent requests depending on model.
- Paged-cache docs show workload-dependent gains: meaningful memory savings,
  modest or no throughput gain in some M1 Max paged-cache tests.

Decision: performance upside is plausible, but not proven for TinyGPT. The
next step is an approved local benchmark, not code integration.

## Recommendation

Partial-proceed:

1. Do not modify `tinygpt serve` yet.
2. Keep native serve as the default and as the eval backend.
3. Add a future integration only after a local benchmark answers:
   - Can vllm-mlx load a TinyGPT-exported model directory?
   - Does `/v1/completions` support `echo + logprobs` well enough for
     lm-eval-harness?
   - What are throughput, startup time, and memory deltas on the same Mac and
     same prompts?
4. If the local run passes, implement `--backend native|vllm-mlx` as an
   explicit opt-in. Do not switch defaults until stability data exists.

## Proposed Local Benchmark Plan

Run only with explicit approval:

```bash
uv tool install vllm-mlx
vllm-mlx --help
```

Then use a small external MLX model first:

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --port 8001 --continuous-batching
vllm-mlx bench-serve --url http://127.0.0.1:8001 --prompts short --concurrency 1,4 --max-tokens 64 --format json
```

Compare with native serve on an existing TinyGPT checkpoint using the same
prompt set and max-token cap. Stop all spawned servers immediately after the
run.
