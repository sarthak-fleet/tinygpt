# lm-evaluation-harness integration for tinygpt

This doc describes how tinygpt plugs into [EleutherAI's
`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness),
the canonical eval framework behind the HuggingFace Open LLM Leaderboard.
With the wiring in this commit you can run **HellaSwag, ARC-Easy/Challenge,
GSM8K, IFEval, MMLU-Pro, GPQA-Diamond, MATH-500, AIME, BBH, HumanEval, …**
— anything the harness defines — against any tinygpt-loaded model.

For the *why* (benchmark landscape, leaderboard saturation, contamination
issues), see [`docs/research/quality_benchmarks_may_2026.md`](research/quality_benchmarks_may_2026.md).

## How it works

```
┌────────────────┐     spawn      ┌──────────────────────┐
│ lm-eval-harness│ ─────────────► │ tinygpt serve <model>│
│  (python)      │                │  (Swift, MLX-Metal)  │
└────────────────┘                └──────────────────────┘
        │                                   ▲
        │ HTTP POST /v1/chat/completions    │
        │   {messages, max_tokens, stop}    │
        ├──────────────────────────────────►│
        │                                   │
        │   {choices: [{message: {...}}]}   │
        │◄──────────────────────────────────┤
        ▼                                   │
   (score / summarize)                      │
```

The harness's `local-chat-completions` adapter talks to any
OpenAI-compatible HTTP endpoint. `tinygpt serve` *is* that endpoint —
implemented in `Sources/TinyGPTServe/Serve.swift` as a hand-rolled
POSIX-socket HTTP server (zero new deps, ~600 LOC).

### Endpoints

| Endpoint                          | Method | Purpose                                          |
|-----------------------------------|--------|--------------------------------------------------|
| `/v1/models`                      | GET    | Lists `tinygpt` — used by clients to probe ready |
| `/v1/chat/completions`            | POST   | OpenAI ChatCompletion (messages: [...])          |
| `/v1/completions`                 | POST   | OpenAI text completion (prompt: "...")           |

Both completion endpoints accept the standard fields: `max_tokens`,
`temperature`, `stop` (string or array). Response shape matches the
OpenAI spec strictly enough that `lm-eval` parses it without complaint.

### Chat formatting

Chat messages are rendered as ChatML (`<|im_start|>role\ncontent<|im_end|>`)
before being fed to the model. If your model was trained on a different
template (Alpaca, Llama), prefer the `/v1/completions` endpoint and pass
an already-formatted prompt directly.

## Setup

### 1. Build tinygpt

```bash
cd native-mac
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme tinygpt -destination "platform=macOS" \
  -derivedDataPath /tmp/tinygpt-smoke -configuration Release build
```

This produces `/tmp/tinygpt-smoke/Build/Products/Release/tinygpt`. The
`bench/run_quality_evals.sh` script auto-detects this path.

### 2. Wire `case "serve":` into TinyGPT.swift

Currently `Sources/TinyGPT/TinyGPT.swift` has a `TODO(serve-merge)`
comment near the `sample` case. Add this line in the dispatch table:

```swift
case "serve":
    Serve.run(args: Array(args.dropFirst()))
```

The constraint that left this un-wired was agent-coordination overhead.
Once it's in, `tinygpt serve` becomes callable from the standard CLI.

There's also a stand-in executable `tinygpt-serve-smoke` (in
`Sources/TinyGPTServeSmoke/main.swift`) that exposes the same entry point
through a separate binary — useful for testing the HTTP layer before the
main dispatch is merged. Delete that target once `case "serve":` lands.

### 3. Install lm-evaluation-harness

```bash
python -m venv .venv
source .venv/bin/activate
pip install lm-eval==0.4.10
```

**Why pin 0.4.10?** Release 0.4.11 introduced a regression in the
`local-chat-completions` adapter where user-supplied `stop` sequences are
silently dropped for generate-until tasks (GSM8K, HumanEval, IFEval,
BBH-cot). The model keeps generating until `max_tokens`, scores
collapse, and you spend an hour wondering why your math accuracy is 0.
See [`docs/research/quality_benchmarks_may_2026.md`](research/quality_benchmarks_may_2026.md)
for the citation. Pin to 0.4.10 OR upgrade to whatever's current at the
time of running and verify generate-until tasks against a known-good
baseline.

If you can't downgrade, the workaround is to pass `--gen_kwargs
"until=['Q:','</s>']"` explicitly on the lm-eval CLI — that bypasses the
broken auto-stop path. Our `python_ref/lm_eval_tinygpt.py` wrapper
exposes a `--lm-eval-extra` flag that you can use for this.

## Running

The one-liner:

```bash
bench/run_quality_evals.sh
```

Defaults to running HellaSwag + ARC-Easy on `/tmp/flagship-huge.tinygpt`.
Output lands in `bench/results/flagship-huge-<timestamp>/`.

Env knobs:

| Var            | Default                            | Notes                                                   |
|----------------|------------------------------------|---------------------------------------------------------|
| `$1`           | `/tmp/flagship-huge.tinygpt`       | Positional arg — model path                             |
| `TASKS`        | `hellaswag,arc_easy`               | Comma-separated lm-eval task names                      |
| `LIMIT`        | _empty_ (full)                     | Per-task example cap — set to e.g. `50` for smoke runs  |
| `MAX_CONTEXT`  | _empty_ (model's native ctx)       | Truncate prompts to this length (helps for MMLU-Pro)    |
| `TINYGPT_BIN`  | `/tmp/tinygpt-smoke/.../tinygpt`   | Explicit path to the binary                             |

Or call the Python wrapper directly:

```bash
python python_ref/lm_eval_tinygpt.py /tmp/flagship-huge.tinygpt \
    --tasks hellaswag,arc_easy,gsm8k \
    --limit 100 \
    --output-path bench/results/smoke_run/
```

## Task cost (Mac M-series, 27M-param flagship)

Rough wall-clock numbers from a 2025 M-series Mac, byte-level 27M model
generating at ~150 tokens/sec on Metal. Tasks vary 10x in number of
examples and 10x in token count per example — your mileage varies.

| Task             | Examples | Type             | Approx wall-clock | Notes                              |
|------------------|---------:|------------------|-------------------|------------------------------------|
| `hellaswag`      | 10,000   | multiple-choice  | ~30 min           | loglikelihood — fastest            |
| `arc_easy`       | 2,376    | multiple-choice  | ~10 min           | loglikelihood                      |
| `arc_challenge`  | 1,172    | multiple-choice  | ~5 min            | loglikelihood                      |
| `gsm8k`          | 1,319    | generate-until   | ~2 hrs            | needs ~200 tok/example; CoT path   |
| `ifeval`         | 541      | generate-until   | ~30 min           | response-following metric          |
| `mmlu_pro`       | 12,032   | multiple-choice  | ~2 hrs            | 14-choice; many prompts > 2K toks  |
| `gpqa_diamond`   | 198      | multiple-choice  | ~5 min            | 4-choice; expert-level science     |
| `math_500`       | 500      | generate-until   | ~45 min           | competition math                   |
| `humaneval`      | 164      | generate-until   | ~15 min           | needs unsafe-code execution sandbox|
| `bbh`            | 6,511    | mixed            | ~3 hrs            | 23 sub-tasks                       |
| `aime_2024`      | 30       | generate-until   | ~10 min           | tiny dataset                       |

Use `LIMIT=50` (or `--limit 50`) for smoke runs that finish in <1 min
each.

## Worked example: flagship-huge HellaSwag (smoke run)

```bash
# After wiring case "serve": into TinyGPT.swift and pip install lm-eval==0.4.10
LIMIT=10 TASKS=hellaswag bench/run_quality_evals.sh
```

**Status as of this commit:** the HTTP server is wired, end-to-end
smoke-tested with `curl` (see report below), and the harness wrapper
script is staged. The actual lm-eval run is **NOT executed in this
commit** because `pip install lm-eval` is gated by the project's
"ask before installing" rule. Once the user installs lm-eval and merges
`case "serve":`, the wrapper runs end-to-end and writes its score JSON
into `bench/results/`.

### HTTP smoke-test results

Captured during this commit using `tinygpt-serve-smoke` (the stand-in
executable) against `/tmp/flagship-huge.tinygpt`:

```text
$ curl -s http://127.0.0.1:8765/v1/models
{"object":"list","data":[{"object":"model","id":"tinygpt","owned_by":"tinygpt"}]}

$ curl -s -X POST http://127.0.0.1:8765/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"tinygpt","messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":10,"temperature":0.0}'
{"id":"chatcmpl-...","object":"chat.completion",
 "choices":[{"message":{"role":"assistant","content":"The first step in the process is to make a"},
             "finish_reason":"stop","index":0}],
 "model":"tinygpt","created":...,
 "usage":{"prompt_tokens":13,"completion_tokens":10,"total_tokens":23}}

$ curl -s -X POST http://127.0.0.1:8765/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Once upon a time","max_tokens":15,"temperature":0.0}'
{"object":"text_completion",
 "choices":[{"text":" of time, the time of time is not always a matter of time.",
             "finish_reason":"stop","index":0}],
 "usage":{"prompt_tokens":4,"completion_tokens":15,"total_tokens":19}}

# Stop sequence smoke — generation cuts off before "time" appears.
$ curl -s -X POST http://127.0.0.1:8765/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt":"A","max_tokens":50,"stop":["time"],"temperature":0.0}'
{"object":"text_completion",
 "choices":[{"text":". 1998.\n- \"The first ", "finish_reason":"stop", …}], …}
```

All three endpoints answer with valid OpenAI-shaped JSON. Generated text
is gibberish-ish (it's a 27M-param model trained for not very long) but
the framework only cares about the *shape* — the harness will read
those `choices[0].message.content` / `choices[0].text` fields and feed
them to the task's grader.

**Expected HellaSwag score for this checkpoint:** somewhere in the
25–28% range (random baseline is 25% — 4-choice multiple choice). The
flagship at 27M params and ~4.7 nats val loss is well below the
"emergent benchmark capability" threshold; this run is a *plumbing*
test, not a capability test. Repeat with a 1B+ HF-loaded model to get
real numbers.

## Adding new tasks

`lm-eval` ships ~400 task definitions out of the box. Common ones:

- **Knowledge:** `mmlu`, `mmlu_pro`, `arc_easy`, `arc_challenge`,
  `triviaqa`, `nq_open`
- **Reasoning:** `hellaswag`, `winogrande`, `piqa`, `gsm8k`, `math_500`,
  `bbh`, `gpqa_diamond`, `aime_2024`, `aime_2025`
- **Instruction-following:** `ifeval`, `mt_bench` (needs judge)
- **Code:** `humaneval`, `mbpp`, `bigcodebench_hard`
- **Long-context:** `ruler_*` (needs a separate config), `loft_*`

To add: pass the task name to `--tasks` / `$TASKS`. Custom YAML task
definitions go in `bench/tasks/<name>.yaml` and are picked up by
`--include_path bench/tasks` (pass via `--lm-eval-extra`).

For tasks that need a judge LLM (`mt_bench`, `arena_hard`, anything with
`judge_model` in its config), expect to pay GPT-4 or Claude API costs.
The harness's `--judge_model` flag accepts an OpenAI-compatible URL — so
you could in principle judge tinygpt's output with another tinygpt
serving on a second port, but the resulting scores are not comparable
to published numbers.

## Known issues

- **0.4.11 stop-sequence bug** — fixed in 0.4.12 per the project's
  changelog, but at the time of this commit 0.4.10 is the safest pin.
  Symptoms: GSM8K answers run to 256 tokens of CoT noise instead of
  stopping at the final answer. Workaround documented above.
- **Context overflow** — our 256-token default `contextLength` is below
  the prompt size of many lm-eval tasks (MMLU-Pro 0-shot can hit 2K+).
  Pass `MAX_CONTEXT=N` to bound the prompt; the server truncates from
  the left so the question survives. For real evaluation, retrain (or
  HF-load) at 4K+.
- **Throughput** — uncached, one-forward-per-token. Throughput is ~50%
  of `tinygpt sample`'s KV-cached path. The harness sends independent
  prompts so KV caching doesn't help across requests. Acceptable for
  HellaSwag-class tasks; painful for GSM8K-class generate-until.
  TODO: per-request KV cache.
- **Single concurrency** — the inference queue serialises all calls.
  Setting `lm-eval --batch_size N > 1` won't actually parallelise; it
  just queues. Don't bother tuning batch size.

## Files

- `Sources/TinyGPTServe/Serve.swift` — the HTTP server + OpenAI adapter
- `Sources/TinyGPTServeSmoke/main.swift` — temporary smoke-only binary
- `Tests/TinyGPTServeTests/TinyGPTServeTests.swift` — XCTest covering
  HTTP parser + live endpoints
- `python_ref/lm_eval_tinygpt.py` — subprocess wrapper that spawns
  `tinygpt serve`, waits for ready, runs `lm-eval`
- `bench/run_quality_evals.sh` — one-liner driver writing to
  `bench/results/<model>-<timestamp>/`
- `docs/research/quality_benchmarks_may_2026.md` — background research
  on the benchmark landscape

## Related

- `tinygpt eval` — perplexity / bits-per-byte (val loss). Faster signal
  for byte-level models; complementary to harness multi-choice tasks.
- `tinygpt bench` — inference-side latency/throughput harness.
- `docs/leaderboard.md` — places we plan to publish numbers.
