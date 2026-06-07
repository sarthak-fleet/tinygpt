# Cookbook - TinyGPT with smolagents

What you get:

- A TinyGPT specialist served through the OpenAI-compatible API.
- A Hugging Face smolagents agent pointed at that local endpoint.
- One tool-calling demo you can run from a clean clone.
- A benchmark command to compare the specialist against a general model.
- A clear limit: this does not train the specialist for you.

What this is not: a claim that a tiny model beats a frontier model on broad
agent tasks. This is for narrow specialists: function calls, schemas, repo
idioms, and other repeated local patterns.

## 1. Train Or Pick A Specialist

For function calling, use the distillation flow in
[`distillation-fc.md`](distillation-fc.md). The short version is:

```bash
tinygpt distill \
  --teacher microsoft/Phi-3-mini-4k-instruct \
  --student ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt \
  --task function-calling \
  --out /tmp/tinygpt-fc-specialist.tinygpt
```

If you already have a checkpoint, set:

```bash
export TINYGPT_MODEL=/tmp/tinygpt-fc-specialist.tinygpt
```

## 2. Serve The Model

```bash
tinygpt serve "$TINYGPT_MODEL" --host 127.0.0.1 --port 8080
```

TinyGPT exposes `/v1/chat/completions`, `/v1/completions`, and `/v1/models`.
smolagents can use either `OpenAIServerModel` or `LiteLLMModel`; this recipe
uses `OpenAIServerModel` because it directly accepts `api_base`.

## 3. Run smolagents

Install:

```bash
python3 -m pip install smolagents openai
```

Minimal agent:

```python
from smolagents import CodeAgent, OpenAIServerModel, tool

model = OpenAIServerModel(
    model_id="tinygpt",
    api_base="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

@tool
def lookup_status(ticket_id: str) -> str:
    """Return the status for a support ticket."""
    return {"A-100": "ready", "B-200": "blocked"}.get(ticket_id, "unknown")

agent = CodeAgent(tools=[lookup_status], model=model)
print(agent.run("Check ticket A-100 and answer in one sentence."))
```

Runnable example:

```bash
examples/smolagents-tinygpt/run.sh
```

## 4. Benchmark

Use the same prompt set for the specialist and the general baseline:

```bash
tinygpt run-bench \
  --model "$TINYGPT_MODEL" \
  --tasks tool-routing \
  --limit 50 \
  --out docs/artifacts/smolagents-specialist-bench.jsonl
```

Record:

| Model | Exact tool call | Invalid JSON | Mean latency |
|---|---:|---:|---:|
| TinyGPT specialist | fill after run | fill after run | fill after run |
| General baseline | fill after run | fill after run | fill after run |

## Honest Limitations

- smolagents will only be as good as the model's tool-call formatting.
- Native TinyGPT serve is single-process and best for local demos/evals.
- Do not claim a win until the benchmark table is filled with real numbers.

See also: [Pydantic AI](cookbook-pydantic-ai.md) and
[personal code specialist](cookbook-personal-code-specialist.md).
