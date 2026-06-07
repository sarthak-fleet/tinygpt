# Cookbook - TinyGPT with Pydantic AI

What you get:

- A TinyGPT specialist behind Pydantic AI's OpenAI-compatible provider.
- A typed result model that validates the response.
- One runnable local demo.
- A structured-output benchmark template.
- A clear limit: this is not a replacement for broad reasoning models.

What this is not: automatic correctness. Pydantic validates shape; your eval
still has to measure whether the values are right.

## 1. Pick The Specialist

Structured output works best when the model was trained on the schema family
you care about. Function-calling and constrained-output data are the closest
starting point:

```bash
export TINYGPT_MODEL=/tmp/tinygpt-structured-specialist.tinygpt
```

## 2. Serve TinyGPT

```bash
tinygpt serve "$TINYGPT_MODEL" --host 127.0.0.1 --port 8080
```

## 3. Configure Pydantic AI

Pydantic AI's `OpenAIProvider` accepts a custom `base_url`, so TinyGPT can be
used like any other OpenAI-compatible server:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

class TicketRoute(BaseModel):
    team: str
    priority: int
    summary: str

model = OpenAIChatModel(
    "tinygpt",
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
    ),
)

agent = Agent(model, output_type=TicketRoute)
result = agent.run_sync(
    "Route this ticket: customer cannot log in after SSO migration."
)
print(result.output)
```

Runnable example:

```bash
examples/pydantic-ai-tinygpt/run.sh
```

## 4. Benchmark

Measure schema compliance and semantic accuracy separately:

```bash
tinygpt run-bench \
  --model "$TINYGPT_MODEL" \
  --tasks json-schema-routing \
  --limit 50 \
  --out docs/artifacts/pydantic-ai-specialist-bench.jsonl
```

Record:

| Model | Valid schema | Correct route | Mean latency |
|---|---:|---:|---:|
| TinyGPT specialist | fill after run | fill after run | fill after run |
| General baseline | fill after run | fill after run | fill after run |

## Honest Limitations

- Pydantic AI can validate the object, but it cannot make a weak model smart.
- Some OpenAI-compatible servers need profile tweaks for strict schemas; keep
  TinyGPT prompts simple until the constrained-decoding path is wired into
  serve.
- Do not publish the benchmark table until it contains real numbers.

See also: [smolagents](cookbook-smolagents.md) and
[personal code specialist](cookbook-personal-code-specialist.md).
