# Cookbook - Personal Code Specialist

What you get:

- A per-repo corpus extraction flow.
- A TinyGPT checkpoint served as an OpenAI-compatible local coding model.
- Continue.dev and Aider config snippets.
- A small benchmark template for repo-pattern completions.
- A clear limit: this is not GPT-4 for general coding.

What this is not: an editor plugin. Continue.dev and Aider already know how
to call OpenAI-compatible local servers; TinyGPT only has to serve the model.

## 1. Build A Repo Corpus

The example script walks the current repo, keeps source/doc files, and emits
JSONL rows with `path` and `text`. Run:

```bash
examples/repo-specialist-cli/run.sh corpus .
```

Output:

```text
.tinygpt/repo-specialist/corpus.jsonl
```

## 2. Train Or Fine-Tune

Use the corpus as the domain data for a specialist run. The exact command
depends on the base model you are using; keep it small and eval-driven:

```bash
tinygpt sft \
  --base ~/.cache/tinygpt/runs/huge-base-v1/huge-base-v1.tinygpt \
  --corpus .tinygpt/repo-specialist/corpus.jsonl \
  --steps 2000 \
  --out .tinygpt/repo-specialist/model.tinygpt
```

If you already have a checkpoint:

```bash
export TINYGPT_MODEL=.tinygpt/repo-specialist/model.tinygpt
```

## 3. Serve It

```bash
tinygpt serve "$TINYGPT_MODEL" --host 127.0.0.1 --port 8080
```

## 4. Continue.dev

Continue's OpenAI provider accepts `apiBase`. Add this to
`~/.continue/config.yaml`:

```yaml
name: TinyGPT Local Specialist
version: 0.0.1
schema: v1
models:
  - name: TinyGPT Repo Specialist
    provider: openai
    model: tinygpt
    apiBase: http://127.0.0.1:8080/v1
    apiKey: not-needed
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - tool_use
```

Generate the snippet:

```bash
examples/repo-specialist-cli/run.sh continue
```

## 5. Aider

Aider can connect to an OpenAI-compatible endpoint with environment
variables:

```bash
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=not-needed
aider --model openai/tinygpt
```

Generate the snippet:

```bash
examples/repo-specialist-cli/run.sh aider
```

## 6. Benchmark

Create a held-out file of repo-specific prompts:

```json
{"prompt":"Add a route matching the existing Astro page style.","expected_contains":"frontmatter"}
{"prompt":"Write a Swift CLI subcommand using the existing run(args:) pattern.","expected_contains":"static func run(args:"}
```

Then run:

```bash
tinygpt run-bench \
  --model "$TINYGPT_MODEL" \
  --tasks custom-code-patterns \
  --limit 50 \
  --out docs/artifacts/repo-specialist-bench.jsonl
```

Record:

| Model | Pattern match | Compile/test pass | Mean latency |
|---|---:|---:|---:|
| TinyGPT repo specialist | fill after run | fill after run | fill after run |
| General coding baseline | fill after run | fill after run | fill after run |

## Honest Limitations

- This helps with local idioms, naming, and repetitive patterns. It will not
  become a frontier code model.
- Bad repo data makes a bad specialist. Exclude generated files and vendored
  dependencies.
- Keep the first eval tiny and concrete before scaling training.

See also: [smolagents](cookbook-smolagents.md) and
[Pydantic AI](cookbook-pydantic-ai.md).
