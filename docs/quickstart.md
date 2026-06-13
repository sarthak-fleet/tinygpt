# `tinygpt quickstart` — your first specialist in one command

`quickstart` turns a data file into a trained, runnable specialist on your
Mac with **one command and zero ML knowledge**: it inspects the data,
auto-picks a base from the gallery, infers a LoRA recipe, trains, and
samples the result so you can see whether it helped. It is the CLI sibling
of the Mac app's Factory tab (B6) and shares its decision core
(`RecipeResolver`) with it.

Everything runs on-device. No account, no cloud upload — the only network
is the initial base-model pull.

## One command

```bash
tinygpt quickstart mydata.jsonl --yes
```

That inspects `mydata.jsonl`, resolves a `(base, recipe)`, trains an adapter
to `adapter.lora`, writes a reproducible `tinygpt.project.json`, and prints
a few sample completions from the new specialist.

See the plan **without training** first:

```bash
tinygpt quickstart mydata.jsonl --dry-run
```

## What it accepts

`quickstart` detects the data shape from the first lines:

| Your data looks like | Detected shape | What it does |
|---|---|---|
| `{"messages":[{role,content},…]}` | `chat` | LoRA fine-tune, chatml template |
| messages with `tool_calls` / a `"tools"` key | `toolCall` | LoRA fine-tune, longer `--max-seq` |
| `{"instruction","output"}` or `{"prompt","completion"}` | `instruction` | LoRA fine-tune |
| not JSON (a plain-text corpus) | `rawText` | from-scratch pretrain (use `tinygpt train`) |

If it can't classify the data it tells you the expected formats and exits
non-zero rather than guessing silently.

## What it picks

- **Base** — the smallest gallery model whose tags match the data shape
  (e.g. a `tool`/`agent`-tagged base for tool-call data), preferring smaller
  models so it fits a laptop. Override with `--base <gallery-id | path | hf-id>`.
- **Recipe** — LoRA rank and step budget scale with dataset size
  (`<500` rows → r8/300 steps, `<5000` → r16/800, else r32/1500); `α = 2·rank`,
  `lr 2e-4`, sequence packing on, NEFTune on. The resolved recipe is printed
  before any training so you can eyeball it.

The full mapping lives in `RecipeResolver` (see
`native-mac/Sources/TinyGPTModel/RecipeResolver.swift`) and is unit-tested.

## Output

- `adapter.lora` — the trained adapter (`--out` to change).
- `tinygpt.project.json` — base + adapter pins so the result is reproducible
  and shippable (the B31 project-pin format; passes `tinygpt validate`).

## Flags

| Flag | Meaning |
|---|---|
| `--dry-run` | print the resolved plan + project file; train nothing |
| `--base <id\|path\|hf-id>` | override the auto-picked base |
| `--gallery <path>` | gallery `manifest.json` (default: `./gallery/manifest.json`) |
| `--out <path>` | adapter output (default `adapter.lora`) |
| `--samples <N>` | demo samples after training (default 3) |
| `--yes`, `-y` | skip the train confirmation |

## Verify (no GPU)

```bash
bash evals/quickstart-smoke.sh
```

Asserts the `--dry-run` plan contract against fixture data: chat data picks
the chat base, tool-call data picks the tool base with `max-seq=2048`, the
project preview carries an adapter pin, and a missing file exits non-zero.

## Limits (V1)

- **From-scratch (raw-text) training** isn't wired into `quickstart` yet —
  it prints the plan but routes you to `tinygpt train`.
- **Auto-pull of a bare gallery id** isn't wired: if the auto-picked base is
  a gallery id with no local weights, pass `--base <local-path-or-hf-id>`
  (or `tinygpt pull` it first). Paths and HF ids train directly.
- **Single SFT pass** — no SFT→DPO→quantize chains yet (the recipe resolver
  can grow stages later). See `docs/prds/B33-laptop-finetune-onboarding.md`.
