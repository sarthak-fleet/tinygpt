# Cookbook - Character Specialist

What you get:

- A zero-API-spend path for an NPC, persona, or brand-voice specialist.
- Local teacher labeling through an OpenAI-compatible server such as LM Studio.
- A merged SFT JSONL for LoRA training.
- A lightweight eval rubric that catches boring, off-character, or malformed replies.

What this is not: a magic 70B replacement. The goal is a small specialist that
stays in character on a narrow world better and faster than the base model.

## 1. Pick The Student

Use a small instruct base with a normal HF tokenizer:

```bash
tinygpt download-model Qwen/Qwen2.5-0.5B
```

Useful alternatives:

```text
Qwen/Qwen2.5-1.5B        higher quality, slower
HuggingFaceTB/SmolLM2-360M-Instruct        faster, weaker
```

Keep the base path in a variable:

```bash
export STUDENT_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/<HASH>"
```

## 2. Pull Public Dialogue Data

Start with permissive public roleplay/dialogue datasets:

```bash
mkdir -p .tinygpt/character

tinygpt download-dataset chimbiwide/RolePlay-NPC \
  --out .tinygpt/character/roleplay-npc.jsonl

tinygpt download-dataset agentlans/multi-character-dialogue \
  --out .tinygpt/character/multi-character-dialogue.jsonl

tinygpt download-dataset NousResearch/CharacterCodex \
  --out .tinygpt/character/character-codex.jsonl
```

Avoid commercially risky scraped datasets unless this is research-only.

## 3. Write Character Prompts

Create prompts that describe the world state, character, recent memory, and
the user turn. Example:

```jsonl
{"prompt":"character: mara, a tired blacksmith in aliveville\nmood: amused but guarded\nscene: the player asks why the forge is closed after sunset\nreply in mara's voice, one or two sentences."}
{"prompt":"character: io, a curious shopkeeper\nmood: excited\nscene: the player asks about rumors near the old bridge\nreply in io's voice, one or two sentences."}
```

Save them to:

```text
.tinygpt/character/character-prompts.jsonl
```

## 4. Label With A Local Teacher

Run a local teacher in LM Studio or any OpenAI-compatible server. Then label:

```bash
tinygpt synthesize \
  --teacher http://127.0.0.1:1234/v1 \
  --teacher-model qwen2.5-72b-instruct-q4 \
  --inputs .tinygpt/character/character-prompts.jsonl \
  --input-field prompt \
  --temperature 0.8 \
  --parallel 2 \
  --rate-limit 20 \
  --out .tinygpt/character/synthetic-character.jsonl
```

For a fast first pass, use 500-1000 prompts. For a stronger specialist, use
10K+ prompts and let the local teacher run overnight.

## 5. Merge And Clean

Use existing shell tooling to merge rows, then TinyGPT cleanup commands to
remove obvious duplicates and unsafe text:

```bash
cat \
  .tinygpt/character/roleplay-npc.jsonl \
  .tinygpt/character/multi-character-dialogue.jsonl \
  .tinygpt/character/character-codex.jsonl \
  .tinygpt/character/synthetic-character.jsonl \
  > .tinygpt/character/merged-raw.jsonl

tinygpt filter \
  --in .tinygpt/character/merged-raw.jsonl \
  --out .tinygpt/character/merged-filtered.jsonl \
  --toxicity

tinygpt dedupe \
  .tinygpt/character/merged-filtered.jsonl \
  --out .tinygpt/character/merged-sft.jsonl
```

If a downloaded dataset is not already SFT-shaped, convert it before merge.
The minimal target shape is one JSON object per line with prompt/input text and
the desired assistant response.

## 6. Train LoRA

```bash
tinygpt sft "$STUDENT_DIR" \
  --data .tinygpt/character/merged-sft.jsonl \
  --template chatml \
  --rank 16 \
  --alpha 32 \
  --neftune-alpha 5 \
  --steps 2000 \
  --lr 1e-4 \
  --out .tinygpt/character/character-specialist.lora
```

Start with a 100-step smoke before a long run:

```bash
tinygpt sft "$STUDENT_DIR" \
  --data .tinygpt/character/merged-sft.jsonl \
  --template chatml \
  --rank 16 \
  --alpha 32 \
  --steps 100 \
  --lr 1e-4 \
  --out .tinygpt/character/smoke.lora
```

## 7. Serve

```bash
tinygpt serve "$STUDENT_DIR" \
  --lora .tinygpt/character/character-specialist.lora \
  --host 127.0.0.1 \
  --port 8080 \
  --prompt-cache-dir .tinygpt/character/prompt-cache
```

The prompt cache matters for character apps because the same character card and
world rules usually repeat on every turn.

## 8. Eval

Create held-out prompts:

```jsonl
{"prompt":"mara sees the player carrying broken armor. reply with concern, not exposition.","character":"mara","must_contain":["armor"],"must_not_contain":["as an ai"]}
{"prompt":"io is asked about the old bridge. reply with curiosity and one rumor.","character":"io","must_contain":["bridge"],"must_not_contain":["i don't have"]}
```

Run the base and LoRA through the same prompts, then compare by hand or with a
small rubric script:

```bash
tinygpt synthesize \
  --teacher http://127.0.0.1:8080/v1 \
  --teacher-model tinygpt \
  --inputs .tinygpt/character/eval-prompts.jsonl \
  --input-field prompt \
  --temperature 0.7 \
  --out .tinygpt/character/eval-lora.jsonl
```

Score each row on:

```text
in_character: 0-2
world_consistency: 0-2
specificity: 0-2
voice_naturalness: 0-2
format_ok: 0-1
```

The LoRA is worth keeping only if it beats the base on character voice and
specificity without losing format discipline.

## 9. Iterate

Common fixes:

- Boring replies: add more synthetic scenarios with conflict, goals, and memory.
- Breaks character: add negative examples and tighten the system prompt.
- Too much lore dumping: cap response length in the training examples.
- Repeats catchphrases: dedupe harder and raise data diversity.
- Unsafe or private data: rerun `tinygpt filter --toxicity --sidecar`.

## Current CLI Gaps

This cookbook uses existing commands. Three useful follow-ups remain:

- `synthesize --teacher-local`: load a local teacher in-process instead of via LM Studio.
- `sft-format-merge`: normalize mixed public datasets into one SFT JSONL.
- `eval rubric`: first-class rubric scoring for base-vs-LoRA comparisons.
