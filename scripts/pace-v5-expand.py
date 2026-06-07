#!/usr/bin/env python3
"""Expand v5 gold → SFT training data.

For each gold (user, elements, gold_response), generate N paraphrases of
the user line via the teacher (LM Studio qwen3-30b). The elements + gold
response stay the same; only the user phrasing varies. This teaches the
student to handle phrasing variety while always emitting the perfect
gold response.

Also add 100+ samples from clickyLocal/evals/intent-corpus/seed.jsonl
mapped to template responses (no screen state).

Output: ~/.cache/tinygpt/datasets/pace-v5-sft.jsonl
"""
import json, re, requests, sys
from pathlib import Path

GOLD = Path.home() / ".cache/tinygpt/datasets/pace-v5-gold.jsonl"
INTENT = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/intent-corpus/seed.jsonl")
OUT = Path.home() / ".cache/tinygpt/datasets/pace-v5-sft.jsonl"
TEACHER_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen/qwen3-30b-a3b"

VARIANTS_PER_GOLD = 10

def paraphrase(user_line, n):
    body = {
        "model": MODEL, "temperature": 0.9, "max_tokens": 600,
        "messages": [
            {"role":"system","content":"Generate paraphrases of a short voice command, keeping the same intent. One per line, no numbering, no extra text. Casual conversational tone."},
            {"role":"user","content":user_line},
        ],
    }
    try:
        r = requests.post(TEACHER_URL, json=body, timeout=120)
        text = r.json()["choices"][0]["message"]["content"]
        out = []
        for ln in text.split("\n"):
            ln = ln.strip().lstrip("0123456789.-•* '\"").strip().rstrip("'\"")
            if len(ln) > 4 and len(ln) < 200:
                out.append(ln)
        return out[:n]
    except Exception as e:
        print(f"  ! {e}", file=sys.stderr); return []


def intent_to_response(intent):
    """Map intent label → canonical gold response (no screen, generic)."""
    canned = {
        "screenDescription": '{"spokenText":"this screen has a few buttons and a text area you can interact with","pointAtElementId":-1,"clickElementId":-1}',
        "pureKnowledge": '{"spokenText":"happy to help with that, what specifically do you want to know","pointAtElementId":-1,"clickElementId":-1}',
        "identityProbe": '{"spokenText":"i\'m pace","pointAtElementId":-1,"clickElementId":-1}',
        "siriProbe": '{"spokenText":"i\'m pace, not siri","pointAtElementId":-1,"clickElementId":-1}',
        "clickTarget": '{"spokenText":"i don\'t see that on the screen right now","pointAtElementId":-1,"clickElementId":-1}',
        "actionKey": "pressing the key [KEY:cmd+s]",
        "actionType": "typing [TYPE:hello]",
        "actionScroll": "scrolling [SCROLL:down]",
        "actionClick": "clicking [CLICK:400,40]",
        "actionOpenApp": "opening [OPEN_APP:Safari]",
        "compoundCommand": "doing that for you",
        "ambiguousTarget": '{"spokenText":"a couple things match — which one did you mean","pointAtElementId":-1,"clickElementId":-1}',
        "targetNotPresent": '{"spokenText":"i don\'t see that on this screen","pointAtElementId":-1,"clickElementId":-1}',
    }
    return canned.get(intent)


def main():
    out_rows = []

    # === Step 1: paraphrase each gold ===
    golds = [json.loads(l) for l in open(GOLD)]
    print(f"Expanding {len(golds)} gold labels × {VARIANTS_PER_GOLD} paraphrases each...")
    for i, g in enumerate(golds):
        # original
        out_rows.append({"instruction": g["input"], "response": g["output"]})

        # extract user_line from input
        user_match = re.search(r"user said:\s*(.+)", g["input"])
        if not user_match: continue
        user_line = user_match.group(1).strip()
        # find the elements block (everything before "user said:")
        prefix = g["input"][:user_match.start()]

        print(f"  [{i+1}/{len(golds)}] {g['_fixture']}: paraphrase '{user_line[:40]}'", flush=True)
        for v in paraphrase(user_line, n=VARIANTS_PER_GOLD):
            new_input = prefix + f"user said: {v}"
            out_rows.append({"instruction": new_input, "response": g["output"]})

    # === Step 2: intent-corpus samples ===
    if INTENT.exists():
        ic = [json.loads(l) for l in open(INTENT)]
        print(f"\nAdding {len(ic)} intent-corpus samples (with canned responses)...")
        skipped = 0
        for row in ic:
            resp = intent_to_response(row["intent"])
            if resp is None:
                skipped += 1; continue
            out_rows.append({
                "instruction": f"user said: {row['transcript']}",
                "response": resp,
            })
        print(f"  skipped {skipped} unmapped intents")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in out_rows: f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out_rows)} SFT rows → {OUT}")


if __name__ == "__main__":
    main()
