#!/usr/bin/env python3
"""Extract Pace user prompts from clickyLocal fixtures + synthesize variants.

For each fixture in clickyLocal/evals/fixtures/, pull the user message
and create N variants (paraphrases of the same intent). Output is a
JSONL ready for `tinygpt synthesize` to label via LM Studio teacher.

Each row: {"prompt": "<user message>", "category": "<fixture name>"}
"""
import json
import sys
from pathlib import Path

FIXTURE_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fixtures")
OUT = Path.home() / ".cache" / "tinygpt" / "datasets" / "pace-prompts.jsonl"

# Seed variants per category — minimal hand-curated set. Tomorrow's
# `tinygpt synthesize` will dramatically expand this by letting the
# teacher generate more under the same prompt scaffolding.
VARIANTS = {
    "qa-no-screen": [
        "what is html?", "explain css", "what is python?",
        "tell me about transformers", "how does git work?",
        "what's the difference between javascript and typescript?",
        "describe the http protocol", "what is a database index?",
        "how does dns resolve a domain?", "explain machine learning",
    ],
    "screen-referential": [
        "save it for me", "click that save button", "open this file",
        "scroll down to the end", "what does this dialog say?",
        "close that error", "fix the typo on the screen",
        "summarize what i'm looking at", "translate this text",
        "what's this button do?",
    ],
    "multi-turn-continuation": [
        "tell me more", "what about the second one?", "give me an example",
        "but why?", "and then?", "what's the alternative?",
        "summarize that in one sentence", "show me the code",
        "which one should i pick?", "is there a faster way?",
    ],
    "action-mode-off": [
        "save the document", "click save", "press enter",
        "open the menu", "type my name", "scroll to top",
        "select all text", "copy this", "paste it here",
        "minimize this window",
    ],
}


def main():
    if not FIXTURE_DIR.exists():
        print(f"error: clickyLocal fixtures not at {FIXTURE_DIR}", file=sys.stderr)
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for fx_path in sorted(FIXTURE_DIR.glob("*.json")):
        with fx_path.open() as f:
            fx = json.load(f)
        name = fx_path.stem
        # Original user prompt from the fixture
        msgs = fx.get("request", {}).get("messages", [])
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        if user_msg:
            rows.append({"prompt": user_msg, "category": name, "source": "fixture"})
        # Variants
        for v in VARIANTS.get(name, []):
            rows.append({"prompt": v, "category": name, "source": "variant"})

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} prompts → {OUT}")
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print("by category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:30s} {count}")


if __name__ == "__main__":
    main()
