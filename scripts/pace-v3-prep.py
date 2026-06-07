#!/usr/bin/env python3
"""pace-v3-prep.py — build v3 input pool from clickyLocal's actual
fm-fixtures (production format) + expand via teacher prompt variation.

fm-fixtures use the production prompt format:
    USER: <natural language>
    ELEMENT: [N] role|x,y|label|text
    ...
    EXPECT_*: ...

We extract USER + ELEMENT blocks from each fixture, then ask the
teacher for 8 paraphrases per scenario, also generating realistic
NEW screens with element lists.

Output: ~/.cache/tinygpt/datasets/pace-prompts-v3.jsonl
Each row: {"prompt": "<full user turn with element context>", "category": "<fixture name>", "source": "..."}
"""
import json
import re
import requests
import sys
from pathlib import Path

FM_FIX_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures")
OUT = Path.home() / ".cache" / "tinygpt" / "datasets" / "pace-prompts-v3.jsonl"
TEACHER_URL = "http://127.0.0.1:1234/v1/chat/completions"
TEACHER_MODEL = "qwen/qwen3-30b-a3b"


def parse_fixture(text: str) -> dict | None:
    """Parse an fm-fixture .txt file into {user, elements, free_text}."""
    user_match = re.search(r"^USER:\s*(.+?)$", text, re.MULTILINE)
    elements = re.findall(r"^ELEMENT:\s*(.+?)$", text, re.MULTILINE)
    free_text = "true" in (re.search(r"^FREE_TEXT_MODE:\s*(\S+)", text, re.MULTILINE) or [""])[0].lower() if re.search(r"^FREE_TEXT_MODE:", text, re.MULTILINE) else False
    if not user_match:
        return None
    return {
        "user": user_match.group(1).strip(),
        "elements": elements,
        "free_text": free_text,
    }


def build_user_turn(fx: dict) -> str:
    """Reconstruct what Pace actually sends as the user message."""
    parts = []
    if fx["elements"]:
        parts.append("on-screen elements:")
        for e in fx["elements"]:
            parts.append(e)
        parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


def expand_user_intent(user_text: str, n: int = 6) -> list[str]:
    """Ask teacher for N paraphrases of a user request, preserving intent."""
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system",
             "content": "generate paraphrases of the following voice command, keeping the same intent. one paraphrase per line, no numbering."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.85,
        "max_tokens": 400,
    }
    try:
        r = requests.post(TEACHER_URL, json=payload, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        lines = [ln.strip().lstrip("0123456789.-•* ").strip()
                 for ln in text.split("\n") if ln.strip()]
        return [ln for ln in lines if len(ln) > 4][:n]
    except Exception as e:
        print(f"  ! expand failed: {e}", file=sys.stderr)
        return []


def main():
    if not FM_FIX_DIR.exists():
        print(f"error: {FM_FIX_DIR} not found", file=sys.stderr)
        sys.exit(1)

    fixtures = sorted(FM_FIX_DIR.glob("*.txt"))
    print(f"loaded {len(fixtures)} fm-fixtures")

    out_rows = []
    for i, fx_path in enumerate(fixtures):
        text = fx_path.read_text()
        fx = parse_fixture(text)
        if fx is None:
            print(f"  skip {fx_path.name} (no USER:)")
            continue
        cat = fx_path.stem
        # Original
        prompt = build_user_turn(fx)
        out_rows.append({"prompt": prompt, "category": cat, "source": "fm_fixture"})
        # Paraphrases — same elements, varied user text
        print(f"[{i+1}/{len(fixtures)}] {cat}: expanding...", flush=True)
        variants = expand_user_intent(fx["user"], n=6)
        for v in variants:
            new_fx = dict(fx, user=v)
            out_rows.append({"prompt": build_user_turn(new_fx),
                            "category": cat, "source": "v3_paraphrase"})

    # Also add no-screen Q&A scenarios (case A): no element list
    qa_seeds = [
        "what is html?", "explain css", "what is python?",
        "tell me about transformers", "how does git work?",
        "what's the difference between javascript and typescript?",
        "describe the http protocol", "what is a database index?",
        "how does dns resolve a domain?", "explain machine learning",
        "what is recursion", "how do hash tables work",
    ]
    for q in qa_seeds:
        out_rows.append({"prompt": f"user said: {q}", "category": "qa-no-screen",
                        "source": "v3_qa_seed"})
        for v in expand_user_intent(q, n=4):
            out_rows.append({"prompt": f"user said: {v}", "category": "qa-no-screen",
                            "source": "v3_qa_paraphrase"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    by_cat = {}
    for r in out_rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"\nwrote {len(out_rows)} prompts → {OUT}")
    print("by category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:35s} {count}")


if __name__ == "__main__":
    main()
